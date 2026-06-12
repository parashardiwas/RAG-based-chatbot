"""Hybrid retrieval service — pgvector cosine search + BM25 keyword search.

Combines dense vector retrieval with sparse lexical matching via
Reciprocal Rank Fusion (RRF) to produce a single ranked list of
:class:`RetrievedChunk` objects.

Tables queried:
  * ``document_chunks`` — main knowledge-base chunks with ``embedding`` column.
  * ``qa_pairs`` — curated Q-A pairs with ``combined_embedding`` column.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Any, ClassVar
from uuid import UUID
import asyncio
import logging

logger = logging.getLogger(__name__)

from sqlalchemy import text as sa_text
from sqlalchemy.ext.asyncio import AsyncSession

# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class RetrievedChunk:
    """A single retrieval result with metadata and scoring."""

    chunk_id: Any  # UUID from database
    content: str
    similarity_score: float
    source_file: str | None = None
    topic: str | None = None
    subject: str | None = None
    source_type: str = "document_chunk"  # or "qa_pair"
    embedding: list[float] | None = None


@dataclass
class RetrievalResult:
    """Aggregated output from the retrieval pipeline."""

    chunks: list[RetrievedChunk] = field(default_factory=list)
    retrieval_confidence: float = 0.0


# ---------------------------------------------------------------------------
# SQL fragments (pgvector cosine distance operator: <=>)
# ---------------------------------------------------------------------------

_VECTOR_SEARCH_CHUNKS_SQL = """
SELECT
    dc.id AS chunk_id,
    dc.content,
    1 - (dc.embedding <=> CAST(:query_embedding AS vector)) AS similarity,
    dc.source_file,
    t.name AS topic,
    s.name AS subject,
    'document_chunk' AS source_type
FROM document_chunks dc
LEFT JOIN topics t ON dc.topic_id = t.id
LEFT JOIN subjects s ON dc.subject_id = s.id
WHERE dc.embedding IS NOT NULL
    {filters}
ORDER BY dc.embedding <=> CAST(:query_embedding AS vector)
LIMIT :top_k
"""

_VECTOR_SEARCH_QA_SQL = """
SELECT
    qa.id AS chunk_id,
    qa.question || ' ' || qa.answer AS content,
    1 - LEAST(
        qa.question_embedding <=> CAST(:query_embedding AS vector),
        qa.combined_embedding <=> CAST(:query_embedding AS vector)
    ) AS similarity,
    NULL AS source_file,
    t.name AS topic,
    s.name AS subject,
    'qa_pair' AS source_type
FROM qa_pairs qa
LEFT JOIN topics t ON qa.topic_id = t.id
LEFT JOIN subjects s ON qa.subject_id = s.id
WHERE qa.combined_embedding IS NOT NULL AND qa.is_deleted = false
    {filters}
ORDER BY LEAST(
    qa.question_embedding <=> CAST(:query_embedding AS vector),
    qa.combined_embedding <=> CAST(:query_embedding AS vector)
)
LIMIT :top_k
"""

_FTS_SEARCH_CHUNKS_SQL = """
SELECT dc.id AS chunk_id, dc.content, dc.source_file,
       t.name AS topic, s.name AS subject,
       'document_chunk' AS source_type,
       ts_rank_cd(to_tsvector('english', dc.content), websearch_to_tsquery('english', :query)) AS similarity
FROM document_chunks dc
LEFT JOIN topics t ON dc.topic_id = t.id
LEFT JOIN subjects s ON dc.subject_id = s.id
WHERE to_tsvector('english', dc.content) @@ websearch_to_tsquery('english', :query)
    {filters}
ORDER BY similarity DESC
LIMIT :top_k
"""

_FTS_SEARCH_QA_SQL = """
SELECT qa.id AS chunk_id, qa.question || ' ' || qa.answer AS content,
       NULL AS source_file, t.name AS topic, s.name AS subject,
       'qa_pair' AS source_type,
       ts_rank_cd(to_tsvector('english', qa.question || ' ' || qa.answer), websearch_to_tsquery('english', :query)) AS similarity
FROM qa_pairs qa
LEFT JOIN topics t ON qa.topic_id = t.id
LEFT JOIN subjects s ON qa.subject_id = s.id
WHERE qa.is_deleted = false
  AND to_tsvector('english', qa.question || ' ' || qa.answer) @@ websearch_to_tsquery('english', :query)
    {filters}
ORDER BY similarity DESC
LIMIT :top_k
"""


# ---------------------------------------------------------------------------
# RetrievalService
# ---------------------------------------------------------------------------


class RetrievalService:
    """Hybrid retriever: pgvector + BM25, merged via Reciprocal Rank Fusion.

    Usage::

        retriever = RetrievalService(db_session)
        result = await retriever.retrieve("What is photosynthesis?", query_emb)
    """

    _cross_encoder: ClassVar[Any] = None

    def __init__(self, db_session_factory) -> None:
        self._session_factory = db_session_factory

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def compute_confidence(self, chunks: list[RetrievedChunk]) -> float:
        """Public wrapper for confidence computation."""
        return self._compute_retrieval_confidence(chunks)

    def invalidate_bm25_cache(self) -> None:
        """Clear the in-memory BM25 index so new documents are picked up."""
        pass  # No-op, Postgres FTS handles updates instantly.

    async def retrieve(
        self,
        query: str,
        query_embedding: list[float],
        top_k: int = 5,
        subject_filter: str | None = None,
        topic_filter: str | None = None,
    ) -> list[RetrievedChunk]:
        """Run hybrid retrieval and return merged, reranked chunks.

        Parameters
        ----------
        query:
            The user's natural-language query.
        query_embedding:
            Pre-computed dense embedding for *query* (384-d).
        top_k:
            Number of final results to return.
        subject_filter / topic_filter:
            Optional metadata filters applied to both search paths.

        Returns
        -------
        RetrievalResult
            Ranked list of :class:`RetrievedChunk` objects with an overall
            ``retrieval_confidence`` score.
        """
        filters = self._build_filter_clauses(subject_filter, topic_filter)

        # Run both search strategies in parallel
        vector_results, fts_results = await asyncio.gather(
            self._vector_search(
                query_embedding, top_k=top_k * 2, filters=filters
            ),
            self._fts_search(
                query, top_k=top_k * 2, filters=filters
            )
        )

        merged = self._reciprocal_rank_fusion(vector_results, fts_results, k=60)
        
        # Cross-Encoder Reranking
        # Re-score the top 20 candidates from the hybrid search
        candidates = merged[:20]
        if candidates:
            # Lazy load the cross-encoder model
            if self.__class__._cross_encoder is None:
                from sentence_transformers import CrossEncoder
                logger.info("Loading cross-encoder model 'cross-encoder/ms-marco-MiniLM-L-6-v2' ...")
                self.__class__._cross_encoder = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2", max_length=512)
            
            # Prepare pairs of (query, chunk_content)
            pairs = [(query, chunk.content) for chunk in candidates]
            # Get logits from the cross encoder (not bound to [0,1], higher is better)
            # Run in a separate thread since inference is CPU blocking
            scores = await asyncio.to_thread(self.__class__._cross_encoder.predict, pairs)
            
            import math
            # Apply the new scores
            for i, chunk in enumerate(candidates):
                # We overwrite the similarity score for the final ranking.
                # Cross-encoder scores are logits. We apply a sigmoid function
                # to convert them to a [0, 1] probability scale which works properly
                # with the downstream confidence calculation.
                logit = float(scores[i])
                chunk.similarity_score = 1.0 / (1.0 + math.exp(-logit))
            
            # Re-sort candidates by the new cross-encoder score descending
            candidates.sort(key=lambda c: c.similarity_score, reverse=True)
            
        top_chunks = candidates[:top_k]

        return top_chunks

    async def retrieve_qa_pairs(
        self,
        query_embedding: list[float] | None = None,
        top_k: int = 1,
        subject_filter: str | None = None,
        topic_filter: str | None = None,
        query_text: str | None = None,
    ) -> list[RetrievedChunk]:
        """Fast path: check if this exact/similar question was already answered."""
        filters = self._build_filter_clauses(subject_filter, topic_filter)
        if query_embedding is not None:
            return await self._vector_search(
                query_embedding, top_k=top_k, filters=filters, target="qa"
            )
        elif query_text is not None:
            async with self._session_factory() as session:
                sql = """
                SELECT qa.id AS chunk_id, qa.question || ' ' || qa.answer AS content,
                       0.85 AS similarity_score, NULL AS source_file, t.name AS topic,
                       s.name AS subject, 'qa_pair' AS source_type
                FROM qa_pairs qa
                LEFT JOIN topics t ON qa.topic_id = t.id
                LEFT JOIN subjects s ON qa.subject_id = s.id
                WHERE qa.is_deleted = false AND qa.question ILIKE :query_text
                """ + filters["sql"] + " LIMIT :top_k"
                params = {"query_text": query_text, "top_k": top_k, **filters["params"]}
                query = sa_text(sql)
                rows = await session.execute(query, params)
                return [
                    RetrievedChunk(
                        chunk_id=row["chunk_id"],
                        content=row["content"],
                        similarity_score=float(row["similarity_score"]),
                        source_file=row["source_file"],
                        topic=row["topic"],
                        subject=row["subject"],
                        source_type=row["source_type"]
                    ) for row in rows.mappings()
                ]
        return []

    # ------------------------------------------------------------------
    # Vector search (pgvector)
    # ------------------------------------------------------------------

    async def _vector_search(
        self,
        embedding: list[float],
        top_k: int,
        filters: dict[str, Any],
        target: str = "both"
    ) -> list[RetrievedChunk]:
        """Search document_chunks and qa_pairs using pgvector cosine distance."""
        embedding_str = str(embedding)
        filter_sql = filters["sql"]
        params: dict[str, Any] = {
            "query_embedding": embedding_str,
            "top_k": top_k,
            **filters["params"],
        }

        results: list[RetrievedChunk] = []

        templates = []
        if target in ("both", "chunks"):
            templates.append(_VECTOR_SEARCH_CHUNKS_SQL)
        if target in ("both", "qa"):
            templates.append(_VECTOR_SEARCH_QA_SQL)

        async with self._session_factory() as session:
            for sql_template in templates:
                query = sa_text(sql_template.format(filters=filter_sql))
                rows = await session.execute(query, params)
                for row in rows.mappings():
                    results.append(
                        RetrievedChunk(
                            chunk_id=row["chunk_id"],
                            content=row["content"],
                            similarity_score=float(row["similarity"]),
                            source_file=row.get("source_file"),
                            topic=row.get("topic"),
                            subject=row.get("subject"),
                            source_type=row.get("source_type", "document_chunk"),
                        )
                    )

        # Sort descending by similarity.
        results.sort(key=lambda c: c.similarity_score, reverse=True)
        return results[:top_k]

    # ------------------------------------------------------------------
    # FTS keyword search
    # ------------------------------------------------------------------

    async def _fts_search(
        self,
        query: str,
        top_k: int,
        filters: dict[str, Any],
    ) -> list[RetrievedChunk]:
        """Rank documents using Postgres Full Text Search (FTS)."""
        filter_sql = filters["sql"]
        params: dict[str, Any] = {
            "query": query,
            "top_k": top_k,
            **filters["params"],
        }

        results: list[RetrievedChunk] = []
        async with self._session_factory() as session:
            for sql_template in (_FTS_SEARCH_CHUNKS_SQL, _FTS_SEARCH_QA_SQL):
                query_obj = sa_text(sql_template.format(filters=filter_sql))
                rows = await session.execute(query_obj, params)
                for row in rows.mappings():
                    results.append(
                        RetrievedChunk(
                            chunk_id=row["chunk_id"],
                            content=row["content"],
                            similarity_score=float(row["similarity"]),
                            source_file=row.get("source_file"),
                            topic=row.get("topic"),
                            subject=row.get("subject"),
                            source_type=row.get("source_type", "document_chunk"),
                        )
                    )

        # Sort descending by similarity.
        results.sort(key=lambda c: c.similarity_score, reverse=True)
        # Normalize scores to [0, 1] for RRF
        max_score = results[0].similarity_score if results else 1.0
        if max_score <= 0.0:
            max_score = 1.0
            
        for c in results:
            c.similarity_score = max(0.0, min(c.similarity_score / max_score, 1.0))
            
        return results[:top_k]

    # ------------------------------------------------------------------
    # Reciprocal Rank Fusion
    # ------------------------------------------------------------------

    @staticmethod
    def _reciprocal_rank_fusion(
        vector_results: list[RetrievedChunk],
        bm25_results: list[RetrievedChunk],
        k: int = 60,
    ) -> list[RetrievedChunk]:
        """Merge two ranked lists using Reciprocal Rank Fusion.

        RRF score for document *d*:  ``sum(1 / (k + rank_i))`` across
        each ranking that contains *d*.

        Parameters
        ----------
        k : int
            Dampening constant (default 60, per the original RRF paper).
        """
        fused_scores: dict[int, float] = {}
        chunk_map: dict[int, RetrievedChunk] = {}

        for rank, chunk in enumerate(vector_results, start=1):
            fused_scores[chunk.chunk_id] = fused_scores.get(chunk.chunk_id, 0.0) + 1.0 / (k + rank)
            chunk_map[chunk.chunk_id] = chunk

        for rank, chunk in enumerate(bm25_results, start=1):
            fused_scores[chunk.chunk_id] = fused_scores.get(chunk.chunk_id, 0.0) + 1.0 / (k + rank)
            chunk_map.setdefault(chunk.chunk_id, chunk)

        # Sort by fused score descending.
        sorted_ids = sorted(fused_scores, key=lambda cid: fused_scores[cid], reverse=True)

        merged: list[RetrievedChunk] = []
        for cid in sorted_ids:
            chunk = chunk_map[cid]
            # Keep the original absolute similarity score for confidence gating!
            # Do NOT overwrite it with the normalized RRF score, because RRF destroys semantic distance margins.
            merged.append(chunk)
            
        return merged

    # ------------------------------------------------------------------
    # Confidence computation
    # ------------------------------------------------------------------

    @staticmethod
    def _compute_retrieval_confidence(chunks: list[RetrievedChunk]) -> float:
        """Compute a retrieval confidence score in [0, 1].

        Combines:
          * **Absolute score** of the top-1 result.
          * **Score margin** between top-1 and top-2 (higher margin → more
            confident that top-1 is clearly best).
        """
        if not chunks:
            return 0.0

        top_score = chunks[0].similarity_score
        
        # Scale the raw cosine/bm25 similarity (which normally peaks around 0.4-0.6 for good doc matches)
        # to a 0.0 - 1.0 confidence range.
        # 0.20 is considered 0% confident, and 0.65 is considered 100% confident.
        scaled_score = max(0.0, min((top_score - 0.20) / (0.65 - 0.20), 1.0))

        if len(chunks) >= 2:
            margin = top_score - chunks[1].similarity_score
        else:
            margin = top_score  # only one result
            
        # Scale the margin. A margin of 0.15 in cosine space is massive.
        scaled_margin = max(0.0, min(margin / 0.15, 1.0))

        # Weighted combination: 70% absolute quality, 30% separation margin.
        confidence = 0.7 * scaled_score + 0.3 * scaled_margin
        return round(max(0.0, min(confidence, 1.0)), 4)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _build_filter_clauses(
        subject_filter: str | None,
        topic_filter: str | None,
    ) -> dict[str, Any]:
        """Build optional SQL WHERE clauses and parameter dict."""
        clauses: list[str] = []
        params: dict[str, Any] = {}

        if subject_filter:
            clauses.append("AND s.name = :subject_filter")
            params["subject_filter"] = subject_filter
        if topic_filter:
            clauses.append("AND t.name = :topic_filter")
            params["topic_filter"] = topic_filter

        return {"sql": "\n    ".join(clauses), "params": params}
