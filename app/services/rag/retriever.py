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
from typing import Any
from uuid import UUID

import numpy as np
from rank_bm25 import BM25Okapi
from sqlalchemy import text as sa_text
from sqlalchemy.ext.asyncio import AsyncSession

import asyncio
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# BM25 tokenization constants (module-level for performance)
# ---------------------------------------------------------------------------
_STOP_WORDS: frozenset[str] = frozenset({
    "what", "is", "are", "the", "a", "an", "of", "in", "to", "and", "or",
    "for", "on", "with", "how", "why", "when", "where", "who", "which",
    "do", "does", "did", "can", "could", "would", "should", "it", "its",
    "this", "that", "these", "those", "be", "been", "being", "was", "were",
    "has", "have", "had", "not", "but", "if", "so", "as", "at", "by",
    "from", "about", "into", "through", "during", "before", "after",
    "above", "below", "between", "up", "down", "out", "off", "over",
    "under", "again", "then", "once", "here", "there", "all", "each",
    "every", "both", "few", "more", "most", "other", "some", "such",
    "no", "nor", "only", "own", "same", "than", "too", "very",
    "will", "just", "don", "shall", "may", "might", "must",
})
_WORD_PATTERN = re.compile(r'\b\w+\b')


def _tokenize(text: str) -> list[str]:
    """Tokenize text for BM25: lowercase, extract words, remove stop words."""
    words = _WORD_PATTERN.findall(text.lower())
    return [w for w in words if w not in _STOP_WORDS]


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

_BM25_CORPUS_CHUNKS_SQL = """
SELECT dc.id AS chunk_id, dc.content, dc.source_file,
       t.name AS topic, s.name AS subject,
       'document_chunk' AS source_type
FROM document_chunks dc
LEFT JOIN topics t ON dc.topic_id = t.id
LEFT JOIN subjects s ON dc.subject_id = s.id
WHERE 1 = 1
    {filters}
"""

_BM25_CORPUS_QA_SQL = """
SELECT qa.id AS chunk_id, qa.question || ' ' || qa.answer AS content,
       NULL AS source_file, t.name AS topic, s.name AS subject,
       'qa_pair' AS source_type
FROM qa_pairs qa
LEFT JOIN topics t ON qa.topic_id = t.id
LEFT JOIN subjects s ON qa.subject_id = s.id
WHERE qa.is_deleted = false
    {filters}
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

    def __init__(self, db_session_factory) -> None:
        self._session_factory = db_session_factory
        # In-memory BM25 corpus cache (populated on first call).
        self._bm25_cache: dict[str, tuple[BM25Okapi, list[dict[str, Any]]]] = {}
        self._bm25_locks: dict[str, asyncio.Lock] = {}  # Per-key locks for thread-safe cache init

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def compute_confidence(self, chunks: list[RetrievedChunk]) -> float:
        """Public wrapper for confidence computation."""
        return self._compute_retrieval_confidence(chunks)

    def invalidate_bm25_cache(self) -> None:
        """Clear the in-memory BM25 index so new documents are picked up."""
        self._bm25_cache.clear()
        logger.info("BM25 cache invalidated — will rebuild on next query.")

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

        # Run both search strategies.
        vector_results = await self._vector_search(
            query_embedding, top_k=top_k * 2, filters=filters
        )
        bm25_results = await self._bm25_search(
            query, top_k=top_k * 2, filters=filters
        )

        merged = self._reciprocal_rank_fusion(vector_results, bm25_results, k=60)
        top_chunks = merged[:top_k]

        return top_chunks

    async def retrieve_qa_pairs(
        self,
        query_embedding: list[float],
        top_k: int = 1,
        subject_filter: str | None = None,
        topic_filter: str | None = None,
    ) -> list[RetrievedChunk]:
        """Fast path: check if this exact/similar question was already answered."""
        filters = self._build_filter_clauses(subject_filter, topic_filter)
        return await self._vector_search(
            query_embedding, top_k=top_k, filters=filters, target="qa"
        )

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
    # BM25 keyword search
    # ------------------------------------------------------------------

    async def _bm25_search(
        self,
        query: str,
        top_k: int,
        filters: dict[str, Any],
    ) -> list[RetrievedChunk]:
        """Rank documents using BM25 over the tokenised corpus.

        The corpus is fetched from the database on first invocation and
        cached in memory (keyed by filter fingerprint).
        
        Uses per-key locks to prevent concurrent cache initialization.
        """
        cache_key = str(sorted(filters["params"].items()))

        # Ensure lock exists for this cache key
        if cache_key not in self._bm25_locks:
            self._bm25_locks[cache_key] = asyncio.Lock()
        
        # Use lock to prevent concurrent cache initialization
        async with self._bm25_locks[cache_key]:
            if cache_key not in self._bm25_cache:
                corpus_docs = await self._load_bm25_corpus(filters)
                if not corpus_docs:
                    return []
                tokenised = [_tokenize(doc["content"]) for doc in corpus_docs]
                bm25 = BM25Okapi(tokenised)
                self._bm25_cache[cache_key] = (bm25, corpus_docs)

        bm25, corpus_docs = self._bm25_cache[cache_key]

        tokenised_query = _tokenize(query)
        if not tokenised_query:
            return []

        scores = bm25.get_scores(tokenised_query)

        # Normalise scores to [0, 1].
        max_score = float(np.max(scores)) if len(scores) > 0 else 1.0
        if max_score == 0:
            max_score = 1.0

        ranked_indices = np.argsort(scores)[::-1][:top_k]

        results: list[RetrievedChunk] = []
        for idx in ranked_indices:
            norm_score = float(scores[int(idx)]) / max_score
            # Filter out zero-score results — they have no keyword overlap
            # and only pollute RRF merge and drag down confidence.
            if norm_score < 0.01:
                continue
            doc = corpus_docs[int(idx)]
            results.append(
                RetrievedChunk(
                    chunk_id=doc["chunk_id"],
                    content=doc["content"],
                    similarity_score=norm_score,
                    source_file=doc.get("source_file"),
                    topic=doc.get("topic"),
                    subject=doc.get("subject"),
                    source_type=doc.get("source_type", "document_chunk"),
                )
            )
        return results

    async def _load_bm25_corpus(
        self, filters: dict[str, Any]
    ) -> list[dict[str, Any]]:
        """Fetch all chunks + QA pairs for BM25 indexing."""
        docs: list[dict[str, Any]] = []
        async with self._session_factory() as session:
            for sql_template in (_BM25_CORPUS_CHUNKS_SQL, _BM25_CORPUS_QA_SQL):
                query = sa_text(sql_template.format(filters=filters["sql"]))
                rows = await session.execute(query, filters["params"])
                for row in rows.mappings():
                    docs.append(dict(row))
        return docs

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
