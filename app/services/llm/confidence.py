"""Confidence scoring — validates that generated answers are grounded.

Computes cosine similarity between the answer embedding and the best
source embedding, combined with retrieval-stage scores, to produce an
overall :class:`ConfidenceResult`.

Threshold default: 0.80 (80 % match).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class ConfidenceResult:
    """Output of the confidence-scoring pipeline."""

    answer_confidence: float
    retrieval_confidence: float
    meets_threshold: bool
    details: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# ConfidenceScorer
# ---------------------------------------------------------------------------


class ConfidenceScorer:
    """Score how well an answer is grounded in retrieved sources.

    Usage::

        scorer = ConfidenceScorer()
        result = scorer.score_answer(
            answer_embedding=answer_emb,
            source_embeddings=[src_emb_1, src_emb_2],
            retrieval_scores=[0.92, 0.85],
        )
    """

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def score_answer(
        self,
        answer_embedding: list[float],
        source_embeddings: list[list[float]],
        retrieval_scores: list[float],
        threshold: float,
    ) -> ConfidenceResult:
        """Compute answer-level and retrieval-level confidence.

        Parameters
        ----------
        answer_embedding:
            Dense vector of the generated answer (384-d).
        source_embeddings:
            Dense vectors of the source chunks used for generation.
        retrieval_scores:
            Similarity / RRF scores from the retrieval stage.
        threshold:
            Minimum cosine similarity to consider the answer grounded
            (default 0.80).

        Returns
        -------
        ConfidenceResult
        """
        if not source_embeddings:
            return ConfidenceResult(
                answer_confidence=0.0,
                retrieval_confidence=0.0,
                meets_threshold=False,
                details={"reason": "No source embeddings provided."},
            )

        # --- answer-source similarity (best match) ---
        similarities = [
            self._cosine_similarity(answer_embedding, src_emb)
            for src_emb in source_embeddings
        ]
        best_similarity = float(max(similarities))
        best_source_idx = int(np.argmax(similarities))

        # --- retrieval confidence (average of retrieval scores) ---
        retrieval_confidence = float(np.mean(retrieval_scores)) if retrieval_scores else 0.0

        meets = self.check_threshold(best_similarity, threshold)

        return ConfidenceResult(
            answer_confidence=round(best_similarity, 4),
            retrieval_confidence=round(retrieval_confidence, 4),
            meets_threshold=meets,
            details={
                "best_source_index": best_source_idx,
                "best_source_similarity": round(best_similarity, 4),
                "all_similarities": [round(s, 4) for s in similarities],
                "threshold": threshold,
            },
        )

    # ------------------------------------------------------------------
    # Cosine similarity
    # ------------------------------------------------------------------

    @staticmethod
    def _cosine_similarity(a: list[float], b: list[float]) -> float:
        """Compute cosine similarity between two vectors using numpy.

        Returns a value in [-1, 1]. Identical directions → 1.0.
        """
        vec_a = np.asarray(a, dtype=np.float64)
        vec_b = np.asarray(b, dtype=np.float64)

        dot = np.dot(vec_a, vec_b)
        norm_a = np.linalg.norm(vec_a)
        norm_b = np.linalg.norm(vec_b)

        if norm_a == 0 or norm_b == 0:
            return 0.0

        return float(dot / (norm_a * norm_b))

    # ------------------------------------------------------------------
    # Threshold check
    # ------------------------------------------------------------------

    @staticmethod
    def check_threshold(
        confidence: float,
        threshold: float,
    ) -> bool:
        """Return ``True`` if *confidence* meets or exceeds *threshold*."""
        return confidence >= threshold
