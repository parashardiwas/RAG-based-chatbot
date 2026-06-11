"""Model router — selects the optimal LLM based on query characteristics.

Routes requests to a *cheap* or *strong* model depending on:

* **Retrieval confidence**: high confidence → cheap model suffices.
* **Query complexity**: presence of multi-hop indicators (e.g. "compare",
  "difference between") or excessive length triggers the stronger model.

Currently both tiers point to ``llama3.1:8b`` (single local model), but the
architecture is ready for heterogeneous model pools.
"""

from __future__ import annotations

import logging
import re

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------
DEFAULT_CHEAP_MODEL: str = "command-r7b"
DEFAULT_STRONG_MODEL: str = "command-r7b"

# Retrieval confidence below this threshold triggers the strong model.
_CONFIDENCE_THRESHOLD: float = 0.7

# Query length (in whitespace-delimited tokens) above which the query is
# considered complex regardless of other signals.
_LONG_QUERY_TOKEN_COUNT: int = 40

# Regex patterns that signal multi-hop or comparative reasoning.
_COMPLEXITY_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r"\bcompare\b", re.IGNORECASE),
    re.compile(r"\bdifference\s+between\b", re.IGNORECASE),
    re.compile(r"\bcontrast\b", re.IGNORECASE),
    re.compile(r"\brelationship\s+between\b", re.IGNORECASE),
    re.compile(r"\bpros\s+and\s+cons\b", re.IGNORECASE),
    re.compile(r"\badvantages?\s+and\s+disadvantages?\b", re.IGNORECASE),
    re.compile(r"\banalyze\b", re.IGNORECASE),
    re.compile(r"\banalyse\b", re.IGNORECASE),
    re.compile(r"\bevaluate\b", re.IGNORECASE),
    re.compile(r"\bsummarize\b", re.IGNORECASE),
    re.compile(r"\bsummarise\b", re.IGNORECASE),
    re.compile(r"\bhow\s+does\s+.+\s+differ\b", re.IGNORECASE),
]


class ModelRouter:
    """Select the most appropriate model for a given query.

    Parameters
    ----------
    cheap_model:
        Model tag for simple, high-confidence queries.
    strong_model:
        Model tag for complex or low-confidence queries.

    Usage::

        router = ModelRouter()
        model = router.select_model(retrieval_confidence=0.85, query="What is AI?")
    """

    def __init__(
        self,
        cheap_model: str = DEFAULT_CHEAP_MODEL,
        strong_model: str = DEFAULT_STRONG_MODEL,
    ) -> None:
        self.cheap_model = cheap_model
        self.strong_model = strong_model

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def select_model(self, retrieval_confidence: float, query: str) -> str:
        """Choose a model based on retrieval confidence and query complexity.

        Decision logic:

        1. If retrieval confidence is **low** (< 0.7), always use the strong
           model — the LLM needs to work harder with weaker context.
        2. If the query contains **multi-hop indicators** or is very long,
           use the strong model even when confidence is high.
        3. Otherwise, fall back to the cheap model.

        Returns
        -------
        str
            Model tag.
        """
        is_complex = self._is_complex_query(query)

        if retrieval_confidence < _CONFIDENCE_THRESHOLD:
            logger.debug(
                "Low retrieval confidence (%.2f < %.2f) → strong model '%s'.",
                retrieval_confidence,
                _CONFIDENCE_THRESHOLD,
                self.strong_model,
            )
            return self.strong_model

        if is_complex:
            logger.debug(
                "Complex query detected → strong model '%s'.", self.strong_model
            )
            return self.strong_model

        logger.debug(
            "High confidence (%.2f) + simple query → cheap model '%s'.",
            retrieval_confidence,
            self.cheap_model,
        )
        return self.cheap_model

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    @staticmethod
    def _is_complex_query(query: str) -> bool:
        """Heuristically detect multi-hop or complex queries.

        Checks for keyword patterns and excessive query length.
        """
        # Long queries are often multi-part.
        if len(query.split()) >= _LONG_QUERY_TOKEN_COUNT:
            return True

        # Check for multi-hop / comparative keywords.
        for pattern in _COMPLEXITY_PATTERNS:
            if pattern.search(query):
                return True

        return False
