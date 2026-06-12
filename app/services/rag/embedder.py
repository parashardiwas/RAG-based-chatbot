"""Embedding service using sentence-transformers (all-MiniLM-L6-v2).

Provides text-to-vector encoding with Redis caching to avoid redundant
computation. The model is loaded lazily on first use and shared across
the application via a singleton pattern.

Embedding dimension: 384.
"""

from __future__ import annotations

import hashlib
import json
import logging
import asyncio
from typing import ClassVar

import redis.asyncio as aioredis
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
from app.config import get_settings
EMBEDDING_DIM: int = 384
REDIS_KEY_PREFIX: str = "emb:"
CACHE_TTL_SECONDS: int = 60 * 60 * 24 * 7  # 7 days


def _cache_key(text: str) -> str:
    """Return a Redis key for the given text using SHA-256 (first 16 hex chars)."""
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()[:32]
    return f"{REDIS_KEY_PREFIX}{digest}"


class EmbeddingService:
    """Singleton wrapper around *sentence-transformers* for text embedding.

    Usage::

        svc = EmbeddingService(redis_url="redis://localhost:6379/0")
        vec = await svc.embed_text("hello world")
    """

    # ---- singleton bookkeeping ----
    _instance: ClassVar[EmbeddingService | None] = None
    _model: ClassVar[SentenceTransformer | None] = None

    def __new__(cls, *args, **kwargs) -> EmbeddingService:  # noqa: ANN002, ANN003
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self, redis_url: str = "redis://localhost:6379/0") -> None:
        # Guard against re-initialisation on repeated __init__ calls.
        if hasattr(self, "_initialised") and self._initialised:
            if hasattr(self, "_redis_url") and self._redis_url != redis_url:
                logger.warning(
                    "EmbeddingService singleton was created with a different "
                    "redis_url (%s). The original URL is being used.",
                    redis_url,
                )
            return
        self._redis_url = redis_url  # store for comparison
        self._redis: aioredis.Redis = aioredis.from_url(
            redis_url, decode_responses=True
        )
        self._lock = asyncio.Lock()
        self._initialised: bool = True
        self._load_model()

    # ------------------------------------------------------------------
    # Model lifecycle
    # ------------------------------------------------------------------

    def _load_model(self) -> None:
        """Load the sentence-transformers model (lazy, once per process)."""
        if self.__class__._model is not None:
            return
        model_name = get_settings().embedding_model
        try:
            logger.info("Loading sentence-transformers model '%s' …", model_name)
            self.__class__._model = SentenceTransformer(model_name)
            logger.info("Model loaded successfully (dim=%d).", EMBEDDING_DIM)
        except Exception:
            logger.exception("Failed to load embedding model '%s'.", model_name)
            raise

    @property
    def model(self) -> SentenceTransformer:
        """Return the loaded model, raising if unavailable."""
        if self.__class__._model is None:
            raise RuntimeError(
                f"Embedding model '{get_settings().embedding_model}' is not loaded. "
                "Call _load_model() first."
            )
        return self.__class__._model

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def embed_text(self, text: str) -> list[float]:
        """Embed a single text string, returning a 384-d float vector.

        Results are cached in Redis keyed by a truncated SHA-256 hash.
        """
        cached = await self._get_cached(text)
        if cached is not None:
            return cached

        # No lock needed — model.encode() is thread-safe for inference
        embedding = await asyncio.to_thread(
            self.model.encode, text, normalize_embeddings=True
        )
        embedding = embedding.tolist()
        await self._set_cached(text, embedding)
        return embedding

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Embed a batch of texts, returning one 384-d vector per input.

        Individual texts that are already cached will be served from Redis;
        only the uncached subset is sent through the model.
        
        IMPORTANT: Lock is only held during model inference, not cache I/O,
        to enable true parallelism for batch embedding requests.
        """
        results: list[list[float] | None] = [None] * len(texts)
        uncached_indices: list[int] = []
        uncached_texts: list[str] = []

        # 1. Check cache for every text (NO LOCK - parallel cache reads).
        for idx, text in enumerate(texts):
            cached = await self._get_cached(text)
            if cached is not None:
                results[idx] = cached
            else:
                uncached_indices.append(idx)
                uncached_texts.append(text)

        # 2. Encode the uncached texts in a single batch call.
        # Lock only during model inference, not cache writes.
        if uncached_texts:
            # No lock needed — model.encode() is thread-safe for inference
            embeddings = await asyncio.to_thread(
                self.model.encode, uncached_texts, normalize_embeddings=True
            )
            embeddings = embeddings.tolist()
            for i, idx in enumerate(uncached_indices):
                results[idx] = embeddings[i]
                await self._set_cached(uncached_texts[i], embeddings[i])

        # At this point every slot is filled.
        return results  # type: ignore[return-value]

    async def embed_query(self, query: str) -> list[float]:
        """Embed a search query.

        Semantically separate from :meth:`embed_text` to allow future
        query-specific optimisations (e.g. instruction-prefixed models).
        """
        return await self.embed_text(query)

    # ------------------------------------------------------------------
    # Redis cache helpers
    # ------------------------------------------------------------------

    async def _get_cached(self, text: str) -> list[float] | None:
        """Return the cached embedding for *text*, or ``None``."""
        try:
            raw = await self._redis.get(_cache_key(text))
            if raw is not None:
                return json.loads(raw)
        except aioredis.RedisError:
            logger.warning("Redis read error for embedding cache; skipping cache.")
        return None

    async def _set_cached(self, text: str, embedding: list[float]) -> None:
        """Store an embedding in Redis with TTL."""
        try:
            await self._redis.set(
                _cache_key(text),
                json.dumps(embedding),
                ex=CACHE_TTL_SECONDS,
            )
        except aioredis.RedisError:
            logger.warning("Redis write error for embedding cache; skipping cache.")
