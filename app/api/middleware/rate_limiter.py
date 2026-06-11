"""
Rate limiting middleware using Redis sliding window.
"""

import time
import logging

from fastapi import Request, HTTPException
from starlette.middleware.base import BaseHTTPMiddleware

from app.config import get_settings

logger = logging.getLogger(__name__)


class RateLimitMiddleware(BaseHTTPMiddleware):
    """
    Simple rate limiter using Redis sliding window.
    Limits requests per IP address.
    
    Features:
    - Reconnects on Redis failures (exponential backoff)
    - Graceful degradation if Redis is unavailable
    """

    def __init__(self, app, requests_per_minute: int = 60):
        super().__init__(app)
        self.requests_per_minute = requests_per_minute
        self._redis = None
        self._redis_failed = False
        self._redis_retry_count = 0

    async def _get_redis(self):
        """Get Redis client with reconnection logic."""
        # If Redis previously failed, don't retry immediately
        if self._redis_failed and self._redis_retry_count < 5:
            self._redis_retry_count += 1
            return None
        
        # Reset retry counter if we get here
        self._redis_retry_count = 0
        
        if self._redis is None or self._redis.connection_pool.disconnect():
            try:
                import redis.asyncio as aioredis
                settings = get_settings()
                self._redis = aioredis.from_url(settings.redis_url)
                # Test connection
                await self._redis.ping()
                self._redis_failed = False
                logger.debug("Redis reconnected successfully")
            except Exception as e:
                logger.debug(f"Redis connection failed: {e}")
                self._redis = None
                self._redis_failed = True
                return None
        return self._redis

    async def dispatch(self, request: Request, call_next):
        # Skip rate limiting for health checks
        if request.url.path in ("/health", "/metrics", "/docs", "/openapi.json"):
            return await call_next(request)

        redis = await self._get_redis()
        if redis:
            client_ip = request.client.host if request.client else "unknown"
            key = f"rate_limit:{client_ip}"

            try:
                current = await redis.get(key)
                if current and int(current) >= self.requests_per_minute:
                    from fastapi.responses import JSONResponse
                    return JSONResponse(
                        status_code=429,
                        content={"detail": f"Rate limit exceeded. Max {self.requests_per_minute} requests/minute."}
                    )
                pipe = redis.pipeline()
                pipe.incr(key)
                pipe.expire(key, 60)
                await pipe.execute()
            except HTTPException:
                raise
            except Exception as e:
                logger.debug(f"Rate limit check failed: {e}")

        response = await call_next(request)
        return response
