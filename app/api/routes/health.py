"""
Health check and metrics endpoints.
"""

import logging
import time

from fastapi import APIRouter, Depends
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import get_settings
from app.core.cost_tracker import get_cost_tracker
from app.core.queue_manager import get_queue_manager
from app.db.database import get_db
from app.schemas.response import HealthResponse

logger = logging.getLogger(__name__)
router = APIRouter(tags=["Health"])

_llm_health_cache: dict = {"status": "unknown", "expires": 0.0}


@router.get(
    "/health",
    response_model=HealthResponse,
    summary="Health check",
    description="Check the health of all system components.",
)
async def health_check(db: AsyncSession = Depends(get_db)):
    """Check connectivity to all services."""
    settings = get_settings()
    health = {
        "database": "unknown",
        "redis": "unknown",
        "llm": "unknown",
        "gpu_available": False,
    }

    # Check PostgreSQL
    try:
        await db.execute(text("SELECT 1"))
        health["database"] = "healthy"
    except Exception as e:
        health["database"] = f"unhealthy: {e}"

    # Check Redis
    try:
        import redis.asyncio as aioredis
        r = aioredis.from_url(settings.redis_url)
        await r.ping()
        health["redis"] = "healthy"
        await r.close()
    except Exception as e:
        health["redis"] = f"unhealthy: {e}"

    # LLM health — cached for 60 seconds
    now = time.time()
    if now < _llm_health_cache["expires"]:
        health["llm"] = _llm_health_cache["status"]
    else:
        try:
            import openai
            client = openai.AsyncOpenAI(api_key=settings.openai_api_key)
            await client.models.list()
            llm_status = f"healthy ({settings.openai_model})"
        except Exception as e:
            llm_status = f"unavailable: {e}"
        _llm_health_cache.update({"status": llm_status, "expires": now + 60.0})
        health["llm"] = llm_status

    # Check GPU (Apple Silicon MPS)
    try:
        import torch
        health["gpu_available"] = torch.backends.mps.is_available()
    except Exception:
        health["gpu_available"] = False

    overall = "healthy" if all(
        "healthy" in str(v) for k, v in health.items() if k != "gpu_available"
    ) else "degraded"

    return HealthResponse(
        status=overall,
        database=health["database"],
        redis=health["redis"],
        llm=health["llm"],
        gpu_available=health["gpu_available"],
    )


@router.get(
    "/metrics",
    summary="System metrics",
    description="Get system metrics: latency, cost, queue depth, cache hit rates.",
)
async def get_metrics():
    """Get current system metrics."""
    queue = get_queue_manager()
    cost_tracker = await get_cost_tracker()

    return {
        "queue": queue.get_metrics(),
        "cost": await cost_tracker.get_daily_summary(),
    }
