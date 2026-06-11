"""
Cost tracking for API usage, token consumption, and resource utilization.
Since we use self-hosted models, this primarily tracks compute time and token counts.
"""

import time
from dataclasses import dataclass, field
from typing import Any

import redis.asyncio as aioredis

from app.config import get_settings


@dataclass
class RequestCost:
    """Cost breakdown for a single request."""
    request_id: str = ""
    embedding_tokens: int = 0
    llm_input_tokens: int = 0
    llm_output_tokens: int = 0
    asr_seconds: float = 0.0
    model_used: str = ""
    total_latency_ms: int = 0

    # Cost estimates (even for self-hosted, track equivalent API cost)
    estimated_embedding_cost: float = 0.0
    estimated_llm_cost: float = 0.0
    estimated_asr_cost: float = 0.0

    @property
    def total_estimated_cost(self) -> float:
        return (
            self.estimated_embedding_cost
            + self.estimated_llm_cost
            + self.estimated_asr_cost
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "request_id": self.request_id,
            "embedding_tokens": self.embedding_tokens,
            "llm_input_tokens": self.llm_input_tokens,
            "llm_output_tokens": self.llm_output_tokens,
            "asr_seconds": self.asr_seconds,
            "model_used": self.model_used,
            "total_latency_ms": self.total_latency_ms,
            "estimated_cost_usd": self.total_estimated_cost,
        }


class CostTracker:
    """
    Tracks resource usage and estimated costs.
    
    Even with self-hosted models, tracking equivalent API costs helps
    understand the value of self-hosting and plan for scaling.
    """

    # Equivalent API pricing for reference (USD per unit)
    PRICING = {
        "embedding_per_1k_tokens": 0.00002,   # OpenAI text-embedding-3-small equivalent
        "llm_input_per_1k_tokens": 0.00015,    # GPT-4o-mini equivalent
        "llm_output_per_1k_tokens": 0.0006,    # GPT-4o-mini equivalent
        "asr_per_minute": 0.006,               # Whisper API equivalent
    }

    def __init__(self):
        self._redis: aioredis.Redis | None = None
        self._daily_costs: float = 0.0
        self._total_requests: int = 0

    async def init(self):
        """Initialize Redis connection for persistent tracking."""
        settings = get_settings()
        try:
            self._redis = aioredis.from_url(
                settings.redis_url, decode_responses=True
            )
        except Exception:
            self._redis = None

    def calculate_cost(self, cost: RequestCost) -> RequestCost:
        """Calculate estimated costs for a request."""
        cost.estimated_embedding_cost = (
            cost.embedding_tokens / 1000 * self.PRICING["embedding_per_1k_tokens"]
        )
        cost.estimated_llm_cost = (
            cost.llm_input_tokens / 1000 * self.PRICING["llm_input_per_1k_tokens"]
            + cost.llm_output_tokens / 1000 * self.PRICING["llm_output_per_1k_tokens"]
        )
        cost.estimated_asr_cost = (
            cost.asr_seconds / 60 * self.PRICING["asr_per_minute"]
        )
        return cost

    async def log_cost(self, cost: RequestCost):
        """Log cost to Redis for aggregation."""
        if self._redis:
            today = time.strftime("%Y-%m-%d")
            pipe = self._redis.pipeline()
            pipe.incrbyfloat(f"cost:daily:{today}", cost.total_estimated_cost)
            pipe.incr(f"cost:requests:{today}")
            pipe.incrbyfloat("cost:total", cost.total_estimated_cost)
            pipe.incr("cost:total_requests")
            await pipe.execute()

        self._daily_costs += cost.total_estimated_cost
        self._total_requests += 1

    async def get_daily_summary(self) -> dict[str, Any]:
        """Get today's cost summary."""
        today = time.strftime("%Y-%m-%d")
        if self._redis:
            daily_cost = await self._redis.get(f"cost:daily:{today}") or "0"
            daily_requests = await self._redis.get(f"cost:requests:{today}") or "0"
            total_cost = await self._redis.get("cost:total") or "0"
            total_requests = await self._redis.get("cost:total_requests") or "0"
            return {
                "date": today,
                "daily_estimated_cost_usd": float(daily_cost),
                "daily_requests": int(daily_requests),
                "total_estimated_cost_usd": float(total_cost),
                "total_requests": int(total_requests),
                "note": "Costs are API-equivalent estimates. Actual cost is compute/electricity for self-hosted models.",
            }
        return {
            "date": today,
            "daily_estimated_cost_usd": self._daily_costs,
            "daily_requests": self._total_requests,
            "note": "Redis not available, showing in-memory totals only.",
        }


# Global singleton
_cost_tracker: CostTracker | None = None


async def get_cost_tracker() -> CostTracker:
    """Get the global CostTracker instance."""
    global _cost_tracker
    if _cost_tracker is None:
        _cost_tracker = CostTracker()
        await _cost_tracker.init()
    return _cost_tracker
