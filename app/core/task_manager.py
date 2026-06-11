"""
Task Manager for asynchronous RAG queries.
Manages the lifecycle of background RAG tasks using Redis for state storage.
"""

import asyncio
import json
import logging
import uuid
from typing import Any, Dict

import redis.asyncio as aioredis
from pydantic import BaseModel

from app.config import get_settings
from app.core.orchestrator import get_orchestrator
from app.schemas.request import QueryRequest
from app.schemas.response import QueryResponse

logger = logging.getLogger(__name__)

class TaskState(BaseModel):
    task_id: str
    status: str
    result: Dict[str, Any] | None = None
    error: str | None = None


class TaskManager:
    def __init__(self):
        self._settings = get_settings()
        self._redis: aioredis.Redis | None = None
        # TTL for task states: 1 hour
        self._ttl = 3600

    def _get_redis(self) -> aioredis.Redis:
        if self._redis is None:
            self._redis = aioredis.from_url(
                self._settings.redis_url, 
                decode_responses=True
            )
        return self._redis

    async def get_task_status(self, task_id: str) -> TaskState | None:
        """Fetch the current status of a task from Redis."""
        redis = self._get_redis()
        key = f"task:{task_id}"
        
        try:
            data = await redis.get(key)
            if not data:
                return None
            
            parsed = json.loads(data)
            return TaskState(**parsed)
        except Exception as e:
            logger.error(f"Failed to fetch task {task_id}: {e}")
            return None

    async def submit_task(self, request: QueryRequest) -> str:
        """Generate a task ID, save initial state, and dispatch background worker."""
        task_id = str(uuid.uuid4())
        
        initial_state = TaskState(
            task_id=task_id,
            status="PENDING"
        )
        
        redis = self._get_redis()
        await redis.setex(
            f"task:{task_id}", 
            self._ttl, 
            initial_state.model_dump_json()
        )
        
        # Dispatch to background
        asyncio.create_task(self._process_task_background(task_id, request))
        
        return task_id

    async def _update_state(self, state: TaskState):
        """Helper to save state to Redis."""
        try:
            redis = self._get_redis()
            await redis.setex(
                f"task:{state.task_id}", 
                self._ttl, 
                state.model_dump_json()
            )
        except Exception as e:
            logger.error(f"Failed to update task {state.task_id} state: {e}")

    async def _process_task_background(self, task_id: str, request: QueryRequest):
        """The actual background worker that runs the RAG pipeline."""
        state = TaskState(task_id=task_id, status="PROCESSING")
        await self._update_state(state)
        
        try:
            orchestrator = await get_orchestrator()
            
            # Execute the heavy RAG pipeline
            result = await orchestrator.process_text_query(
                text=request.text.strip(),
                language=request.language,
                subject_filter=request.subject,
                topic_filter=request.topic,
            )
            
            # Convert Result object to a dictionary matching QueryResponse schema
            # We construct a dict so it can be cleanly serialized to JSON and loaded back
            result_dict = {
                "answer": result.answer,
                "language": result.language,
                "confidence": round(result.confidence, 3),
                "retrieval_confidence": round(result.retrieval_confidence, 3),
                "sources": result.sources,
                "model_used": result.model_used,
                "latency_ms": result.latency_ms,
                "cached": result.cached,
            }
            
            state.status = "COMPLETED"
            state.result = result_dict
            await self._update_state(state)
            
        except Exception as e:
            logger.error(f"Task {task_id} failed: {e}", exc_info=True)
            state.status = "FAILED"
            state.error = str(e)
            await self._update_state(state)


# Global singleton
_task_manager: TaskManager | None = None

def get_task_manager() -> TaskManager:
    global _task_manager
    if _task_manager is None:
        _task_manager = TaskManager()
    return _task_manager
