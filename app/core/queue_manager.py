"""
Concurrency control for managing parallel request limits.

Uses asyncio.Semaphore to cap concurrent request processing at MAX_CONCURRENT_REQUESTS (default 50).
Requests beyond the limit are queued with priority ordering.
"""

import asyncio
import time
import uuid
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Any

from app.config import get_settings


class RequestPriority(IntEnum):
    """Priority levels for request queue. Lower number = higher priority."""
    INTERACTIVE_TEXT = 1
    INTERACTIVE_VOICE = 2
    BATCH_FILE = 3
    BATCH_INGESTION = 4


@dataclass
class QueuedRequest:
    """A request waiting in the queue."""
    request_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    priority: RequestPriority = RequestPriority.INTERACTIVE_TEXT
    enqueued_at: float = field(default_factory=time.time)
    position: int = 0


class QueueManager:
    """
    Manages concurrent request processing with a semaphore and overflow queue.
    
    - Up to MAX_CONCURRENT_REQUESTS processed simultaneously
    - Excess requests get a 202 Accepted with queue position
    - Tracks metrics: active count, queue depth, wait times
    """

    def __init__(self):
        settings = get_settings()
        self._semaphore = asyncio.Semaphore(settings.max_concurrent_requests)
        self._active_count = 0
        self._queue_depth = 0
        self._total_processed = 0
        self._total_queued = 0
        self._lock = asyncio.Lock()

    @property
    def active_count(self) -> int:
        return self._active_count

    @property
    def queue_depth(self) -> int:
        return self._queue_depth

    @property
    def is_at_capacity(self) -> bool:
        return self._active_count >= get_settings().max_concurrent_requests

    async def acquire(self) -> bool:
        """
        Try to acquire a processing slot.
        Returns True if acquired immediately, False if at capacity.
        """
        # Try non-blocking acquire first
        if self._semaphore._value > 0:
            await self._semaphore.acquire()
            async with self._lock:
                self._active_count += 1
            return True
        return False

    async def acquire_wait(self, timeout: float | None = None) -> bool:
        """Acquire a slot, waiting up to timeout seconds."""
        try:
            if timeout:
                await asyncio.wait_for(self._semaphore.acquire(), timeout=timeout)
            else:
                await self._semaphore.acquire()
            async with self._lock:
                self._active_count += 1
            return True
        except asyncio.TimeoutError:
            return False

    async def release(self):
        """Release a processing slot."""
        self._semaphore.release()
        async with self._lock:
            self._active_count -= 1
            self._total_processed += 1

    def get_queue_position(self) -> QueuedRequest:
        """Create a queue entry for a request that can't be processed immediately."""
        self._total_queued += 1
        self._queue_depth += 1
        return QueuedRequest(
            position=self._queue_depth,
            priority=RequestPriority.INTERACTIVE_TEXT,
        )

    def get_metrics(self) -> dict[str, Any]:
        """Return current queue metrics."""
        return {
            "active_requests": self._active_count,
            "queue_depth": self._queue_depth,
            "total_processed": self._total_processed,
            "total_queued": self._total_queued,
            "max_concurrent": get_settings().max_concurrent_requests,
        }
    
    async def reset_metrics(self) -> None:
        """Reset cumulative metrics to prevent unbounded growth."""
        async with self._lock:
            self._total_processed = 0
            self._total_queued = 0
            self._queue_depth = 0


# Global singleton
_queue_manager: QueueManager | None = None


def get_queue_manager() -> QueueManager:
    """Get the global QueueManager instance."""
    global _queue_manager
    if _queue_manager is None:
        _queue_manager = QueueManager()
    return _queue_manager
