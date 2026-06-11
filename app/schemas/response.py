"""
Pydantic v2 response schemas for the RAG chatbot API.
"""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Query response
# ---------------------------------------------------------------------------


class SourceInfo(BaseModel):
    """A single retrieved source/chunk used to compose the answer."""

    chunk_id: str = Field(..., description="UUID of the matched chunk.")
    content_preview: str = Field(
        ..., description="Truncated preview of the chunk content."
    )
    similarity: float = Field(
        ..., ge=0.0, le=1.0, description="Cosine similarity score."
    )
    source_file: str | None = Field(
        None, description="Original filename, if available."
    )
    topic: str | None = Field(None, description="Topic name, if available.")


class QueryResponse(BaseModel):
    """Response returned for a user query."""

    answer: str = Field(..., description="Generated answer text.")
    language: str = Field(..., description="Language of the answer.")
    confidence: float = Field(
        ..., ge=0.0, le=1.0, description="Overall answer confidence."
    )
    retrieval_confidence: float = Field(
        ..., ge=0.0, le=1.0, description="Retrieval-stage confidence."
    )
    sources: list[SourceInfo] = Field(
        default_factory=list, description="Sources used for the answer."
    )
    model_used: str = Field(..., description="LLM model identifier.")
    latency_ms: int = Field(
        ..., ge=0, description="End-to-end latency in milliseconds."
    )
    cached: bool = Field(
        False, description="Whether the response was served from cache."
    )


class TaskSubmitResponse(BaseModel):
    """Returned when an async task is submitted."""
    
    task_id: str = Field(..., description="UUID of the background task.")
    status: str = Field(..., description="Initial status, usually 'PENDING'.")
    message: str = Field(..., description="Message indicating task was queued.")


class TaskStatusResponse(BaseModel):
    """Returned when polling for task status."""
    
    task_id: str = Field(..., description="UUID of the background task.")
    status: str = Field(..., description="Current status: PENDING, PROCESSING, COMPLETED, or FAILED.")
    result: QueryResponse | None = Field(None, description="The final QueryResponse if COMPLETED.")
    error: str | None = Field(None, description="Error message if FAILED.")


# ---------------------------------------------------------------------------
# QA pair response
# ---------------------------------------------------------------------------


class QAPairResponse(BaseModel):
    """Serialised QA pair returned by CRUD endpoints."""

    id: str = Field(..., description="UUID of the QA pair.")
    question: str
    answer: str
    subject: str | None = None
    topic: str | None = None
    language: str = "en"
    version: int = 1
    created_at: datetime
    updated_at: datetime | None = None


# ---------------------------------------------------------------------------
# Document ingestion response
# ---------------------------------------------------------------------------


class IngestResponse(BaseModel):
    """Returned after a document has been successfully ingested."""

    document_id: str = Field(..., description="UUID of the created document.")
    status: str = Field(
        ..., description="Processing status (e.g. 'completed', 'processing')."
    )
    chunks_created: int = Field(
        ..., ge=0, description="Number of chunks generated."
    )
    subject: str | None = None
    topic: str | None = None


# ---------------------------------------------------------------------------
# Health check
# ---------------------------------------------------------------------------


class HealthResponse(BaseModel):
    """System health summary."""

    status: str = Field(..., description="Overall status ('ok' or 'degraded').")
    database: str = Field(..., description="Database connectivity status.")
    redis: str = Field(..., description="Redis connectivity status.")
    llm: str = Field(..., description="LLM service status.")
    gpu_available: bool = Field(
        ..., description="Whether a GPU is detected and usable."
    )


# ---------------------------------------------------------------------------
# Error
# ---------------------------------------------------------------------------


class ErrorResponse(BaseModel):
    """Standard error envelope."""

    error: str = Field(..., description="Short error description.")
    detail: str | None = Field(
        None, description="Extended detail / traceback info."
    )
    request_id: str | None = Field(
        None, description="Correlation ID for log tracing."
    )
