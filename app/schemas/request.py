"""
Pydantic v2 request schemas for the RAG chatbot API.
"""

from __future__ import annotations

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Query
# ---------------------------------------------------------------------------


class QueryRequest(BaseModel):
    """Incoming user query — text or voice (after ASR)."""

    text: str | None = Field(None, description="The user's question text.")
    language: str | None = Field(
        None, description="ISO 639-1 language code (auto-detected if omitted)."
    )
    topic: str | None = Field(None, description="Optional topic filter.")
    subject: str | None = Field(None, description="Optional subject filter.")


class CompareRequest(BaseModel):
    """Payload for comparing a user's answer against the system's ground truth."""
    question: str = Field(..., description="The original question.")
    user_answer: str = Field(..., description="The user's answer to evaluate.")
    language: str | None = Field(None, description="Optional ISO 639-1 language code.")


# ---------------------------------------------------------------------------
# Document ingestion
# ---------------------------------------------------------------------------


class IngestFileRequest(BaseModel):
    """Metadata sent alongside a file upload for ingestion."""

    subject: str | None = Field(None, description="Subject to file under.")
    topic: str | None = Field(None, description="Topic to file under.")
    language: str | None = Field(
        None, description="Language of the document content."
    )
    source_type: str | None = Field(
        None,
        description="E.g. 'pdf', 'docx', 'txt'.",
    )


# ---------------------------------------------------------------------------
# QA pair management
# ---------------------------------------------------------------------------


class QAPairCreate(BaseModel):
    """Payload for creating a new QA pair."""

    question: str = Field(..., min_length=1, description="The question text.")
    answer: str = Field(..., min_length=1, description="The answer text.")
    subject: str | None = Field(None, description="Optional subject name.")
    topic: str | None = Field(None, description="Optional topic name.")
    language: str | None = Field(
        None, description="ISO 639-1 language code (defaults to 'en')."
    )


class QAPairUpdate(BaseModel):
    """Payload for updating an existing QA pair."""

    question: str | None = Field(None, description="Updated question text.")
    answer: str | None = Field(None, description="Updated answer text.")
    edit_reason: str | None = Field(
        None, description="Reason for the edit (stored in qa_versions)."
    )


class BulkQAUpload(BaseModel):
    """Wrapper for uploading multiple QA pairs at once."""

    pairs: list[QAPairCreate] = Field(
        ..., min_length=1, description="List of QA pairs to create."
    )
