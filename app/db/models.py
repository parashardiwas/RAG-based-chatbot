"""
SQLAlchemy 2.0 ORM models for the RAG chatbot.

All tables use UUID primary keys with server-side generation via
``gen_random_uuid()`` (PostgreSQL 13+). Vector columns use pgvector's
``Vector(384)`` type for all-MiniLM-L6-v2 embeddings.
"""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import Optional

from pgvector.sqlalchemy import Vector
from sqlalchemy import (
    ARRAY,
    Boolean,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
    UniqueConstraint,
    text,
)
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import (
    DeclarativeBase,
    Mapped,
    mapped_column,
    relationship,
)
from sqlalchemy.sql import func


# ---------------------------------------------------------------------------
# Base class
# ---------------------------------------------------------------------------


class Base(DeclarativeBase):
    """Declarative base for all ORM models."""

    pass


# ---------------------------------------------------------------------------
# Subjects & Topics
# ---------------------------------------------------------------------------


class Subject(Base):
    """Top-level organisational unit (e.g. "Machine Learning")."""

    __tablename__ = "subjects"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        server_default=text("gen_random_uuid()"),
    )
    name: Mapped[str] = mapped_column(String(255), unique=True, nullable=False)
    description: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
    )

    # Relationships
    topics: Mapped[list[Topic]] = relationship(
        "Topic", back_populates="subject", cascade="all, delete-orphan"
    )
    qa_pairs: Mapped[list[QAPair]] = relationship(
        "QAPair", back_populates="subject"
    )
    document_chunks: Mapped[list[DocumentChunk]] = relationship(
        "DocumentChunk", back_populates="subject"
    )
    documents: Mapped[list[Document]] = relationship(
        "Document", back_populates="subject"
    )


class Topic(Base):
    """Second-level organisational unit, belongs to a Subject."""

    __tablename__ = "topics"
    __table_args__ = (
        UniqueConstraint("subject_id", "name", name="uq_topic_subject_name"),
    )

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        server_default=text("gen_random_uuid()"),
    )
    subject_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("subjects.id", ondelete="CASCADE"),
        nullable=False,
    )
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    description: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
    )

    # Relationships
    subject: Mapped[Subject] = relationship("Subject", back_populates="topics")
    qa_pairs: Mapped[list[QAPair]] = relationship(
        "QAPair", back_populates="topic"
    )
    document_chunks: Mapped[list[DocumentChunk]] = relationship(
        "DocumentChunk", back_populates="topic"
    )
    documents: Mapped[list[Document]] = relationship(
        "Document", back_populates="topic"
    )


# ---------------------------------------------------------------------------
# QA Pairs & Versions
# ---------------------------------------------------------------------------


class QAPair(Base):
    """A question-answer pair with vector embeddings for retrieval."""

    __tablename__ = "qa_pairs"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        server_default=text("gen_random_uuid()"),
    )
    question: Mapped[str] = mapped_column(Text, nullable=False)
    answer: Mapped[str] = mapped_column(Text, nullable=False)
    subject_id: Mapped[Optional[uuid.UUID]] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("subjects.id", ondelete="SET NULL"),
        nullable=True,
    )
    topic_id: Mapped[Optional[uuid.UUID]] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("topics.id", ondelete="SET NULL"),
        nullable=True,
    )
    language: Mapped[str] = mapped_column(
        String(20), default="en", server_default=text("'en'"), nullable=False
    )
    source_type: Mapped[str] = mapped_column(
        String(50), default="manual", server_default=text("'manual'"), nullable=False
    )
    is_deleted: Mapped[bool] = mapped_column(
        Boolean, default=False, server_default=text("false"), nullable=False
    )
    version: Mapped[int] = mapped_column(
        Integer, default=1, server_default=text("1"), nullable=False
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
    )
    updated_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True),
        onupdate=func.now(),
        nullable=True,
    )
    created_by: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)

    # Vector embeddings (384-dim for all-MiniLM-L6-v2)
    question_embedding = mapped_column(Vector(384), nullable=True)
    answer_embedding = mapped_column(Vector(384), nullable=True)
    combined_embedding = mapped_column(Vector(384), nullable=True)

    # Relationships
    subject: Mapped[Optional[Subject]] = relationship(
        "Subject", back_populates="qa_pairs"
    )
    topic: Mapped[Optional[Topic]] = relationship(
        "Topic", back_populates="qa_pairs"
    )
    versions: Mapped[list[QAVersion]] = relationship(
        "QAVersion", back_populates="qa_pair", cascade="all, delete-orphan"
    )


class QAVersion(Base):
    """Immutable snapshot of a QA pair edit for audit purposes."""

    __tablename__ = "qa_versions"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        server_default=text("gen_random_uuid()"),
    )
    qa_pair_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("qa_pairs.id", ondelete="CASCADE"),
        nullable=False,
    )
    question: Mapped[str] = mapped_column(Text, nullable=False)
    answer: Mapped[str] = mapped_column(Text, nullable=False)
    version: Mapped[int] = mapped_column(Integer, nullable=False)
    edited_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
    )
    edited_by: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)
    edit_reason: Mapped[Optional[str]] = mapped_column(Text, nullable=True)

    # Relationships
    qa_pair: Mapped[QAPair] = relationship("QAPair", back_populates="versions")


# ---------------------------------------------------------------------------
# Documents & Chunks
# ---------------------------------------------------------------------------


class Document(Base):
    """Metadata for an uploaded/ingested document."""

    __tablename__ = "documents"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        server_default=text("gen_random_uuid()"),
    )
    filename: Mapped[str] = mapped_column(String(500), nullable=False)
    file_type: Mapped[Optional[str]] = mapped_column(String(50), nullable=True)
    file_path: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    file_hash: Mapped[Optional[str]] = mapped_column(String(64), nullable=True)
    subject_id: Mapped[Optional[uuid.UUID]] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("subjects.id", ondelete="SET NULL"),
        nullable=True,
    )
    topic_id: Mapped[Optional[uuid.UUID]] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("topics.id", ondelete="SET NULL"),
        nullable=True,
    )
    language: Mapped[Optional[str]] = mapped_column(String(20), nullable=True)
    chunk_count: Mapped[int] = mapped_column(
        Integer, default=0, server_default=text("0"), nullable=False
    )
    status: Mapped[str] = mapped_column(
        String(50),
        default="processing",
        server_default=text("'processing'"),
        nullable=False,
    )
    metadata_: Mapped[Optional[dict]] = mapped_column(
        "metadata", JSONB, nullable=True
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
    )

    # Relationships
    subject: Mapped[Optional[Subject]] = relationship(
        "Subject", back_populates="documents"
    )
    topic: Mapped[Optional[Topic]] = relationship(
        "Topic", back_populates="documents"
    )
    chunks: Mapped[list[DocumentChunk]] = relationship(
        "DocumentChunk", back_populates="document", cascade="all, delete-orphan"
    )


class DocumentChunk(Base):
    """A single chunk of a document with its vector embedding."""

    __tablename__ = "document_chunks"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        server_default=text("gen_random_uuid()"),
    )
    document_id: Mapped[Optional[uuid.UUID]] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("documents.id", ondelete="CASCADE"),
        nullable=True,
    )
    chunk_index: Mapped[int] = mapped_column(Integer, nullable=False)
    content: Mapped[str] = mapped_column(Text, nullable=False)
    subject_id: Mapped[Optional[uuid.UUID]] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("subjects.id", ondelete="SET NULL"),
        nullable=True,
    )
    topic_id: Mapped[Optional[uuid.UUID]] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("topics.id", ondelete="SET NULL"),
        nullable=True,
    )
    language: Mapped[Optional[str]] = mapped_column(String(20), nullable=True)
    source_file: Mapped[Optional[str]] = mapped_column(String(500), nullable=True)
    source_type: Mapped[Optional[str]] = mapped_column(String(50), nullable=True)
    metadata_: Mapped[Optional[dict]] = mapped_column(
        "metadata", JSONB, nullable=True
    )

    # Vector embedding (384-dim for all-MiniLM-L6-v2)
    embedding = mapped_column(Vector(384), nullable=True)

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
    )

    # Relationships
    document: Mapped[Optional[Document]] = relationship(
        "Document", back_populates="chunks"
    )
    subject: Mapped[Optional[Subject]] = relationship(
        "Subject", back_populates="document_chunks"
    )
    topic: Mapped[Optional[Topic]] = relationship(
        "Topic", back_populates="document_chunks"
    )


# ---------------------------------------------------------------------------
# Audit & Query Logs
# ---------------------------------------------------------------------------


class AuditLog(Base):
    """Immutable audit trail for entity mutations."""

    __tablename__ = "audit_log"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        server_default=text("gen_random_uuid()"),
    )
    entity_type: Mapped[str] = mapped_column(String(50), nullable=False)
    entity_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), nullable=False
    )
    action: Mapped[str] = mapped_column(String(50), nullable=False)
    old_data: Mapped[Optional[dict]] = mapped_column(JSONB, nullable=True)
    new_data: Mapped[Optional[dict]] = mapped_column(JSONB, nullable=True)
    performed_by: Mapped[Optional[str]] = mapped_column(
        String(255), nullable=True
    )
    performed_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
    )


class QueryLog(Base):
    """Log entry for every user query processed by the system."""

    __tablename__ = "query_log"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        server_default=text("gen_random_uuid()"),
    )
    input_text: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    input_type: Mapped[Optional[str]] = mapped_column(String(50), nullable=True)
    detected_language: Mapped[Optional[str]] = mapped_column(
        String(20), nullable=True
    )
    retrieved_chunk_ids = mapped_column(
        ARRAY(UUID(as_uuid=True)), nullable=True
    )
    matched_qa_pair_id: Mapped[Optional[uuid.UUID]] = mapped_column(
        UUID(as_uuid=True), nullable=True
    )
    answer: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    retrieval_confidence: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True
    )
    answer_confidence: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True
    )
    model_used: Mapped[Optional[str]] = mapped_column(
        String(100), nullable=True
    )
    total_tokens: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    estimated_cost_usd: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True
    )
    latency_ms: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
    )
