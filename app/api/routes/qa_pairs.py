"""
Q/A pair management — CRUD operations with versioning, soft delete, and bulk upload.
"""

import logging
import uuid
from datetime import datetime, timezone

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy import func, select, update
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import joinedload

from app.db.database import get_db
from app.api.auth import verify_api_key
from app.db.models import AuditLog, QAPair, QAVersion
from app.schemas.request import BulkQAUpload, QAPairCreate, QAPairUpdate
from app.schemas.response import QAPairResponse

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/v1/qa", tags=["Q/A Pairs"])


def _parse_uuid(value: str) -> uuid.UUID:
    """Parse a UUID string, raising 422 for invalid format."""
    try:
        return uuid.UUID(value)
    except ValueError:
        raise HTTPException(status_code=422, detail=f"Invalid UUID format: {value}")


@router.get(
    "",
    response_model=list[QAPairResponse],
    summary="List Q/A pairs",
    description="Get all Q/A pairs with optional filtering by subject, topic, and language.",
)
async def list_qa_pairs(
    subject: str | None = Query(None, description="Filter by subject"),
    topic: str | None = Query(None, description="Filter by topic"),
    language: str | None = Query(None, description="Filter by language"),
    include_deleted: bool = Query(False, description="Include soft-deleted pairs"),
    page: int = Query(1, ge=1, description="Page number"),
    per_page: int = Query(20, ge=1, le=100, description="Items per page"),
    db: AsyncSession = Depends(get_db),
):
    """List all Q/A pairs with pagination and filtering."""
    query = select(QAPair).options(joinedload(QAPair.subject), joinedload(QAPair.topic))

    if not include_deleted:
        query = query.where(QAPair.is_deleted == False)
    if language:
        query = query.where(QAPair.language == language)

    # Pagination
    query = query.order_by(QAPair.created_at.desc())
    query = query.offset((page - 1) * per_page).limit(per_page)

    result = await db.execute(query)
    pairs = result.scalars().all()

    return [
        QAPairResponse(
            id=str(pair.id),
            question=pair.question,
            answer=pair.answer,
            subject=pair.subject.name if pair.subject else None,
            topic=pair.topic.name if pair.topic else None,
            language=pair.language or "en",
            version=pair.version or 1,
            created_at=pair.created_at or datetime.now(timezone.utc),
            updated_at=pair.updated_at or datetime.now(timezone.utc),
        )
        for pair in pairs
    ]


@router.get(
    "/{qa_id}",
    response_model=QAPairResponse,
    summary="Get a Q/A pair",
    description="Get a single Q/A pair by ID, including its version history.",
)
async def get_qa_pair(
    qa_id: str,
    db: AsyncSession = Depends(get_db),
):
    """Get a single Q/A pair with its version history."""
    result = await db.execute(
        select(QAPair).where(QAPair.id == _parse_uuid(qa_id))
    )
    pair = result.scalar_one_or_none()

    if not pair:
        raise HTTPException(status_code=404, detail="Q/A pair not found")

    return QAPairResponse(
        id=str(pair.id),
        question=pair.question,
        answer=pair.answer,
        subject=None,
        topic=None,
        language=pair.language or "en",
        version=pair.version or 1,
        created_at=pair.created_at or datetime.now(timezone.utc),
        updated_at=pair.updated_at or datetime.now(timezone.utc),
    )


@router.get(
    "/{qa_id}/versions",
    summary="Get version history",
    description="Get all previous versions of a Q/A pair.",
)
async def get_qa_versions(
    qa_id: str,
    db: AsyncSession = Depends(get_db),
):
    """Get version history for a Q/A pair."""
    result = await db.execute(
        select(QAVersion)
        .where(QAVersion.qa_pair_id == _parse_uuid(qa_id))
        .order_by(QAVersion.version.desc())
    )
    versions = result.scalars().all()

    return [
        {
            "version": v.version,
            "question": v.question,
            "answer": v.answer,
            "edited_at": v.edited_at.isoformat() if v.edited_at else None,
            "edited_by": v.edited_by,
            "edit_reason": v.edit_reason,
        }
        for v in versions
    ]


@router.post(
    "",
    response_model=QAPairResponse,
    status_code=201,
    summary="Create a Q/A pair",
    description="Create a new Q/A pair. Embeddings are auto-generated.",
)
async def create_qa_pair(
    request: QAPairCreate,
    db: AsyncSession = Depends(get_db),
    api_key: str = Depends(verify_api_key),
):
    """Create a new Q/A pair with auto-generated embeddings."""
    from app.services.rag.embedder import EmbeddingService

    embedding_service = EmbeddingService()

    # Generate embeddings for question, answer, and combined
    q_emb = await embedding_service.embed_text(request.question)
    a_emb = await embedding_service.embed_text(request.answer)
    combined_emb = await embedding_service.embed_text(f"{request.question} {request.answer}")

    pair = QAPair(
        id=uuid.uuid4(),
        question=request.question,
        answer=request.answer,
        language=request.language or "en",
        source_type="manual",
        question_embedding=q_emb,
        answer_embedding=a_emb,
        combined_embedding=combined_emb,
    )

    db.add(pair)

    # Audit log
    audit = AuditLog(
        id=uuid.uuid4(),
        entity_type="qa_pair",
        entity_id=pair.id,
        action="create",
        new_data={"question": request.question, "answer": request.answer},
    )
    db.add(audit)
    await db.commit()
    await db.refresh(pair)

    logger.info(f"Created Q/A pair {pair.id}")

    return QAPairResponse(
        id=str(pair.id),
        question=pair.question,
        answer=pair.answer,
        subject=request.subject,
        topic=request.topic,
        language=pair.language or "en",
        version=1,
        created_at=pair.created_at or datetime.now(timezone.utc),
        updated_at=pair.updated_at or datetime.now(timezone.utc),
    )


@router.put(
    "/{qa_id}",
    response_model=QAPairResponse,
    summary="Update a Q/A pair",
    description="Edit a Q/A pair. Creates a version snapshot and re-generates embeddings.",
)
async def update_qa_pair(
    qa_id: str,
    request: QAPairUpdate,
    db: AsyncSession = Depends(get_db),
    api_key: str = Depends(verify_api_key),
):
    """Update a Q/A pair with version tracking."""
    result = await db.execute(
        select(QAPair).where(QAPair.id == _parse_uuid(qa_id))
    )
    pair = result.scalar_one_or_none()

    if not pair:
        raise HTTPException(status_code=404, detail="Q/A pair not found")

    if pair.is_deleted:
        raise HTTPException(status_code=410, detail="Q/A pair has been deleted")

    # Save current version to history
    version = QAVersion(
        id=uuid.uuid4(),
        qa_pair_id=pair.id,
        question=pair.question,
        answer=pair.answer,
        version=pair.version or 1,
        edit_reason=request.edit_reason,
    )
    db.add(version)

    # Update the pair
    old_data = {"question": pair.question, "answer": pair.answer}

    if request.question is not None:
        pair.question = request.question
    if request.answer is not None:
        pair.answer = request.answer

    pair.version = (pair.version or 1) + 1
    pair.updated_at = datetime.now(timezone.utc)

    # Re-generate embeddings
    from app.services.rag.embedder import EmbeddingService

    embedding_service = EmbeddingService()
    pair.question_embedding = await embedding_service.embed_text(pair.question)
    pair.answer_embedding = await embedding_service.embed_text(pair.answer)
    pair.combined_embedding = await embedding_service.embed_text(
        f"{pair.question} {pair.answer}"
    )

    # Audit log
    audit = AuditLog(
        id=uuid.uuid4(),
        entity_type="qa_pair",
        entity_id=pair.id,
        action="update",
        old_data=old_data,
        new_data={"question": pair.question, "answer": pair.answer},
    )
    db.add(audit)
    await db.commit()
    await db.refresh(pair)

    logger.info(f"Updated Q/A pair {pair.id} to version {pair.version}")

    return QAPairResponse(
        id=str(pair.id),
        question=pair.question,
        answer=pair.answer,
        subject=None,
        topic=None,
        language=pair.language or "en",
        version=pair.version or 1,
        created_at=pair.created_at or datetime.now(timezone.utc),
        updated_at=pair.updated_at or datetime.now(timezone.utc),
    )


@router.delete(
    "/{qa_id}",
    summary="Delete a Q/A pair",
    description="Soft-delete a Q/A pair. Use ?hard=true for permanent deletion.",
)
async def delete_qa_pair(
    qa_id: str,
    hard_delete: bool = Query(False, description="Permanently delete instead of soft delete"),
    db: AsyncSession = Depends(get_db),
    api_key: str = Depends(verify_api_key),
):
    """Soft-delete (or hard-delete) a Q/A pair."""
    result = await db.execute(
        select(QAPair).where(QAPair.id == _parse_uuid(qa_id))
    )
    pair = result.scalar_one_or_none()

    if not pair:
        raise HTTPException(status_code=404, detail="Q/A pair not found")

    if hard_delete:
        # Permanent deletion
        await db.delete(pair)
        action = "hard_delete"
    else:
        # Soft delete
        pair.is_deleted = True
        pair.updated_at = datetime.now(timezone.utc)
        action = "soft_delete"

    # Audit log
    audit = AuditLog(
        id=uuid.uuid4(),
        entity_type="qa_pair",
        entity_id=_parse_uuid(qa_id),
        action=action,
        old_data={"question": pair.question, "answer": pair.answer},
    )
    db.add(audit)
    await db.commit()

    logger.info(f"{'Hard' if hard_delete else 'Soft'}-deleted Q/A pair {qa_id}")
    return {"status": "deleted", "qa_id": qa_id, "type": "hard" if hard_delete else "soft"}


@router.post(
    "/{qa_id}/restore",
    response_model=QAPairResponse,
    summary="Restore a deleted Q/A pair",
)
async def restore_qa_pair(
    qa_id: str,
    db: AsyncSession = Depends(get_db),
    api_key: str = Depends(verify_api_key),
):
    """Restore a soft-deleted Q/A pair."""
    result = await db.execute(
        select(QAPair).where(QAPair.id == _parse_uuid(qa_id))
    )
    pair = result.scalar_one_or_none()

    if not pair:
        raise HTTPException(status_code=404, detail="Q/A pair not found")

    if not pair.is_deleted:
        raise HTTPException(status_code=400, detail="Q/A pair is not deleted")

    pair.is_deleted = False
    pair.updated_at = datetime.now(timezone.utc)

    audit = AuditLog(
        id=uuid.uuid4(),
        entity_type="qa_pair",
        entity_id=pair.id,
        action="restore",
    )
    db.add(audit)
    await db.commit()
    await db.refresh(pair)

    return QAPairResponse(
        id=str(pair.id),
        question=pair.question,
        answer=pair.answer,
        subject=None,
        topic=None,
        language=pair.language or "en",
        version=pair.version or 1,
        created_at=pair.created_at or datetime.now(timezone.utc),
        updated_at=pair.updated_at or datetime.now(timezone.utc),
    )


@router.post(
    "/bulk",
    summary="Bulk upload Q/A pairs",
    description="Upload multiple Q/A pairs at once. All pairs are embedded and stored.",
)
async def bulk_upload_qa(
    request: BulkQAUpload,
    db: AsyncSession = Depends(get_db),
    api_key: str = Depends(verify_api_key),
):
    """Bulk create Q/A pairs with auto-generated embeddings."""
    from app.services.rag.embedder import EmbeddingService

    embedding_service = EmbeddingService()

    # Batch embed all texts
    questions = [p.question for p in data.pairs]
    answers = [p.answer for p in data.pairs]
    combined = [f"{p.question} {p.answer}" for p in data.pairs]

    q_embeddings = await embedding_service.embed_batch(questions)
    a_embeddings = await embedding_service.embed_batch(answers)
    c_embeddings = await embedding_service.embed_batch(combined)

    created = []
    for i, pair_data in enumerate(data.pairs):
        pair = QAPair(
            id=uuid.uuid4(),
            question=pair_data.question,
            answer=pair_data.answer,
            language=pair_data.language or "en",
            source_type="manual",
            question_embedding=q_embeddings[i],
            answer_embedding=a_embeddings[i],
            combined_embedding=c_embeddings[i],
        )
        db.add(pair)
        created.append(str(pair.id))

    await db.commit()
    logger.info(f"Bulk created {len(created)} Q/A pairs")

    return {
        "status": "created",
        "count": len(created),
        "ids": created,
    }


