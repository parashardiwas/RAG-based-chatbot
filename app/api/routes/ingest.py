"""
Ingestion endpoint — upload files and documents to be chunked, embedded, and stored.
"""

import logging
import os
import pathlib
import re as _re
import uuid

from fastapi import APIRouter, BackgroundTasks, Depends, File, Form, HTTPException, UploadFile
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import get_settings
from app.db.database import get_db
from app.schemas.request import IngestFileRequest
from app.schemas.response import IngestResponse

import aiofiles
logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/v1/ingest", tags=["Ingestion"])


def _safe_filename(filename: str) -> str:
    """Strip directory components and dangerous characters from an uploaded filename."""
    name = pathlib.Path(filename).name  # strips all path components
    name = _re.sub(r'[^\w\-_. ]', '_', name)
    return name or "upload"


async def _process_file_background(
    file_path: str,
    filename: str,
    document_id: str,
    subject: str | None,
    topic: str | None,
    language: str | None,
):
    """Background task to process an uploaded file."""
    try:
        from app.services.media.file_parser import FileParser
        from app.services.rag.embedder import EmbeddingService
        from app.db.database import async_session_maker
        from app.db.models import Document, DocumentChunk
        import uuid as uuid_mod

        parser = FileParser()
        embedding_service = EmbeddingService()

        # Parse file into chunks
        chunks = await parser.parse_file(file_path, filename)
        logger.info(f"Parsed {len(chunks)} chunks from {filename}")

        # Generate embeddings in batch
        texts = [chunk["content"] for chunk in chunks]
        embeddings = await embedding_service.embed_batch(texts)

        # Store in database
        async with async_session_maker() as session:
            # Look up subject ONCE before the loop to avoid N+1 queries
            subject_id = None
            topic_id = None
            if subject:
                from app.db.models import Subject
                from sqlalchemy import select
                subj_res = await session.execute(select(Subject).where(Subject.name == subject))
                subj_row = subj_res.scalar_one_or_none()
                if not subj_row:
                    subj_row = Subject(name=subject)
                    session.add(subj_row)
                    await session.flush()
                subject_id = subj_row.id
            
            if topic:
                from app.db.models import Topic
                from sqlalchemy import select
                if not subject_id:
                    # Topic requires subject, get default or first subject
                    from app.db.models import Subject
                    subj_res = await session.execute(select(Subject).limit(1))
                    subj_row = subj_res.scalar_one_or_none()
                    subject_id = subj_row.id if subj_row else None
                
                if subject_id:
                    topic_res = await session.execute(
                        select(Topic).where(
                            (Topic.subject_id == subject_id) & 
                            (Topic.name == topic)
                        )
                    )
                    topic_row = topic_res.scalar_one_or_none()
                    if not topic_row:
                        topic_row = Topic(subject_id=subject_id, name=topic)
                        session.add(topic_row)
                        await session.flush()
                    topic_id = topic_row.id
            
            for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
                db_chunk = DocumentChunk(
                    id=uuid_mod.uuid4(),
                    document_id=uuid_mod.UUID(document_id),
                    chunk_index=i,
                    content=chunk["content"],
                    language=language or chunk.get("language"),
                    source_file=filename,
                    source_type=chunk.get("source_type", "document"),
                    metadata_=chunk.get("metadata", {}),
                    embedding=embedding,
                    subject_id=subject_id,
                    topic_id=topic_id,
                )
                session.add(db_chunk)

            # Update document status
            from sqlalchemy import update
            await session.execute(
                update(Document)
                .where(Document.id == uuid_mod.UUID(document_id))
                .values(status="completed", chunk_count=len(chunks))
            )
            await session.commit()

        # Invalidate the BM25 cache so new chunks are searchable immediately
        try:
            from app.core.orchestrator import get_orchestrator
            orchestrator = await get_orchestrator()
            if hasattr(orchestrator, '_retrieval_service') and orchestrator._retrieval_service:
                orchestrator._retrieval_service.invalidate_bm25_cache()
        except Exception as e:
            logger.warning(f"Failed to invalidate BM25 cache: {e}")

        if len(chunks) == 0:
            logger.warning(
                f"File {filename} produced 0 chunks — it may be a scanned PDF "
                f"(image-based) or an unsupported format."
            )

        logger.info(f"Successfully ingested {filename}: {len(chunks)} chunks stored")

    except Exception as e:
        logger.error(f"Failed to process file {filename}: {e}", exc_info=True)
        # Update document status to failed
        try:
            from app.db.database import async_session_maker
            from app.db.models import Document
            from sqlalchemy import update
            import uuid as uuid_mod

            async with async_session_maker() as session:
                await session.execute(
                    update(Document)
                    .where(Document.id == uuid_mod.UUID(document_id))
                    .values(status="failed", metadata={"error": str(e)})
                )
                await session.commit()
        except Exception:
            pass


@router.post(
    "/file",
    response_model=IngestResponse,
    summary="Upload and ingest a file",
    description="Upload a PDF, DOCX, TXT, CSV, or other file. "
                "The file is parsed, chunked, embedded, and stored for RAG retrieval.",
)
async def ingest_file(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(..., description="File to ingest"),
    subject: str | None = Form(None, description="Subject category"),
    topic: str | None = Form(None, description="Topic within subject"),
    language: str | None = Form(None, description="Document language"),
    db: AsyncSession = Depends(get_db),
):
    """Upload and process a file for RAG ingestion."""
    settings = get_settings()

    # Validate file size
    content = await file.read()
    size_mb = len(content) / (1024 * 1024)
    if size_mb > settings.max_upload_size_mb:
        raise HTTPException(
            status_code=413,
            detail=f"File too large: {size_mb:.1f}MB (max {settings.max_upload_size_mb}MB)",
        )

    # Save file (use absolute path so it works regardless of server CWD)
    document_id = str(uuid.uuid4())
    abs_upload_dir = os.path.abspath(os.path.join(settings.upload_dir, document_id))
    os.makedirs(abs_upload_dir, exist_ok=True)
    safe_name = _safe_filename(file.filename or "upload")
    file_path = os.path.join(abs_upload_dir, safe_name)
    # Validate resolved path stays within upload directory
    resolved = pathlib.Path(file_path).resolve()
    allowed = pathlib.Path(abs_upload_dir).resolve()
    if not str(resolved).startswith(str(allowed)):
        raise HTTPException(status_code=400, detail="Invalid filename")

    # Use async file I/O to avoid blocking event loop
    async with aiofiles.open(file_path, "wb") as f:
        await f.write(content)

    # Check for duplicate document by file hash
    import hashlib
    from app.db.models import Document
    from sqlalchemy import select

    file_hash_value = hashlib.sha256(content).hexdigest()
    existing_doc = await db.execute(
        select(Document).where(Document.file_hash == file_hash_value)
    )
    if existing_doc.scalar_one_or_none():
        raise HTTPException(
            status_code=409,
            detail="This document has already been ingested. Delete it first to re-ingest."
        )

    # Create document record
    doc = Document(
        id=uuid.UUID(document_id),
        filename=file.filename or "upload",
        file_type=file.content_type,
        file_path=file_path,
        file_hash=file_hash_value,
        language=language,
        status="processing",
    )
    db.add(doc)
    await db.commit()

    # Process in background
    background_tasks.add_task(
        _process_file_background,
        file_path=file_path,
        filename=file.filename or "upload",
        document_id=document_id,
        subject=subject,
        topic=topic,
        language=language,
    )

    return IngestResponse(
        document_id=document_id,
        status="processing",
        chunks_created=0,
        subject=subject,
        topic=topic,
    )


@router.get(
    "/status/{document_id}",
    summary="Check ingestion status",
)
async def check_ingestion_status(
    document_id: str,
    db: AsyncSession = Depends(get_db),
):
    """Check the processing status of an uploaded document."""
    from app.db.models import Document
    from sqlalchemy import select

    result = await db.execute(
        select(Document).where(Document.id == uuid.UUID(document_id))
    )
    doc = result.scalar_one_or_none()

    if not doc:
        raise HTTPException(status_code=404, detail="Document not found")

    return {
        "document_id": str(doc.id),
        "filename": doc.filename,
        "status": doc.status,
        "chunks_created": doc.chunk_count,
        "created_at": doc.created_at.isoformat() if doc.created_at else None,
    }


@router.get(
    "/documents",
    summary="List uploaded documents",
)
async def list_documents(
    limit: int = 20,
    offset: int = 0,
    db: AsyncSession = Depends(get_db),
):
    """Get a paginated list of uploaded documents.
    
    Args:
        limit: Maximum number of documents to return (default 20)
        offset: Number of documents to skip (default 0)
    """
    from app.db.models import Document
    from sqlalchemy import select, func

    # Get total count
    count_result = await db.execute(select(func.count(Document.id)))
    total_count = count_result.scalar()

    # Get paginated results
    result = await db.execute(
        select(Document)
        .order_by(Document.created_at.desc())
        .limit(limit)
        .offset(offset)
    )
    docs = result.scalars().all()

    return {
        "documents": [
            {
                "id": str(doc.id),
                "filename": doc.filename,
                "status": doc.status,
                "chunks_created": doc.chunk_count,
                "created_at": doc.created_at.isoformat() if doc.created_at else None,
            }
            for doc in docs
        ],
        "total": total_count,
        "limit": limit,
        "offset": offset,
        "has_more": offset + limit < total_count,
    }


@router.get(
    "/documents/{document_id}/view",
    summary="View an uploaded document",
)
async def view_document(
    document_id: str,
    db: AsyncSession = Depends(get_db),
):
    """View an uploaded document directly in the browser."""
    from app.db.models import Document
    from sqlalchemy import select
    import os
    from fastapi.responses import FileResponse
    import uuid

    result = await db.execute(
        select(Document).where(Document.id == uuid.UUID(document_id))
    )
    doc = result.scalar_one_or_none()

    if not doc:
        raise HTTPException(status_code=404, detail="Document not found")
        
    if not doc.file_path or not os.path.exists(doc.file_path):
        raise HTTPException(status_code=404, detail="File content not found on server")
        
    # Determine basic media type (for PDFs)
    media_type = doc.file_type
    if not media_type and doc.filename.lower().endswith('.pdf'):
        media_type = "application/pdf"
        
    return FileResponse(
        path=doc.file_path, 
        filename=doc.filename, 
        media_type=media_type, 
        content_disposition_type="inline"
    )


@router.delete(
    "/documents/{document_id}",
    summary="Delete an uploaded document",
)
async def delete_document(
    document_id: str,
    db: AsyncSession = Depends(get_db),
):
    """Delete a document and all its chunks from the database."""
    from app.db.models import Document
    from sqlalchemy import select
    import os
    import uuid

    result = await db.execute(
        select(Document).where(Document.id == uuid.UUID(document_id))
    )
    doc = result.scalar_one_or_none()

    if not doc:
        raise HTTPException(status_code=404, detail="Document not found")

    # Guard: prevent deleting a document while it's still being processed
    if doc.status == "processing":
        raise HTTPException(
            status_code=409,
            detail="Cannot delete a document that is still being processed. Please wait for ingestion to complete.",
        )

    # Delete physical file if it exists
    if doc.file_path and os.path.exists(doc.file_path):
        try:
            os.remove(doc.file_path)
            # Try to remove the directory if it's empty
            dir_path = os.path.dirname(doc.file_path)
            if not os.listdir(dir_path):
                os.rmdir(dir_path)
        except Exception as e:
            logger.warning(f"Failed to delete physical file {doc.file_path}: {e}")

    # Delete from database (chunks will cascade)
    await db.delete(doc)
    await db.commit()

    return {"status": "success", "message": "Document deleted successfully"}

