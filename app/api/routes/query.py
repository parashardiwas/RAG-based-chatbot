"""
Query endpoint — the main /ask route for processing questions.
Handles text, audio, video, and file-based queries.
"""

import logging
import time
import uuid

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import get_settings
from app.core.orchestrator import get_orchestrator
from app.core.queue_manager import get_queue_manager
from app.db.database import get_db
from app.schemas.request import QueryRequest, CompareRequest
from app.schemas.response import ErrorResponse, QueryResponse, SourceInfo
from typing import Dict, Any

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/v1", tags=["Query"])

@router.post(
    "/compare",
    response_model=Dict[str, Any],
    summary="Compare user answer to ground truth",
    description="Translates the Q&A, retrieves the true answer, and returns YES/NO match."
)
async def compare_answer(
    request: CompareRequest,
    db: AsyncSession = Depends(get_db),
):
    try:
        from app.services.language.translator import TranslatorService
        translator = TranslatorService()
        
        # 1. Translate Q&A to English
        q_trans = await translator.translate_to_english(request.question)
        a_trans = await translator.translate_to_english(request.user_answer)
        
        eng_question = q_trans["english_text"]
        eng_user_answer = a_trans["english_text"]
        
        # 2. Get true answer using orchestrator
        orchestrator = await get_orchestrator()
        result = await orchestrator.process_text_query(
            text=eng_question,
            language="en"
        )
        eng_true_answer = result.answer
        
        # 3. Compare with LLM
        is_match = await translator.compare_answers(
            question=eng_question,
            answer_a=eng_user_answer,
            answer_b=eng_true_answer
        )
        
        return {
            "match": "YES" if is_match else "NO",
            "detected_language": q_trans["original_language"],
            "true_answer_preview": eng_true_answer[:100] + "..."
        }
    except Exception as e:
        logger.error(f"Comparison error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal comparison error")


@router.post(
    "/ask",
    response_model=QueryResponse,
    responses={
        202: {"description": "Request queued (at capacity)"},
        429: {"description": "Rate limit exceeded"},
        500: {"model": ErrorResponse},
    },
    summary="Ask a question",
    description="Submit a text question and get a RAG-powered answer. "
                "Supports Hindi, English, and Hinglish. Response is in the same language as input.",
)
async def ask_question(
    request: QueryRequest,
    db: AsyncSession = Depends(get_db),
):
    """Process a text-based question through the RAG pipeline."""
    queue_manager = get_queue_manager()

    # Check concurrency limits
    acquired = await queue_manager.acquire()
    if not acquired:
        queued = queue_manager.get_queue_position()
        return {
            "status": "queued",
            "queue_position": queued.position,
            "estimated_wait_seconds": queued.position * 3,
            "request_id": queued.request_id,
        }

    try:
        if not request.text or not request.text.strip():
            raise HTTPException(status_code=400, detail="Question text is required")

        orchestrator = await get_orchestrator()
        try:
            result = await orchestrator.process_text_query(
                text=request.text.strip(),
                language=request.language,
                subject_filter=request.subject,
                topic_filter=request.topic,
            )
        except Exception as e:
            import logging
            logging.getLogger(__name__).error(f"Orchestrator failed: {e}", exc_info=True)
            raise HTTPException(status_code=500, detail="Internal processing error")

        return QueryResponse(
            answer=result.answer,
            language=result.language,
            confidence=round(result.confidence, 3),
            retrieval_confidence=round(result.retrieval_confidence, 3),
            sources=[
                SourceInfo(**s) for s in result.sources
            ],
            model_used=result.model_used,
            latency_ms=result.latency_ms,
            cached=result.cached,
        )

    finally:
        await queue_manager.release()


@router.post(
    "/ask/audio",
    response_model=QueryResponse,
    summary="Ask via audio",
    description="Upload an audio file to ask a question. "
                "Audio is transcribed using Whisper, then processed through RAG.",
)
async def ask_audio(
    file: UploadFile = File(..., description="Audio file (mp3, wav, m4a, etc.)"),
    language: str | None = Form(None, description="Override language detection"),
    subject: str | None = Form(None),
    topic: str | None = Form(None),
    db: AsyncSession = Depends(get_db),
):
    """Process an audio question through ASR + RAG pipeline."""
    queue_manager = get_queue_manager()
    acquired = await queue_manager.acquire()
    if not acquired:
        queued = queue_manager.get_queue_position()
        raise HTTPException(status_code=202, detail=f"Queued at position {queued.position}")

    try:
        # Save uploaded file
        from app.services.media.audio_processor import AudioProcessor
        
        audio_processor = AudioProcessor()
        transcript_result = await audio_processor.transcribe_upload(file)

        if transcript_result["confidence"] < get_settings().asr_confidence_threshold:
            return QueryResponse(
                answer=f"Transcription confidence is low ({transcript_result['confidence']:.0%}). "
                       f"Transcribed text: \"{transcript_result['text']}\"\n\n"
                       f"Please confirm this is correct or try again with clearer audio.",
                language=transcript_result.get("language", "en"),
                confidence=transcript_result["confidence"],
                retrieval_confidence=0.0,
                sources=[],
                model_used="whisper (transcription only)",
                latency_ms=transcript_result.get("latency_ms", 0),
                cached=False,
            )

        orchestrator = await get_orchestrator()
        result = await orchestrator.process_text_query(
            text=transcript_result["text"],
            language=language or transcript_result.get("language"),
            subject_filter=subject,
            topic_filter=topic,
        )

        return QueryResponse(
            answer=result.answer,
            language=result.language,
            confidence=round(result.confidence, 3),
            retrieval_confidence=round(result.retrieval_confidence, 3),
            sources=[SourceInfo(**s) for s in result.sources],
            model_used=result.model_used,
            latency_ms=result.latency_ms,
            cached=result.cached,
        )

    finally:
        await queue_manager.release()


@router.post(
    "/ask/video",
    response_model=QueryResponse,
    summary="Ask via video",
    description="Upload a video file. Audio track is transcribed; "
                "if no audio, key frames are OCR'd for text extraction.",
)
async def ask_video(
    file: UploadFile = File(..., description="Video file (mp4, avi, mkv, etc.)"),
    language: str | None = Form(None),
    subject: str | None = Form(None),
    topic: str | None = Form(None),
    db: AsyncSession = Depends(get_db),
):
    """Process a video question through video processing + RAG pipeline."""
    queue_manager = get_queue_manager()
    acquired = await queue_manager.acquire()
    if not acquired:
        queued = queue_manager.get_queue_position()
        raise HTTPException(status_code=202, detail=f"Queued at position {queued.position}")

    try:
        from app.services.media.video_processor import VideoProcessor

        video_processor = VideoProcessor()
        video_result = await video_processor.process_upload(file)

        if not video_result["text"].strip():
            return QueryResponse(
                answer="Could not extract any text from the video. "
                       "Please ensure the video has audio or visible text.",
                language="en",
                confidence=0.0,
                retrieval_confidence=0.0,
                sources=[],
                model_used="none",
                latency_ms=video_result.get("latency_ms", 0),
                cached=False,
            )

        orchestrator = await get_orchestrator()
        result = await orchestrator.process_text_query(
            text=video_result["text"],
            language=language or video_result.get("language"),
            subject_filter=subject,
            topic_filter=topic,
        )

        return QueryResponse(
            answer=result.answer,
            language=result.language,
            confidence=round(result.confidence, 3),
            retrieval_confidence=round(result.retrieval_confidence, 3),
            sources=[SourceInfo(**s) for s in result.sources],
            model_used=result.model_used,
            latency_ms=result.latency_ms,
            cached=result.cached,
        )

    finally:
        await queue_manager.release()
