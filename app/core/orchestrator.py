"""
Main orchestrator that coordinates the entire request pipeline:
Input → Media Processing → Language Detection → RAG Retrieval → Generation → Confidence Check → Response
"""

import asyncio
import logging
import time
import uuid
from dataclasses import dataclass
from typing import Any

from app.config import get_settings
from app.core.cost_tracker import RequestCost, get_cost_tracker
from app.core.queue_manager import get_queue_manager

logger = logging.getLogger(__name__)


@dataclass
class OrchestratorResult:
    """Complete result from the orchestration pipeline."""
    request_id: str
    answer: str
    language: str
    confidence: float
    retrieval_confidence: float
    sources: list[dict[str, Any]]
    model_used: str
    latency_ms: int
    cached: bool = False
    input_text: str = ""
    detected_language: str = ""
    fallback_used: bool = False
    cost: RequestCost | None = None


_background_tasks: set[asyncio.Task] = set()

class Orchestrator:
    """
    Central orchestrator that routes requests through the processing pipeline.
    
    Pipeline stages:
    1. Input normalization (text/audio/video/file → text)
    2. Language detection
    3. Cache check
    4. Topic routing
    5. Hybrid retrieval (vector + BM25)
    6. Model selection (cheap vs strong based on confidence)
    7. Answer generation
    8. Confidence scoring (80% match check)
    9. Fallback handling ("I don't know")
    10. Cost tracking + logging
    """

    def __init__(self):
        self._settings = get_settings()
        self._embedding_service = None
        self._retrieval_service = None
        self._llm_generator = None
        self._confidence_scorer = None
        self._translator_service = None
        self._redis = None

    def _fire_and_forget(self, coro):
        task = asyncio.create_task(coro)
        _background_tasks.add(task)
        task.add_done_callback(_background_tasks.discard)

    async def init(self, db_session_factory, redis_client=None):
        """Initialize all services. Call once at app startup."""
        try:
            from app.services.rag.embedder import EmbeddingService
            from app.services.rag.retriever import RetrievalService
            from app.services.llm.generator import LLMGenerator
            from app.services.llm.confidence import ConfidenceScorer
            from app.services.language.translator import TranslatorService

            redis_url = self._settings.redis_url if hasattr(self._settings, 'redis_url') else 'redis://localhost:6379/0'
            self._embedding_service = EmbeddingService(redis_url=redis_url)
            self._retrieval_service = RetrievalService(db_session_factory)
            self._llm_generator = LLMGenerator()
            self._confidence_scorer = ConfidenceScorer()
            self._translator_service = TranslatorService()
            self._redis = redis_client
            self._db_session_factory = db_session_factory
        except Exception as e:
            logger.error(f"Failed to initialize orchestrator services: {e}", exc_info=True)
            raise

    async def process_text_query(
        self,
        text: str,
        language: str | None = None,
        subject_filter: str | None = None,
        topic_filter: str | None = None,
    ) -> OrchestratorResult:
        """
        Process a text query through the full RAG pipeline (English-centric).
        
        Args:
            text: The user's question
            language: Override language (auto-detected if None)
            subject_filter: Restrict retrieval to a subject
            topic_filter: Restrict retrieval to a topic
            
        Returns:
            OrchestratorResult with answer, confidence, sources, etc.
        """
        request_id = str(uuid.uuid4())
        start_time = time.time()
        cost = RequestCost(request_id=request_id)

        try:
            # ── Step 1: Translate to English ────────────────────
            trans_result = await self._translator_service.translate_to_english(text)
            english_question = trans_result["english_text"]
            original_language = language or trans_result["original_language"]
            
            logger.info(f"[{request_id}] Detected language: {original_language}. "
                       f"English question: {english_question}")

            # ── Step 2: Cache Check (Exact Match) ───────────────
            cached_answer = await self._check_cache(english_question, original_language, topic_filter)
            if cached_answer:
                latency = int((time.time() - start_time) * 1000)
                return OrchestratorResult(
                    request_id=request_id,
                    answer=cached_answer["answer"],
                    language=original_language,
                    confidence=cached_answer["confidence"],
                    retrieval_confidence=cached_answer["retrieval_confidence"],
                    sources=cached_answer["sources"],
                    model_used=cached_answer["model_used"],
                    latency_ms=latency,
                    cached=True,
                    input_text=text,
                    detected_language=original_language,
                )

            # ── Step 3: Exact Match QA Cache Check ──────────────
            exact_match = await self._check_exact_qa_cache(english_question, language)
            if exact_match:
                logger.info(f"[{request_id}] Exact QA Match found! Confidence: {exact_match['confidence']:.2f}")
                
                # Retranslate Answer
                final_answer = await self._translator_service.translate_from_english(exact_match['answer'], original_language)
                
                latency = int((time.time() - start_time) * 1000)
                cost.total_latency_ms = latency
                return OrchestratorResult(
                    request_id=request_id,
                    answer=final_answer,
                    language=original_language,
                    confidence=exact_match['confidence'],
                    retrieval_confidence=exact_match['retrieval_confidence'],
                    sources=exact_match.get('sources', []),
                    model_used="qa_pair_cache",
                    latency_ms=latency,
                    cached=True,
                    input_text=text,
                    detected_language=original_language,
                    cost=cost
                )

            # ── Step 4: Sequential Embed & Semantic QA Lookup ──────────────
            query_embedding = await self._embedding_service.embed_query(english_question)
            qa_results = await self._retrieval_service.retrieve_qa_pairs(
                query_embedding=query_embedding,
                query_text=english_question,
                top_k=1,
                subject_filter=subject_filter,
                topic_filter=topic_filter
            )
            cost.embedding_tokens = len(english_question.split()) * 2

            # High confidence threshold for returning a QA Pair
            qa_confidence_threshold = 0.75
            if qa_results and qa_results[0].similarity_score >= qa_confidence_threshold:
                best_qa = qa_results[0]
                logger.info(f"[{request_id}] Q/A Pair match found! Confidence: {best_qa.similarity_score:.2f}")
                
                # Fetch exact answer from DB (content has question + answer)
                async with self._db_session_factory() as session:
                    from app.db.models import QAPair
                    from sqlalchemy import select
                    stmt = select(QAPair).where(QAPair.id == best_qa.chunk_id)
                    res = await session.execute(stmt)
                    qa_row = res.scalar_one_or_none()
                    
                if qa_row:
                    english_answer = qa_row.answer
                    
                    # ── Retranslate Answer ────────────────────────
                    final_answer = await self._translator_service.translate_from_english(english_answer, original_language)
                    
                    # Cache in redis for exact subsequent queries
                    result_dict = {
                        "answer": final_answer,
                        "confidence": best_qa.similarity_score,
                        "retrieval_confidence": best_qa.similarity_score,
                        "sources": [],
                        "model_used": "qa_pair_cache"
                    }
                    self._fire_and_forget(self._cache_result(english_question, original_language, topic_filter, result_dict))
                    
                    latency = int((time.time() - start_time) * 1000)
                    cost.total_latency_ms = latency
                    return OrchestratorResult(
                        request_id=request_id,
                        answer=final_answer,
                        language=original_language,
                        confidence=best_qa.similarity_score,
                        retrieval_confidence=best_qa.similarity_score,
                        sources=[],
                        model_used="qa_pair_cache",
                        latency_ms=latency,
                        cached=True,
                        input_text=text,
                        detected_language=original_language,
                        cost=cost
                    )

            # ── Step 5: Hybrid RAG Retrieval ────────────────────
            chunks = await self._retrieval_service.retrieve(
                query=english_question,
                query_embedding=query_embedding,
                top_k=10,
                subject_filter=subject_filter,
                topic_filter=topic_filter,
            )

            retrieval_confidence = self._retrieval_service.compute_confidence(chunks)
            logger.info(f"[{request_id}] Retrieved {len(chunks)} chunks, confidence: {retrieval_confidence:.2f}")

            # Confidence Gate — Early Exit
            if retrieval_confidence < self._settings.retrieval_confidence_low:
                latency = int((time.time() - start_time) * 1000)
                fallback_answer = self._get_fallback_answer(original_language, text)
                cost.total_latency_ms = latency
                return OrchestratorResult(
                    request_id=request_id,
                    answer=fallback_answer,
                    language=original_language,
                    confidence=retrieval_confidence,
                    retrieval_confidence=retrieval_confidence,
                    sources=[],
                    model_used="none (low confidence fallback)",
                    latency_ms=latency,
                    input_text=text,
                    detected_language=original_language,
                    fallback_used=True,
                    cost=cost,
                )

            # ── Step 6: Model Selection & Generation ────────────
            model = self._settings.openai_model
            cost.model_used = model

            if retrieval_confidence > 0.65:
                valid_chunks = [c for c in chunks if c.similarity_score >= 0.25][:1]
            else:
                valid_chunks = [c for c in chunks if c.similarity_score >= 0.25][:2]
            
            if not valid_chunks:
                latency = int((time.time() - start_time) * 1000)
                return OrchestratorResult(
                    request_id=request_id,
                    answer=self._get_fallback_answer(original_language, text),
                    language=original_language,
                    confidence=0.0,
                    retrieval_confidence=retrieval_confidence,
                    sources=[],
                    model_used="none (low context match)",
                    latency_ms=latency,
                    input_text=text,
                    detected_language=original_language,
                    fallback_used=True,
                    cost=cost,
                )

            context_block = "\n\n".join(c.content for c in valid_chunks)
            generation_result = await self._llm_generator.generate(
                prompt=english_question,
                context=context_block,
                language="en", # Always generate in English internally
                model=model,
            )
            cost.llm_input_tokens = generation_result.total_tokens // 2
            cost.llm_output_tokens = generation_result.total_tokens - cost.llm_input_tokens
            english_answer = generation_result.answer

            # ── Step 7: Confidence Scoring ──────────────────────
            try:
                texts_to_embed = [english_answer] + [c.content for c in valid_chunks[:2]]
                embeddings = await self._embedding_service.embed_batch(texts_to_embed)
                answer_embedding = embeddings[0]
                source_embeddings = embeddings[1:]
                confidence_result = self._confidence_scorer.score_answer(
                    answer_embedding=answer_embedding,
                    source_embeddings=source_embeddings,
                    retrieval_scores=[c.similarity_score for c in valid_chunks[:2]],
                    threshold=self._settings.answer_match_threshold,
                )
                answer_confidence = confidence_result.answer_confidence
                meets_threshold = confidence_result.meets_threshold
            except Exception as e:
                logger.warning(f"[{request_id}] Confidence scoring failed, falling back: {e}")
                answer_confidence = min(retrieval_confidence * 1.1, 1.0)
                meets_threshold = answer_confidence >= self._settings.answer_match_threshold

            # ── Step 8: Cache and Store Q/A Pair (Async) ────────
            # Gate the insert on meets_threshold to prevent self-poisoning
            if meets_threshold:
                try:
                    # Store the successful English Q/A pair
                    async with self._db_session_factory() as session:
                        from app.db.models import QAPair
                        from sqlalchemy import select
                        
                        # Prevent exact duplicate insertions
                        stmt = select(QAPair).where(QAPair.question == english_question)
                        res = await session.execute(stmt)
                        existing_qa = res.scalar_one_or_none()
                        
                        if not existing_qa:
                            new_qa = QAPair(
                                question=english_question,
                                answer=english_answer,
                                question_embedding=query_embedding,
                                answer_embedding=answer_embedding,
                                combined_embedding=query_embedding, # simplification
                                language="en",
                                source_type="rag_generated"
                            )
                            session.add(new_qa)
                            await session.commit()
                except Exception as e:
                    logger.warning(f"[{request_id}] Failed to save Q/A Pair: {e}")

            # ── Step 9: Final Confidence Gate & Retranslate ───
            final_answer = english_answer
            if not meets_threshold:
                logger.warning(
                    f"[{request_id}] Answer confidence {answer_confidence:.2f} "
                    f"below threshold {self._settings.answer_match_threshold}"
                )
                disclaimer = self._get_low_confidence_disclaimer(original_language)
                final_answer = f"{final_answer}\n\n{disclaimer}"

            final_answer = await self._translator_service.translate_from_english(final_answer, original_language)

            # Cache the result — only include valid_chunks (actually used)
            if meets_threshold:
                self._fire_and_forget(self._cache_result(english_question, original_language, topic_filter, {
                    "answer": final_answer,
                    "confidence": answer_confidence,
                    "retrieval_confidence": retrieval_confidence,
                    "sources": [{"chunk_id": str(c.chunk_id), "content_preview": c.content[:200],
                                "similarity": c.similarity_score, "source_file": c.source_file,
                                "topic": c.topic} for c in valid_chunks],
                    "model_used": generation_result.model_used,
                }))

            # ── Step 10: Cache + Log ──────────────────────────
            latency = int((time.time() - start_time) * 1000)
            cost.total_latency_ms = latency

            # Track cost
            cost_tracker = await get_cost_tracker()
            cost = cost_tracker.calculate_cost(cost)
            self._fire_and_forget(cost_tracker.log_cost(cost))

            # Only report chunks that were actually used for generation
            sources = [
                {
                    "chunk_id": str(c.chunk_id),
                    "content_preview": c.content[:200],
                    "similarity": round(c.similarity_score, 3),
                    "source_file": c.source_file,
                    "topic": c.topic,
                }
                for c in valid_chunks
            ]

            # Log query for audit trail
            self._fire_and_forget(self._log_query(
                request_id=request_id,
                input_text=text,
                detected_language=original_language,
                retrieved_chunk_ids=[c.chunk_id for c in valid_chunks],
                answer=final_answer,
                retrieval_confidence=retrieval_confidence,
                answer_confidence=answer_confidence,
                model_used=generation_result.model_used,
                total_tokens=generation_result.total_tokens,
                latency_ms=latency,
            ))

            return OrchestratorResult(
                request_id=request_id,
                answer=final_answer,
                language=original_language,
                confidence=answer_confidence,
                retrieval_confidence=retrieval_confidence,
                sources=sources,
                model_used=generation_result.model_used,
                latency_ms=latency,
                input_text=text,
                detected_language=original_language,
                fallback_used=not meets_threshold,
                cost=cost,
            )

        except Exception as e:
            latency = int((time.time() - start_time) * 1000)
            logger.error(f"[{request_id}] Pipeline error: {e}", exc_info=True)
            raise

    async def process_text_query_stream(
        self,
        text: str,
        language: str | None = None,
        subject_filter: str | None = None,
        topic_filter: str | None = None,
    ):
        """Streaming version of the pipeline. Yields SSE events."""
        import json
        start_time = time.time()
        
        # ── Step 1: Translate to English ────────────────────
        trans_result = await self._translator_service.translate_to_english(text)
        english_question = trans_result["english_text"]
        original_language = language or trans_result["original_language"]
        
        # ── Step 2: Cache Check (Exact Match) ───────────────
        exact_match = await self._check_exact_qa_cache(english_question, original_language)
        if exact_match:
            latency = int((time.time() - start_time) * 1000)
            yield f"data: {json.dumps({'metadata': {'latency_ms': latency, 'confidence': exact_match['confidence'], 'retrieval_confidence': exact_match['retrieval_confidence'], 'model_used': 'qa_pair_cache', 'sources': exact_match.get('sources', []), 'cached': True}})}\n\n"
            yield f"data: {json.dumps({'chunk': exact_match['answer']})}\n\n"
            yield "data: [DONE]\n\n"
            return
            
        # ── Step 3: Embed & QA Lookup ───────────────────────
        query_embedding = await self._embedding_service.embed_query(english_question)
        qa_results = await self._retrieval_service.retrieve_qa_pairs(
            query_embedding=query_embedding,
            query_text=english_question,
            top_k=1,
            subject_filter=subject_filter,
            topic_filter=topic_filter
        )
        
        if qa_results and qa_results[0].similarity_score >= 0.75:
            best_qa = qa_results[0]
            async with self._db_session_factory() as session:
                from app.db.models import QAPair
                from sqlalchemy import select
                stmt = select(QAPair).where(QAPair.id == best_qa.chunk_id)
                res = await session.execute(stmt)
                qa_row = res.scalar_one_or_none()
            if qa_row:
                # Translate the English QA answer back to the user's language
                translated_answer = await self._translator_service.translate_from_english(
                    qa_row.answer, original_language
                )
                latency = int((time.time() - start_time) * 1000)
                yield f"data: {json.dumps({'metadata': {'latency_ms': latency, 'confidence': best_qa.similarity_score, 'retrieval_confidence': best_qa.similarity_score, 'model_used': 'qa_pair_cache', 'sources': [], 'cached': True}})}\n\n"
                yield f"data: {json.dumps({'chunk': translated_answer})}\n\n"
                yield "data: [DONE]\n\n"
                return

        # ── Step 4: Hybrid RAG Retrieval ────────────────────
        chunks = await self._retrieval_service.retrieve(
            query=english_question,
            query_embedding=query_embedding,
            top_k=10,
            subject_filter=subject_filter,
            topic_filter=topic_filter,
        )

        retrieval_confidence = self._retrieval_service.compute_confidence(chunks)
        if retrieval_confidence < self._settings.retrieval_confidence_low:
            fallback = self._get_fallback_answer(original_language, text)
            latency = int((time.time() - start_time) * 1000)
            yield f"data: {json.dumps({'metadata': {'latency_ms': latency, 'retrieval_confidence': retrieval_confidence, 'confidence': 0.0, 'model_used': 'none (low confidence fallback)', 'sources': [], 'cached': False}})}\n\n"
            yield f"data: {json.dumps({'chunk': fallback})}\n\n"
            yield "data: [DONE]\n\n"
            return

        if retrieval_confidence > 0.65:
            valid_chunks = [c for c in chunks if c.similarity_score >= 0.25][:1]
        else:
            valid_chunks = [c for c in chunks if c.similarity_score >= 0.25][:2]
            
        if not valid_chunks:
            fallback = self._get_fallback_answer(original_language, text)
            latency = int((time.time() - start_time) * 1000)
            yield f"data: {json.dumps({'metadata': {'latency_ms': latency, 'retrieval_confidence': retrieval_confidence, 'confidence': 0.0, 'model_used': 'none (low context match)', 'sources': [], 'cached': False}})}\n\n"
            yield f"data: {json.dumps({'chunk': fallback})}\n\n"
            yield "data: [DONE]\n\n"
            return

        context_block = "\n\n".join(c.content for c in valid_chunks)
        
        sources = [
            {
                "chunk_id": str(c.chunk_id),
                "content_preview": c.content[:200],
                "similarity": round(c.similarity_score, 3),
                "source_file": c.source_file,
                "topic": c.topic,
            }
            for c in valid_chunks
        ]
        latency = int((time.time() - start_time) * 1000)
        metadata = {
            "metadata": {
                "latency_ms": latency,
                "retrieval_confidence": retrieval_confidence,
                "confidence": retrieval_confidence, # Estimate
                "model_used": self._settings.openai_model,
                "sources": sources,
                "cached": False
            }
        }
        yield f"data: {json.dumps(metadata)}\n\n"

        # ── Step 5: Generation Stream ───────────────────────
        full_answer_chunks = []
        async for chunk in self._llm_generator.generate_stream(
            prompt=english_question,
            context=context_block,
            language=original_language
        ):
            full_answer_chunks.append(chunk)
            yield f"data: {json.dumps({'chunk': chunk})}\n\n"
            
        full_answer = "".join(full_answer_chunks)
        
        # ── Step 6: Store Q/A Pair ──────────────────────────
        try:
            # Generate answer embedding (using query embedding for now or embed it)
            # In process_text_query it does embed_batch, here we just use query_embedding as simplification to avoid blocking stream end too long
            async with self._db_session_factory() as session:
                from app.db.models import QAPair
                from sqlalchemy import select
                
                stmt = select(QAPair).where(QAPair.question == english_question)
                res = await session.execute(stmt)
                existing_qa = res.scalar_one_or_none()
                
                if not existing_qa:
                    new_qa = QAPair(
                        question=english_question,
                        answer=full_answer,
                        question_embedding=query_embedding,
                        answer_embedding=query_embedding, # Simplify
                        combined_embedding=query_embedding,
                        language="en",
                        source_type="rag_generated"
                    )
                    session.add(new_qa)
                    await session.commit()
        except Exception as e:
            logger.warning(f"Failed to save streamed Q/A Pair: {e}")

        yield "data: [DONE]\n\n"

    def _get_fallback_answer(self, language: str, query: str) -> str:
        """Generate a structured 'I don't know' response in the appropriate language."""
        if language == "hi":
            return (
                "मुझे इस प्रश्न का उत्तर देने के लिए पर्याप्त जानकारी नहीं मिली।\n\n"
                "कृपया:\n"
                "1. अपना प्रश्न अधिक विशिष्ट बनाएं\n"
                "2. संबंधित दस्तावेज़ अपलोड करें\n"
                "3. कोई विशेष विषय चुनें"
            )
        elif language == "hinglish":
            return (
                "Mujhe is question ka answer dene ke liye enough information nahi mili.\n\n"
                "Please:\n"
                "1. Apna question thoda specific karein\n"
                "2. Related documents upload karein\n"
                "3. Koi specific topic select karein"
            )
        else:
            return (
                "I don't have enough information to answer this question accurately.\n\n"
                "You could try:\n"
                "1. Making your question more specific\n"
                "2. Uploading relevant documents\n"
                "3. Selecting a specific subject or topic"
            )

    def _get_low_confidence_disclaimer(self, language: str) -> str:
        """Disclaimer for low-confidence answers."""
        if language == "hi":
            return "⚠️ यह उत्तर कम विश्वास स्तर पर आधारित है। कृपया स्रोतों की जांच करें।"
        elif language == "hinglish":
            return "⚠️ Yeh answer low confidence level par based hai. Please sources verify karein."
        else:
            return "⚠️ This answer has low confidence. Please verify against the sources provided."

    async def _check_exact_qa_cache(self, question: str, language: str) -> dict | None:
        """O(1) exact match before spending 50ms on embedding."""
        if not self._redis:
            return None
        import hashlib
        import json
        key = f"exact_qa:{hashlib.sha256(f'{question}:{language}'.lower().strip().encode()).hexdigest()[:16]}"
        cached = await self._redis.get(key)
        return json.loads(cached) if cached else None

    async def _check_cache(self, text: str, language: str, topic: str | None) -> dict | None:
        """Check Redis cache for a previously computed answer."""
        if not self._redis:
            return None
        try:
            import hashlib
            import json
            cache_key = f"answer:{hashlib.sha256(f'{text}:{language}:{topic}'.encode()).hexdigest()[:32]}"
            cached = await self._redis.get(cache_key)
            if cached:
                logger.info(f"Cache hit for query")
                return json.loads(cached)
        except Exception as e:
            logger.debug(f"Cache check failed: {e}")
        return None

    async def _cache_result(self, text: str, language: str, topic: str | None, result: dict):
        """Cache the answer in Redis with 1-hour TTL."""
        if not self._redis:
            return
        try:
            import hashlib
            import json
            cache_key = f"answer:{hashlib.sha256(f'{text}:{language}:{topic}'.encode()).hexdigest()[:32]}"
            await self._redis.setex(cache_key, 3600, json.dumps(result, default=str))
            
            # Also write to exact_qa for O(1) exact match lookup
            exact_key = f"exact_qa:{hashlib.sha256(f'{text}:{language}'.lower().strip().encode()).hexdigest()[:16]}"
            await self._redis.setex(exact_key, 3600, json.dumps(result, default=str))
        except Exception as e:
            logger.debug(f"Cache write failed: {e}")

    async def _log_query(
        self,
        request_id: str,
        input_text: str,
        detected_language: str,
        retrieved_chunk_ids: list,
        answer: str,
        retrieval_confidence: float,
        answer_confidence: float,
        model_used: str,
        total_tokens: int,
        latency_ms: int,
    ) -> None:
        """Log query execution to the audit database."""
        try:
            async with self._db_session_factory() as session:
                from app.db.models import QueryLog
                from uuid import UUID as UUIDType
                
                # Convert chunk IDs to proper UUID format
                chunk_ids = []
                for chunk_id in retrieved_chunk_ids:
                    try:
                        if isinstance(chunk_id, str):
                            chunk_ids.append(UUIDType(chunk_id))
                        else:
                            chunk_ids.append(chunk_id)
                    except (ValueError, TypeError):
                        pass
                
                query_log = QueryLog(
                    input_text=input_text,
                    input_type="text",
                    detected_language=detected_language,
                    retrieved_chunk_ids=chunk_ids if chunk_ids else None,
                    answer=answer,
                    retrieval_confidence=retrieval_confidence,
                    answer_confidence=answer_confidence,
                    model_used=model_used,
                    total_tokens=total_tokens,
                    latency_ms=latency_ms,
                )
                session.add(query_log)
                await session.commit()
        except Exception as e:
            logger.warning(f"[{request_id}] Failed to log query: {e}")


# Global singleton
_orchestrator: Orchestrator | None = None


async def get_orchestrator() -> Orchestrator:
    """Get the global Orchestrator instance."""
    global _orchestrator
    if _orchestrator is None:
        _orchestrator = Orchestrator()
    return _orchestrator


async def shutdown_orchestrator() -> None:
    """Cleanup orchestrator resources. Call during app shutdown."""
    global _orchestrator
    if _orchestrator is not None:
        if _orchestrator._redis:
            try:
                await _orchestrator._redis.close()
            except Exception:
                pass
        if _orchestrator._llm_generator:
            try:
                await _orchestrator._llm_generator.cleanup()
            except Exception as e:
                logger.error(f"Failed to cleanup LLMGenerator: {e}")
                
        if _orchestrator._translator_service:
            try:
                await _orchestrator._translator_service.cleanup()
            except Exception as e:
                logger.error(f"Failed to cleanup TranslatorService: {e}")
                
        _orchestrator = None
        logger.info("Orchestrator shutdown complete")
