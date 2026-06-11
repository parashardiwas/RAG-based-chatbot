"""LLM generation service — interacts with OpenAI API.

Builds grounded prompts from retrieved context chunks, calls the model,
and returns structured :class:`GenerationResult` objects.

Default model: ``gpt-5-nano``.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass

from openai import AsyncOpenAI
from app.config import get_settings

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
DEFAULT_MODEL: str = "gpt-5-nano"

# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class GenerationResult:
    """Structured output from a single LLM generation call."""

    answer: str
    model_used: str
    total_tokens: int
    generation_time_ms: float


# ---------------------------------------------------------------------------
# Prompt template
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT_TEMPLATE: str = """\
You are a strict, factual data extraction assistant. Your ONLY job is to extract answers from the provided sources.

CRITICAL RULES - READ CAREFULLY:
1. **Language**: Respond in {language}. Never switch languages.
2. **Absolute Grounding**: Base your answer EXCLUSIVELY on the text in the sources below. You are FORBIDDEN from using outside knowledge. If you use outside knowledge, you will fail the task.
3. **No Guessing**: If the exact answer cannot be found in the sources, you must NOT guess. You must reply exactly with:
"{fallback_message}"
4. **Citations**: When referencing a source, cite it inline as [Source N].
5. **No Hallucinations**: Do not mention names, places, dates, or facts that are not explicitly stated in the text.

--- SOURCES ---
{sources}
--- END SOURCES ---
"""

_FALLBACK_MESSAGES: dict[str, str] = {
    "en": "I don't have enough information to answer this question.",
    "hi": "इस प्रश्न का उत्तर देने के लिए मेरे पास पर्याप्त जानकारी नहीं है।",
    "es": "No tengo suficiente información para responder a esta pregunta.",
    "fr": "Je n'ai pas assez d'informations pour répondre à cette question.",
}


# ---------------------------------------------------------------------------
# LLMGenerator
# ---------------------------------------------------------------------------


class LLMGenerator:
    """Generate answers by calling OpenAI models.

    Usage::

        gen = LLMGenerator()
        result = await gen.generate(
            prompt="What is photosynthesis?",
            context="Photosynthesis is the process …",
            language="en",
        )
    """

    def __init__(self) -> None:
        self._settings = get_settings()
        self._client: AsyncOpenAI | None = None
    
    async def cleanup(self) -> None:
        """Close the HTTP client. Call during app shutdown."""
        if self._client:
            await self._client.close()
            self._client = None
            logger.info("LLMGenerator client closed")

    def _get_client(self) -> AsyncOpenAI:
        if self._client is None:
            self._client = AsyncOpenAI(api_key=self._settings.openai_api_key)
        return self._client

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def generate(
        self,
        prompt: str,
        context: str,
        language: str = "en",
        model: str = DEFAULT_MODEL,
    ) -> GenerationResult:
        
        full_system_prompt = self._build_system_prompt(
            context_chunks=[context] if isinstance(context, str) else context,
            language=language,
        )

        model = self._settings.openai_model
        
        start = time.perf_counter()
        
        try:
            client = self._get_client()
            response = await client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": full_system_prompt},
                    {"role": "user", "content": prompt}
                ]
            )
            answer = response.choices[0].message.content.strip() if response.choices else ""
            total_tokens = response.usage.total_tokens if response.usage else 0
        except Exception as e:
            logger.error(f"OpenAI generation failed: {e}")
            raise
            
        elapsed_ms = (time.perf_counter() - start) * 1000

        return GenerationResult(
            answer=answer,
            model_used=f"openai/{model}",
            total_tokens=total_tokens,
            generation_time_ms=round(elapsed_ms, 2),
        )

    # ------------------------------------------------------------------
    # Prompt construction
    # ------------------------------------------------------------------

    @staticmethod
    def _build_system_prompt(
        context_chunks: list[str],
        language: str,
    ) -> str:
        """Assemble the system prompt.

        Each context chunk is labelled as ``[Source N]`` for citation.
        """
        numbered_sources = "\n".join(
            f"[Source {i}] {chunk}" for i, chunk in enumerate(context_chunks, 1)
        )
        fallback_message = _FALLBACK_MESSAGES.get(
            language, _FALLBACK_MESSAGES["en"]
        )
        
        language_names = {
            "en": "English",
            "hi": "Hindi",
            "es": "Spanish",
            "fr": "French",
            "hinglish": "Hinglish (a mix of Hindi and English written in Roman script)"
        }
        full_language_name = language_names.get(language, "English")

        return _SYSTEM_PROMPT_TEMPLATE.format(
            language=full_language_name,
            sources=numbered_sources,
            fallback_message=fallback_message,
        )
