"""Translation service using OpenAI for language identification and translation."""

import logging
from openai import AsyncOpenAI
from app.config import get_settings

logger = logging.getLogger(__name__)

class TranslatorService:
    def __init__(self):
        self._settings = get_settings()
        self._client: AsyncOpenAI | None = None
        self._model = self._settings.openai_model
        
    def _get_client(self) -> AsyncOpenAI:
        if self._client is None:
            self._client = AsyncOpenAI(api_key=self._settings.openai_api_key)
        return self._client
        
    async def cleanup(self):
        """Cleanup resources like the AsyncOpenAI client."""
        if self._client:
            await self._client.close()
            self._client = None
        
    async def translate_to_english(self, text: str) -> dict[str, str]:
        """
        Takes raw text, identifies language using langdetect (no LLM).
        Returns: {"english_text": str, "original_language": str}
        
        OPTIMIZATION: Use langdetect ONLY - skip LLM translation for now.
        For production, integrate Google Translate API (super fast) instead.
        """
        import langdetect
        
        try:
            detected_lang = langdetect.detect(text)
            if detected_lang == 'en':
                return {"english_text": text, "original_language": "en"}
            
            # Map language codes to language names
            lang_names = {
                'hi': 'Hindi',
                'gu': 'Gujarati',
                'ta': 'Tamil',
                'te': 'Telugu',
                'kn': 'Kannada',
                'ml': 'Malayalam',
                'mr': 'Marathi',
                'bn': 'Bengali',
            }
            
            detected_name = lang_names.get(detected_lang, detected_lang)
            logger.info(f"Detected language: {detected_lang} ({detected_name})")
            
            # For now, return as-is with language code
            # TODO: Replace with Google Translate API for actual translation
            return {
                "english_text": text,  # Return original for now
                "original_language": detected_lang
            }
        except Exception as e:
            logger.warning(f"langdetect failed: {e}")
            return {"english_text": text, "original_language": "en"}

    async def translate_from_english(self, text: str, target_language: str) -> str:
        """Translates English text to target language."""
        if target_language == "en":
            return text
        
        # OPTIMIZATION: Skip translation for now in production
        # This is the second-biggest latency killer
        # TODO: Use Google Translate API instead of LLM
        return text

    async def compare_answers(self, question: str, answer_a: str, answer_b: str) -> bool:
        """Compare two answers to see if they mean the same thing for a given question."""
        prompt = f"""You are an automated grading system.
Question: "{question}"
Answer A (User): "{answer_a}"
Answer B (System Truth): "{answer_b}"

Do Answer A and Answer B convey the same core factual meaning or truth in response to the Question?
Output EXACTLY 'YES' or 'NO' and nothing else.
"""
        try:
            client = self._get_client()
            response = await client.chat.completions.create(
                model=self._model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
            )
            result = response.choices[0].message.content.strip().upper()
            return "YES" in result
        except Exception as e:
            logger.error(f"Comparison failed: {e}")
            return False
