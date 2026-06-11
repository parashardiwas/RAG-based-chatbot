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
        Takes raw text, identifies language, translates to English.
        Returns: {"english_text": str, "original_language": str}
        """
        import langdetect
        
        try:
            detected_lang = langdetect.detect(text)
            if detected_lang == 'en':
                return {"english_text": text, "original_language": "en"}
        except Exception as e:
            logger.warning(f"langdetect failed: {e}")
            pass

        prompt = f"""You are a precise translation engine. 
Analyze the following text. 
1. Identify the language (e.g., 'en', 'hi', 'es').
2. Translate the text to English. If it is already in English, output it exactly as is.

Format your response EXACTLY like this with no other text:
LANG: <language_code>
EN: <english_translation>

Text: "{text}"
"""
        try:
            client = self._get_client()
            response = await client.chat.completions.create(
                model=self._model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
            )
            result = response.choices[0].message.content.strip()
            
            lang = "en"
            en_text = text
            
            for line in result.split("\n"):
                if line.startswith("LANG:"):
                    lang = line.replace("LANG:", "").strip().lower()
                elif line.startswith("EN:"):
                    en_text = line.replace("EN:", "").strip()
                    
            return {"english_text": en_text, "original_language": lang}
        except Exception as e:
            logger.error(f"Translation to English failed: {e}")
            return {"english_text": text, "original_language": "en"}

    async def translate_from_english(self, text: str, target_language: str) -> str:
        """Translates English text to target language."""
        if target_language == "en":
            return text
            
        prompt = f"""You are a precise translation engine.
Translate the following English text into the language code '{target_language}'.
Output ONLY the translated text, nothing else.

Text: "{text}"
"""
        try:
            client = self._get_client()
            response = await client.chat.completions.create(
                model=self._model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            logger.error(f"Translation from English failed: {e}")
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
