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
        Fast hybrid language detection:
        1. Uses a heuristic for Hinglish.
        2. Uses langdetect for English fast-path.
        3. Only uses LLM for translation if non-English.
        """
        import langdetect
        import re
        
        # 1. Check for Hinglish using common romanized Hindi stop words
        text_lower = text.lower()
        hinglish_keywords = {"kya", "hai", "kaise", "kyu", "mera", "tera", "haan", "nahi", "karo", "yeh", "woh", "aap", "ka", "ki", "ke", "ho"}
        words = set(re.findall(r'\b\w+\b', text_lower))
        is_hinglish = len(words.intersection(hinglish_keywords)) > 0
        
        detected_lang = "en"
        if is_hinglish:
            detected_lang = "hinglish"
            logger.info("Heuristic detected: Hinglish")
        else:
            try:
                lang = langdetect.detect(text)
                if lang == 'en':
                    return {"english_text": text, "original_language": "en"}
                detected_lang = lang
                logger.info(f"Langdetect detected: {lang}")
            except Exception:
                # Fallback to English if detection fails
                return {"english_text": text, "original_language": "en"}
                
        # 2. If we reach here, it's non-English or Hinglish. Use a lightweight LLM translation prompt.
        prompt = f"""You are a fast translation engine.
Translate the following text to English.
If it is in Hindi (Devanagari) or Hinglish (Roman Hindi), translate it to pure English.
Output ONLY the English translation, nothing else. No markdown, no quotes.

Text: "{text}"
"""
        try:
            client = self._get_client()
            response = await client.chat.completions.create(
                model=self._model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
            )
            en_text = response.choices[0].message.content.strip()
            return {"english_text": en_text, "original_language": detected_lang}
        except Exception as e:
            logger.error(f"Translation to English failed: {e}")
            return {"english_text": text, "original_language": detected_lang}

    async def translate_from_english(self, text: str, target_language: str) -> str:
        """Translates English text back to the target language if necessary."""
        if target_language == "en":
            return text
            
        lang_instruction = "Hindi in Devanagari script" if target_language == "hi" else f"'{target_language}'"
        if target_language == "hinglish":
            lang_instruction = "Hinglish (Hindi written in Roman/English alphabet)"
            
        prompt = f"""You are a fast translation engine.
Translate the following English text into {lang_instruction}.
Output ONLY the translated text, nothing else. No markdown, no quotes.

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
        """
        Compare two answers for factual equivalence, ignoring phrasing differences.
        Both inputs must already be in English before calling this method.
        """
        prompt = f"""You are a semantic fact-checker. Your only job is to determine if two answers convey the same core facts.

RULES:
- Ignore all differences in phrasing, word order, verbosity, or style.
- Focus ONLY on whether the key factual claim(s) in Answer A are present in Answer B.
- A short, correct answer (e.g. "Paris") MUST match a longer one ("The capital is Paris.").
- Partial answers that contain the correct fact still count as a match.
- Output ONLY the single word YES or NO. No explanation, no punctuation.

Question: {question}
Answer A: {answer_a}
Answer B: {answer_b}

Are the core facts in Answer A and Answer B equivalent?"""

        try:
            client = self._get_client()
            response = await client.chat.completions.create(
                model=self._model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                max_tokens=5,  # We only need YES or NO — cap tokens for speed
            )
            result = response.choices[0].message.content.strip().upper()
            return result.startswith("YES")
        except Exception as e:
            logger.error(f"Comparison failed: {e}")
            return False
