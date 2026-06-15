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
        hinglish_keywords = {
            # Common Hinglish words
            "kya", "hai", "hain", "kaise", "kyu", "kyun", "kab", "kahan", "kaun",
            "mera", "tera", "uska", "hamara", "tumhara", "unka",
            "haan", "nahi", "nahin", "mat", "karo", "karna", "kiye", "kiya",
            "yeh", "woh", "ye", "wo", "iska", "uske", "isme", "usme",
            "aap", "aapka", "aapki", "aapko", "aapse", 
            "main", "mujhe", "mujhse", "mujhko", "mere", "meri",
            "tum", "tumhe", "tumse", "tumko", "tumhare", "tumhari",
            "ho", "hota", "hoti", "hote", "tha", "thi", "the", "thay",
            "kar", "karte", "karti", "karta", "karke", "karne",
            "ja", "jao", "jata", "jati", "jate", "jaana", "jaane",
            "aa", "aao", "aata", "aati", "aate", "aana", "aane",
            "de", "do", "deta", "deti", "dete", "dena", "dene",
            "le", "lo", "leta", "leti", "lete", "lena", "lene",
            "bhi", "bhi", "sirf", "bas", "bilkul", "thoda", "zyada", "kam",
            "accha", "bura", "sahi", "galat", "theek", "samjha", "pata",
            "lagta", "lagti", "lagte", "laga", "lagi", "lage",
            "chahiye", "chahiye", "chalo", "ruko", "dekho", "suno",
            "batao", "bolo", "kaho", "pucho", "janiye", "samjho"
        }
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
        prompt = f"""You are a specialized multilingual translation engine with expertise in Hindi and Hinglish.

IMPORTANT LANGUAGE RECOGNITION:
- Hindi: Text written in Devanagari script (हिंदी)
- Hinglish: Hindi words written in English alphabet (examples: "kya hai", "mujhe pata nahi", "acha lagta hai")
- Mixed: Some English + some Hindi/Hinglish in the same sentence

TASK: Translate the following text to clear, natural English.

RULES:
1. If the text contains ANY Hindi (Devanagari) or Hinglish (romanized Hindi), translate it completely to English
2. Preserve the meaning and context accurately
3. For mixed sentences, translate only the Hindi/Hinglish parts while keeping English parts intact
4. Output ONLY the English translation, no explanations or formatting

Text: "{text}"
"""
        try:
            client = self._get_client()
            response = await client.chat.completions.create(
                model=self._model,
                messages=[{"role": "user", "content": prompt}],
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
            
        prompt = f"""You are a specialized multilingual translation engine with expertise in Hindi and Hinglish.

TARGET LANGUAGE SPECIFICATIONS:
- Hindi (hi): Use proper Devanagari script (हिंदी में लिखें)
- Hinglish: Use Hindi words written in English alphabet (example: "Main samjha ki aap sahi keh rahe hain")
- Other languages: Use the specified language code

TASK: Translate this English text into {lang_instruction}.

RULES:
1. Maintain the exact meaning and context
2. Use natural, conversational tone appropriate for the target language
3. For Hinglish: Mix Hindi and English naturally as native speakers do
4. For Hindi: Use proper Devanagari script with correct grammar
5. Output ONLY the translated text, no explanations or formatting

English text: "{text}"
"""
        try:
            client = self._get_client()
            response = await client.chat.completions.create(
                model=self._model,
                messages=[{"role": "user", "content": prompt}],
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
                max_completion_tokens=2000,
            )
            content = response.choices[0].message.content
            if not content:
                raise ValueError("Model returned empty content during comparison")
                
            result = content.strip().upper()
            return result.startswith("YES")
        except Exception as e:
            logger.error(f"Comparison failed: {e}")
            raise
