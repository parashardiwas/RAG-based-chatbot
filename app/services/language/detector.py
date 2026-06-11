"""
Language detection for Hindi, English, and Hinglish.

Uses langdetect for primary detection with custom Hinglish heuristics.
Hinglish is detected by looking for mixing patterns:
- Roman script with Hindi-origin words
- Devanagari mixed with English words
- Common Hinglish patterns (hai, kya, nahi, etc.)
"""

import re
import logging
from typing import Any

logger = logging.getLogger(__name__)

# Common Hinglish words and patterns (written in Roman script)
HINGLISH_MARKERS = {
    # Common Hindi words in Roman script
    "hai", "hain", "ka", "ki", "ke", "ko", "se", "me", "mein",
    "kya", "kaise", "kab", "kahan", "kyun", "kyu", "kaun",
    "nahi", "nhi", "nahin", "mat", "na",
    "hum", "tum", "aap", "mera", "tera", "uska", "hamara",
    "acha", "accha", "theek", "thik", "sahi",
    "karo", "karna", "karenge", "karein", "karte",
    "bolo", "bolna", "bata", "batao",
    "dekho", "dekhna", "dekh", "dikhao",
    "jao", "jana", "aao", "aana",
    "chahiye", "chahte", "chahta",
    "lekin", "par", "magar", "isliye", "kyunki",
    "bahut", "bohot", "zyada", "kam", "thoda",
    "abhi", "pehle", "baad", "kal", "aaj",
    "samajh", "samjho", "pata", "maloom",
    "kuch", "sab", "sabhi", "dono",
    "wala", "wali", "wale",
    "yeh", "woh", "ye", "wo",
    "bhai", "yaar", "dost",
}

# Devanagari Unicode range
DEVANAGARI_PATTERN = re.compile(r'[\u0900-\u097F]')

# English word pattern
ENGLISH_WORD_PATTERN = re.compile(r'\b[a-zA-Z]{2,}\b')


def detect_language(text: str) -> dict[str, Any]:
    """
    Detect the language of input text.
    
    Returns:
        dict with:
        - language: "en", "hi", or "hinglish"
        - confidence: 0.0 to 1.0
        - details: additional detection info
    """
    if not text or not text.strip():
        return {"language": "en", "confidence": 0.0, "details": {"reason": "empty input"}}

    text_stripped = text.strip()

    # Check for Devanagari script (strong Hindi signal)
    devanagari_chars = len(DEVANAGARI_PATTERN.findall(text_stripped))
    total_chars = len(text_stripped.replace(" ", ""))

    if total_chars == 0:
        return {"language": "en", "confidence": 0.5, "details": {"reason": "whitespace only"}}

    devanagari_ratio = devanagari_chars / total_chars

    # Pure Devanagari → Hindi
    if devanagari_ratio > 0.7:
        # Check if there's also significant English mixed in
        english_words = ENGLISH_WORD_PATTERN.findall(text_stripped)
        if english_words and len(english_words) >= 2:
            return {
                "language": "hinglish",
                "confidence": 0.85,
                "details": {
                    "reason": "devanagari_with_english_words",
                    "devanagari_ratio": round(devanagari_ratio, 2),
                    "english_words_found": english_words[:5],
                },
            }
        return {
            "language": "hi",
            "confidence": 0.95,
            "details": {
                "reason": "primarily_devanagari",
                "devanagari_ratio": round(devanagari_ratio, 2),
            },
        }

    # Roman script — check for Hinglish markers
    words = text_stripped.lower().split()
    hinglish_word_count = sum(1 for w in words if w.strip(".,?!") in HINGLISH_MARKERS)
    hinglish_ratio = hinglish_word_count / max(len(words), 1)

    if hinglish_ratio > 0.15 and len(words) >= 3:
        return {
            "language": "hinglish",
            "confidence": min(0.6 + hinglish_ratio, 0.95),
            "details": {
                "reason": "hinglish_markers_detected",
                "hinglish_ratio": round(hinglish_ratio, 2),
                "markers_found": [w for w in words if w.strip(".,?!") in HINGLISH_MARKERS][:5],
            },
        }

    # Some Devanagari mixed with Roman → Hinglish
    if 0.05 < devanagari_ratio < 0.7:
        return {
            "language": "hinglish",
            "confidence": 0.80,
            "details": {
                "reason": "mixed_scripts",
                "devanagari_ratio": round(devanagari_ratio, 2),
            },
        }

    # Fall back to langdetect for primary language detection
    try:
        from langdetect import detect_langs
        results = detect_langs(text_stripped)

        if results:
            top = results[0]
            lang_code = str(top.lang)
            confidence = top.prob

            # Map langdetect codes to our codes
            if lang_code == "hi":
                return {
                    "language": "hi",
                    "confidence": confidence,
                    "details": {"reason": "langdetect", "raw_results": str(results[:3])},
                }
            elif lang_code == "en":
                # Double-check for Hinglish that langdetect missed
                if hinglish_ratio > 0.08:
                    return {
                        "language": "hinglish",
                        "confidence": 0.7,
                        "details": {
                            "reason": "langdetect_en_with_hinglish_markers",
                            "hinglish_ratio": round(hinglish_ratio, 2),
                        },
                    }
                return {
                    "language": "en",
                    "confidence": confidence,
                    "details": {"reason": "langdetect", "raw_results": str(results[:3])},
                }
            else:
                # Other language detected — default to English
                return {
                    "language": "en",
                    "confidence": confidence * 0.8,
                    "details": {
                        "reason": "langdetect_other",
                        "detected": lang_code,
                        "raw_results": str(results[:3]),
                    },
                }

    except Exception as e:
        logger.debug(f"langdetect failed: {e}")

    # Final fallback
    return {
        "language": "en",
        "confidence": 0.5,
        "details": {"reason": "fallback_default"},
    }
