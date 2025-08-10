from typing import Tuple

from langdetect import detect_langs
from langdetect.lang_detect_exception import LangDetectException


def detect_language(text: str) -> Tuple[str, float]:
    """
    Detect the language of the given text using langdetect.

    Args:
        text (str): The text to analyze.

    Returns:
        Tuple[str, float]: The detected language code (e.g., 'en', 'hu') and the related probability.
                          Returns ('en', 0.0) if detection fails or confidence is low (<0.4).
    """
    try:
        languages = detect_langs(text)
        if not languages:
            return ("en", 0.0)

        lang, prob = languages[0].lang, languages[0].prob
        return (lang, prob) if prob >= 0.4 else ("en", 0.0)

    except LangDetectException:
        return ("en", 0.0)
