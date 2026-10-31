"""
utils/text_cleaner.py
----------------------
Simple, explainable text cleaning used before TF-IDF vectorization.

Why this matters for the interview/viva:
TF-IDF works on word tokens. Raw review text has punctuation, links,
extra whitespace, and mixed casing that create "noisy" tokens
(e.g. "Great!" vs "great" would be treated as two different words
without cleaning). Cleaning normalizes the text so the model learns
from meaningful word patterns instead of formatting noise.
"""

import re

MAX_REVIEW_LENGTH = 5000  # basic input-size guard (see Security section)


def clean_text(text: str) -> str:
    """
    Clean a raw review string for ML processing.

    Steps:
    1. Guard against non-string / empty input
    2. Truncate excessively long input (basic DoS / abuse protection)
    3. Lowercase
    4. Remove URLs
    5. Remove HTML tags
    6. Remove punctuation / special characters (keep letters, numbers, spaces)
    7. Collapse multiple spaces
    8. Strip leading/trailing whitespace
    """
    if not text or not isinstance(text, str):
        return ""

    text = text[:MAX_REVIEW_LENGTH]

    text = text.lower()
    text = re.sub(r"http\S+|www\.\S+", " ", text)
    text = re.sub(r"<.*?>", " ", text)
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text)

    return text.strip()


def basic_stats(text: str) -> dict:
    """
    Return a few simple, explainable characteristics of the raw review.
    These are used as small extra signals in the risk score (not the
    core ML prediction).
    """
    if not text:
        return {
            "length": 0,
            "word_count": 0,
            "exclamation_count": 0,
            "all_caps_word_count": 0,
        }

    words = text.split()
    return {
        "length": len(text),
        "word_count": len(words),
        "exclamation_count": text.count("!"),
        "all_caps_word_count": sum(1 for w in words if w.isupper() and len(w) > 2),
    }
