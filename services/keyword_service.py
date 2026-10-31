"""
services/keyword_service.py
-----------------------------
Detects suspicious/promotional phrases commonly seen in fake or
incentivized reviews.

IMPORTANT: A matched keyword is only ONE signal that feeds into the
overall risk score (see prediction_service.py / app.py). It never
classifies a review as fake by itself, matching the project's
"do not over-trust a single signal" rule.
"""

# Configurable list — edit freely without touching any other file.
SUSPICIOUS_PHRASES = [
    "100% guaranteed",
    "best product ever",
    "must buy",
    "buy now",
    "highly recommended",
    "amazing product",
    "five star product",
    "best quality",
    "life changing",
    "changed my life",
    "perfect product",
    "no flaws",
    "guaranteed satisfaction",
    "best purchase",
    "best in class",
    "outstanding outstanding",
]


def detect_suspicious_keywords(text: str) -> dict:
    """
    Scan cleaned/raw text for known suspicious phrases.

    Returns:
        {
            "found": ["must buy", "best product ever"],
            "count": 2
        }
    """
    if not text:
        return {"found": [], "count": 0}

    lowered = text.lower()
    found = [phrase for phrase in SUSPICIOUS_PHRASES if phrase in lowered]

    return {"found": found, "count": len(found)}
