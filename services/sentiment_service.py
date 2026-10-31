"""
services/sentiment_service.py
------------------------------
Simple, explainable sentiment analysis using TextBlob.

TextBlob's polarity score ranges from -1 (very negative) to +1 (very
positive). We bucket that score into Positive / Neutral / Negative,
which is easy to explain in an interview: "we used a lexicon-based
polarity score and thresholded it."
"""

from textblob import TextBlob

POSITIVE_THRESHOLD = 0.15
NEGATIVE_THRESHOLD = -0.15


def analyze_sentiment(text: str) -> dict:
    """
    Return sentiment label + polarity score for a piece of review text.

    Example:
        {"sentiment": "Positive", "score": 0.82}
    """
    if not text or not text.strip():
        return {"sentiment": "Neutral", "score": 0.0}

    blob = TextBlob(text)
    polarity = round(blob.sentiment.polarity, 2)

    if polarity >= POSITIVE_THRESHOLD:
        label = "Positive"
    elif polarity <= NEGATIVE_THRESHOLD:
        label = "Negative"
    else:
        label = "Neutral"

    return {"sentiment": label, "score": polarity}
