"""
services/prediction_service.py
--------------------------------
Loads the trained TF-IDF + Logistic Regression model and produces:
  1. Fake/Genuine prediction with confidence
  2. A combined, explainable risk score (0-100)

Risk score design (documented, single place to tune weights):
---------------------------------------------------------------
  ML model prediction        -> up to 55 points  (major factor)
  Similarity to other reviews-> up to 25 points  (additional factor)
  Suspicious keywords        -> up to 15 points  (additional factor)
  Review characteristics     -> up to 5  points  (additional factor)
  -------------------------------------------------------------
  Total possible             -> 100 points

Bands:
  0-30   -> LOW RISK
  31-60  -> MEDIUM RISK
  61-100 -> HIGH RISK
"""

import os
import joblib

from utils.text_cleaner import clean_text, basic_stats
from services.sentiment_service import analyze_sentiment
from services.keyword_service import detect_suspicious_keywords
from services.similarity_service import find_most_similar

MODEL_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "model")
MODEL_PATH = os.path.join(MODEL_DIR, "fake_review_model.pkl")
VECTORIZER_PATH = os.path.join(MODEL_DIR, "tfidf_vectorizer.pkl")

# ---- Risk score weight configuration (edit here only) ----
WEIGHT_ML = 55
WEIGHT_SIMILARITY = 25
WEIGHT_KEYWORDS = 15
WEIGHT_CHARACTERISTICS = 5

RISK_LOW_MAX = 30
RISK_MEDIUM_MAX = 60


class ModelNotLoadedError(Exception):
    pass


_model = None
_vectorizer = None
_model_load_error = None


def load_model():
    """Load model + vectorizer once and cache them in memory."""
    global _model, _vectorizer, _model_load_error

    if _model is not None and _vectorizer is not None:
        return

    try:
        _model = joblib.load(MODEL_PATH)
        _vectorizer = joblib.load(VECTORIZER_PATH)
        _model_load_error = None
    except FileNotFoundError as e:
        _model_load_error = (
            "Model files not found. Please run 'python train_model.py' first."
        )
    except Exception as e:  # noqa: BLE001
        _model_load_error = f"Failed to load model: {e}"


def is_model_ready() -> bool:
    load_model()
    return _model is not None and _vectorizer is not None


def get_model_error() -> str:
    load_model()
    return _model_load_error


def predict_fake_or_genuine(cleaned_text: str) -> dict:
    """
    Run the trained Logistic Regression model on cleaned review text.

    Returns:
        {"label": "Fake"/"Genuine", "confidence": 0.91}
    """
    load_model()
    if _model is None or _vectorizer is None:
        raise ModelNotLoadedError(_model_load_error or "Model not loaded.")

    if not cleaned_text.strip():
        return {"label": "Genuine", "confidence": 0.5}

    vector = _vectorizer.transform([cleaned_text])
    prediction = _model.predict(vector)[0]
    probabilities = _model.predict_proba(vector)[0]

    confidence = float(max(probabilities))
    label = "Fake" if prediction == 1 else "Genuine"

    return {"label": label, "confidence": round(confidence, 2)}


def calculate_risk_score(ml_result: dict, similarity_result: dict,
                          keyword_result: dict, stats: dict) -> dict:
    """
    Combine all signals into one explainable 0-100 risk score.
    """
    reasons = []
    score = 0.0

    # 1. ML model contribution (major factor)
    if ml_result["label"] == "Fake":
        ml_points = WEIGHT_ML * ml_result["confidence"]
        score += ml_points
        reasons.append("ML model classified the review as likely fake")
    else:
        # Genuine prediction still contributes a small amount of risk
        # if the model wasn't very confident
        ml_points = WEIGHT_ML * (1 - ml_result["confidence"]) * 0.3
        score += ml_points

    # 2. Similarity contribution
    similarity = similarity_result.get("similarity", 0.0)
    if similarity_result.get("is_potential_duplicate"):
        score += WEIGHT_SIMILARITY
        reasons.append("Similar/duplicate review found in database")
    else:
        score += WEIGHT_SIMILARITY * similarity * 0.4

    # 3. Suspicious keyword contribution
    kw_count = keyword_result.get("count", 0)
    if kw_count > 0:
        kw_points = min(WEIGHT_KEYWORDS, kw_count * 5)
        score += kw_points
        reasons.append("Suspicious promotional language detected")

    # 4. Review characteristics contribution
    char_points = 0
    if stats.get("exclamation_count", 0) >= 3:
        char_points += 2
        reasons.append("Excessive exclamation marks")
    if stats.get("all_caps_word_count", 0) >= 2:
        char_points += 2
        reasons.append("Excessive use of capital letters")
    if stats.get("word_count", 0) < 4:
        char_points += 1
        reasons.append("Review is unusually short")
    score += min(WEIGHT_CHARACTERISTICS, char_points)

    score = round(min(100, max(0, score)), 1)

    if score <= RISK_LOW_MAX:
        level = "LOW"
    elif score <= RISK_MEDIUM_MAX:
        level = "MEDIUM"
    else:
        level = "HIGH"

    if not reasons:
        reasons.append("No strong risk indicators found")

    return {"risk_score": score, "risk_level": level, "reasons": reasons}


def analyze_review(raw_text: str, existing_reviews: list) -> dict:
    """
    Full pipeline for a single review: clean -> predict -> sentiment ->
    keywords -> similarity -> risk score.

    This is the single entry point used by both the manual-review route
    and the Flipkart bulk-analysis route.
    """
    cleaned = clean_text(raw_text)
    stats = basic_stats(raw_text)

    ml_result = predict_fake_or_genuine(cleaned)
    sentiment_result = analyze_sentiment(raw_text)
    keyword_result = detect_suspicious_keywords(raw_text)
    similarity_result = find_most_similar(cleaned, [clean_text(r) for r in existing_reviews])

    risk_result = calculate_risk_score(ml_result, similarity_result, keyword_result, stats)

    return {
        "review": raw_text,
        "prediction": ml_result["label"],
        "confidence": ml_result["confidence"],
        "sentiment": sentiment_result["sentiment"],
        "sentiment_score": sentiment_result["score"],
        "suspicious_keywords": keyword_result["found"],
        "similarity_score": similarity_result["similarity"],
        "is_potential_duplicate": similarity_result["is_potential_duplicate"],
        "risk_score": risk_result["risk_score"],
        "risk_level": risk_result["risk_level"],
        "reasons": risk_result["reasons"],
    }
