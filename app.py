"""
app.py
-------
Main Flask application for the AI-Powered E-Commerce Fake Review &
Scam Detection System.

Run:
    python app.py
"""

import os
from flask import Flask, render_template, request, jsonify
from dotenv import load_dotenv

from database import database
from services import flipkart_service
from services.prediction_service import (
    analyze_review,
    is_model_ready,
    get_model_error,
)

load_dotenv()

app = Flask(__name__)
app.secret_key = os.getenv("FLASK_SECRET_KEY", "dev-secret-key-change-me")

MAX_REVIEW_CHARS = 5000

# Ensure the database and table exist before the app serves any request.
database.init_db()


# ---------------------------------------------------------------------
# Page routes
# ---------------------------------------------------------------------

@app.route("/")
def index():
    return render_template("index.html", model_ready=is_model_ready())


@app.route("/dashboard")
def dashboard():
    stats = database.get_statistics()
    recent = database.get_recent_reviews(limit=10)
    return render_template("dashboard.html", stats=stats, recent=recent)


# ---------------------------------------------------------------------
# API routes
# ---------------------------------------------------------------------

@app.route("/analyze-review", methods=["POST"])
def analyze_review_route():
    """Analyze a single manually entered review."""
    data = request.get_json(silent=True) or request.form

    review_text = (data.get("review") or "").strip()

    if not review_text:
        return jsonify({"success": False, "message": "Please enter a review to analyze."}), 400

    if len(review_text) > MAX_REVIEW_CHARS:
        return jsonify({
            "success": False,
            "message": f"Review is too long. Please limit it to {MAX_REVIEW_CHARS} characters.",
        }), 400

    if not is_model_ready():
        return jsonify({
            "success": False,
            "message": get_model_error() or "Model is not available.",
        }), 503

    try:
        existing_reviews = database.get_all_review_texts()
        result = analyze_review(review_text, existing_reviews)

        database.insert_review(
            review_text=review_text,
            rating=None,
            source="manual",
            prediction=result["prediction"],
            confidence=result["confidence"],
            sentiment=result["sentiment"],
            similarity_score=result["similarity_score"],
            risk_score=result["risk_score"],
            risk_level=result["risk_level"],
        )

        return jsonify({"success": True, "result": result})

    except Exception as e:  # noqa: BLE001
        return jsonify({"success": False, "message": f"Analysis failed: {e}"}), 500


@app.route("/fetch-flipkart", methods=["POST"])
def fetch_flipkart_route():
    """
    Fetch reviews for a Flipkart product URL and analyze each one.
    Never crashes the app if fetching fails — always returns a
    friendly JSON message instead.
    """
    data = request.get_json(silent=True) or request.form
    product_url = (data.get("url") or "").strip()

    fetch_result = flipkart_service.fetch_reviews(product_url)

    if not fetch_result["success"]:
        return jsonify({"success": False, "message": fetch_result["message"]}), 200

    if not is_model_ready():
        return jsonify({
            "success": False,
            "message": get_model_error() or "Model is not available.",
        }), 503

    try:
        existing_reviews = database.get_all_review_texts()
        analyzed = []

        for item in fetch_result["reviews"]:
            review_text = (item.get("review") or "").strip()
            if not review_text:
                continue

            result = analyze_review(review_text, existing_reviews)
            result["rating"] = item.get("rating")
            result["username"] = item.get("username")
            result["date"] = item.get("date")
            analyzed.append(result)

            database.insert_review(
                review_text=review_text,
                rating=item.get("rating"),
                source="flipkart",
                prediction=result["prediction"],
                confidence=result["confidence"],
                sentiment=result["sentiment"],
                similarity_score=result["similarity_score"],
                risk_score=result["risk_score"],
                risk_level=result["risk_level"],
            )

            # Keep similarity comparisons useful within this batch too
            existing_reviews.append(review_text)

        summary = _build_product_summary(analyzed)

        return jsonify({"success": True, "reviews": analyzed, "summary": summary})

    except Exception as e:  # noqa: BLE001
        return jsonify({"success": False, "message": f"Analysis failed: {e}"}), 500


@app.route("/api/statistics")
def api_statistics():
    return jsonify(database.get_statistics())


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------

def _build_product_summary(analyzed_reviews: list) -> dict:
    total = len(analyzed_reviews)
    if total == 0:
        return {
            "total_reviews": 0,
            "genuine_count": 0,
            "fake_count": 0,
            "fake_percentage": 0,
            "average_rating": 0,
            "overall_risk": "LOW",
            "positive_count": 0,
            "neutral_count": 0,
            "negative_count": 0,
        }

    fake_count = sum(1 for r in analyzed_reviews if r["prediction"] == "Fake")
    genuine_count = total - fake_count

    ratings = [r["rating"] for r in analyzed_reviews if isinstance(r.get("rating"), (int, float))]
    avg_rating = round(sum(ratings) / len(ratings), 1) if ratings else 0

    positive = sum(1 for r in analyzed_reviews if r["sentiment"] == "Positive")
    neutral = sum(1 for r in analyzed_reviews if r["sentiment"] == "Neutral")
    negative = sum(1 for r in analyzed_reviews if r["sentiment"] == "Negative")

    avg_risk = sum(r["risk_score"] for r in analyzed_reviews) / total
    if avg_risk <= 30:
        overall_risk = "LOW"
    elif avg_risk <= 60:
        overall_risk = "MEDIUM"
    else:
        overall_risk = "HIGH"

    return {
        "total_reviews": total,
        "genuine_count": genuine_count,
        "fake_count": fake_count,
        "fake_percentage": round((fake_count / total) * 100, 1),
        "average_rating": avg_rating,
        "overall_risk": overall_risk,
        "positive_count": positive,
        "neutral_count": neutral,
        "negative_count": negative,
    }


# ---------------------------------------------------------------------
# Error handlers
# ---------------------------------------------------------------------

@app.errorhandler(404)
def not_found(e):
    return jsonify({"success": False, "message": "Not found."}), 404


@app.errorhandler(500)
def server_error(e):
    return jsonify({"success": False, "message": "Internal server error."}), 500


if __name__ == "__main__":
    debug_mode = os.getenv("FLASK_DEBUG", "True").lower() == "true"
    app.run(debug=debug_mode, host="0.0.0.0", port=5000)
