"""
services/similarity_service.py
--------------------------------
Detects near-duplicate reviews using TF-IDF + cosine similarity.

Why cosine similarity: it measures the angle between two review
vectors, regardless of their length, so two reviews that use similar
wording end up with a similarity close to 1.0 (identical direction)
even if one is longer than the other.

We compare a new review against a list of previously stored reviews
and return the single most similar one.
"""

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

SIMILARITY_THRESHOLD = 0.85


def find_most_similar(new_review: str, existing_reviews: list) -> dict:
    """
    Compare `new_review` against a list of existing review strings.

    Args:
        new_review: cleaned or raw text of the new review
        existing_reviews: list of strings (previously stored reviews)

    Returns:
        {
            "similarity": 0.92,
            "most_similar_review": "...",
            "is_potential_duplicate": True
        }
    """
    if not new_review or not existing_reviews:
        return {
            "similarity": 0.0,
            "most_similar_review": None,
            "is_potential_duplicate": False,
        }

    corpus = existing_reviews + [new_review]

    try:
        vectorizer = TfidfVectorizer(stop_words="english")
        tfidf_matrix = vectorizer.fit_transform(corpus)
    except ValueError:
        # Happens if all text is empty/stopwords-only after cleaning
        return {
            "similarity": 0.0,
            "most_similar_review": None,
            "is_potential_duplicate": False,
        }

    new_vector = tfidf_matrix[-1]
    existing_vectors = tfidf_matrix[:-1]

    scores = cosine_similarity(new_vector, existing_vectors).flatten()

    best_idx = int(scores.argmax())
    best_score = float(scores[best_idx])

    return {
        "similarity": round(best_score, 2),
        "most_similar_review": existing_reviews[best_idx],
        "is_potential_duplicate": best_score >= SIMILARITY_THRESHOLD,
    }
