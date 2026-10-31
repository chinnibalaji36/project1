"""
database/database.py
----------------------
SQLite storage layer. All queries are parameterized to prevent SQL
injection (see Security section of the README).
"""

import os
import sqlite3
from datetime import datetime

DB_PATH = os.getenv("DATABASE_PATH", os.path.join(os.path.dirname(__file__), "reviews.db"))


def get_connection():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    """Create the reviews table if it doesn't already exist."""
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS reviews (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            review_text TEXT NOT NULL,
            rating INTEGER,
            source TEXT NOT NULL DEFAULT 'manual',
            prediction TEXT NOT NULL,
            confidence REAL,
            sentiment TEXT,
            similarity_score REAL,
            risk_score REAL,
            risk_level TEXT,
            created_at TEXT NOT NULL
        )
    """)
    conn.commit()
    conn.close()


def insert_review(review_text, rating, source, prediction, confidence,
                   sentiment, similarity_score, risk_score, risk_level):
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("""
        INSERT INTO reviews (
            review_text, rating, source, prediction, confidence,
            sentiment, similarity_score, risk_score, risk_level, created_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        review_text, rating, source, prediction, confidence,
        sentiment, similarity_score, risk_score, risk_level,
        datetime.utcnow().isoformat()
    ))
    conn.commit()
    review_id = cursor.lastrowid
    conn.close()
    return review_id


def get_all_review_texts(limit=500):
    """Used by the similarity service to compare against stored reviews."""
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT review_text FROM reviews ORDER BY id DESC LIMIT ?", (limit,))
    rows = cursor.fetchall()
    conn.close()
    return [row["review_text"] for row in rows]


def get_recent_reviews(limit=10):
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("""
        SELECT id, review_text, rating, source, prediction, confidence,
               sentiment, similarity_score, risk_score, risk_level, created_at
        FROM reviews
        ORDER BY id DESC
        LIMIT ?
    """, (limit,))
    rows = cursor.fetchall()
    conn.close()
    return [dict(row) for row in rows]


def get_statistics():
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("SELECT COUNT(*) AS total FROM reviews")
    total = cursor.fetchone()["total"]

    cursor.execute("SELECT COUNT(*) AS fake FROM reviews WHERE prediction = 'Fake'")
    fake = cursor.fetchone()["fake"]

    cursor.execute("SELECT COUNT(*) AS genuine FROM reviews WHERE prediction = 'Genuine'")
    genuine = cursor.fetchone()["genuine"]

    cursor.execute("SELECT AVG(risk_score) AS avg_risk FROM reviews")
    avg_risk_row = cursor.fetchone()["avg_risk"]
    avg_risk = round(avg_risk_row, 1) if avg_risk_row is not None else 0

    cursor.execute("""
        SELECT sentiment, COUNT(*) AS count FROM reviews
        GROUP BY sentiment
    """)
    sentiment_rows = cursor.fetchall()
    sentiment_breakdown = {row["sentiment"] or "Neutral": row["count"] for row in sentiment_rows}

    cursor.execute("""
        SELECT risk_level, COUNT(*) AS count FROM reviews
        GROUP BY risk_level
    """)
    risk_rows = cursor.fetchall()
    risk_breakdown = {row["risk_level"] or "LOW": row["count"] for row in risk_rows}

    conn.close()

    return {
        "total_reviews": total,
        "fake_reviews": fake,
        "genuine_reviews": genuine,
        "average_risk_score": avg_risk,
        "sentiment_breakdown": sentiment_breakdown,
        "risk_breakdown": risk_breakdown,
    }
