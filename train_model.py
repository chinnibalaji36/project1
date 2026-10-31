"""
train_model.py
----------------
Trains the Fake Review Detection model.

Pipeline:
    Review -> Text Cleaning -> TF-IDF -> Train/Test Split
    -> Logistic Regression -> Evaluation -> Save Model

Run:
    python train_model.py
"""

import os
import pandas as pd
import joblib

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)

from utils.text_cleaner import clean_text

DATASET_PATH = os.path.join("dataset", "reviews.csv")
MODEL_DIR = "model"
MODEL_PATH = os.path.join(MODEL_DIR, "fake_review_model.pkl")
VECTORIZER_PATH = os.path.join(MODEL_DIR, "tfidf_vectorizer.pkl")


def main():
    print("=" * 60)
    print("AI Fake Review Detector - Model Training")
    print("=" * 60)

    if not os.path.exists(DATASET_PATH):
        raise FileNotFoundError(
            f"Dataset not found at '{DATASET_PATH}'. "
            "Add a CSV with 'review' and 'label' columns (0=Genuine, 1=Fake)."
        )

    print(f"\n[1/6] Loading dataset from {DATASET_PATH} ...")
    df = pd.read_csv(DATASET_PATH)

    if "review" not in df.columns or "label" not in df.columns:
        raise ValueError("Dataset must contain 'review' and 'label' columns.")

    df = df.dropna(subset=["review", "label"])
    print(f"      Loaded {len(df)} rows.")
    print(f"      Label distribution:\n{df['label'].value_counts().to_string()}")

    print("\n[2/6] Cleaning review text ...")
    df["cleaned_review"] = df["review"].astype(str).apply(clean_text)
    df = df[df["cleaned_review"].str.strip() != ""]

    print("\n[3/6] Vectorizing text with TF-IDF ...")
    vectorizer = TfidfVectorizer(
        max_features=5000,
        ngram_range=(1, 2),
        stop_words="english",
    )
    X = vectorizer.fit_transform(df["cleaned_review"])
    y = df["label"].astype(int)

    print("\n[4/6] Splitting into train/test sets (80/20) ...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"      Train size: {X_train.shape[0]}, Test size: {X_test.shape[0]}")

    print("\n[5/6] Training Logistic Regression model ...")
    model = LogisticRegression(
        max_iter=1000,
        class_weight="balanced",
        random_state=42,
    )
    model.fit(X_train, y_train)

    print("\n[6/6] Evaluating model on the held-out test set ...")
    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, target_names=["Genuine", "Fake"])
    matrix = confusion_matrix(y_test, y_pred)

    print("\n" + "=" * 60)
    print("EVALUATION RESULTS (actual results on this dataset)")
    print("=" * 60)
    print(f"Accuracy: {accuracy:.2%}")
    print("\nClassification Report:")
    print(report)
    print("Confusion Matrix (rows=actual, cols=predicted) [Genuine, Fake]:")
    print(matrix)
    print("=" * 60)

    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    joblib.dump(vectorizer, VECTORIZER_PATH)

    print(f"\nModel saved to:      {MODEL_PATH}")
    print(f"Vectorizer saved to: {VECTORIZER_PATH}")
    print("\nDone. You can now run 'python app.py' to start the web app.")


if __name__ == "__main__":
    main()
