# AI-Powered E-Commerce Fake Review & Scam Detection System

A beginner-friendly, interview-ready project that detects whether an
e-commerce review is likely **Genuine** or **Fake**, analyzes its
sentiment, checks for suspicious/duplicate content, and produces an
overall risk score — for both manually entered reviews and reviews
fetched from a Flipkart product URL.

> **This is a decision-support tool, not a perfect fraud detector.**
> Predictions are probabilistic and should be combined with human judgment.

---

## 1. Project Overview

E-commerce platforms are full of reviews, and some of them are fake —
written to inflate a product's rating or unfairly damage a competitor.
This project builds a small, explainable pipeline that:

1. Cleans review text
2. Predicts **Fake** or **Genuine** using a trained ML model
3. Scores sentiment (Positive / Neutral / Negative)
4. Flags suspicious promotional phrases
5. Checks for near-duplicate reviews using cosine similarity
6. Combines all of the above into one **risk score (0–100)**
7. Optionally repeats this for every review on a Flipkart product page
8. Stores everything in SQLite and shows it on a dashboard

---

## 2. Features

- Manual review analysis (paste one review, get an instant verdict)
- Flipkart product URL input with graceful fallback if fetching isn't configured
- Bulk analysis of fetched reviews with a per-product summary
- Fake / Genuine prediction with confidence
- Sentiment analysis
- Suspicious keyword detection (configurable phrase list)
- Similar/duplicate review detection (TF-IDF + cosine similarity)
- Transparent, configurable risk scoring with reasons
- Dashboard with charts (Fake vs Genuine, sentiment mix, risk levels)
- SQLite persistence with parameterized queries
- Never crashes if the Flipkart data source isn't configured

---

## 3. Technology Stack

| Layer      | Tools |
|------------|-------|
| Frontend   | HTML5, CSS3, vanilla JavaScript, Chart.js |
| Backend    | Python, Flask |
| ML         | pandas, numpy, scikit-learn (TF-IDF + Logistic Regression) |
| NLP        | TextBlob (sentiment), scikit-learn cosine similarity |
| Database   | SQLite |
| Model store| joblib (`.pkl` files) |

---

## 4. Project Structure

```text
fake-review-detector/
│
├── app.py                     # Flask app & routes
├── train_model.py             # Trains and saves the ML model
├── requirements.txt
├── README.md
├── .gitignore
├── .env.example
│
├── dataset/
│   └── reviews.csv            # Sample training data (review, label)
│
├── model/
│   ├── fake_review_model.pkl  # Trained Logistic Regression model
│   └── tfidf_vectorizer.pkl   # Fitted TF-IDF vectorizer
│
├── services/
│   ├── prediction_service.py  # ML prediction + risk score engine
│   ├── sentiment_service.py   # TextBlob sentiment analysis
│   ├── similarity_service.py  # TF-IDF cosine similarity
│   ├── keyword_service.py     # Suspicious phrase detection
│   └── flipkart_service.py    # Modular Flipkart fetch layer
│
├── database/
│   ├── database.py            # SQLite access (parameterized queries)
│   └── reviews.db             # Created automatically on first run
│
├── templates/
│   ├── index.html             # Manual + Flipkart intake page
│   ├── results.html           # Standalone server-rendered result view
│   └── dashboard.html         # Stats + Chart.js dashboard
│
├── static/
│   ├── css/style.css
│   └── js/script.js
│
└── utils/
    └── text_cleaner.py        # Text cleaning shared by training + inference
```

---

## 5. Installation

```bash
git clone <your-repo-url>
cd fake-review-detector
```

### Create and activate a virtual environment

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# macOS / Linux
source venv/bin/activate
```

### Install dependencies

```bash
pip install -r requirements.txt
```

---

## 6. Dataset Setup

A sample dataset is already included at `dataset/reviews.csv` with two
columns:

```text
review,label
"Amazing product best quality highly recommended must buy now!!!",1
"The product works fine but the battery life could be better.",0
```

- `0` = Genuine
- `1` = Fake

To use your own data, replace this file with the same two-column
format (any CSV works as long as it has `review` and `label` columns).

> **Note on the sample dataset:** it's a small, template-based demo
> dataset meant to make the pipeline easy to run and explain. Because
> of that, it's easier to classify than real-world reviews, so the
> reported accuracy is unusually high (see *Limitations* below). Swap
> in a larger, real dataset (e.g. a public Amazon/Yelp fake-review
> dataset) for realistic numbers.

---

## 7. Train the Model

```bash
python train_model.py
```

This will:

1. Load and clean `dataset/reviews.csv`
2. Vectorize text with TF-IDF
3. Split into train/test sets (80/20)
4. Train a Logistic Regression classifier (`class_weight="balanced"`)
5. Print accuracy, precision, recall, F1-score, and a confusion matrix
6. Save `model/fake_review_model.pkl` and `model/tfidf_vectorizer.pkl`

---

## 8. Environment Variables

Copy the example file and fill in only what you need:

```bash
cp .env.example .env
```

```text
FLASK_SECRET_KEY=some-random-string
FLASK_DEBUG=True

FLIPKART_API_KEY=
FLIPKART_API_HOST=
FLIPKART_API_URL=

DATABASE_PATH=database/reviews.db
```

The Flipkart variables are **optional**. If they're left blank, the
Flipkart tab will still work in the UI but will show:

```text
Unable to fetch reviews automatically.
Please check the API configuration or enter reviews manually.
```

`.env` is already listed in `.gitignore` — never commit real API keys.

---

## 9. Run the App

```bash
python app.py
```

Then open **http://localhost:5000** in your browser.

- `/` — Manual review + Flipkart intake
- `/dashboard` — Statistics and charts
- `/analyze-review` (POST) — JSON API for manual analysis
- `/fetch-flipkart` (POST) — JSON API for Flipkart batch analysis
- `/api/statistics` (GET) — JSON statistics for the dashboard

The **manual review flow works fully even if Flipkart isn't configured.**

---

## 10. Flipkart API Configuration

This project intentionally does **not** ship a hard-coded or invented
Flipkart API endpoint, and it does not scrape Flipkart directly
(scraping the site would likely violate its Terms of Service).

Instead, `services/flipkart_service.py` is a clean, swappable "fetch
layer":

- `is_configured()` checks whether you've provided your own
  legitimate review-data source via `FLIPKART_API_KEY`,
  `FLIPKART_API_HOST`, and `FLIPKART_API_URL`.
- `_fetch_from_api()` is the one function you edit to call whatever
  legitimate provider you have access to (for example, a licensed
  product-data API). It just needs to return a list of
  `{"review", "rating", "username", "date"}` dictionaries — nothing
  else in the app needs to change.
- If nothing is configured, or the call fails for any reason, the
  service returns a friendly error instead of raising an exception,
  so **the app never crashes** because of Flipkart fetching.

---

## 11. How the ML Model Works (TF-IDF)

**What is TF-IDF?**
TF-IDF stands for *Term Frequency – Inverse Document Frequency*. It
turns a sentence into a list of numbers (a vector) where common,
uninformative words (like "the", "and") get a low score, and words
that are frequent in one review but rare across all reviews get a
higher score. This lets the model focus on words that actually help
tell reviews apart, like "guaranteed" or "must buy".

**What is Logistic Regression?**
It's a simple, well-understood classification algorithm. Despite the
name, it's used for classification (Fake vs Genuine), not predicting
numbers. It looks at the TF-IDF numbers for a review and calculates a
probability between 0 and 1 that the review is "Fake". If that
probability is above 0.5, we label it Fake. It's popular for text
classification because it's fast, interpretable, and works well on
high-dimensional sparse data like TF-IDF vectors.

---

## 12. How Similarity Detection Works (Cosine Similarity)

**What is Cosine Similarity?**
Once two reviews are turned into TF-IDF vectors, cosine similarity
measures the angle between those two vectors. If the angle is small
(vectors point in almost the same direction), the reviews use very
similar words in similar proportions, so the similarity score is close
to 1.0. If the angle is close to 90°, the score is close to 0 — the
reviews share almost no meaningful words.

Example:

```text
"Excellent product and amazing quality."
"Excellent product with amazing quality."
```

These two sentences share almost all their words, so their cosine
similarity would be very high (e.g. ~0.9), flagging them as a
**potential duplicate** — not an automatic "Fake" verdict, just one
more signal for the risk score.

---

## 13. How Sentiment Analysis Works

We use TextBlob's built-in polarity score, which ranges from **-1**
(very negative) to **+1** (very positive), based on a lexicon of words
with known positive/negative associations. We bucket that score:

- `>= 0.15` → Positive
- `<= -0.15` → Negative
- otherwise → Neutral

Important: a **positive** sentiment does **not** automatically mean a
review is fake, and a 5-star rating does not automatically mean it's
fake either — genuinely happy customers write positive reviews too.
Sentiment is just one extra piece of context shown alongside the
verdict.

---

## 14. How the Risk Score Is Calculated

The risk score combines four signals into a single 0–100 number. All
weights live in one place — `services/prediction_service.py` — so
they're easy to tune:

```text
ML model prediction         → up to 55 points  (major factor)
Similarity to other reviews → up to 25 points  (additional factor)
Suspicious keywords         → up to 15 points  (additional factor)
Review characteristics      → up to 5  points  (additional factor)
                               -----------------
                               100 points total
```

Bands:

```text
0–30    → LOW RISK
31–60   → MEDIUM RISK
61–100  → HIGH RISK
```

**Worked example:**
A review says *"Amazing product best quality highly recommended must
buy now!!!"*

- ML model predicts **Fake** with 95% confidence → contributes ~52 of
  55 points
- No matching duplicate found → contributes a small amount from the
  similarity term
- 5 suspicious keywords found ("must buy", "buy now", "highly
  recommended", "amazing product", "best quality") → capped at 15
  points
- 3+ exclamation marks → +2 points

Total ≈ 69/100 → **HIGH RISK**, with the specific reasons listed
under the verdict so the result is never a black box.

---

## 15. Database

SQLite table `reviews`:

```sql
CREATE TABLE reviews (
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
);
```

Both manually entered and Flipkart-fetched reviews are stored here and
power the `/dashboard` page and `/api/statistics` endpoint. All
queries use parameter placeholders (`?`) — never raw string
concatenation — to prevent SQL injection.

---

## 16. Security Notes

- API keys are read only from environment variables (`.env`, git-ignored)
- Flipkart URLs are validated with a regex before any request is made
- Review text is truncated to a maximum length before processing
- All SQL queries are parameterized
- The app never executes user-submitted text as code
- Flipkart fetch failures are caught and turned into friendly messages
  instead of stack traces or crashes

---

## 17. Limitations

- The bundled sample dataset is small and template-based, so the
  model separates it almost perfectly. That is a property of the toy
  dataset, not a claim about real-world accuracy — replace it with a
  larger, real dataset for production use.
- Sentiment analysis uses a general-purpose lexicon (TextBlob), not a
  model fine-tuned on e-commerce reviews, so nuanced sarcasm may be
  missed.
- Flipkart fetching depends entirely on whatever legitimate data
  source you configure — none is bundled.
- Cosine similarity only catches *textual* duplication, not fake
  reviews that are reworded to avoid detection.
- This system supports human decision-making; it should not be the
  sole basis for removing a review or banning a seller.

---

## 18. Future Improvements

- Swap Logistic Regression for a fine-tuned transformer model for
  higher accuracy on nuanced text
- Add reviewer-level features (posting frequency, account age) if
  available from a legitimate data source
- Add multi-language support
- Add an admin view to relabel reviews and retrain the model
- Add authentication for the dashboard in a multi-user deployment
- Add rate limiting on the API routes

---

## 19. Interview / Viva Questions (with simple answers)

1. **Why TF-IDF instead of just word counts?**
   TF-IDF down-weights common words that appear in almost every review
   (like "the" or "product") and up-weights rarer, more distinctive
   words, which makes the resulting vectors more informative for
   classification.

2. **Why Logistic Regression instead of a deep learning model?**
   With a small-to-medium labeled dataset, Logistic Regression on
   TF-IDF features is fast to train, easy to explain, resistant to
   overfitting, and performs competitively on text classification —
   a deep model would need far more data to earn its complexity.

3. **What does `class_weight="balanced"` do?**
   It automatically re-weights the loss function so that the
   minority class (whichever label has fewer examples) isn't ignored
   by the model, which matters if Fake/Genuine counts are uneven.

4. **How does cosine similarity differ from Euclidean distance here?**
   Cosine similarity measures the angle between two vectors and
   ignores their magnitude, so it isn't thrown off by one review
   simply being longer than another — it captures the direction
   (word usage pattern), not the length.

5. **Why isn't a 5-star review automatically flagged as fake?**
   Because genuine customers write enthusiastic 5-star reviews all
   the time. Rating and sentiment are only extra context; the risk
   score is built from multiple independent signals so no single
   cue dominates the verdict.

6. **How do you prevent SQL injection in this project?**
   Every database query uses parameterized placeholders (`?`) via
   `sqlite3`, so user input is always passed as data, never
   concatenated into the SQL string.

7. **What happens if the Flipkart API isn't configured?**
   `flipkart_service.is_configured()` returns `False`, and
   `fetch_reviews()` returns a structured `{"success": False, ...}`
   response with a friendly message instead of raising an exception,
   so the rest of the app (and the manual review flow) keeps working.

8. **How is the risk score different from the ML prediction?**
   The ML prediction is a single Fake/Genuine label with a
   confidence score. The risk score is a broader 0–100 signal that
   combines the ML prediction with similarity, keyword, and
   review-characteristic signals so a human reviewer gets more
   context than a single label.

9. **Why clean text before vectorizing?**
   Cleaning (lowercasing, removing URLs/punctuation, collapsing
   whitespace) reduces noise so that, for example, "Great!" and
   "great" are treated as the same token instead of two unrelated
   ones, which improves the quality of the TF-IDF features.

10. **What's a real limitation of this system?**
    It can only detect duplication or fake patterns similar to what
    it has seen; a fake review that's carefully reworded to avoid
    suspicious phrases and near-duplicate wording could still slip
    through, which is why this is framed as decision support, not a
    perfect detector.

---

## 20. Resume Description

> **AI-Powered E-Commerce Fake Review Detection System** — Built a
> full-stack Flask application that classifies e-commerce reviews as
> Fake or Genuine using a TF-IDF + Logistic Regression pipeline
> (scikit-learn), combined with TextBlob sentiment analysis, cosine
> similarity-based duplicate detection, and a configurable, explainable
> risk-scoring engine. Designed a modular Flipkart review-fetching
> layer that fails gracefully when unconfigured, a SQLite-backed
> dashboard with Chart.js visualizations, and a responsive vanilla
> JS/HTML/CSS frontend — all built with parameterized queries,
> environment-based secrets, and clear separation between the ML,
> data, and presentation layers.

---

## 21. Quick Command Reference

```bash
# Setup
python -m venv venv
source venv/bin/activate      # or venv\Scripts\activate on Windows
pip install -r requirements.txt

# Train
python train_model.py

# Run
python app.py
# -> open http://localhost:5000
```
