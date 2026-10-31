// static/js/script.js
// Handles tab switching, form submission (AJAX), and rendering of
// analysis results as "exhibit" cards without a page reload.

document.addEventListener("DOMContentLoaded", () => {
  initClock();
  initTabs();
  initManualAnalysis();
  initFlipkartAnalysis();
  initCharCount();
});

function initClock() {
  const el = document.getElementById("clock");
  if (!el) return;
  const update = () => {
    el.textContent = new Date().toLocaleString("en-IN", {
      dateStyle: "medium",
      timeStyle: "short",
    });
  };
  update();
  setInterval(update, 30000);
}

function initTabs() {
  const buttons = document.querySelectorAll(".tab-btn");
  buttons.forEach((btn) => {
    btn.addEventListener("click", () => {
      buttons.forEach((b) => b.classList.remove("active"));
      btn.classList.add("active");

      document.getElementById("panel-manual").hidden = btn.dataset.tab !== "manual";
      document.getElementById("panel-flipkart").hidden = btn.dataset.tab !== "flipkart";
    });
  });
}

function initCharCount() {
  const textarea = document.getElementById("review-input");
  const counter = document.getElementById("char-count");
  if (!textarea || !counter) return;
  textarea.addEventListener("input", () => {
    counter.textContent = textarea.value.length;
  });
}

function initManualAnalysis() {
  const btn = document.getElementById("analyze-btn");
  const textarea = document.getElementById("review-input");
  const errorBox = document.getElementById("manual-error");
  const resultBox = document.getElementById("manual-result");

  if (!btn) return;

  btn.addEventListener("click", async () => {
    const review = textarea.value.trim();
    errorBox.innerHTML = "";
    resultBox.innerHTML = "";

    if (!review) {
      errorBox.innerHTML = `<div class="error-box">Please enter a review to analyze.</div>`;
      return;
    }

    setLoading(btn, true, "Analyzing");

    try {
      const res = await fetch("/analyze-review", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ review }),
      });
      const data = await res.json();

      if (!data.success) {
        errorBox.innerHTML = `<div class="error-box">${escapeHtml(data.message)}</div>`;
        return;
      }

      resultBox.innerHTML = renderExhibit(data.result, "A");
    } catch (err) {
      errorBox.innerHTML = `<div class="error-box">Something went wrong. Please try again.</div>`;
    } finally {
      setLoading(btn, false, "Analyze Review");
    }
  });
}

function initFlipkartAnalysis() {
  const btn = document.getElementById("fetch-btn");
  const input = document.getElementById("flipkart-url");
  const errorBox = document.getElementById("flipkart-error");
  const resultBox = document.getElementById("flipkart-result");

  if (!btn) return;

  btn.addEventListener("click", async () => {
    const url = input.value.trim();
    errorBox.innerHTML = "";
    resultBox.innerHTML = "";

    if (!url) {
      errorBox.innerHTML = `<div class="error-box">Please enter a Flipkart product URL.</div>`;
      return;
    }

    setLoading(btn, true, "Fetching");

    try {
      const res = await fetch("/fetch-flipkart", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ url }),
      });
      const data = await res.json();

      if (!data.success) {
        errorBox.innerHTML = `<div class="error-box">${escapeHtml(data.message)}</div>`;
        return;
      }

      resultBox.innerHTML = renderProductSummary(data.summary) + renderReviewTable(data.reviews);
    } catch (err) {
      errorBox.innerHTML = `<div class="error-box">Something went wrong. Please try again.</div>`;
    } finally {
      setLoading(btn, false, "Fetch & Analyze");
    }
  });
}

function setLoading(btn, isLoading, label) {
  btn.disabled = isLoading;
  btn.innerHTML = isLoading ? `<span class="spinner"></span>${label}...` : label;
}

function renderExhibit(result, exhibitLetter) {
  const riskClass = "risk-" + result.risk_level.toLowerCase();
  const stampClass = result.prediction === "Genuine" ? "genuine" : "fake";
  const confidencePct = Math.round(result.confidence * 100);
  const similarityPct = Math.round(result.similarity_score * 100);

  const keywordsHtml = result.suspicious_keywords.length
    ? `<div class="keywords">${result.suspicious_keywords
        .map((k) => `<span class="keyword-chip">${escapeHtml(k)}</span>`)
        .join("")}</div>`
    : "";

  const reasonsHtml = result.reasons.map((r) => `<li>${escapeHtml(r)}</li>`).join("");

  return `
    <div class="exhibit ${riskClass}">
      <div class="exhibit-label">Exhibit ${exhibitLetter} — Submitted Review</div>
      <div class="exhibit-review">&ldquo;${escapeHtml(result.review)}&rdquo;</div>

      <span class="stamp ${stampClass}">${result.prediction}</span>

      <div class="evidence-grid">
        <div class="evidence-item"><div class="k">Confidence</div><div class="v">${confidencePct}%</div></div>
        <div class="evidence-item"><div class="k">Sentiment</div><div class="v">${result.sentiment}</div></div>
        <div class="evidence-item"><div class="k">Similarity</div><div class="v">${similarityPct}%</div></div>
        <div class="evidence-item"><div class="k">Risk Score</div><div class="v">${result.risk_score}/100</div></div>
      </div>

      <div class="risk-bar-track"><div class="risk-bar-fill" style="width:${result.risk_score}%;"></div></div>
      <span class="badge ${result.risk_level.toLowerCase()}">${result.risk_level} RISK</span>

      ${keywordsHtml}

      <ul class="reasons">${reasonsHtml}</ul>
    </div>
  `;
}

function renderProductSummary(summary) {
  return `
    <div class="summary-strip">
      <div class="summary-cell"><div class="num">${summary.total_reviews}</div><div class="label">Total Reviews</div></div>
      <div class="summary-cell"><div class="num">${summary.genuine_count}</div><div class="label">Genuine</div></div>
      <div class="summary-cell"><div class="num">${summary.fake_count}</div><div class="label">Fake</div></div>
      <div class="summary-cell"><div class="num">${summary.fake_percentage}%</div><div class="label">Fake Percentage</div></div>
      <div class="summary-cell"><div class="num">${summary.average_rating || "—"}</div><div class="label">Avg Rating</div></div>
      <div class="summary-cell"><div class="num">${summary.overall_risk}</div><div class="label">Overall Risk</div></div>
    </div>
    <p class="hint">Positive: ${summary.positive_count} &nbsp;·&nbsp; Neutral: ${summary.neutral_count} &nbsp;·&nbsp; Negative: ${summary.negative_count}</p>
  `;
}

function renderReviewTable(reviews) {
  const rows = reviews
    .map((r) => {
      const stars = "★".repeat(Math.round(r.rating || 0)) + "☆".repeat(5 - Math.round(r.rating || 0));
      return `
        <tr>
          <td>${escapeHtml(truncate(r.review, 90))}</td>
          <td>${r.rating ? stars : "—"}</td>
          <td>${r.prediction}</td>
          <td>${Math.round(r.confidence * 100)}%</td>
          <td>${r.sentiment}</td>
          <td><span class="badge ${r.risk_level.toLowerCase()}">${r.risk_score}</span></td>
        </tr>
      `;
    })
    .join("");

  return `
    <table class="review-table">
      <thead>
        <tr><th>Review</th><th>Rating</th><th>Prediction</th><th>Confidence</th><th>Sentiment</th><th>Risk</th></tr>
      </thead>
      <tbody>${rows}</tbody>
    </table>
  `;
}

function truncate(str, len) {
  if (!str) return "";
  return str.length > len ? str.slice(0, len) + "…" : str;
}

function escapeHtml(str) {
  const div = document.createElement("div");
  div.textContent = str;
  return div.innerHTML;
}
