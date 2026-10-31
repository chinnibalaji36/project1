"""
services/flipkart_service.py
------------------------------
Flipkart review fetching layer.

DESIGN NOTE (read this before wiring up a real data source):
Flipkart does not publish a public, official API for arbitrary product
reviews, and this project intentionally does NOT invent or hard-code
an unofficial/undocumented endpoint. Scraping Flipkart's website
directly can violate its Terms of Service, so it is not implemented
here either.

Instead, this file is a clean, replaceable "fetch layer":
  - `is_configured()` tells the rest of the app whether a real data
    source has been wired up via environment variables.
  - `fetch_reviews(product_url)` is the single function the rest of
    the app calls. Internally it delegates to `_fetch_from_api(...)`.
  - If you have legitimate access to a product-review data provider
    (e.g. a licensed data API, or your own authorized scraper/service
    that you are permitted to use), implement `_fetch_from_api()` to
    call it and return data in the structure below. Nothing else in
    the app needs to change.

Expected return structure (list of dicts):
    [
        {"review": "...", "rating": 5, "username": "...", "date": "..."},
        ...
    ]

If no provider is configured, or the call fails for any reason, this
service returns a structured error instead of raising, so the rest of
the app can show a friendly message and the user can fall back to
manual review entry. The app must never crash because of this.
"""

import os
import re
import requests

FLIPKART_API_KEY = os.getenv("FLIPKART_API_KEY", "").strip()
FLIPKART_API_HOST = os.getenv("FLIPKART_API_HOST", "").strip()
FLIPKART_API_URL = os.getenv("FLIPKART_API_URL", "").strip()

FLIPKART_URL_PATTERN = re.compile(r"^https?://(www\.)?flipkart\.com/.+", re.IGNORECASE)


def is_valid_flipkart_url(url: str) -> bool:
    if not url or not isinstance(url, str):
        return False
    return bool(FLIPKART_URL_PATTERN.match(url.strip()))


def is_configured() -> bool:
    """True only if the user has supplied their own review-data provider."""
    return bool(FLIPKART_API_KEY and FLIPKART_API_HOST and FLIPKART_API_URL)


def _fetch_from_api(product_url: str) -> list:
    """
    Calls the review-data provider configured via environment variables.

    This is deliberately generic (a simple GET with an API key header)
    because the exact provider is left up to the developer deploying
    this project. Adjust the request shape to match whatever legitimate
    data source you configure.
    """
    headers = {
        "X-API-KEY": FLIPKART_API_KEY,
        "X-API-HOST": FLIPKART_API_HOST,
    }
    params = {"url": product_url}

    response = requests.get(FLIPKART_API_URL, headers=headers, params=params, timeout=15)
    response.raise_for_status()
    data = response.json()

    # Normalize whatever shape the provider returns into our structure.
    # Adjust this mapping to match your provider's actual response.
    reviews = []
    for item in data.get("reviews", []):
        reviews.append({
            "review": item.get("review") or item.get("text") or "",
            "rating": item.get("rating"),
            "username": item.get("username") or item.get("author") or "Anonymous",
            "date": item.get("date") or "",
        })
    return reviews


def fetch_reviews(product_url: str) -> dict:
    """
    Main entry point used by app.py.

    Returns:
        {
            "success": True,
            "reviews": [...],
        }
        or
        {
            "success": False,
            "message": "human-readable reason"
        }
    """
    if not is_valid_flipkart_url(product_url):
        return {
            "success": False,
            "message": "Please enter a valid Flipkart product URL "
                        "(e.g. https://www.flipkart.com/...).",
        }

    if not is_configured():
        return {
            "success": False,
            "message": "Unable to fetch reviews automatically. "
                        "Please check the API configuration or enter reviews manually.",
        }

    try:
        reviews = _fetch_from_api(product_url)
    except requests.exceptions.RequestException:
        return {
            "success": False,
            "message": "Unable to fetch reviews automatically. "
                        "Please check the API configuration or enter reviews manually.",
        }
    except Exception:  # noqa: BLE001 - never let fetching crash the app
        return {
            "success": False,
            "message": "Unable to fetch reviews automatically. "
                        "Please check the API configuration or enter reviews manually.",
        }

    if not reviews:
        return {
            "success": False,
            "message": "No reviews were returned for this product. "
                        "Please try another product or enter reviews manually.",
        }

    return {"success": True, "reviews": reviews}
