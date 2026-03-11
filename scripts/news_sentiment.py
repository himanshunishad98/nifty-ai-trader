"""news_sentiment.py
Multi-source, NLP-powered news sentiment engine for Indian market analysis.
Uses VADER for sentiment scoring and covers 8 topic categories via RSS feeds
and Google News queries.
"""
from __future__ import annotations

import re
from concurrent.futures import ThreadPoolExecutor, as_completed

import feedparser
import requests
from bs4 import BeautifulSoup

try:
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    _VADER = SentimentIntensityAnalyzer()
except ImportError:
    _VADER = None

try:
    from gnews import GNews
    _GNEWS = GNews(language="en", country="IN", period="1d", max_results=5)
except Exception:
    _GNEWS = None

from scripts.config import NEWS_SENTIMENT_CONFIG

HEADERS = NEWS_SENTIMENT_CONFIG.get("headers", {"User-Agent": "Mozilla/5.0 (compatible)"})

# RSS Feeds
RSS_FEEDS = {
    "Economic Times":    "https://economictimes.indiatimes.com/markets/rssfeeds/1977021501.cms",
    "Moneycontrol":      "https://www.moneycontrol.com/rss/MCtopnews.xml",
    "LiveMint":          "https://www.livemint.com/rss/markets",
    "Business Standard": "https://www.business-standard.com/rss/markets-106.rss",
    "NDTV Profit":       "https://www.ndtv.com/business/rss-topics/market",
}

# Google News topic queries covering all India-market-moving categories
GNEWS_QUERIES = [
    "NIFTY 50 market today",
    "Sensex BSE today",
    "FII DII activity India",
    "RBI repo rate MPC",
    "crude oil India inflation",
    "USD INR rupee",
    "gold price India today",
    "US Fed rate interest",
    "India quarterly results earnings",
    "SEBI budget India market",
    "global recession US market",
    "China economy Asia market",
    "FII selling buying India equity",
]

# Keywords to filter out irrelevant articles (only keep India-market-relevant ones)
INDIA_MARKET_KEYWORDS = {
    "nifty", "sensex", "bse", "nse", "india", "rbi", "sebi", "rupee", "fii", "dii",
    "market", "equity", "stock", "shares", "crude", "gold", "inflation", "rate",
    "economy", "gdp", "budget", "rally", "selling", "buying", "earnings", "results",
}

# Simple fallback word lists for when VADER is not available
_FALLBACK_POS = {"gain", "rally", "surge", "up", "rise", "bullish", "strong", "beat", "growth", "rebound", "positive"}
_FALLBACK_NEG = {"fall", "drop", "crash", "down", "slump", "bearish", "weak", "miss", "loss", "negative", "sell-off"}


def _clean(text: str) -> str:
    return BeautifulSoup(text or "", "html.parser").get_text(" ", strip=True)


def _vader_score(text: str) -> float:
    """Returns compound VADER score or fallback keyword score."""
    if _VADER:
        return round(_VADER.polarity_scores(text)["compound"], 3)
    # Fallback to simple keyword count
    words = re.findall(r"[a-zA-Z]+", text.lower())
    pos = sum(w in _FALLBACK_POS for w in words)
    neg = sum(w in _FALLBACK_NEG for w in words)
    total = pos + neg
    return 0.0 if total == 0 else round((pos - neg) / total, 2)


def _is_relevant(text: str) -> bool:
    text_lower = text.lower()
    return any(kw in text_lower for kw in INDIA_MARKET_KEYWORDS)


def _fetch_rss(source: str, url: str, limit: int) -> list[dict]:
    try:
        response = requests.get(url, headers=HEADERS, timeout=8)
        response.raise_for_status()
        feed = feedparser.parse(response.text)
    except Exception:
        return []
    results = []
    for entry in feed.entries[:limit]:
        title = _clean(entry.get("title", ""))
        summary = _clean(entry.get("summary", ""))
        combined = f"{title} {summary}".strip()
        if not title or not _is_relevant(combined):
            continue
        score = _vader_score(combined)
        results.append({
            "source": source,
            "category": "RSS",
            "title": title,
            "score": score,
            "compound": score,
        })
    return results


def _fetch_gnews_query(query: str) -> list[dict]:
    if _GNEWS is None:
        return []
    try:
        articles = _GNEWS.get_news(query) or []
    except Exception:
        return []
    results = []
    for article in articles:
        title = article.get("title", "")
        desc  = article.get("description", "")
        combined = f"{title} {desc}".strip()
        if not title or not _is_relevant(combined):
            continue
        score = _vader_score(combined)
        results.append({
            "source": "Google News",
            "category": _category_for_query(query),
            "title": title,
            "score": score,
            "compound": score,
        })
    return results


def _category_for_query(query: str) -> str:
    q = query.lower()
    if any(k in q for k in ("nifty", "sensex", "bse", "nse")):            return "Equity"
    if any(k in q for k in ("fii", "dii")):                               return "FII/DII"
    if any(k in q for k in ("rbi", "repo", "mpc")):                       return "Monetary Policy"
    if any(k in q for k in ("crude", "oil", "petrol", "inflation")):      return "Commodity/Inflation"
    if any(k in q for k in ("usd", "inr", "rupee")):                      return "Currency"
    if any(k in q for k in ("gold", "silver", "metal")):                  return "Metals"
    if any(k in q for k in ("fed", "recession", "global", "us market")): return "Global Cues"
    if any(k in q for k in ("results", "earnings", "sebi", "budget")):    return "Corporate/Policy"
    return "General"


def _bias_label(score: float) -> str:
    if score >= 0.15:  return "BULLISH"
    if score <= -0.15: return "BEARISH"
    return "NEUTRAL"


def _category_scores(headlines: list[dict]) -> dict[str, float]:
    buckets: dict[str, list[float]] = {}
    for h in headlines:
        cat = h.get("category", "General")
        buckets.setdefault(cat, []).append(h["score"])
    return {cat: round(sum(v) / len(v), 3) for cat, v in buckets.items()}


def fetch_news_sentiment(limit: int = 5) -> dict:
    """Fetch and score headlines from all sources concurrently using VADER NLP."""
    all_headlines: list[dict] = []

    with ThreadPoolExecutor(max_workers=10) as pool:
        futures = {}

        # RSS feeds
        for source, url in RSS_FEEDS.items():
            futures[pool.submit(_fetch_rss, source, url, limit)] = source

        # Google News queries
        for query in GNEWS_QUERIES:
            futures[pool.submit(_fetch_gnews_query, query)] = query

        for future in as_completed(futures):
            try:
                all_headlines.extend(future.result())
            except Exception:
                pass

    if not all_headlines:
        return {
            "score": 0.0,
            "bias": "NEUTRAL",
            "headlines": [],
            "category_scores": {},
            "source_count": 0,
        }

    # Deduplicate by title similarity (simple check)
    seen: set[str] = set()
    unique: list[dict] = []
    for h in all_headlines:
        key = h["title"][:60].lower()
        if key not in seen:
            seen.add(key)
            unique.append(h)

    score = round(sum(h["score"] for h in unique) / len(unique), 3)
    category_scores = _category_scores(unique)

    # Sort: most extreme scores first for better UX display
    unique.sort(key=lambda x: abs(x["score"]), reverse=True)

    return {
        "score": score,
        "bias": _bias_label(score),
        "headlines": unique[:15],
        "category_scores": category_scores,
        "source_count": len({h["source"] for h in unique}),
    }
