"""global_market_data.py
Fetches live global market data (indices, commodities, currency, bonds)
that are significant for Indian equity market prediction.
"""
from __future__ import annotations

import yfinance as yf

GLOBAL_INSTRUMENTS = {
    "S&P 500":       {"symbol": "^GSPC",    "category": "US Equity",    "india_impact": "FII Flow"},
    "Dow Jones":     {"symbol": "^DJI",     "category": "US Equity",    "india_impact": "Risk Sentiment"},
    "NASDAQ":        {"symbol": "^IXIC",    "category": "US Equity",    "india_impact": "Tech / Risk-On"},
    "Nikkei 225":    {"symbol": "^N225",    "category": "Asian Equity",  "india_impact": "Asia Session"},
    "Hang Seng":     {"symbol": "^HSI",     "category": "Asian Equity",  "india_impact": "China Risk"},
    "Crude WTI":     {"symbol": "CL=F",     "category": "Commodity",    "india_impact": "Inflation / CAD"},
    "Brent Crude":   {"symbol": "BZ=F",     "category": "Commodity",    "india_impact": "Oil Benchmark"},
    "Gold":          {"symbol": "GC=F",     "category": "Commodity",    "india_impact": "Safe Haven"},
    "USD/INR":       {"symbol": "USDINR=X", "category": "Currency",     "india_impact": "Rupee Strength"},
    "DXY Index":     {"symbol": "DX=F",     "category": "Currency",     "india_impact": "Dollar Strength"},
    "US 10Y Yield":  {"symbol": "^TNX",     "category": "Bond",         "india_impact": "Capital Flows"},
}

# Map of instrument → feature column name for ML model
GLOBAL_FEATURE_KEYS = {
    "S&P 500":   "sp500_change",
    "Nikkei 225":"nikkei_change",
    "Crude WTI": "crude_change",
    "Gold":      "gold_change",
    "USD/INR":   "usdinr_change",
    "DXY Index": "dxy_change",
    "US 10Y Yield": "us10y_yield",
}


def _direction(change_pct: float) -> str:
    if change_pct >= 0.3:
        return "UP"
    if change_pct <= -0.3:
        return "DOWN"
    return "FLAT"


def _india_sentiment(name: str, change_pct: float) -> float:
    """Transform a global instrument's % change into an India market sentiment score (-1 to 1)."""
    sign = change_pct / 100.0  # normalise
    # Instruments where a rise hurts India
    bearish_for_india = {"Crude WTI", "Brent Crude", "DXY Index", "US 10Y Yield", "USD/INR"}
    if name in bearish_for_india:
        return round(max(-1.0, min(1.0, -sign * 5)), 3)
    # Instruments where a rise helps India
    return round(max(-1.0, min(1.0, sign * 5)), 3)


def fetch_global_market_data() -> dict:
    """
    Returns a dict with:
      - 'instruments': list of instrument dicts with price, change_pct, direction, sentiment
      - 'overall_score': weighted composite India-impact score (-1 to 1)
      - 'features': flat dict of ML feature values
    """
    instruments = []
    feature_values: dict[str, float] = {}
    scores = []

    symbols = [info["symbol"] for info in GLOBAL_INSTRUMENTS.values()]
    try:
        tickers = yf.Tickers(" ".join(symbols))
    except Exception:
        return {"instruments": [], "overall_score": 0.0, "features": {k: 0.0 for k in GLOBAL_FEATURE_KEYS.values()}}

    for name, meta in GLOBAL_INSTRUMENTS.items():
        symbol = meta["symbol"]
        try:
            ticker = tickers.tickers[symbol]
            info = ticker.fast_info
            price = round(float(info.last_price), 2)
            prev_close = float(info.previous_close or price)
            change_pct = round((price - prev_close) / prev_close * 100, 2) if prev_close else 0.0
        except Exception:
            price, change_pct = 0.0, 0.0

        direction = _direction(change_pct)
        sentiment = _india_sentiment(name, change_pct)
        scores.append(sentiment)

        instruments.append({
            "name": name,
            "symbol": symbol,
            "category": meta["category"],
            "india_impact": meta["india_impact"],
            "price": price,
            "change_pct": change_pct,
            "direction": direction,
            "sentiment": sentiment,
        })

        # Populate ML features
        if name in GLOBAL_FEATURE_KEYS:
            feature_values[GLOBAL_FEATURE_KEYS[name]] = change_pct

    # Ensure all ML keys present even if fetch failed
    for v in GLOBAL_FEATURE_KEYS.values():
        feature_values.setdefault(v, 0.0)

    overall_score = round(sum(scores) / len(scores), 3) if scores else 0.0
    return {
        "instruments": instruments,
        "overall_score": overall_score,
        "features": feature_values,
    }
