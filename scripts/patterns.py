from __future__ import annotations

import pandas as pd


def _body(candle: pd.Series) -> float:
    return abs(candle["Close"] - candle["Open"])


def _range(candle: pd.Series) -> float:
    return max(candle["High"] - candle["Low"], 0.0)


def _upper_shadow(candle: pd.Series) -> float:
    return candle["High"] - max(candle["Open"], candle["Close"])


def _lower_shadow(candle: pd.Series) -> float:
    return min(candle["Open"], candle["Close"]) - candle["Low"]


def bullish_engulfing(prev: pd.Series, curr: pd.Series) -> bool:
    return (
        prev["Close"] < prev["Open"]
        and curr["Close"] > curr["Open"]
        and curr["Open"] <= prev["Close"]
        and curr["Close"] >= prev["Open"]
    )


def bearish_engulfing(prev: pd.Series, curr: pd.Series) -> bool:
    return (
        prev["Close"] > prev["Open"]
        and curr["Close"] < curr["Open"]
        and curr["Open"] >= prev["Close"]
        and curr["Close"] <= prev["Open"]
    )


def hammer(candle: pd.Series) -> bool:
    body = _body(candle)
    return body > 0 and _lower_shadow(candle) >= body * 2 and _upper_shadow(candle) <= body


def shooting_star(candle: pd.Series) -> bool:
    body = _body(candle)
    return body > 0 and _upper_shadow(candle) >= body * 2 and _lower_shadow(candle) <= body


def doji(candle: pd.Series) -> bool:
    candle_range = _range(candle)
    return candle_range > 0 and _body(candle) / candle_range <= 0.1


def detect_patterns(df: pd.DataFrame) -> dict:
    if len(df) < 2:
        return {"score": 0.0, "bias": "NEUTRAL", "patterns": []}
    prev, curr = df.iloc[-2], df.iloc[-1]
    patterns = []
    if bullish_engulfing(prev, curr):
        patterns.append({"name": "Bullish Engulfing", "signal": "BULLISH", "score": 1.0})
    if bearish_engulfing(prev, curr):
        patterns.append({"name": "Bearish Engulfing", "signal": "BEARISH", "score": -1.0})
    if hammer(curr):
        patterns.append({"name": "Hammer", "signal": "BULLISH", "score": 0.6})
    if shooting_star(curr):
        patterns.append({"name": "Shooting Star", "signal": "BEARISH", "score": -0.6})
    if doji(curr):
        patterns.append({"name": "Doji", "signal": "NEUTRAL", "score": 0.0})
    if not patterns:
        return {"score": 0.0, "bias": "NEUTRAL", "patterns": []}
    score = round(sum(item["score"] for item in patterns) / len(patterns), 2)
    bias = "BULLISH" if score > 0.2 else "BEARISH" if score < -0.2 else "NEUTRAL"
    return {"score": score, "bias": bias, "patterns": patterns}
