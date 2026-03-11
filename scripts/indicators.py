from __future__ import annotations

import numpy as np
import pandas as pd
from ta.momentum import RSIIndicator
from ta.trend import EMAIndicator, MACD
from ta.volatility import BollingerBands


def calculate_indicators(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df.copy()
    data = df.copy()
    volume = data.get("Volume", pd.Series(1, index=data.index)).replace(0, np.nan).fillna(1)
    macd = MACD(close=data["Close"])
    bands = BollingerBands(close=data["Close"], window=20, window_dev=2)
    typical_price = (data["High"] + data["Low"] + data["Close"]) / 3
    data["rsi"] = RSIIndicator(close=data["Close"], window=14).rsi()
    data["macd"] = macd.macd()
    data["macd_signal"] = macd.macd_signal()
    data["ema9"] = EMAIndicator(close=data["Close"], window=9).ema_indicator()
    data["ema21"] = EMAIndicator(close=data["Close"], window=21).ema_indicator()
    data["vwap"] = (typical_price * volume).cumsum() / volume.cumsum()
    data["bb_high"] = bands.bollinger_hband()
    data["bb_low"] = bands.bollinger_lband()
    data["bb_mid"] = bands.bollinger_mavg()
    return data


def _label(score: float) -> str:
    if score > 0.2:
        return "BULLISH"
    if score < -0.2:
        return "BEARISH"
    return "NEUTRAL"


def _rsi_score(value: float) -> float:
    if pd.isna(value):
        return 0.0
    if value > 55:
        return 1.0
    if value < 45:
        return -1.0
    return 0.0


def _compare(a: float, b: float) -> float:
    if pd.isna(a) or pd.isna(b):
        return 0.0
    return 1.0 if a > b else -1.0


def summarize_signals(df: pd.DataFrame) -> dict:
    data = calculate_indicators(df)
    if data.empty:
        return {"score": 0.0, "bias": "NEUTRAL", "signals": {}}
    row = data.iloc[-1]
    signal_scores = {
        "RSI": _rsi_score(row.get("rsi", np.nan)),
        "MACD": _compare(row.get("macd", np.nan), row.get("macd_signal", np.nan)),
        "EMA": _compare(row.get("ema9", np.nan), row.get("ema21", np.nan)),
        "VWAP": _compare(row.get("Close", np.nan), row.get("vwap", np.nan)),
        "Bollinger": _compare(row.get("Close", np.nan), row.get("bb_mid", np.nan)),
    }
    score = round(float(sum(signal_scores.values()) / len(signal_scores)), 2)
    signals = {name: _label(value) for name, value in signal_scores.items()}
    return {"score": score, "bias": _label(score), "signals": signals, "data": data}
