from __future__ import annotations

from scripts.indicators import calculate_indicators


def classify_market_regime(price_data, vix_value: float | None, structure: dict) -> dict:
    if price_data.empty:
        return {"regime": "RANGE", "expected_move": 0.0, "expected_move_15m": 0.0}
    data = calculate_indicators(price_data)
    row = data.iloc[-1]
    spot = float(row["Close"])
    ema_spread = abs(float(row.get("ema9", spot)) - float(row.get("ema21", spot))) / spot
    if vix_value is None:
        vix_value = 15.0
    if vix_value >= 20:
        regime = "HIGH_VOLATILITY"
    elif vix_value <= 13:
        regime = "LOW_VOLATILITY"
    elif structure.get("trend") != "NEUTRAL" and ema_spread > 0.0015:
        regime = "TRENDING"
    else:
        regime = "RANGE"
    expected_move = round(spot * (vix_value / 100) / (365 ** 0.5), 2)
    expected_move_15m = round(expected_move * (15 / 390) ** 0.5, 2)
    return {"regime": regime, "expected_move": expected_move, "expected_move_15m": expected_move_15m}
