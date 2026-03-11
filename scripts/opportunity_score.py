from __future__ import annotations

WEIGHTS = {
    "trend_alignment": 0.35,
    "option_chain_pressure": 0.30,
    "volatility_regime": 0.20,
    "pattern_quality": 0.15,
}


def _strength(value: float | int | None) -> float:
    if value is None:
        return 0.0
    value = float(value)
    if -1 <= value <= 1:
        return abs(value) * 100
    return max(0.0, min(value, 100.0))


def _label(score: float) -> str:
    if score < 40:
        return "Low"
    if score < 70:
        return "Moderate"
    return "High"


def calculate_opportunity(trend, option_pressure, volatility, pattern_quality) -> dict:
    score = round(
        _strength(trend) * WEIGHTS["trend_alignment"]
        + _strength(option_pressure) * WEIGHTS["option_chain_pressure"]
        + _strength(volatility) * WEIGHTS["volatility_regime"]
        + _strength(pattern_quality) * WEIGHTS["pattern_quality"],
        1,
    )
    return {"score": score, "label": _label(score), "weights": WEIGHTS}
