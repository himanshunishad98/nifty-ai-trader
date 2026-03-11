from __future__ import annotations

WEIGHTS = {"indicators": 0.4, "options": 0.3, "patterns": 0.2, "sentiment": 0.1}


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


def calculate_signal_quality(indicators, options, patterns, sentiment) -> dict:
    score = round(
        _strength(indicators) * WEIGHTS["indicators"]
        + _strength(options) * WEIGHTS["options"]
        + _strength(patterns) * WEIGHTS["patterns"]
        + _strength(sentiment) * WEIGHTS["sentiment"],
        1,
    )
    return {"score": score, "label": _label(score), "weights": WEIGHTS}
