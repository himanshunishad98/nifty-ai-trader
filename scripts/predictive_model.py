from __future__ import annotations

import math
import pickle
from pathlib import Path

import pandas as pd
from sklearn.ensemble import RandomForestClassifier

FEATURE_COLUMNS = [
    "rsi",
    "macd",
    "ema_trend",
    "pcr",
    "oi_pressure",
    "gamma_flip_distance",
    "total_gex",
    "india_vix",
    "sentiment_score",
    "pattern_signal",
    # Global market features
    "sp500_change",
    "nikkei_change",
    "crude_change",
    "gold_change",
    "usdinr_change",
    "dxy_change",
    "us10y_yield",
]
ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT / "models"
MODEL_PATH = MODEL_DIR / "predictor.pkl"


def _clean(value: float | int | None) -> float:
    if value is None or pd.isna(value):
        return 0.0
    return float(value)


def build_feature_row(payload: dict, global_features: dict | None = None) -> dict:
    values = payload["indicator_values"]
    gamma = payload["options_intelligence"]["gamma"]
    spot = payload["spot"] or 1.0
    gamma_flip = gamma.get("gamma_flip")
    total_gex = gamma.get("total_gex", 0.0)
    gf = global_features or {}
    return {
        "rsi": _clean(values.get("rsi", 0.0)),
        "macd": _clean(values.get("macd", 0.0)),
        "ema_trend": 1.0 if _clean(values.get("ema9", 0.0)) > _clean(values.get("ema21", 0.0)) else 0.0,
        "pcr": _clean(payload["options_intelligence"].get("pcr", 0.0)),
        "oi_pressure": _clean(payload["options_intelligence"]["smart_money_pressure"]["score"]),
        "gamma_flip_distance": 0.0 if gamma_flip is None else round((gamma_flip - spot) / spot, 4),
        "total_gex": math.copysign(math.log1p(abs(_clean(total_gex))), _clean(total_gex)),
        "india_vix": _clean(payload["volatility"].get("value", 0.0)),
        "sentiment_score": _clean(payload["news"].get("score", 0.0)),
        "pattern_signal": _clean(payload["patterns"].get("score", 0.0)),
        # Global features
        "sp500_change":  _clean(gf.get("sp500_change", 0.0)),
        "nikkei_change": _clean(gf.get("nikkei_change", 0.0)),
        "crude_change":  _clean(gf.get("crude_change", 0.0)),
        "gold_change":   _clean(gf.get("gold_change", 0.0)),
        "usdinr_change": _clean(gf.get("usdinr_change", 0.0)),
        "dxy_change":    _clean(gf.get("dxy_change", 0.0)),
        "us10y_yield":   _clean(gf.get("us10y_yield", 0.0)),
    }


def _load_model():
    if not MODEL_PATH.exists():
        return None
    with MODEL_PATH.open("rb") as handle:
        return pickle.load(handle)


def maybe_retrain_model(conn, min_rows: int = 40) -> dict:
    query = "SELECT * FROM feature_snapshots WHERE target IS NOT NULL"
    frame = pd.read_sql_query(query, conn)
    if len(frame) < min_rows or frame["target"].nunique() < 2:
        return {"trained": False, "samples": len(frame)}
    model = RandomForestClassifier(n_estimators=200, random_state=42, min_samples_leaf=2)
    model.fit(frame[FEATURE_COLUMNS], frame["target"])
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    with MODEL_PATH.open("wb") as handle:
        pickle.dump(model, handle)
    return {"trained": True, "samples": len(frame), "features": FEATURE_COLUMNS}


def predict_probabilities(feature_row: dict, composite_score: float) -> dict:
    model = _load_model()
    if model is None:
        bullish = max(0.05, min(0.95, 0.5 + composite_score / 2))
        return {"bullish_probability": round(bullish, 3), "bearish_probability": round(1 - bullish, 3), "model_ready": False}
    frame = pd.DataFrame([feature_row])[FEATURE_COLUMNS]
    probabilities = model.predict_proba(frame)[0]
    classes = list(model.classes_)
    bullish = probabilities[classes.index(1)] if 1 in classes else 0.5
    bearish = probabilities[classes.index(0)] if 0 in classes else 0.5
    return {"bullish_probability": round(float(bullish), 3), "bearish_probability": round(float(bearish), 3), "model_ready": True}
