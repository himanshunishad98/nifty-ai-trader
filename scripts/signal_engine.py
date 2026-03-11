from __future__ import annotations

from scripts.analysis_engine import analyze_snapshot
from scripts.config import ANALYSIS_CONFIG
from scripts.data_engine import get_connection, refresh_snapshot
from scripts.deep_learning_model import maybe_retrain_mlp, predict_dl
from scripts.global_market_data import fetch_global_market_data
from scripts.market_regime import classify_market_regime
from scripts.opportunity_score import calculate_opportunity
from scripts.predictive_model import build_feature_row, maybe_retrain_model, predict_probabilities
from scripts.prediction_tracker import performance_summary, record_prediction, resolve_predictions, store_feature_snapshot, timeline
from scripts.reasoning import build_reasoning
from scripts.rl_agent import predict_rl, update_rl_agent
from scripts.signal_quality import calculate_signal_quality
from scripts.strike_selector import suggest_trade

WEIGHTS = ANALYSIS_CONFIG.get("weights", {"indicators": 0.4, "option_chain": 0.3, "patterns": 0.2, "news_sentiment": 0.1})


def _signal_label(score: float) -> str:
    if score >= 0.2:
        return "BUY CALL"
    if score <= -0.2:
        return "BUY PUT"
    return "NO TRADE"


def _confidence(score: float, signal: str) -> float:
    if signal == "NO TRADE":
        return round(max(0.0, 1 - min(abs(score) / 0.2, 1)) * 100, 1)
    return round(abs(score) * 100, 1)


def _support_resistance(price_data, option_data: dict, structure: dict) -> dict:
    if price_data.empty:
        return {
            "price_support": structure.get("support"),
            "price_resistance": structure.get("resistance"),
            "option_support": option_data.get("support"),
            "option_resistance": option_data.get("resistance"),
        }
    recent = price_data.tail(20)
    return {
        "price_support": structure.get("support") or round(float(recent["Low"].min()), 2),
        "price_resistance": structure.get("resistance") or round(float(recent["High"].max()), 2),
        "option_support": option_data.get("support"),
        "option_resistance": option_data.get("resistance"),
    }


def _volatility_snapshot(vix_data, regime: str) -> dict:
    if vix_data.empty:
        return {"value": None, "regime": "Unknown", "trend": "NEUTRAL", "score": 0.5, "data": vix_data}
    recent = vix_data.tail(10)
    value = round(float(vix_data["Close"].iloc[-1]), 2)
    start = float(recent["Close"].iloc[0])
    end = float(recent["Close"].iloc[-1])
    trend = "RISING" if end > start else "FALLING" if end < start else "FLAT"
    if value < 13:
        band, score = "Low", 0.6
    elif value < 18:
        band, score = "Moderate", 1.0
    elif value < 22:
        band, score = "Elevated", 0.7
    else:
        band, score = "High", 0.35
    return {"value": value, "regime": band, "trend": trend, "score": score, "data": vix_data, "market_regime": regime}


def _trend_alignment(signal: str, structure_trend: str, indicator_score: float) -> float:
    if signal == "BUY CALL" and structure_trend == "BULLISH":
        return 1.0
    if signal == "BUY PUT" and structure_trend == "BEARISH":
        return 1.0
    if structure_trend == "NEUTRAL":
        return abs(indicator_score) * 0.6
    return abs(indicator_score) * 0.2


def generate_signal() -> dict:
    conn = get_connection()
    snapshot = refresh_snapshot(conn)
    resolve_predictions(conn)
    analysis = analyze_snapshot(snapshot)
    market_data = snapshot["market_data"]
    data_5m = market_data["5m"]
    indicators = analysis["indicators"]
    indicator_score = indicators["score"]
    patterns = analysis["patterns"]
    news = snapshot["news"]
    option_chain = snapshot["option_chain"]
    options_intelligence = analysis["options_intelligence"]
    market_structure = analysis["market_structure"]
    market_regime = classify_market_regime(data_5m, snapshot.get("vix"), market_structure)
    volatility = _volatility_snapshot(snapshot["vix_data"], market_regime["regime"])
    composite_score = round(
        indicator_score * WEIGHTS["indicators"]
        + option_chain.get("score", 0.0) * WEIGHTS["option_chain"]
        + patterns["score"] * WEIGHTS["patterns"]
        + news["score"] * WEIGHTS["news_sentiment"],
        2,
    )
    signal = _signal_label(composite_score)
    opportunity = calculate_opportunity(
        trend=_trend_alignment(signal, market_structure["trend"], indicator_score),
        option_pressure=option_chain.get("score", 0.0),
        volatility=volatility["score"],
        pattern_quality=patterns["score"],
    )
    signal_quality = calculate_signal_quality(
        indicators=indicator_score,
        options=option_chain.get("score", 0.0),
        patterns=patterns["score"],
        sentiment=news["score"],
    )
    base_payload = {
        "spot": round(float(data_5m["Close"].iloc[-1]), 2) if not data_5m.empty else snapshot.get("nifty_price"),
        "signal": signal,
        "confidence": _confidence(composite_score, signal),
        "score": composite_score,
        "market_data": market_data,
        "indicators": indicators,
        "indicator_values": analysis["indicator_values"],
        "patterns": patterns,
        "news": news,
        "option_chain": option_chain,
        "options_intelligence": options_intelligence,
        "market_structure": market_structure,
        "volatility": volatility,
        "market_regime": market_regime,
    }
    trade = suggest_trade(signal, base_payload["spot"], option_chain, market_regime, base_payload["confidence"])
    global_market = fetch_global_market_data()
    global_features = global_market.get("features", {})
    feature_row = build_feature_row(base_payload, global_features)
    store_feature_snapshot(conn, snapshot["timestamp"], base_payload["spot"], feature_row, signal, base_payload["confidence"], composite_score)
    model_status = maybe_retrain_model(conn)
    dl_status    = maybe_retrain_mlp(conn)
    rl_status    = update_rl_agent(conn)
    prediction   = predict_probabilities(feature_row, composite_score)
    dl_prediction = predict_dl(feature_row)
    rl_prediction = predict_rl(feature_row)
    model_status.update(dl_status)
    model_status.update(rl_status)
    record_prediction(conn, snapshot["timestamp"], signal, base_payload["confidence"], trade, base_payload["spot"], prediction)
    performance = performance_summary(conn)
    signal_timeline = timeline(conn)
    reasoning = build_reasoning(signal, indicators, patterns, option_chain, news, volatility)
    conn.close()
    return {
        **base_payload,
        "weights": WEIGHTS,
        "opportunity": opportunity,
        "signal_quality": signal_quality,
        "component_scores": {
            "Indicators": indicator_score,
            "Options": option_chain.get("score", 0.0),
            "Patterns": patterns["score"],
            "Sentiment": news["score"],
        },
        "trade": trade,
        "feature_row": feature_row,
        "prediction": prediction,
        "prediction_performance": performance,
        "signal_timeline": signal_timeline,
        "model_status": model_status,
        "data_timestamp": snapshot["timestamp"],
        "data_freshness": snapshot["freshness"],
        "data_age_seconds": snapshot["age_seconds"],
        "reasoning": reasoning,
        "levels": _support_resistance(data_5m, option_chain, market_structure),
        "global_market": global_market,
        "dl_prediction": dl_prediction,
        "rl_prediction": rl_prediction,
    }
