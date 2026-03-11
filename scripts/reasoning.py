from __future__ import annotations


def _pattern_text(patterns: dict) -> str:
    names = [item["name"] for item in patterns.get("patterns", [])]
    if not names:
        return "price action is neutral"
    return f"patterns show {', '.join(names[:2]).lower()}"


def build_reasoning(signal: str, indicators: dict, patterns: dict, option_chain: dict, sentiment: dict, volatility: dict) -> str:
    indicator_bias = indicators["5m"]["bias"].lower()
    option_bias = option_chain.get("bias", "NEUTRAL").lower()
    sentiment_bias = sentiment.get("bias", "NEUTRAL").lower()
    volatility_regime = volatility.get("regime", "Unknown").lower()
    parts = [
        f"{signal} is driven by {indicator_bias} intraday indicators.",
        f"Option chain pressure is {option_bias} with PCR at {option_chain.get('pcr', 0)}.",
        f"{_pattern_text(patterns)} while news sentiment is {sentiment_bias}.",
        f"India VIX is in a {volatility_regime} regime.",
    ]
    return " ".join(parts)


def answer_question(question: str, payload: dict) -> str:
    text = question.lower()
    if any(token in text for token in ["signal", "bias", "call", "put"]):
        return f"Current bias is {payload['signal']} with {payload['confidence']}% confidence. {payload['reasoning']}"
    if any(token in text for token in ["trade", "strike", "entry", "stop", "target"]):
        trade = payload["trade"]
        if not trade.get("strike"):
            return "No trade is suggested right now because the engine is neutral."
        return f"Suggested setup is {trade['strike']} for {trade['expiry']} with entry near {trade['entry_level']}, stop {trade['stop_loss']}, target {trade['target']}."
    if any(token in text for token in ["vix", "volatility", "regime"]):
        regime = payload["market_regime"]
        return f"Market regime is {regime['regime']} and India VIX is {payload['volatility']['value']} with an expected move of {regime['expected_move']} points."
    if any(token in text for token in ["option", "pcr", "gamma", "flow"]):
        options = payload["options_intelligence"]
        gamma = options["gamma"]
        return f"PCR is {options['pcr']}, smart money pressure is {options['smart_money_pressure']['label']}, gamma flip is {gamma['gamma_flip']}, and dealer positioning is {gamma['dealer_positioning']}."
    if any(token in text for token in ["performance", "accuracy", "model", "prediction"]):
        perf = payload["prediction_performance"]
        prediction = payload["prediction"]
        return f"Prediction accuracy is {perf['accuracy']}% across {perf['resolved']} resolved calls. Current model probabilities are {prediction['bullish_probability']*100:.1f}% bullish and {prediction['bearish_probability']*100:.1f}% bearish."
    return payload["reasoning"]
