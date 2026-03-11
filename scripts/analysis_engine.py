from __future__ import annotations

import math
from datetime import datetime, timezone

import pandas as pd

from scripts.config import ANALYSIS_CONFIG
from scripts.indicators import calculate_indicators, summarize_signals
from scripts.market_structure import analyze_market_structure
from scripts.patterns import detect_patterns

CONTRACT_SIZE = ANALYSIS_CONFIG.get("contract_size", 75)


def _norm_pdf(value: float) -> float:
    return math.exp(-0.5 * value * value) / math.sqrt(2 * math.pi)


def _gamma(spot: float, strike: float, iv: float, years: float) -> float:
    if min(spot, strike, iv, years) <= 0:
        return 0.0
    sigma = iv / 100
    if sigma <= 0:
        return 0.0
    try:
        d1 = (math.log(spot / strike) + 0.5 * sigma * sigma * years) / (sigma * math.sqrt(years))
    except (ValueError, ZeroDivisionError):
        return 0.0
    return _norm_pdf(d1) / (spot * sigma * math.sqrt(years))


def _years_to_expiry(expiry: str) -> float:
    if not expiry:
        return 1 / 365
    expiry_dt = datetime.strptime(expiry, "%d-%b-%Y").replace(tzinfo=timezone.utc)
    days = max((expiry_dt - datetime.now(timezone.utc)).total_seconds() / 86400, 1)
    return days / 365


def _label(score: float) -> str:
    if score > 0.15:
        return "BULLISH"
    if score < -0.15:
        return "BEARISH"
    return "NEUTRAL"


def _volume_spikes(rows: list[dict]) -> list[dict]:
    if not rows:
        return []
    frame = pd.DataFrame(rows)
    frame["total_volume"] = frame["call_volume"] + frame["put_volume"]
    threshold = frame["total_volume"].quantile(0.9)
    spikes = frame[frame["total_volume"] >= threshold].sort_values("total_volume", ascending=False).head(8)
    return spikes[["strike", "call_volume", "put_volume", "total_volume"]].to_dict("records")


def _gamma_profile(rows: list[dict], spot: float, expiry: str) -> dict:
    if not rows or not spot:
        return {
            "rows": [],
            "gamma_flip": None,
            "positive_wall": None,
            "negative_wall": None,
            "dealer_positioning": "NEUTRAL",
            "total_gex": 0.0,
        }
    years = _years_to_expiry(expiry)
    profile = []
    for row in sorted(rows, key=lambda item: item["strike"]):
        iv = next((value for value in [row.get("call_iv"), row.get("put_iv")] if value and value > 0), 12.0)
        gamma = _gamma(spot, row["strike"], iv, years)
        gex = gamma * (row["call_oi"] - row["put_oi"]) * CONTRACT_SIZE * spot
        profile.append({"strike": row["strike"], "gamma": gamma, "gex": gex})
    total_gex = sum(item["gex"] for item in profile)
    positive_wall = max(profile, key=lambda item: item["gex"]) if profile else None
    negative_wall = min(profile, key=lambda item: item["gex"]) if profile else None
    gamma_flip = None
    for left, right in zip(profile, profile[1:]):
        if left["gex"] == 0 or left["gex"] * right["gex"] < 0:
            gamma_flip = right["strike"]
            break
    return {
        "rows": profile,
        "gamma_flip": gamma_flip,
        "positive_wall": positive_wall["strike"] if positive_wall else None,
        "negative_wall": negative_wall["strike"] if negative_wall else None,
        "dealer_positioning": "LONG_GAMMA" if total_gex >= 0 else "SHORT_GAMMA",
        "total_gex": total_gex,
    }


def analyze_snapshot(snapshot: dict) -> dict:
    data_1m = snapshot["market_data"]["1m"]
    data_5m = snapshot["market_data"]["5m"]
    indicators_1m = summarize_signals(data_1m)
    indicators_5m = summarize_signals(data_5m)
    indicator_score = round(indicators_1m["score"] * 0.4 + indicators_5m["score"] * 0.6, 2)
    indicator_data = calculate_indicators(data_5m)
    latest = indicator_data.iloc[-1] if not indicator_data.empty else pd.Series(dtype=float)
    patterns = detect_patterns(data_5m)
    structure = analyze_market_structure(data_5m)
    option_chain = snapshot["option_chain"]
    rows = option_chain.get("rows", [])
    call_flow = sum(row["call_change_oi"] * max(row["call_volume"], 1) for row in rows)
    put_flow = sum(row["put_change_oi"] * max(row["put_volume"], 1) for row in rows)
    smart_money_score = round((put_flow - call_flow) / (abs(call_flow) + abs(put_flow) or 1), 2)
    gamma = _gamma_profile(rows, snapshot["nifty_price"] or option_chain.get("underlying_value") or 0, option_chain.get("expiry", ""))
    options_intelligence = {
        "pcr": option_chain.get("pcr", 0.0),
        "max_pain": option_chain.get("max_pain"),
        "oi_change": option_chain.get("net_change_total", 0),
        "volume_spikes": _volume_spikes(rows),
        "smart_money_pressure": {"score": smart_money_score, "label": _label(smart_money_score)},
        "gamma": gamma,
        "score": option_chain.get("score", 0.0),
        "bias": option_chain.get("bias", "NEUTRAL"),
    }
    return {
        "indicators": {"1m": indicators_1m, "5m": indicators_5m, "score": indicator_score},
        "indicator_values": {
            "rsi": round(float(latest.get("rsi", 0) or 0), 2),
            "macd": round(float(latest.get("macd", 0) or 0), 2),
            "ema9": round(float(latest.get("ema9", 0) or 0), 2),
            "ema21": round(float(latest.get("ema21", 0) or 0), 2),
            "vwap": round(float(latest.get("vwap", 0) or 0), 2),
            "bb_mid": round(float(latest.get("bb_mid", 0) or 0), 2),
        },
        "patterns": patterns,
        "market_structure": structure,
        "options_intelligence": options_intelligence,
    }
