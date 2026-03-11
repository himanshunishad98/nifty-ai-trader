from __future__ import annotations


def _step(rows: list[dict]) -> int:
    strikes = sorted({int(row["strike"]) for row in rows if row.get("strike")})
    diffs = [right - left for left, right in zip(strikes, strikes[1:]) if right > left]
    return min(diffs) if diffs else 50


def _atm_strike(spot: float, rows: list[dict]) -> int:
    strikes = sorted({int(row["strike"]) for row in rows if row.get("strike")})
    if not strikes:
        return round(spot / 50) * 50
    return min(strikes, key=lambda strike: abs(strike - spot))


def suggest_trade(signal: str, spot: float | None, option_chain: dict, market_regime: dict, confidence: float) -> dict:
    if signal == "NO TRADE" or spot is None:
        return {"strike": None, "expiry": option_chain.get("expiry"), "entry_level": None, "stop_loss": None, "target": None}
    rows = option_chain.get("rows", [])
    step = _step(rows)
    atm = _atm_strike(spot, rows)
    use_atm = confidence >= 70
    offset = 0 if use_atm else step
    move = max(market_regime.get("expected_move_15m", 0.0), spot * 0.002)
    if signal == "BUY CALL":
        strike = atm + offset
        return {
            "strike": f"{strike} CE",
            "expiry": option_chain.get("expiry"),
            "entry_level": round(spot, 2),
            "stop_loss": round(spot - move, 2),
            "target": round(spot + move * 1.8, 2),
        }
    strike = atm - offset
    return {
        "strike": f"{strike} PE",
        "expiry": option_chain.get("expiry"),
        "entry_level": round(spot, 2),
        "stop_loss": round(spot + move, 2),
        "target": round(spot - move * 1.8, 2),
    }
