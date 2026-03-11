from __future__ import annotations

from datetime import datetime

import requests

from scripts.config import OPTION_CHAIN_CONFIG

try:
    from curl_cffi import requests as curl_requests
except ImportError:
    curl_requests = None

BASE_URL = OPTION_CHAIN_CONFIG.get("base_url", "https://www.nseindia.com")
HEADERS = OPTION_CHAIN_CONFIG.get("headers", {
    "User-Agent": "Mozilla/5.0",
    "Accept": "application/json,text/plain,*/*",
    "Referer": f"{BASE_URL}/option-chain",
    "Accept-Language": "en-US,en;q=0.9",
})


def _empty(expiry: str = "") -> dict:
    return {
        "expiry": expiry,
        "underlying_value": None,
        "pcr": 0.0,
        "max_pain": None,
        "highest_call_oi": None,
        "highest_put_oi": None,
        "highest_call_oi_value": 0,
        "highest_put_oi_value": 0,
        "call_change_total": 0,
        "put_change_total": 0,
        "net_change_total": 0,
        "support": None,
        "resistance": None,
        "score": 0.0,
        "bias": "NEUTRAL",
        "rows": [],
    }


def _nearest_expiry(expiries: list[str]) -> str:
    parsed = []
    for expiry in expiries:
        try:
            parsed.append((datetime.strptime(expiry, "%d-%b-%Y"), expiry))
        except ValueError:
            continue
    parsed.sort(key=lambda item: item[0])
    return parsed[0][1] if parsed else (expiries[0] if expiries else "")


def _max_pain(rows: list[dict]) -> float | None:
    if not rows:
        return None
    return min(
        rows,
        key=lambda base: sum(
            max(0, base["strike"] - row["strike"]) * row["call_oi"]
            + max(0, row["strike"] - base["strike"]) * row["put_oi"]
            for row in rows
        ),
    )["strike"]


def _new_session():
    if curl_requests is not None:
        session = curl_requests.Session(impersonate="chrome124")
    else:
        session = requests.Session()
    session.headers.update(HEADERS)
    return session


def fetch_option_chain() -> dict:
    session = _new_session()
    try:
        session.get(f"{BASE_URL}/option-chain", timeout=10)
        contract_info = session.get(
            f"{BASE_URL}/api/option-chain-contract-info",
            params={"symbol": "NIFTY"},
            timeout=10,
        )
        expiry = _nearest_expiry(contract_info.json().get("expiryDates", []))
        response = session.get(
            f"{BASE_URL}/api/option-chain-v3",
            params={"type": "Indices", "symbol": "NIFTY", "expiry": expiry},
            timeout=10,
        )
        payload = response.json()
    except Exception:
        return _empty()
    records = payload.get("records", {})
    expiry = expiry or _nearest_expiry(records.get("expiryDates", []))
    rows = []
    for item in records.get("data", []):
        row_expiry = item.get("expiryDate") or item.get("expiryDates")
        if expiry and row_expiry != expiry:
            continue
        strike = item.get("strikePrice")
        if strike is None:
            continue
        ce = item.get("CE", {})
        pe = item.get("PE", {})
        rows.append(
            {
                "strike": strike,
                "call_oi": ce.get("openInterest", 0) or 0,
                "put_oi": pe.get("openInterest", 0) or 0,
                "call_change_oi": ce.get("changeinOpenInterest", 0) or 0,
                "put_change_oi": pe.get("changeinOpenInterest", 0) or 0,
                "call_volume": ce.get("totalTradedVolume", 0) or 0,
                "put_volume": pe.get("totalTradedVolume", 0) or 0,
                "call_iv": ce.get("impliedVolatility", 0) or 0,
                "put_iv": pe.get("impliedVolatility", 0) or 0,
            }
        )
    if not rows:
        return _empty(expiry)
    highest_call = max(rows, key=lambda row: row["call_oi"])
    highest_put = max(rows, key=lambda row: row["put_oi"])
    call_total = sum(row["call_oi"] for row in rows)
    put_total = sum(row["put_oi"] for row in rows)
    call_change_total = sum(row["call_change_oi"] for row in rows)
    put_change_total = sum(row["put_change_oi"] for row in rows)
    pcr = round(put_total / call_total, 2) if call_total else 0.0
    pcr_score = max(-1.0, min(1.0, (pcr - 1.0) / 0.2))
    oi_score = 1.0 if highest_put["put_oi"] > highest_call["call_oi"] else -1.0 if highest_put["put_oi"] < highest_call["call_oi"] else 0.0
    score = round((pcr_score + oi_score) / 2, 2)
    bias = "BULLISH" if score > 0.2 else "BEARISH" if score < -0.2 else "NEUTRAL"
    return {
        "expiry": expiry,
        "underlying_value": records.get("underlyingValue"),
        "pcr": pcr,
        "max_pain": _max_pain(rows),
        "highest_call_oi": highest_call["strike"],
        "highest_put_oi": highest_put["strike"],
        "highest_call_oi_value": highest_call["call_oi"],
        "highest_put_oi_value": highest_put["put_oi"],
        "call_change_total": call_change_total,
        "put_change_total": put_change_total,
        "net_change_total": put_change_total - call_change_total,
        "support": highest_put["strike"],
        "resistance": highest_call["strike"],
        "score": score,
        "bias": bias,
        "rows": rows,
    }
