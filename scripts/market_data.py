from __future__ import annotations

import pandas as pd
import yfinance as yf

from scripts.config import MARKET_DATA_CONFIG

TICKER = MARKET_DATA_CONFIG.get("ticker", "^NSEI")
VIX_TICKER = MARKET_DATA_CONFIG.get("vix_ticker", "^INDIAVIX")
PERIODS = MARKET_DATA_CONFIG.get("periods", {"1m": "5d", "5m": "1mo"})
VIX_PERIODS = MARKET_DATA_CONFIG.get("vix_periods", {"1d": "1mo", "5m": "5d"})


def _format_history(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=["Datetime", "Open", "High", "Low", "Close", "Volume"])
    df = df.reset_index()
    time_col = "Datetime" if "Datetime" in df.columns else df.columns[0]
    df = df.rename(columns={time_col: "Datetime"})
    keep = [col for col in ["Datetime", "Open", "High", "Low", "Close", "Volume"] if col in df.columns]
    out = df[keep].copy()
    if "Volume" not in out:
        out["Volume"] = 1
    out["Volume"] = out["Volume"].fillna(0)
    return out


def fetch_index_data(ticker: str, interval: str, period: str) -> pd.DataFrame:
    try:
        df = yf.Ticker(ticker).history(period=period, interval=interval, auto_adjust=False)
    except Exception:
        df = pd.DataFrame()
    return _format_history(df)


def fetch_nifty_data(interval: str = "5m") -> pd.DataFrame:
    period = PERIODS.get(interval, "1mo")
    return fetch_index_data(TICKER, interval, period)


def fetch_market_data() -> dict[str, pd.DataFrame]:
    return {interval: fetch_nifty_data(interval) for interval in ("1m", "5m")}


def fetch_india_vix_data(interval: str = "1d") -> pd.DataFrame:
    period = VIX_PERIODS.get(interval, "1mo")
    return fetch_index_data(VIX_TICKER, interval, period)
