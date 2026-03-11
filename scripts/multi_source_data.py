"""multi_source_data.py
Cascading multi-source historical data fetcher for NIFTY 50.

Tries sources in order of reliability and data depth:
  1. NSE India direct API (unofficial) -- up to 15+ years
  2. nsepy library -- up to 15 years, chunked
  3. Stooq via pandas_datareader -- 10+ years
  4. yfinance -- 5 years (most reliable fallback)
  5. jugaad-data -- NSE library, chunked

All sources normalise to the same OHLCV DataFrame with DatetimeIndex.
"""
from __future__ import annotations

import io
import time
from datetime import date, datetime, timedelta
from pathlib import Path

import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[1]
CACHE_DIR = ROOT / "data" / "historical_cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

NIFTY_NSE_URL  = "https://www.niftyindices.com/Backpage.aspx/getHistoricalData"
NIFTY_NSE_HEADERS = {
    "Content-Type": "application/json; charset=utf-8",
    "User-Agent": "Mozilla/5.0",
    "Referer": "https://www.niftyindices.com/",
}

STOOQ_NIFTY = "^NF"          # Stooq symbol for NIFTY 50
NSEI_VIX    = "^INDIAVIX"


def _standardize(df: pd.DataFrame) -> pd.DataFrame:
    """Rename columns to Open/High/Low/Close/Volume, ensure DatetimeIndex."""
    rename = {}
    for col in df.columns:
        lc = col.lower().strip()
        if lc in ("open", "open price"):         rename[col] = "Open"
        elif lc in ("high", "high price"):       rename[col] = "High"
        elif lc in ("low",  "low price"):        rename[col] = "Low"
        elif lc in ("close", "close price", "closing price"): rename[col] = "Close"
        elif lc in ("volume", "vol"):            rename[col] = "Volume"
    df = df.rename(columns=rename)
    if "Close" not in df.columns:
        return pd.DataFrame()
    for col in ["Open", "High", "Low", "Close"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col].astype(str).str.replace(",", ""), errors="coerce")
    if "Volume" not in df.columns:
        df["Volume"] = 0
    df = df[["Open", "High", "Low", "Close", "Volume"]].dropna(subset=["Close"])
    df.index = pd.to_datetime(df.index).tz_localize(None)
    df = df.sort_index()
    return df


# ─── Source 1: NSE Indices website ─────────────────────────────────────────

def _fetch_nseindices(start: date, end: date) -> pd.DataFrame:
    """
    POST to niftyindices.com historical data endpoint.
    Returns daily OHLCV. No auth required.
    """
    url = "https://niftyindices.com/IndexConstituent/GetIndexData"
    payload = {
        "indexName": "NIFTY 50",
        "startDate": start.strftime("%d-%m-%Y"),
        "endDate":   end.strftime("%d-%m-%Y"),
    }
    try:
        r = requests.post(url, json=payload, headers=NIFTY_NSE_HEADERS, timeout=20)
        r.raise_for_status()
        data = r.json()
        if isinstance(data, dict):
            data = data.get("d", data.get("data", []))
        if not data:
            return pd.DataFrame()
        df = pd.DataFrame(data)
        # Column names vary; try to detect date column
        date_col = next((c for c in df.columns if "date" in c.lower()), None)
        if date_col:
            df.index = pd.to_datetime(df[date_col], dayfirst=True, errors="coerce")
            df = df.drop(columns=[date_col])
        return _standardize(df)
    except Exception as exc:
        print(f"  [nseindices] {exc}")
        return pd.DataFrame()


# ─── Source 2: NSE India historical index data (direct CSV) ────────────────

def _fetch_nse_csv_chunk(start: date, end: date) -> pd.DataFrame:
    """
    Download NIFTY 50 historical data from NSE India's public CSV endpoint.
    Works for any date range, data goes back to 1990s.
    """
    url = (
        f"https://www.nseindia.com/api/index-names?index=NIFTY%2050"
        f"&from={start.strftime('%d-%m-%Y')}&to={end.strftime('%d-%m-%Y')}"
    )
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
        "Referer":    "https://www.nseindia.com/",
        "Accept":     "application/json",
    }
    try:
        session = requests.Session()
        # Hit index page first to get cookies
        session.get("https://www.nseindia.com", headers=headers, timeout=15)
        time.sleep(0.5)
        csv_url = (
            f"https://www.nseindia.com/api/historical/indicesHistory?"
            f"indexType=NIFTY%2050&from={start.strftime('%d-%m-%Y')}"
            f"&to={end.strftime('%d-%m-%Y')}"
        )
        r = session.get(csv_url, headers=headers, timeout=25)
        r.raise_for_status()
        jdata = r.json()
        rows = jdata.get("data", {}).get("indexCloseOnlineRecords", [])
        if not rows:
            return pd.DataFrame()
        df = pd.DataFrame(rows)
        df.index = pd.to_datetime(df.get("EOD_TIMESTAMP", df.get("PrevClose", "")),
                                   dayfirst=True, errors="coerce")
        return _standardize(df)
    except Exception as exc:
        print(f"  [nse_api] {exc}")
        return pd.DataFrame()


# ─── Source 3: nsepy library ────────────────────────────────────────────────

def _fetch_nsepy(start: date, end: date) -> pd.DataFrame:
    """
    Use nsepy to download NIFTY 50 index data in 3-year chunks.
    nsepy can go back to 2000.
    """
    try:
        from nsepy import get_history
        chunks = []
        chunk_start = start
        while chunk_start < end:
            chunk_end = min(chunk_start + timedelta(days=365 * 3), end)
            try:
                df = get_history(symbol="NIFTY 50", start=chunk_start, end=chunk_end, index=True)
                if not df.empty:
                    chunks.append(df)
            except Exception:
                pass
            chunk_start = chunk_end + timedelta(days=1)
            time.sleep(0.5)  # be polite to NSE servers
        if not chunks:
            return pd.DataFrame()
        merged = pd.concat(chunks)
        return _standardize(merged)
    except ImportError:
        return pd.DataFrame()
    except Exception as exc:
        print(f"  [nsepy] {exc}")
        return pd.DataFrame()


# ─── Source 4: Stooq via pandas_datareader ──────────────────────────────────

def _fetch_stooq(start: date, end: date) -> pd.DataFrame:
    """
    Download NIFTY 50 data from Stooq.
    Stooq provides 10+ years of daily data for free.
    """
    try:
        from pandas_datareader import data as pdr
        df = pdr.DataReader(STOOQ_NIFTY, "stooq", start=start, end=end)
        if df.empty:
            return pd.DataFrame()
        return _standardize(df)
    except Exception as exc:
        print(f"  [stooq] {exc}")
        return pd.DataFrame()


# ─── Source 5: yfinance (existing, reliable fallback) ──────────────────────

def _fetch_yfinance(start: date, end: date) -> pd.DataFrame:
    """yfinance fallback — 5 years max, most reliable."""
    try:
        import yfinance as yf
        df = yf.download("^NSEI", start=start.strftime("%Y-%m-%d"),
                          end=end.strftime("%Y-%m-%d"),
                          interval="1d", auto_adjust=True, progress=False)
        if df.empty:
            return pd.DataFrame()
        df.columns = [c[0] if isinstance(c, tuple) else c for c in df.columns]
        return _standardize(df)
    except Exception as exc:
        print(f"  [yfinance] {exc}")
        return pd.DataFrame()


# ─── Source 6: jugaad-data ──────────────────────────────────────────────────

def _fetch_jugaad(start: date, end: date) -> pd.DataFrame:
    """
    jugaad-data: NSE-specific Python library, fetches index data in chunks.
    Works for 5+ years.
    """
    try:
        from jugaad_data.nse import index_raw
        chunks = []
        chunk_start = start
        while chunk_start < end:
            chunk_end = min(chunk_start + timedelta(days=365), end)
            try:
                data = index_raw("NIFTY 50", chunk_start, chunk_end)
                if data:
                    df = pd.DataFrame(data)
                    chunks.append(df)
            except Exception:
                pass
            chunk_start = chunk_end + timedelta(days=1)
            time.sleep(0.3)
        if not chunks:
            return pd.DataFrame()
        merged = pd.concat(chunks)
        if "HistoricalDate" in merged.columns:
            merged.index = pd.to_datetime(merged["HistoricalDate"], dayfirst=True)
            merged = merged.drop(columns=["HistoricalDate"], errors="ignore")
        return _standardize(merged)
    except ImportError:
        return pd.DataFrame()
    except Exception as exc:
        print(f"  [jugaad] {exc}")
        return pd.DataFrame()


# ─── Master Fetcher ─────────────────────────────────────────────────────────

SOURCES = [
    ("nsepy",    _fetch_nsepy),
    ("stooq",    _fetch_stooq),
    ("yfinance", _fetch_yfinance),
    ("jugaad",   _fetch_jugaad),
    ("nseindices", _fetch_nseindices),
]


def fetch_nifty_history(years: float = 10.0,
                         end_date: date | None = None,
                         use_cache: bool = True) -> pd.DataFrame:
    """
    Fetch NIFTY 50 daily OHLCV data from multiple sources in cascade.
    Returns the longest clean dataset found.

    Args:
        years:     How many years of history to fetch (default 10)
        end_date:  End date (default today)
        use_cache: Cache result to disk to avoid re-downloading

    Returns:
        pd.DataFrame with columns [Open, High, Low, Close, Volume]
    """
    if end_date is None:
        end_date = date.today()
    start_date = end_date - timedelta(days=int(years * 365))

    cache_file = CACHE_DIR / f"nifty_{start_date}_{end_date}.parquet"
    if use_cache and cache_file.exists():
        try:
            df = pd.read_parquet(cache_file)
            print(f"  [cache] Loaded {len(df):,} rows from disk cache")
            return df
        except Exception:
            pass

    best_df = pd.DataFrame()

    for name, fetcher in SOURCES:
        print(f"  Trying {name} ...", end=" ", flush=True)
        try:
            df = fetcher(start_date, end_date)
            if df.empty:
                print("empty")
                continue
            print(f"{len(df):,} bars "
                  f"({df.index[0].date()} to {df.index[-1].date()})")
            if len(df) > len(best_df):
                best_df = df
            if len(best_df) >= years * 220:   # ~220 trading days per year — enough
                break
        except Exception as exc:
            print(f"error: {exc}")

    if best_df.empty:
        print("  [WARN] All sources failed. No historical data available.")
        return pd.DataFrame()

    # Deduplicate and sort
    best_df = best_df[~best_df.index.duplicated(keep="first")].sort_index()

    if use_cache:
        try:
            best_df.to_parquet(cache_file)
        except Exception:
            pass  # parquet optional

    print(f"\n  Total: {len(best_df):,} trading days "
          f"({best_df.index[0].date()} to {best_df.index[-1].date()})")
    return best_df


# ─── VIX fetcher ────────────────────────────────────────────────────────────

def fetch_vix_history(years: float = 10.0) -> pd.DataFrame:
    """Fetch India VIX daily data (yfinance is the most reliable source)."""
    try:
        import yfinance as yf
        end_d = date.today()
        start_d = end_d - timedelta(days=int(years * 365) + 30)
        df = yf.download(NSEI_VIX,
                          start=start_d.strftime("%Y-%m-%d"),
                          end=end_d.strftime("%Y-%m-%d"),
                          interval="1d", auto_adjust=True, progress=False)
        if df.empty:
            return pd.DataFrame()
        df.columns = [c[0] if isinstance(c, tuple) else c for c in df.columns]
        df.index = pd.to_datetime(df.index).tz_localize(None)
        return df[["Close"]].rename(columns={"Close": "vix"})
    except Exception as exc:
        print(f"  [VIX] {exc}")
        return pd.DataFrame()


# ─── Standalone test ─────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("Testing multi-source data fetcher ...")
    df = fetch_nifty_history(years=10.0)
    if not df.empty:
        print(f"\nFinal dataset: {len(df):,} bars")
        print(df.tail(5))
    else:
        print("No data fetched.")
