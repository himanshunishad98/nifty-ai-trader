"""backtest_engine.py
Walk-forward historical backtester for NIFTY AI Trader.

Downloads 1-2 years of NIFTY 5m + VIX + global market data,
replays it bar-by-bar computing all available features,
labels each bar (correct direction call after N minutes),
stores everything in feature_snapshots, then triggers
RF + MLP + RL training.

Usage (from project root):
    .venv\\Scripts\\python.exe -m scripts.backtest_engine
or:
    .venv\\Scripts\\python.exe scripts/backtest_engine.py
"""
from __future__ import annotations

import argparse
import math
import sqlite3
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.data_engine import DB_PATH, DATA_DIR, get_connection, init_db
from scripts.indicators import calculate_indicators
from scripts.multi_source_data import fetch_nifty_history, fetch_vix_history
from scripts.predictive_model import FEATURE_COLUMNS, maybe_retrain_model
from scripts.deep_learning_model import maybe_retrain_mlp
from scripts.rl_agent import update_rl_agent

# Config
NIFTY_SYMBOL   = "^NSEI"
VIX_SYMBOL     = "^INDIAVIX"
PERIOD_YEARS   = 10            # multi-source: up to 15 years available
FORWARD_BARS   = 5             # 5 trading days forward window
MIN_RETURN_PCT = 1.5           # min % move over 5 days to label as directional
SIGNAL_WEIGHTS = {            # replicate signal_engine composite weights
    "indicators": 0.35,
    "sentiment":  0.10,       # zeroed out in backtest (no historical news)
    "options":    0.25,       # zeroed out in backtest (no historical OI)
    "patterns":   0.15,
    "volatility": 0.15,
}

GLOBAL_SYMBOLS = {
    "sp500_change":  "^GSPC",
    "nikkei_change": "^N225",
    "crude_change":  "CL=F",
    "gold_change":   "GC=F",
    "usdinr_change": "USDINR=X",
    "dxy_change":    "DX=F",
    "us10y_yield":   "^TNX",
}


# Download helpers

def _download_nifty_daily(years: float = 5.0) -> pd.DataFrame:
    """Download NIFTY daily candles — available for 5+ years."""
    try:
        period = f"{min(int(years), 5)}y"
        df = yf.download(NIFTY_SYMBOL, period=period, interval="1d",
                         auto_adjust=True, progress=False)
        if df.empty:
            return pd.DataFrame()
        df.columns = [c[0] if isinstance(c, tuple) else c for c in df.columns]
        df.index = pd.to_datetime(df.index)
        if df.index.tz is not None:
            df.index = df.index.tz_localize(None)
        print(f"   NIFTY daily: {len(df):,} bars ({df.index[0].date()} to {df.index[-1].date()})")
        return df
    except Exception as exc:
        print(f"  [WARN] NIFTY daily download: {exc}")
        return pd.DataFrame()


def _download_nifty_hourly() -> pd.DataFrame:
    """Best-effort: try 60-day 1h candles to supplement daily with intraday."""
    try:
        df = yf.download(NIFTY_SYMBOL, period="59d", interval="1h",
                         auto_adjust=True, progress=False)
        if df.empty:
            return pd.DataFrame()
        df.columns = [c[0] if isinstance(c, tuple) else c for c in df.columns]
        df.index = pd.to_datetime(df.index)
        if df.index.tz is not None:
            df.index = df.index.tz_localize(None)
        return df
    except Exception:
        return pd.DataFrame()


def _download_daily(symbol: str, years: float = 5.0) -> pd.DataFrame:
    try:
        end   = datetime.today()
        start = end - timedelta(days=int(years * 365) + 30)
        df = yf.download(symbol, start=start.strftime("%Y-%m-%d"),
                         end=end.strftime("%Y-%m-%d"),
                         interval="1d", auto_adjust=True, progress=False)
        if df.empty:
            return pd.DataFrame()
        df.columns = [c[0] if isinstance(c, tuple) else c for c in df.columns]
        df.index = pd.to_datetime(df.index)
        if df.index.tz is not None:
            df.index = df.index.tz_localize(None)
        return df
    except Exception as exc:
        print(f"  [WARN] {symbol}: {exc}")
        return pd.DataFrame()


#  Feature computation 

def _compute_indicators(window: pd.DataFrame) -> dict:
    """Compute technical indicators on a rolling window of daily bars."""
    if len(window) < 20:
        return {}
    try:
        ind = calculate_indicators(window)
        latest = ind.iloc[-1]
        rsi  = float(latest.get("rsi",  50) or 50)
        macd = float(latest.get("macd",  0) or 0)
        ema9 = float(latest.get("ema9",  0) or 0)
        ema21= float(latest.get("ema21", 0) or 0)
        return {
            "rsi":       round(rsi, 2),
            "macd":      round(macd, 2),
            "ema_trend": 1.0 if ema9 > ema21 else 0.0,
        }
    except Exception:
        return {}


def _vix_at(vix_daily: pd.DataFrame, ts: pd.Timestamp) -> float:
    """Nearest daily VIX value to the given timestamp."""
    if vix_daily.empty:
        return 15.0
    date = ts.date()
    try:
        row = vix_daily[vix_daily.index.date <= date]
        return float(row["Close"].iloc[-1]) if not row.empty else 15.0
    except Exception:
        return 15.0


def _global_at(global_daily: dict[str, pd.DataFrame], ts: pd.Timestamp) -> dict:
    """Daily % change for each global instrument at timestamp date."""
    result = {k: 0.0 for k in GLOBAL_SYMBOLS}
    date = ts.date()
    for feat, df in global_daily.items():
        if df.empty:
            continue
        try:
            row = df[df.index.date <= date]
            if len(row) >= 2:
                chg = float((row["Close"].iloc[-1] - row["Close"].iloc[-2]) /
                            row["Close"].iloc[-2] * 100)
                result[feat] = round(chg, 2)
        except Exception:
            pass
    return result


def _indicator_score(rsi: float, macd: float, ema_trend: float) -> float:
    """Quick directional score from indicators, range -1 to +1."""
    score = 0.0
    score += (rsi - 50) / 50 * 0.5
    score += np.sign(macd) * 0.3
    score += (ema_trend - 0.5) * 2 * 0.2
    return round(max(-1.0, min(1.0, score)), 3)


def _pattern_score(window: pd.DataFrame) -> float:
    """Lightweight pattern score: bullish/bearish engulfing based on last 2 bars."""
    if len(window) < 2:
        return 0.0
    prev = window.iloc[-2]
    curr = window.iloc[-1]
    try:
        prev_body = float(curr["Close"]) - float(curr["Open"])
        engulf    = (float(prev["Open"]) - float(prev["Close"])) * prev_body
        return 0.3 if engulf > 0 else -0.3 if engulf < 0 else 0.0
    except Exception:
        return 0.0


def _composite_signal(ind_score: float) -> str:
    if ind_score > 0.15:
        return "BUY CALL"
    if ind_score < -0.15:
        return "BUY PUT"
    return "NO TRADE"


#  DB helpers 

def _clear_backtest_rows(conn: sqlite3.Connection) -> None:
    """Remove previously inserted backtest rows so we can re-run cleanly."""
    conn.execute("DELETE FROM feature_snapshots WHERE timestamp LIKE 'BT_%'")
    conn.execute("DELETE FROM predictions      WHERE timestamp LIKE 'BT_%'")
    conn.commit()


def _insert_feature_row(conn: sqlite3.Connection, ts_label: str, price: float,
                        row: dict, signal: str, confidence: float,
                        composite: float, target: int,
                        actual_after: float) -> None:
    values = [ts_label, price]
    values.extend(row.get(c, 0.0) for c in FEATURE_COLUMNS)
    values.extend([composite, signal, confidence, actual_after, target])
    try:
        conn.execute("""
            INSERT OR REPLACE INTO feature_snapshots (
                timestamp, nifty_price, rsi, macd, ema_trend, pcr, oi_pressure,
                gamma_flip_distance, total_gex, india_vix, sentiment_score, pattern_signal,
                sp500_change, nikkei_change, crude_change, gold_change,
                usdinr_change, dxy_change, us10y_yield,
                composite_score, signal, confidence, actual_price_after, target
            ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
        """, values)
    except Exception as exc:
        pass  # skip malformed rows silently


def _label_target(current_price: float, future_price: float,
                  signal: str, threshold: float = MIN_RETURN_PCT) -> int:
    """1 = prediction correct, 0 = incorrect."""
    if signal == "NO TRADE":
        return 1   # NO TRADE is never "wrong"
    pct_change = (future_price - current_price) / current_price * 100
    if signal == "BUY CALL":
        return 1 if pct_change >= threshold else 0
    if signal == "BUY PUT":
        return 1 if pct_change <= -threshold else 0
    return 0


#  Main Backtest Loop 

def run_backtest(years: float = 10.0, forward_bars: int = FORWARD_BARS,
                 verbose: bool = True) -> dict:
    years = min(years, 15.0)

    print(f"\n{'='*60}")
    print(f"  NIFTY AI Trader -- Backtest Engine (Multi-Source)")
    print(f"  Period: {years:.0f} years | Forward: {forward_bars} trading days")
    print(f"{'='*60}\n")

    # 1. Download NIFTY via multi-source cascade
    print("[1/5] Fetching NIFTY history (cascade: nsepy > stooq > yfinance > jugaad) ...")
    nifty = fetch_nifty_history(years=years, use_cache=True)
    if nifty.empty:
        print("  [ERROR] All sources failed. Check internet connection.")
        return {"success": False}

    print("[2/5] Fetching India VIX history ...")
    vix_df = fetch_vix_history(years=years)
    # Build a simple date->vix lookup
    vix_lookup: dict = {}
    if not vix_df.empty:
        for ts, row in vix_df.iterrows():
            vix_lookup[ts.date()] = float(row.get("vix", 15.0))

    print("[3/5] Downloading global market data (daily) ...")
    global_daily: dict[str, pd.DataFrame] = {}
    for feat, sym in GLOBAL_SYMBOLS.items():
        global_daily[feat] = _download_daily(sym, years)
    ok = len([k for k, v in global_daily.items() if not v.empty])
    print(f"   {ok}/{len(GLOBAL_SYMBOLS)} global feeds OK")

    # 2. Setup DB
    print("\n  Setting up database ...")
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    conn = get_connection()
    _clear_backtest_rows(conn)

    # 4. Walk-Forward Simulation
    prices = nifty["Close"].values.astype(float)
    n      = len(prices)
    window_size = 30    # 30 daily bars = ~6 weeks of context for indicators

    inserted = 0
    skipped  = 0
    bullish_count = 0
    bearish_count = 0
    no_trade_count = 0

    print(f"\n[4/5] Walk-forward on {n:,} daily bars ({forward_bars}-day outcome window) ...")

    for i in range(window_size, n - forward_bars):
        ts  = nifty.index[i]
        price = prices[i]
        future_price = prices[i + forward_bars]

        # Skip zero / nan
        if price <= 0 or math.isnan(price) or math.isnan(future_price):
            skipped += 1
            continue

        # Rolling window of OHLCV
        window = nifty.iloc[i - window_size: i + 1].copy()

        # Compute features
        ind = _compute_indicators(window)
        if not ind:
            skipped += 1
            continue

        rsi      = ind["rsi"]
        macd     = ind["macd"]
        ema_trend= ind["ema_trend"]
        vix_val  = vix_lookup.get(ts.date(), 15.0)
        gf       = _global_at(global_daily, ts)
        pat_score= _pattern_score(window)
        ind_score= _indicator_score(rsi, macd, ema_trend)
        composite= round(
            ind_score * SIGNAL_WEIGHTS["indicators"] +
            pat_score * SIGNAL_WEIGHTS["patterns"] +
            (15.0 - vix_val) / 30.0 * SIGNAL_WEIGHTS["volatility"],
            3
        )
        signal    = _composite_signal(composite)
        confidence= min(abs(composite) * 100, 95.0)
        target    = _label_target(price, future_price, signal)

        feature_row = {
            "rsi":                 rsi,
            "macd":                macd,
            "ema_trend":           ema_trend,
            "pcr":                 1.0,        # neutral default (no historical OI)
            "oi_pressure":         0.0,
            "gamma_flip_distance": 0.0,
            "total_gex":           0.0,
            "india_vix":           vix_val,
            "sentiment_score":     0.0,        # no historical news
            "pattern_signal":      pat_score,
            **gf,
        }

        ts_label = f"BT_{ts.strftime('%Y%m%d_%H%M')}"
        _insert_feature_row(conn, ts_label, price, feature_row,
                            signal, confidence, composite,
                            target, float(future_price))

        inserted += 1
        if signal == "BUY CALL":  bullish_count += 1
        elif signal == "BUY PUT": bearish_count += 1
        else:                     no_trade_count += 1

        if verbose and inserted % 500 == 0:
            print(f"   {inserted:,} bars processed ")

    conn.commit()
    print(f"\nSimulation complete:")
    print(f"   Inserted  : {inserted:,} feature rows")
    print(f"   Skipped   : {skipped:,} bars")
    print(f"   BUY CALL  : {bullish_count:,}  |  BUY PUT: {bearish_count:,}  |  NO TRADE: {no_trade_count:,}")

    # 4. Compute accuracy
    df = pd.read_sql_query(
        "SELECT signal, target FROM feature_snapshots WHERE timestamp LIKE 'BT_%'", conn)
    directional = df[df["signal"] != "NO TRADE"]
    accuracy = round(directional["target"].mean() * 100, 1) if len(directional) else 0.0
    print(f"   Historical accuracy (directional calls): {accuracy}%")

    # 5. Train Models
    print("\n[5/5] Training all AI models on historical data ...")
    print("   Training Random Forest ...")
    rf_status = maybe_retrain_model(conn)
    print(f"   RF: {rf_status}")

    print("   Training Deep Learning MLP ...")
    dl_status = maybe_retrain_mlp(conn, min_rows=40)
    print(f"   MLP: {dl_status}")

    print("   Running RL agent update ...")
    rl_status = update_rl_agent(conn)
    print(f"   RL: {rl_status}")

    conn.close()

    summary = {
        "success":       True,
        "bars_processed": inserted,
        "accuracy":      accuracy,
        "rf":            rf_status,
        "dl":            dl_status,
        "rl":            rl_status,
    }

    print(f"\n{'='*60}")
    print("   Backtest & Training Complete!")
    print(f"  Accuracy: {accuracy}% | RF trained: {rf_status.get('trained')} "
          f"| MLP trained: {dl_status.get('dl_trained')}")
    print(f"{'='*60}\n")
    return summary


# Entry Point

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="NIFTY AI Trader -- Backtest Engine (Multi-Source)")
    parser.add_argument("--years", type=float, default=10.0,
                        help="Years of daily history to replay (default: 10, max: 15)")
    parser.add_argument("--forward", type=int, default=FORWARD_BARS,
                        help=f"Trading days ahead to measure outcome (default: {FORWARD_BARS})")
    parser.add_argument("--no-cache", action="store_true",
                        help="Force re-download even if cache exists")
    parser.add_argument("--quiet", action="store_true", help="Suppress progress output")
    args = parser.parse_args()

    result = run_backtest(
        years=min(args.years, 15.0),
        forward_bars=args.forward,
        verbose=not args.quiet,
    )
    sys.exit(0 if result["success"] else 1)

