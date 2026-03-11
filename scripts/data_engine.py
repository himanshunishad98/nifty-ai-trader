from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from scripts.market_data import fetch_india_vix_data, fetch_market_data
from scripts.news_sentiment import fetch_news_sentiment
from scripts.option_chain import fetch_option_chain

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
DB_PATH = DATA_DIR / "nifty_ai_trader.db"


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _frame_to_json(df: pd.DataFrame, limit: int) -> str:
    return df.tail(limit).to_json(orient="records", date_format="iso")


def _frame_from_json(payload: str) -> pd.DataFrame:
    if not payload:
        return pd.DataFrame()
    frame = pd.DataFrame(json.loads(payload))
    if "Datetime" in frame.columns:
        frame["Datetime"] = pd.to_datetime(frame["Datetime"])
    return frame


def get_connection() -> sqlite3.Connection:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    init_db(conn)
    return conn


def init_db(conn: sqlite3.Connection) -> None:
    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS snapshots (
            timestamp TEXT PRIMARY KEY,
            nifty_price REAL,
            vix REAL,
            market_data_1m TEXT NOT NULL,
            market_data_5m TEXT NOT NULL,
            vix_data TEXT NOT NULL,
            option_chain TEXT NOT NULL,
            news TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS feature_snapshots (
            timestamp TEXT PRIMARY KEY,
            nifty_price REAL,
            rsi REAL,
            macd REAL,
            ema_trend REAL,
            pcr REAL,
            oi_pressure REAL,
            gamma_flip_distance REAL,
            total_gex REAL,
            india_vix REAL,
            sentiment_score REAL,
            pattern_signal REAL,
            sp500_change REAL,
            nikkei_change REAL,
            crude_change REAL,
            gold_change REAL,
            usdinr_change REAL,
            dxy_change REAL,
            us10y_yield REAL,
            composite_score REAL,
            signal TEXT,
            confidence REAL,
            actual_price_after REAL,
            target INTEGER
        );
        CREATE TABLE IF NOT EXISTS predictions (
            timestamp TEXT PRIMARY KEY,
            signal TEXT,
            confidence REAL,
            strike TEXT,
            expiry TEXT,
            price_at_signal REAL,
            prediction_window INTEGER,
            bullish_probability REAL,
            bearish_probability REAL,
            actual_price_after REAL,
            correct_prediction INTEGER
        );
        """
    )
    conn.commit()
    # Migrate existing databases: add global market columns if they don't exist yet
    global_cols = ["sp500_change", "nikkei_change", "crude_change", "gold_change",
                   "usdinr_change", "dxy_change", "us10y_yield"]
    existing = {row[1] for row in conn.execute("PRAGMA table_info(feature_snapshots)").fetchall()}
    for col in global_cols:
        if col not in existing:
            conn.execute(f"ALTER TABLE feature_snapshots ADD COLUMN {col} REAL DEFAULT 0.0")
    conn.commit()


import asyncio

async def _build_snapshot_async() -> dict:
    market_data, vix_data, option_chain, news = await asyncio.gather(
        asyncio.to_thread(fetch_market_data),
        asyncio.to_thread(fetch_india_vix_data),
        asyncio.to_thread(fetch_option_chain),
        asyncio.to_thread(fetch_news_sentiment),
    )
    nifty_price = round(float(market_data["5m"]["Close"].iloc[-1]), 2) if not market_data["5m"].empty else None
    vix_value = round(float(vix_data["Close"].iloc[-1]), 2) if not vix_data.empty else None
    return {
        "timestamp": _now().isoformat(),
        "nifty_price": nifty_price,
        "vix": vix_value,
        "market_data": market_data,
        "vix_data": vix_data,
        "option_chain": option_chain,
        "news": news,
    }


def build_snapshot() -> dict:
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None

    if loop and loop.is_running():
        # Fallback if already in an event loop (e.g., Streamlit sometimes)
        import concurrent.futures
        with concurrent.futures.ThreadPoolExecutor() as pool:
            market_data = pool.submit(fetch_market_data).result()
            vix_data = pool.submit(fetch_india_vix_data).result()
            option_chain = pool.submit(fetch_option_chain).result()
            news = pool.submit(fetch_news_sentiment).result()
            
            nifty_price = round(float(market_data["5m"]["Close"].iloc[-1]), 2) if not market_data["5m"].empty else None
            vix_value = round(float(vix_data["Close"].iloc[-1]), 2) if not vix_data.empty else None
            return {
                "timestamp": _now().isoformat(),
                "nifty_price": nifty_price,
                "vix": vix_value,
                "market_data": market_data,
                "vix_data": vix_data,
                "option_chain": option_chain,
                "news": news,
            }
    else:
        return asyncio.run(_build_snapshot_async())


def store_snapshot(conn: sqlite3.Connection, snapshot: dict) -> None:
    conn.execute(
        """
        INSERT OR REPLACE INTO snapshots (
            timestamp, nifty_price, vix, market_data_1m, market_data_5m, vix_data, option_chain, news
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            snapshot["timestamp"],
            snapshot["nifty_price"],
            snapshot["vix"],
            _frame_to_json(snapshot["market_data"]["1m"], 240),
            _frame_to_json(snapshot["market_data"]["5m"], 200),
            _frame_to_json(snapshot["vix_data"], 90),
            json.dumps(snapshot["option_chain"]),
            json.dumps(snapshot["news"]),
        ),
    )
    conn.commit()


def _row_to_snapshot(row: sqlite3.Row | None) -> dict | None:
    if row is None:
        return None
    return {
        "timestamp": row["timestamp"],
        "nifty_price": row["nifty_price"],
        "vix": row["vix"],
        "market_data": {
            "1m": _frame_from_json(row["market_data_1m"]),
            "5m": _frame_from_json(row["market_data_5m"]),
        },
        "vix_data": _frame_from_json(row["vix_data"]),
        "option_chain": json.loads(row["option_chain"]),
        "news": json.loads(row["news"]),
    }


def get_latest_snapshot(conn: sqlite3.Connection) -> dict | None:
    row = conn.execute("SELECT * FROM snapshots ORDER BY timestamp DESC LIMIT 1").fetchone()
    return _row_to_snapshot(row)


def snapshot_age_seconds(timestamp: str | None) -> float | None:
    if not timestamp:
        return None
    return (_now() - datetime.fromisoformat(timestamp)).total_seconds()


def freshness_label(age_seconds: float | None) -> str:
    if age_seconds is None:
        return "STALE"
    if age_seconds < 60:
        return "LIVE"
    if age_seconds < 300:
        return "DELAYED"
    return "STALE"


def refresh_snapshot(conn: sqlite3.Connection, max_age_seconds: int = 60) -> dict:
    snapshot = get_latest_snapshot(conn)
    age = snapshot_age_seconds(snapshot["timestamp"]) if snapshot else None
    if snapshot is None or age is None or age >= max_age_seconds:
        snapshot = build_snapshot()
        store_snapshot(conn, snapshot)
        age = 0.0
    snapshot["freshness"] = freshness_label(age)
    snapshot["age_seconds"] = age
    return snapshot
