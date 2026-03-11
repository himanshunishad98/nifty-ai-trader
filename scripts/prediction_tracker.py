from __future__ import annotations

from datetime import datetime, timedelta

import pandas as pd

from scripts.predictive_model import FEATURE_COLUMNS


def store_feature_snapshot(conn, timestamp: str, price: float | None, feature_row: dict, signal: str, confidence: float, composite_score: float) -> None:
    values = [timestamp, price]
    values.extend(feature_row[column] for column in FEATURE_COLUMNS)
    values.extend([composite_score, signal, confidence])
    conn.execute(
        """
        INSERT OR REPLACE INTO feature_snapshots (
            timestamp, nifty_price, rsi, macd, ema_trend, pcr, oi_pressure, gamma_flip_distance,
            total_gex, india_vix, sentiment_score, pattern_signal,
            sp500_change, nikkei_change, crude_change, gold_change, usdinr_change, dxy_change, us10y_yield,
            composite_score, signal, confidence
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        values,
    )
    conn.commit()


def record_prediction(conn, timestamp: str, signal: str, confidence: float, trade: dict, price_at_signal: float | None, probabilities: dict, prediction_window: int = 15) -> None:
    conn.execute(
        """
        INSERT OR REPLACE INTO predictions (
            timestamp, signal, confidence, strike, expiry, price_at_signal, prediction_window,
            bullish_probability, bearish_probability
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            timestamp,
            signal,
            confidence,
            trade.get("strike"),
            trade.get("expiry"),
            price_at_signal,
            prediction_window,
            probabilities["bullish_probability"],
            probabilities["bearish_probability"],
        ),
    )
    conn.commit()


def _target_row(conn, timestamp: str, minutes: int):
    target_time = (datetime.fromisoformat(timestamp) + timedelta(minutes=minutes)).isoformat()
    return conn.execute(
        "SELECT timestamp, nifty_price FROM snapshots WHERE timestamp >= ? ORDER BY timestamp ASC LIMIT 1",
        (target_time,),
    ).fetchone()


def _correct(signal: str, start_price: float, end_price: float) -> int:
    if signal == "BUY CALL":
        return int(end_price > start_price)
    if signal == "BUY PUT":
        return int(end_price < start_price)
    return int(abs(end_price - start_price) / start_price < 0.001)


def resolve_predictions(conn) -> None:
    rows = conn.execute("SELECT * FROM predictions WHERE actual_price_after IS NULL").fetchall()
    for row in rows:
        target = _target_row(conn, row["timestamp"], row["prediction_window"])
        if target is None or row["price_at_signal"] is None:
            continue
        actual = float(target["nifty_price"])
        correct = _correct(row["signal"], float(row["price_at_signal"]), actual)
        conn.execute(
            "UPDATE predictions SET actual_price_after = ?, correct_prediction = ? WHERE timestamp = ?",
            (actual, correct, row["timestamp"]),
        )
        conn.execute(
            "UPDATE feature_snapshots SET actual_price_after = ?, target = ? WHERE timestamp = ?",
            (actual, int(actual > float(row["price_at_signal"])), row["timestamp"]),
        )
    conn.commit()


def performance_summary(conn) -> dict:
    resolved = conn.execute("SELECT COUNT(*) AS total, COALESCE(SUM(correct_prediction), 0) AS correct FROM predictions WHERE correct_prediction IS NOT NULL").fetchone()
    pending = conn.execute("SELECT COUNT(*) AS total FROM predictions WHERE correct_prediction IS NULL").fetchone()
    accuracy = round((resolved["correct"] / resolved["total"]) * 100, 1) if resolved["total"] else 0.0
    return {"accuracy": accuracy, "resolved": resolved["total"], "correct": resolved["correct"], "pending": pending["total"]}


def timeline(conn, limit: int = 40) -> pd.DataFrame:
    frame = pd.read_sql_query(
        "SELECT timestamp, signal, confidence, correct_prediction FROM predictions ORDER BY timestamp DESC LIMIT ?",
        conn,
        params=(limit,),
    )
    if frame.empty:
        return frame
    frame["timestamp"] = pd.to_datetime(frame["timestamp"])
    return frame.sort_values("timestamp")
