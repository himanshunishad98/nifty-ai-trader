from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go


def _swing_points(df: pd.DataFrame, column: str, window: int, mode: str) -> pd.DataFrame:
    if len(df) < window * 2 + 1:
        return pd.DataFrame(columns=["Datetime", column])
    values = []
    for idx in range(window, len(df) - window):
        span = df.iloc[idx - window : idx + window + 1]
        value = df.iloc[idx][column]
        target = span[column].max() if mode == "high" else span[column].min()
        if value == target:
            values.append(df.iloc[idx][["Datetime", column]].to_dict())
    return pd.DataFrame(values)


def _trend_direction(df: pd.DataFrame) -> str:
    if len(df) < 10:
        return "NEUTRAL"
    recent = df.tail(10)
    slope = recent["Close"].iloc[-1] - recent["Close"].iloc[0]
    midpoint = recent["Close"].mean()
    close = recent["Close"].iloc[-1]
    if slope > 0 and close >= midpoint:
        return "BULLISH"
    if slope < 0 and close <= midpoint:
        return "BEARISH"
    return "NEUTRAL"


def analyze_market_structure(df: pd.DataFrame, window: int = 3) -> dict:
    if df.empty:
        return {
            "trend": "NEUTRAL",
            "support": None,
            "resistance": None,
            "swing_highs": pd.DataFrame(columns=["Datetime", "High"]),
            "swing_lows": pd.DataFrame(columns=["Datetime", "Low"]),
        }
    swing_highs = _swing_points(df, "High", window, "high")
    swing_lows = _swing_points(df, "Low", window, "low")
    recent = df.tail(20)
    support = swing_lows["Low"].tail(3).min() if not swing_lows.empty else recent["Low"].min()
    resistance = swing_highs["High"].tail(3).max() if not swing_highs.empty else recent["High"].max()
    return {
        "trend": _trend_direction(df),
        "support": round(float(support), 2) if pd.notna(support) else None,
        "resistance": round(float(resistance), 2) if pd.notna(resistance) else None,
        "swing_highs": swing_highs.tail(8),
        "swing_lows": swing_lows.tail(8),
    }


def apply_market_structure(fig: go.Figure, structure: dict) -> go.Figure:
    swing_highs = structure["swing_highs"]
    swing_lows = structure["swing_lows"]
    if not swing_highs.empty:
        fig.add_trace(
            go.Scatter(
                x=swing_highs["Datetime"],
                y=swing_highs["High"],
                mode="markers",
                marker=dict(color="#ff4d4f", size=8, symbol="triangle-down"),
                name="Swing Highs",
            )
        )
    if not swing_lows.empty:
        fig.add_trace(
            go.Scatter(
                x=swing_lows["Datetime"],
                y=swing_lows["Low"],
                mode="markers",
                marker=dict(color="#22c55e", size=8, symbol="triangle-up"),
                name="Swing Lows",
            )
        )
    if structure.get("support") is not None:
        fig.add_hline(y=structure["support"], line_color="#22c55e", line_dash="dot")
    if structure.get("resistance") is not None:
        fig.add_hline(y=structure["resistance"], line_color="#ff4d4f", line_dash="dot")
    return fig
