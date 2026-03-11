from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go

DEFAULT_THEME = {
    "panel": "#F2EFE7",
    "grid": "#C8C2B0",
    "text": "#1A1A14",
    "muted": "#6B6554",
    "green": "#2D6A4F",
    "red": "#8B1A1A",
    "yellow": "#7A5C00",
}


def _window(rows: list[dict], spot: float | None, depth: int = 9) -> list[dict]:
    if not rows:
        return []
    ordered = sorted(rows, key=lambda row: row["strike"])
    if spot is None:
        center = len(ordered) // 2
    else:
        center = min(range(len(ordered)), key=lambda idx: abs(ordered[idx]["strike"] - spot))
    start = max(0, center - depth)
    end = min(len(ordered), center + depth + 1)
    return ordered[start:end]


def _scale(values, signed: bool = False) -> list[float]:
    if not values:
        return []
    array = np.array(values, dtype=float)
    if signed:
        peak = np.max(np.abs(array)) or 1.0
        return (array / peak).tolist()
    peak = np.max(array) or 1.0
    return (array / peak).tolist()


def heatmap_frame(rows: list[dict], spot: float | None = None) -> pd.DataFrame:
    subset = _window(rows, spot)
    if not subset:
        return pd.DataFrame(columns=["Strike", "Call OI", "Put OI", "Change OI"])
    call_oi = _scale([row["call_oi"] for row in subset])
    put_oi = _scale([row["put_oi"] for row in subset])
    net_change = _scale([row["put_change_oi"] - row["call_change_oi"] for row in subset], signed=True)
    frame = pd.DataFrame(
        {
            "Strike": [row["strike"] for row in subset],
            "Call OI": [-value for value in call_oi],
            "Put OI": put_oi,
            "Change OI": net_change,
            "Raw Call OI": [row["call_oi"] for row in subset],
            "Raw Put OI": [row["put_oi"] for row in subset],
            "Raw Change OI": [row["put_change_oi"] - row["call_change_oi"] for row in subset],
        }
    )
    return frame


def build_option_heatmap(rows: list[dict], spot: float | None = None, theme: dict | None = None) -> go.Figure:
    palette = {**DEFAULT_THEME, **(theme or {})}
    frame = heatmap_frame(rows, spot)
    fig = go.Figure()
    if frame.empty:
        fig.update_layout(
            template="plotly_white",
            paper_bgcolor=palette["panel"],
            plot_bgcolor=palette["panel"],
            font={"family": "Times New Roman", "color": palette["text"]},
            height=440,
            margin=dict(l=10, r=10, t=40, b=10),
        )
        fig.add_annotation(text="No option chain data", showarrow=False, x=0.5, y=0.5, xref="paper", yref="paper")
        return fig
    matrix = frame[["Call OI", "Put OI", "Change OI"]]
    hover = np.array(
        [
            [f"Strike {row['Strike']}<br>Call OI {row['Raw Call OI']:,}", f"Strike {row['Strike']}<br>Put OI {row['Raw Put OI']:,}", f"Strike {row['Strike']}<br>Net Chg OI {row['Raw Change OI']:,}"]
            for _, row in frame.iterrows()
        ]
    )
    fig.add_trace(
        go.Heatmap(
            z=matrix.values,
            x=matrix.columns,
            y=frame["Strike"],
            text=hover,
            hoverinfo="text",
            colorscale=[(0.0, palette["red"]), (0.5, palette["yellow"]), (1.0, palette["green"])],
            zmin=-1,
            zmax=1,
        )
    )
    fig.update_layout(
        template="plotly_white",
        paper_bgcolor=palette["panel"],
        plot_bgcolor=palette["panel"],
        font={"family": "Times New Roman", "color": palette["text"]},
        height=440,
        margin=dict(l=10, r=10, t=40, b=10),
        xaxis_title="Option Pressure",
        yaxis_title="Strike",
    )
    fig.update_xaxes(gridcolor="rgba(200, 194, 176, 0.35)", linecolor=palette["grid"], tickfont={"color": palette["muted"]}, title_font={"color": palette["muted"]})
    fig.update_yaxes(gridcolor="rgba(200, 194, 176, 0.35)", linecolor=palette["grid"], tickfont={"color": palette["muted"]}, title_font={"color": palette["muted"]})
    return fig
