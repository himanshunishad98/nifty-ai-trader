from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import streamlit.components.v1 as components
from streamlit_option_menu import option_menu

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from scripts.market_structure import apply_market_structure
from scripts.option_heatmap import build_option_heatmap
from scripts.predictive_model import FEATURE_COLUMNS
from scripts.reasoning import answer_question
from scripts.signal_engine import generate_signal
from scripts.config import UI_CONFIG

THEME = UI_CONFIG.get("theme", {
    "bg": "#0B0F19", "panel": "#151A26", "panel_alt": "#1C2333", "grid": "#2A3441",
    "text": "#E2E8F0", "muted": "#94A3B8", "accent": "#38BDF8", "accent_alt": "#0EA5E9",
    "green": "#10B981", "red": "#F43F5E", "yellow": "#F59E0B",
    "font": "'Inter', 'Segoe UI', sans-serif"
})
TABS = UI_CONFIG.get("tabs", ["Dashboard", "Signals", "Options Flow", "AI Model", "Performance", "Assistant"])
REFRESH_INTERVAL = UI_CONFIG.get("refresh_interval_seconds", 60)

st.set_page_config(page_title="NIFTY AI Trader", layout="wide", initial_sidebar_state="expanded")

# Initialize Session State routing — default to Dashboard so trading data shows immediately
if 'current_page' not in st.session_state:
    st.session_state.current_page = "Dashboard"


@st.cache_data(ttl=30, show_spinner=False)
def load_data() -> dict:
    return generate_signal()


def auto_refresh(seconds: int = 60) -> None:
    try:
        from streamlit_autorefresh import st_autorefresh
        st_autorefresh(interval=seconds * 1000, limit=None, key="auto_refresh")
    except ImportError:
        st.warning("Please install `streamlit-autorefresh` for seamless auto-refresh.")
        # Fallback to the old method if not installed
        components.html(
            f"""
            <script>
            setTimeout(function() {{
                window.parent.location.reload();
            }}, {seconds * 1000});
            </script>
            """,
            height=0,
        )


def apply_theme() -> None:
    st.markdown(
        f"""
        <style>
        /* Base typography - leverage inheritance instead of wildcards to preserve Streamlit's specific span-level emotion cache for icons (arrow_down fix) */
        html, body, .stApp, div[class*="st-"], p, h1, h2, h3, h4, h5, h6, li, label, button, input, textarea, table, td, th {{
            font-family: {THEME['font']} !important;
        }}
        /* Specifically restore Streamlit's material font globally just in case of any bleed-over into icon containers */
        [data-testid="stIconMaterial"], .material-symbols-rounded, .material-icons, span[class*="stIcon"] {{
            font-family: "Material Symbols Rounded" !important;
        }}
        .stApp {{ background: #0d1117; color: {THEME['text']}; }}
        [data-testid="stHeader"] {{ display: none !important; }}
        [data-testid="stToolbar"] {{ display: none !important; }}
        [data-testid="stDecoration"] {{ display: none !important; }}
        #MainMenu {{ display: none !important; }}
        footer {{ display: none !important; }}
        [data-testid="stSidebar"] {{ background: #161b22; border-right: 1px solid #30363d; }}
        .block-container {{ padding-top: 2.5rem; padding-bottom: 1.5rem; max-width: 96%; }}
        h1, h2, h3, h4, h5, h6, p, span, label, div {{ color: {THEME['text']}; }}
        .terminal-card {{
            background: #161b22;
            border: 1px solid #30363d;
            border-radius: 14px;
            padding: 20px 24px;
            min-height: auto;
            position: relative;
            overflow: hidden;
            box-shadow: 0 4px 16px rgba(0,0,0,0.3);
            display: flex;
            flex-direction: column;
            justify-content: center;
        }}
        .terminal-card::before {{
            content: "";
            position: absolute;
            top: 0;
            left: 0;
            right: 0;
            height: 2px;
            background: linear-gradient(90deg, transparent, #58a6ff, transparent);
            opacity: 0.6;
        }}
        .terminal-title {{ font-size: 0.8rem; color: #8b949e; text-transform: uppercase; letter-spacing: 0.12em; font-weight: 600; margin-bottom: 4px; }}
        .terminal-value {{ font-size: 1.6rem; font-weight: 800; margin: 0; color: #ffffff; line-height: 1.2; word-wrap: break-word; }}
        .terminal-sub {{ color: #8b949e; font-size: 0.9rem; line-height: 1.4; margin-top: 6px; font-weight: 500; }}
        .badge {{
            display: inline-block;
            padding: 4px 12px;
            border-radius: 9999px;
            font-size: 0.78rem;
            font-weight: 700;
            border: 1px solid #30363d;
            letter-spacing: 0.06em;
            text-transform: uppercase;
        }}
        .section-head {{
            display: flex;
            align-items: center;
            gap: 10px;
            font-size: 0.85rem;
            font-weight: 700;
            margin-bottom: 0.8rem;
            margin-top: 0.6rem;
            color: #58a6ff;
            text-transform: uppercase;
            letter-spacing: 0.12em;
        }}
        .section-head::after {{
            content: "";
            flex: 1;
            height: 1px;
            background: linear-gradient(90deg, #30363d, transparent);
        }}
        .theme-topbar {{
            display: flex;
            align-items: center;
            gap: 14px;
            background: {THEME['panel']};
            border: 1px solid {THEME['grid']};
            border-radius: 12px;
            padding: 12px 14px;
            margin-bottom: 1rem;
        }}
        .theme-brand {{
            color: {THEME['accent']};
            font-size: 1rem;
            font-weight: 800;
            letter-spacing: 0.16em;
            text-transform: uppercase;
            white-space: nowrap;
        }}
        
        /* Glassmorphism Classes */
        .glass-card {{
            background: rgba(22, 27, 34, 0.4);
            backdrop-filter: blur(12px);
            -webkit-backdrop-filter: blur(12px);
            border: 1px solid rgba(255, 255, 255, 0.08);
            border-top: 1px solid rgba(255, 255, 255, 0.15);
            border-left: 1px solid rgba(255, 255, 255, 0.1);
            border-radius: 16px;
            padding: 24px;
            box-shadow: 0 8px 32px rgba(0, 0, 0, 0.2);
            transition: transform 0.3s ease, box-shadow 0.3s ease;
        }}
        .glass-card:hover {{
            transform: translateY(-4px);
            box-shadow: 0 12px 40px rgba(0, 0, 0, 0.3);
            border-top: 1px solid rgba(88, 166, 255, 0.3);
        }}
        .hero-title {{
            background: linear-gradient(90deg, #58a6ff, #00ff66);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            font-size: 4rem;
            font-weight: 900;
            margin-bottom: 0px;
        }}
        .hero-subtitle {{
            font-size: 1.2rem;
            color: #8b949e;
            max-width: 600px;
            margin-bottom: 2rem;
        }}
        .theme-ticker {{
            display: flex;
            flex-wrap: wrap;
            gap: 8px;
            align-items: center;
        }}
        .tape-pill {{
            display: inline-flex;
            align-items: center;
            gap: 8px;
            padding: 7px 10px;
            background: {THEME['panel_alt']};
            border: 1px solid {THEME['grid']};
            border-radius: 8px;
        }}
        .tape-label {{
            color: {THEME['muted']};
            font-size: 0.72rem;
            letter-spacing: 0.14em;
            text-transform: uppercase;
            font-weight: 700;
        }}
        .tape-value {{
            color: {THEME['accent']};
            font-size: 0.9rem;
            font-weight: 700;
        }}
        .page-title {{
            color: {THEME['accent']};
            font-size: 3rem;
            font-weight: 800;
            margin-bottom: 0.18rem;
            line-height: 1.1;
        }}
        .page-subtitle {{
            color: {THEME['muted']};
            margin-bottom: 1rem;
            font-size: 1rem;
        }}
        .sidebar-brand {{
            padding: 0.2rem 0 1rem 0;
            border-bottom: 1px solid {THEME['grid']};
            margin-bottom: 1rem;
        }}
        .sidebar-logo {{
            color: {THEME['muted']};
            font-size: 0.76rem;
            letter-spacing: 0.24em;
            text-transform: uppercase;
            font-weight: 700;
            margin-bottom: 0.45rem;
        }}
        .sidebar-title {{
            color: {THEME['accent']};
            font-size: 1.45rem;
            font-weight: 800;
            margin-bottom: 0.25rem;
        }}
        .sidebar-note, .sidebar-footer {{
            color: {THEME['muted']};
            font-size: 0.86rem;
            line-height: 1.6;
        }}
        .sidebar-footer {{
            border-top: 1px solid {THEME['grid']};
            margin-top: 1rem;
            padding-top: 0.8rem;
            text-transform: uppercase;
            letter-spacing: 0.12em;
        }}
        .stButton > button {{
            background: #21262d;
            color: #ffffff;
            border: 1px solid #484f58;
            border-radius: 8px;
            font-weight: 700;
            letter-spacing: 0.06em;
            text-transform: uppercase;
            padding: 0.5rem 1rem;
        }}
        .stButton > button:hover {{
            border-color: #58a6ff;
            background: #30363d;
            color: #58a6ff;
        }}
        .stTabs [data-baseweb="tab-list"] {{
            gap: 4px;
            border-bottom: 1px solid #30363d;
            padding-bottom: 0;
            background: transparent;
        }}
        .stTabs [data-baseweb="tab"] {{
            background: transparent;
            border: none;
            border-bottom: 2px solid transparent;
            border-radius: 0;
            color: #8b949e;
            font-size: 0.95rem;
            font-weight: 600;
            padding: 0.6rem 1.2rem;
        }}
        .stTabs [aria-selected="true"] {{
            background: transparent;
            color: #58a6ff !important;
            border-bottom: 2px solid #58a6ff !important;
        }}
        div[data-testid="stDataFrame"] {{
            border: 1px solid #30363d;
            border-radius: 12px;
            overflow: hidden;
            box-shadow: 0 2px 8px rgba(0,0,0,0.2);
        }}
        [data-testid="stChatInput"] {{
            background: #161b22;
            border-top: 1px solid #30363d;
        }}
        /* Modern metric cards */
        [data-testid="stMetric"] {{
            background: #161b22;
            border: 1px solid #30363d;
            border-radius: 12px;
            padding: 16px 20px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.15);
        }}
        [data-testid="stMetricLabel"] {{
            color: #8b949e !important;
        }}
        [data-testid="stMetricValue"] {{
            color: #ffffff !important;
            font-weight: 800 !important;
        }}
        [data-testid="stMetricDelta"] {{
            color: #8b949e !important;
        }}
        /* Expander styling */
        [data-testid="stExpander"] {{
            background: #161b22;
            border: 1px solid #30363d;
            border-radius: 12px;
            overflow: hidden;
        }}
        /* Info box */
        .stAlert {{
            background: #161b22 !important;
            border: 1px solid #30363d !important;
            border-radius: 12px !important;
        }}
        .market-tape-container {{
            background: #161b22;
            border: 1px solid #30363d;
            border-radius: 16px;
            padding: 24px;
            margin-bottom: 24px;
            box-shadow: 0 8px 24px rgba(0,0,0,0.4);
        }}
        .market-tape-row {{
            display: flex;
            flex-wrap: wrap;
            gap: 20px;
            align-items: center;
            justify-content: center;
        }}
        .market-tape-divider {{
            height: 1px;
            background: #30363d;
            margin: 20px 0;
            width: 100%;
        }}
        .tape-chip {{
            background: #21262d;
            border: 1px solid #484f58;
            border-radius: 999px;
            padding: 12px 28px;
            display: flex;
            align-items: center;
            gap: 14px;
            box-shadow: 0 4px 12px rgba(0,0,0,0.2);
            transition: all 0.2s ease;
        }}
        .tape-chip:hover {{
            box-shadow: 0 4px 16px rgba(255,255,255,0.05);
            border-color: #8b949e;
        }}
        .tape-chip.signal-chip {{
            padding: 14px 36px;
            border-width: 2px;
        }}
        .tape-chip-label {{
            color: #8b949e;
            font-size: 0.9rem;
            text-transform: uppercase;
            letter-spacing: 0.1em;
            font-weight: 600;
        }}
        .tape-chip-value {{
            color: #ffffff;
            font-size: 1.2rem;
            font-weight: 800;
        }}
        .tape-chip-value.large {{
            font-size: 1.5rem;
        }}
        .tape-status-badge {{
            padding: 4px 12px;
            border-radius: 999px;
            font-size: 0.85rem;
            font-weight: 800;
            text-transform: uppercase;
            letter-spacing: 0.1em;
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )


def tone_color(label: str) -> str:
    text = str(label).upper()
    if any(token in text for token in ["BUY CALL", "BULLISH", "POSITIVE", "HIGH", "LIVE", "LONG_GAMMA", "TRENDING", "LOW_VOLATILITY"]):
        return THEME["green"]
    if any(token in text for token in ["BUY PUT", "BEARISH", "NEGATIVE", "LOW", "STALE", "SHORT_GAMMA", "HIGH_VOLATILITY"]):
        return THEME["red"]
    return THEME["yellow"]


def badge(label: str) -> str:
    color = tone_color(label)
    return f"<span class='badge' style='color:{color}; background:{color}16;'>{label}</span>"


def numeric_color(value: float) -> str:
    if value > 0:
        return THEME["green"]
    if value < 0:
        return THEME["red"]
    return THEME["yellow"]


def _format_value(value) -> str:
    if value is None:
        return "NA"
    if isinstance(value, float):
        return f"{value:.2f}" if abs(value) < 100000 else f"{value:,.0f}"
    return str(value)


def stat_card(title: str, value: str, subtitle: str) -> None:
    st.metric(label=title, value=value, delta=subtitle, delta_color="off")


def trade_summary(payload: dict) -> tuple[str, str]:
    trade = payload["trade"]
    strike = trade.get("strike")
    if not strike:
        return "No Trade", "No option strike suggested right now."
    return f"BUY {strike}", f"Expiry {trade.get('expiry')} | Entry {trade.get('entry_level')}"


def render_plot(container, figure: go.Figure, key: str) -> None:
    container.plotly_chart(figure, use_container_width=True, key=key)


def apply_chart_theme(fig: go.Figure, height: int, title: str | None = None, plot_bg: str | None = None) -> go.Figure:
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="#161b22",
        plot_bgcolor=plot_bg or "#0d1117",
        font={"family": THEME['font'], "color": "#c9d1d9"},
        title={"text": title or "", "font": {"size": 14, "color": "#58a6ff"}},
        height=height,
        margin=dict(l=10, r=10, t=30, b=10),
        legend=dict(bgcolor="rgba(0,0,0,0)", font={"color": "#c9d1d9"}),
    )
    fig.update_xaxes(gridcolor="#21262d", linecolor="#30363d", tickfont={"color": "#8b949e"}, title_font={"color": "#8b949e"})
    fig.update_yaxes(gridcolor="#21262d", linecolor="#30363d", tickfont={"color": "#8b949e"}, title_font={"color": "#8b949e"})
    return fig


def render_market_tape(payload: dict) -> None:
    trade_strike = payload.get("trade", {}).get("strike")
    strike_str = f"{trade_strike}" if trade_strike else "N/A"
    signal = payload.get("signal", "NO TRADE").upper()
    spot = _format_value(payload.get("spot"))
    
    # Signal Colors
    sig_color = THEME.get("yellow", "#FFC107")
    sig_bg = "rgba(255, 193, 7, 0.1)"
    if "CALL" in signal or "BULL" in signal:
        sig_color = "#00ff66" 
        sig_bg = "rgba(0, 255, 102, 0.1)"
    elif "PUT" in signal or "BEAR" in signal:
        sig_color = "#ff3333" 
        sig_bg = "rgba(255, 51, 51, 0.1)"

    # Freshness
    freshness = payload.get("data_freshness", "STALE").upper()
    fresh_bg, fresh_color = ("rgba(0, 255, 102, 0.15)", "#00ff66") if freshness == "LIVE" else ("rgba(255, 51, 51, 0.15)", "#ff3333")

    regime = payload.get("market_regime", {}).get("regime", "N/A")
    regime_color = tone_color(regime)
    vix_val = _format_value(payload.get("volatility", {}).get("value"))
    pcr_val = _format_value(payload.get("options_intelligence", {}).get("pcr"))

    html = f"""
    <div class="market-tape-container">
        <div class="market-tape-row">
            <div class="tape-chip">
                <span class="tape-chip-label">Spot</span>
                <span class="tape-chip-value large">{spot}</span>
            </div>
            <div class="tape-chip signal-chip" style="border-color:{sig_color}; background:{sig_bg}; box-shadow: 0 0 15px {sig_bg};">
                <span class="tape-chip-label" style="color:rgba(255,255,255,0.7);">Signal</span>
                <span class="tape-chip-value large" style="color:{sig_color};">{signal}</span>
            </div>
            <div class="tape-chip">
                <span class="tape-chip-label">Strike</span>
                <span class="tape-chip-value large">{strike_str}</span>
            </div>
        </div>
        <div class="market-tape-divider"></div>
        <div class="market-tape-row">
            <div class="tape-chip">
                <span class="tape-chip-label">Regime</span>
                <span class="tape-chip-value" style="color:{regime_color}">{regime}</span>
            </div>
            <div class="tape-chip">
                <span class="tape-chip-label">VIX</span>
                <span class="tape-chip-value">{vix_val}</span>
            </div>
            <div class="tape-chip">
                <span class="tape-chip-label">PCR</span>
                <span class="tape-chip-value">{pcr_val}</span>
            </div>
            <div class="tape-chip">
                <span class="tape-chip-label">Data Status</span>
                <span class="tape-status-badge" style="background:{fresh_bg}; color:{fresh_color};">{freshness}</span>
            </div>
        </div>
    </div>
    """
    st.markdown(html, unsafe_allow_html=True)


def gauge(title: str, score: float, label: str, suffix: str = "") -> go.Figure:
    color = tone_color(label)
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=score,
            number={"suffix": suffix, "font": {"size": 36, "color": "#ffffff"}},
            title={"text": f"{title}<br><span style='font-size:13px;color:{color}'>{label}</span>"},
            gauge={
                "axis": {"range": [0, 100], "tickcolor": "#484f58"},
                "bar": {"color": color},
                "bgcolor": "#21262d",
                "borderwidth": 0,
                "steps": [
                    {"range": [0, 40], "color": "#1a1e24"},
                    {"range": [40, 70], "color": "#21262d"},
                    {"range": [70, 100], "color": "#272c33"},
                ],
            },
        )
    )
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="#161b22",
        plot_bgcolor="#161b22",
        font={"family": THEME['font'], "color": "#c9d1d9"},
        height=250,
        margin=dict(l=15, r=15, t=60, b=10),
    )
    return fig


def build_structure_chart(df: pd.DataFrame, structure: dict) -> go.Figure:
    fig = go.Figure()
    if df.empty:
        apply_chart_theme(fig, 520, plot_bg=THEME["panel_alt"])
        fig.add_annotation(text="No market data available", showarrow=False, x=0.5, y=0.5, xref="paper", yref="paper")
        return fig
    fig.add_trace(
        go.Candlestick(
            x=df["Datetime"],
            open=df["Open"],
            high=df["High"],
            low=df["Low"],
            close=df["Close"],
            name="NIFTY",
            increasing_line_color=THEME["green"],
            decreasing_line_color=THEME["red"],
        )
    )
    fig = apply_market_structure(fig, structure)
    fig.add_annotation(xref="paper", yref="paper", x=0.01, y=0.98, text=f"Trend: {structure['trend']}", showarrow=False, font={"color": tone_color(structure['trend'])}, bgcolor=THEME["panel"])
    apply_chart_theme(fig, 520, "Market Structure", THEME["panel_alt"])
    fig.update_layout(xaxis_rangeslider_visible=False)
    return fig


def build_vix_chart(df: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    if df.empty:
        apply_chart_theme(fig, 220)
        return fig
    fig.add_trace(go.Scatter(x=df["Datetime"], y=df["Close"], mode="lines", line=dict(color=THEME["yellow"], width=2), fill="tozeroy", fillcolor="rgba(122,92,0,0.14)"))
    apply_chart_theme(fig, 220, "India VIX")
    return fig


def build_gamma_chart(payload: dict) -> go.Figure:
    rows = payload["options_intelligence"]["gamma"]["rows"]
    fig = go.Figure()
    if not rows:
        apply_chart_theme(fig, 260)
        return fig
    frame = pd.DataFrame(rows)
    if payload["spot"] is not None:
        frame = frame.assign(distance=(frame["strike"] - payload["spot"]).abs()).nsmallest(18, "distance").sort_values("strike")
    fig.add_trace(go.Bar(x=frame["strike"], y=frame["gex"], marker_color=[numeric_color(value) for value in frame["gex"]]))
    apply_chart_theme(fig, 260, "Gamma Exposure")
    fig.update_layout(yaxis_title="GEX")
    return fig


def build_options_flow_chart(payload: dict) -> go.Figure:
    option_chain = payload["option_chain"]
    fig = go.Figure(
        go.Bar(
            x=["Call Chg OI", "Put Chg OI"],
            y=[option_chain.get("call_change_total", 0), option_chain.get("put_change_total", 0)],
            marker_color=[THEME["red"], THEME["green"]],
        )
    )
    apply_chart_theme(fig, 260, f"Smart Money: {payload['options_intelligence']['smart_money_pressure']['label']}")
    return fig


def build_prediction_chart(prediction: dict) -> go.Figure:
    fig = go.Figure(
        go.Bar(
            x=["Bullish", "Bearish"],
            y=[prediction["bullish_probability"] * 100, prediction["bearish_probability"] * 100],
            marker_color=[THEME["green"], THEME["red"]],
            text=[f"{prediction['bullish_probability']*100:.1f}%", f"{prediction['bearish_probability']*100:.1f}%"],
            textposition="outside",
        )
    )
    apply_chart_theme(fig, 260, "Prediction Bias")
    fig.update_layout(yaxis_title="Probability")
    return fig


def build_timeline_chart(frame: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    if frame.empty:
        apply_chart_theme(fig, 260)
        return fig
    fig.add_trace(
        go.Scatter(
            x=frame["timestamp"],
            y=frame["confidence"],
            mode="lines+markers",
            marker=dict(color=[tone_color(value) for value in frame["signal"]], size=9),
            text=frame["signal"],
            hovertemplate="%{text}<br>Confidence %{y}%<extra></extra>",
            line=dict(color=THEME["accent_alt"]),
        )
    )
    apply_chart_theme(fig, 260, "Signal Timeline")
    fig.update_layout(yaxis_title="Confidence")
    return fig


def component_frame(payload: dict) -> pd.DataFrame:
    mapping = {"Indicators": "indicators", "Options": "option_chain", "Patterns": "patterns", "Sentiment": "news_sentiment"}
    rows = []
    for component, raw in payload["component_scores"].items():
        weight_key = mapping[component]
        rows.append(
            {
                "Component": component,
                "Raw Score": raw,
                "Strength": round(abs(raw) * 100, 1),
                "Weight": f"{int(payload['weights'][weight_key] * 100)}%",
                "Weighted Impact": round(raw * payload["weights"][weight_key], 2),
            }
        )
    return pd.DataFrame(rows)


def indicator_frame(payload: dict) -> pd.DataFrame:
    rows = []
    for timeframe in ("1m", "5m"):
        for indicator, signal in payload["indicators"][timeframe]["signals"].items():
            rows.append({"Timeframe": timeframe, "Indicator": indicator, "Signal": signal})
    return pd.DataFrame(rows)


def pattern_frame(payload: dict) -> pd.DataFrame:
    patterns = payload["patterns"]["patterns"]
    if not patterns:
        return pd.DataFrame([{"Pattern": "None", "Signal": "NEUTRAL", "Score": 0.0}])
    return pd.DataFrame(patterns).rename(columns={"name": "Pattern", "signal": "Signal", "score": "Score"})


def option_frame(payload: dict) -> pd.DataFrame:
    rows = payload["option_chain"]["rows"]
    if not rows:
        return pd.DataFrame(columns=["Strike", "Call OI", "Put OI", "Call Chg OI", "Put Chg OI", "Call Vol", "Put Vol"])
    frame = pd.DataFrame(rows).sort_values("strike")
    return frame.rename(
        columns={
            "strike": "Strike",
            "call_oi": "Call OI",
            "put_oi": "Put OI",
            "call_change_oi": "Call Chg OI",
            "put_change_oi": "Put Chg OI",
            "call_volume": "Call Vol",
            "put_volume": "Put Vol",
        }
    )


def feature_frame(payload: dict) -> pd.DataFrame:
    return pd.DataFrame([payload["feature_row"]])[FEATURE_COLUMNS]


def trade_frame(payload: dict) -> pd.DataFrame:
    trade = payload["trade"]
    return pd.DataFrame(
        [
            {"Field": "Action", "Value": payload["signal"]},
            {"Field": "Strike", "Value": trade.get("strike")},
            {"Field": "Expiry", "Value": trade.get("expiry")},
            {"Field": "Entry", "Value": trade.get("entry_level")},
            {"Field": "Stop Loss", "Value": trade.get("stop_loss")},
            {"Field": "Target", "Value": trade.get("target")},
        ]
    )


def reasoning_panel(payload: dict) -> None:
    st.info(payload['reasoning'], icon="🧠")


def render_dashboard_tab(payload: dict) -> None:
    top_1, top_2, top_3 = st.columns(3)
    render_plot(top_1, gauge("Signal Strength", payload["confidence"], payload["signal"], "%"), "dashboard_signal_strength")
    render_plot(top_2, gauge("Trade Opportunity", payload["opportunity"]["score"], payload["opportunity"]["label"]), "dashboard_trade_opportunity")
    render_plot(top_3, gauge("Signal Quality", payload["signal_quality"]["score"], payload["signal_quality"]["label"]), "dashboard_signal_quality")

    middle_left, middle_right = st.columns([1.7, 1])
    render_plot(middle_left, build_structure_chart(payload["market_data"]["5m"], payload["market_structure"]), "dashboard_structure")
    render_plot(middle_right, build_option_heatmap(payload["option_chain"]["rows"], payload["spot"], theme=THEME), "dashboard_option_heatmap")

    lower_top = st.columns(3)
    with lower_top[0]:
        st.markdown("<div class='section-head'>India VIX Panel</div>", unsafe_allow_html=True)
        stat_card("Current VIX", str(payload["volatility"]["value"] or "NA"), f"{payload['volatility']['regime']} | {payload['market_regime']['regime']}")
        render_plot(st, build_vix_chart(payload["volatility"]["data"]), "dashboard_vix")
    with lower_top[1]:
        st.markdown("<div class='section-head'>Gamma Exposure Panel</div>", unsafe_allow_html=True)
        stat_card("Dealer Positioning", payload["options_intelligence"]["gamma"]["dealer_positioning"], f"Flip {payload['options_intelligence']['gamma']['gamma_flip']}")
        render_plot(st, build_gamma_chart(payload), "dashboard_gamma")
    with lower_top[2]:
        st.markdown("<div class='section-head'>Options Flow Panel</div>", unsafe_allow_html=True)
        stat_card("PCR / Max Pain", f"{payload['options_intelligence']['pcr']} / {payload['options_intelligence']['max_pain']}", payload["options_intelligence"]["smart_money_pressure"]["label"])
        render_plot(st, build_options_flow_chart(payload), "dashboard_options_flow")

    lower_mid = st.columns(3)
    with lower_mid[0]:
        st.markdown("<div class='section-head'>Suggested Trade Panel</div>", unsafe_allow_html=True)
        st.dataframe(trade_frame(payload), hide_index=True, use_container_width=True)
    with lower_mid[1]:
        st.markdown("<div class='section-head'>AI Prediction Panel</div>", unsafe_allow_html=True)
        stat_card("Model Status", "READY" if payload["prediction"]["model_ready"] else "LEARNING", f"Samples {payload['model_status'].get('samples', 0)}")
        render_plot(st, build_prediction_chart(payload["prediction"]), "dashboard_prediction")
    with lower_mid[2]:
        st.markdown("<div class='section-head'>AI Reasoning Panel</div>", unsafe_allow_html=True)
        reasoning_panel(payload)

    lower_bottom = st.columns([0.8, 1.2])
    with lower_bottom[0]:
        st.markdown("<div class='section-head'>Prediction Performance Panel</div>", unsafe_allow_html=True)
        perf = payload["prediction_performance"]
        stat_card("Accuracy", f"{perf['accuracy']}%", f"Resolved {perf['resolved']} | Pending {perf['pending']}")
    with lower_bottom[1]:
        st.markdown("<div class='section-head'>Signal Timeline</div>", unsafe_allow_html=True)
        render_plot(st, build_timeline_chart(payload["signal_timeline"]), "dashboard_timeline")

    with st.expander("Detailed analysis"):
        st.dataframe(component_frame(payload), hide_index=True, use_container_width=True)
        st.dataframe(option_frame(payload), hide_index=True, use_container_width=True)


def render_signals_tab(payload: dict) -> None:
    row_1, row_2 = st.columns([1, 1.1])
    with row_1:
        stat_card("Signal", payload["signal"], f"Confidence {payload['confidence']}% | Composite {payload['score']}")
        st.dataframe(indicator_frame(payload), hide_index=True, use_container_width=True)
    with row_2:
        st.dataframe(component_frame(payload), hide_index=True, use_container_width=True)
        st.dataframe(pattern_frame(payload), hide_index=True, use_container_width=True)
    with st.expander("Reasoning and news", expanded=True):
        reasoning_panel(payload)
        news = payload["news"]["headlines"] or [{"source": "NA", "title": "No headlines", "score": 0.0}]
        st.dataframe(pd.DataFrame(news), hide_index=True, use_container_width=True)


def render_options_tab(payload: dict) -> None:
    row_1, row_2 = st.columns([1, 1])
    render_plot(row_1, build_option_heatmap(payload["option_chain"]["rows"], payload["spot"], theme=THEME), "options_heatmap")
    render_plot(row_2, build_gamma_chart(payload), "options_gamma")
    row_3, row_4 = st.columns([1, 1])
    render_plot(row_3, build_options_flow_chart(payload), "options_flow")
    with row_4:
        spikes = payload["options_intelligence"]["volume_spikes"] or [{"strike": None, "call_volume": 0, "put_volume": 0, "total_volume": 0}]
        st.dataframe(pd.DataFrame(spikes), hide_index=True, use_container_width=True)
    with st.expander("Detailed option chain", expanded=True):
        st.dataframe(option_frame(payload), hide_index=True, use_container_width=True)


def render_ai_model_tab(payload: dict) -> None:
    row_1, row_2 = st.columns([1, 1])
    render_plot(row_1, build_prediction_chart(payload["prediction"]), "ai_model_prediction")
    with row_2:
        status = payload["model_status"]
        stat_card("Model", "READY" if payload["prediction"]["model_ready"] else "LEARNING", f"Training samples {status.get('samples', 0)}")
        st.dataframe(feature_frame(payload), hide_index=True, use_container_width=True)


def render_performance_tab(payload: dict) -> None:
    perf = payload["prediction_performance"]
    top = st.columns(3)
    top[0].metric("Accuracy", f"{perf['accuracy']}%")
    top[1].metric("Resolved", perf["resolved"])
    top[2].metric("Pending", perf["pending"])
    render_plot(st, build_timeline_chart(payload["signal_timeline"]), "performance_timeline")
    with st.expander("Prediction log", expanded=True):
        frame = payload["signal_timeline"].copy()
        if not frame.empty:
            st.dataframe(frame, hide_index=True, use_container_width=True)


def render_assistant_tab(payload: dict) -> None:
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    question = st.chat_input("Ask about signals, options flow, volatility, trade idea, or model performance")
    if question:
        answer = answer_question(question, payload)
        st.session_state.chat_history.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)
        with st.chat_message("assistant"):
            st.markdown(answer)
        st.session_state.chat_history.append({"role": "assistant", "content": answer})

# ==========================================
# NEW SPA ROUTING ARCHITECTURE
# ==========================================

def render_home_page():
    # Hero Section — split into separate calls to avoid Streamlit HTML escaping issue
    st.markdown('''
    <div style="text-align: center; padding: 60px 20px 40px 20px; position: relative;">
        <div style="position: absolute; top: -50%; left: 50%; transform: translateX(-50%); width: 600px; height: 600px; background: radial-gradient(circle, rgba(88,166,255,0.15) 0%, rgba(13,17,23,0) 70%); z-index: -1;"></div>
        <div class="badge" style="background: rgba(88, 166, 255, 0.1); color: #58a6ff; border: 1px solid #58a6ff; margin-bottom: 20px; display: inline-block;">NIFTY AI v2.0 Live</div>
        <h1 class="hero-title">Master the Market.</h1>
        <p class="hero-subtitle" style="margin: 0 auto 30px auto;">Institutional-grade options intelligence and AI pricing models brought directly to your browser. Navigate volatility with unparalleled precision.</p>
    </div>
    ''', unsafe_allow_html=True)
    
    # Feature Grid
    st.markdown('<div class="section-head" style="margin: 40px 0 20px 0; justify-content:center;">Platform Capabilities</div>', unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown('''
        <div class="glass-card" style="height: 100%;">
            <div style="width: 48px; height: 48px; border-radius: 12px; background: rgba(88, 166, 255, 0.1); display: flex; align-items: center; justify-content: center; margin-bottom: 20px;">
                <span class="material-symbols-rounded" style="color: #58a6ff; font-size: 28px;">monitoring</span>
            </div>
            <h3 style="color: #ffffff; margin: 0 0 10px 0; font-size: 1.2rem;">Live Gamma Tracking</h3>
            <p style="color: #8b949e; margin: 0; line-height: 1.6; font-size: 0.95rem;">Visualize dealer positioning and implied volatility spikes in real-time across the entire NIFTY option chain. Trade the levels that matter.</p>
        </div>
        ''', unsafe_allow_html=True)
    with col2:
        st.markdown('''
        <div class="glass-card" style="height: 100%;">
            <div style="width: 48px; height: 48px; border-radius: 12px; background: rgba(0, 255, 102, 0.1); display: flex; align-items: center; justify-content: center; margin-bottom: 20px;">
                <span class="material-symbols-rounded" style="color: #00ff66; font-size: 28px;">psychology</span>
            </div>
            <h3 style="color: #ffffff; margin: 0 0 10px 0; font-size: 1.2rem;">AI Predictive Models</h3>
            <p style="color: #8b949e; margin: 0; line-height: 1.6; font-size: 0.95rem;">Leverage custom trained institutional ML models to predict next-bar directional bias. Understand the probability before entering.</p>
        </div>
        ''', unsafe_allow_html=True)
    with col3:
        st.markdown('''
        <div class="glass-card" style="height: 100%;">
            <div style="width: 48px; height: 48px; border-radius: 12px; background: rgba(255, 51, 51, 0.1); display: flex; align-items: center; justify-content: center; margin-bottom: 20px;">
                <span class="material-symbols-rounded" style="color: #ff3333; font-size: 28px;">troubleshoot</span>
            </div>
            <h3 style="color: #ffffff; margin: 0 0 10px 0; font-size: 1.2rem;">Options Flow Analyser</h3>
            <p style="color: #8b949e; margin: 0; line-height: 1.6; font-size: 0.95rem;">Detect smart money accumulation, call/put writing pressure, and max pain convergence before the breakout actually happens in spot.</p>
        </div>
        ''', unsafe_allow_html=True)

    # Progress/Community Section
    st.markdown('<div class="section-head" style="margin: 60px 0 20px 0;">Platform Statistics</div>', unsafe_allow_html=True)
    stat_col1, stat_col2, stat_col3, stat_col4 = st.columns(4)
    with stat_col1: stat_card("Live Signals Today", "14", "+2 from yesterday")
    with stat_col2: stat_card("Model Accuracy", "82.4%", "Past 30 days")
    with stat_col3: stat_card("Active Put/Call Edge", "0.85", "Bearish skew")
    with stat_col4: stat_card("Gamma Exposure", "-3.2M", "Short Gamma Regime")

def render_lessons_page():
    st.markdown('''
    <div style="margin-bottom: 40px;">
        <h1 class="page-title" style="margin-bottom: 8px;">Educational Modules</h1>
        <p class="page-subtitle">Master the mechanics of options order flow, market maker hedging, and volatility.</p>
    </div>
    ''', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    with col1:
        st.markdown('''
        <div class="glass-card" style="position: relative; overflow: hidden; height: 100%;">
            <div style="position: absolute; top:0; right:0; width: 150px; height: 150px; background: rgba(88, 166, 255, 0.05); border-radius: 50%; transform: translate(50%, -50%);"></div>
            
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 15px;">
                <span class="badge" style="background: rgba(88, 166, 255, 0.1); color: #58a6ff; border: 1px solid #58a6ff;">Beginner</span>
                <span style="color: #8b949e; font-size: 0.8em; display: flex; align-items: center; gap: 4px;">
                    <span class="material-symbols-rounded" style="font-size:16px;">schedule</span> 15 min
                </span>
            </div>
            
            <h3 style="color: #ffffff; margin: 0 0 10px 0; font-size: 1.4rem;">Market Structure & Liquidity</h3>
            <p style="color: #8b949e; margin: 0 0 20px 0; line-height: 1.6; font-size: 0.95rem;">Understand how institutional orders shape Support and Resistance, and how to identify true liquidity pools vs retail stops.</p>
            
            <div style="width: 100%; background: rgba(255,255,255,0.05); height: 6px; border-radius: 3px; overflow: hidden; margin-bottom: 8px;">
                <div style="width: 100%; background: #58a6ff; height: 100%;"></div>
            </div>
            <div style="color: #8b949e; font-size: 0.8rem; text-align: right; font-weight: 700;">100% Complete</div>
        </div>
        ''', unsafe_allow_html=True)
    with col2:
        st.markdown('''
        <div class="glass-card" style="position: relative; overflow: hidden; height: 100%;">
            <div style="position: absolute; top:0; right:0; width: 150px; height: 150px; background: rgba(255, 51, 51, 0.05); border-radius: 50%; transform: translate(50%, -50%);"></div>
            
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 15px;">
                <span class="badge" style="background: rgba(255, 51, 51, 0.1); color: #ff3333; border: 1px solid #ff3333;">Advanced</span>
                <span style="color: #8b949e; font-size: 0.8em; display: flex; align-items: center; gap: 4px;">
                    <span class="material-symbols-rounded" style="font-size:16px;">schedule</span> 45 min
                </span>
            </div>
            
            <h3 style="color: #ffffff; margin: 0 0 10px 0; font-size: 1.4rem;">Gamma Squeeze Mechanics</h3>
            <p style="color: #8b949e; margin: 0 0 20px 0; line-height: 1.6; font-size: 0.95rem;">Deep dive into how Dealer Hedging (Delta Neutralizing) drives explosive price moves when the market enters Short Gamma regimes.</p>
            
            <div style="width: 100%; background: rgba(255,255,255,0.05); height: 6px; border-radius: 3px; overflow: hidden; margin-bottom: 8px;">
                <div style="width: 35%; background: #ff3333; height: 100%; box-shadow: 0 0 10px #ff3333;"></div>
            </div>
            <div style="color: #8b949e; font-size: 0.8rem; text-align: right; font-weight: 700;">35% Complete</div>
        </div>
        ''', unsafe_allow_html=True)
        
    st.markdown('<div class="section-head" style="margin: 40px 0 20px 0;">Knowledge Check</div>', unsafe_allow_html=True)
    
    with st.expander("Quiz: Identifying Gamma Regimes", expanded=False):
        st.radio("When the market is in a 'Short Gamma' regime, dealers tend to:", 
                 ["Buy the dip and sell the rip (Suppress Volatility)", 
                  "Buy strength and sell weakness (Expand Volatility)", 
                  "Ignore price changes"], index=None)
        st.button("Submit Answer", type="primary")


def render_global_intelligence_panel(payload: dict):
    """Render world indices, commodities, currency and bond tiles."""
    global_market = payload.get("global_market", {})
    instruments = global_market.get("instruments", [])
    overall_score = global_market.get("overall_score", 0.0)

    bias_color = "#39d353" if overall_score > 0.05 else "#ff3333" if overall_score < -0.05 else "#e3b341"
    bias_label = "🟢 BULLISH for India" if overall_score > 0.05 else "🔴 BEARISH for India" if overall_score < -0.05 else "🟡 NEUTRAL"

    st.markdown(f'''
    <div style="display:flex; align-items:center; gap:16px; margin-bottom:20px;">
        <div class="section-head" style="margin:0;">Global Market Intelligence</div>
        <div style="background: rgba(255,255,255,0.05); border:1px solid {bias_color}33; color:{bias_color};
                    border-radius:20px; padding:4px 14px; font-size:0.8rem; font-weight:700;">{bias_label}</div>
    </div>
    ''', unsafe_allow_html=True)

    if not instruments:
        st.info("Global market data loading... Please refresh in a moment.")
        return

    # Group into rows of 4
    for i in range(0, len(instruments), 4):
        row = instruments[i:i+4]
        cols = st.columns(len(row))
        for col, inst in zip(cols, row):
            chg = inst["change_pct"]
            color = "#39d353" if chg > 0 else "#ff3333" if chg < 0 else "#8b949e"
            arrow = "▲" if chg > 0 else "▼" if chg < 0 else "—"
            col.markdown(f'''
            <div class="glass-card" style="padding:14px 16px; text-align:center;">
                <div style="color:#8b949e; font-size:0.7rem; font-weight:600; text-transform:uppercase; letter-spacing:0.05em; margin-bottom:4px;">{inst["name"]}</div>
                <div style="color:#ffffff; font-size:1.05rem; font-weight:800;">{inst["price"]:,.2f}</div>
                <div style="color:{color}; font-size:0.85rem; font-weight:700;">{arrow} {abs(chg):.2f}%</div>
                <div style="color:#484f58; font-size:0.65rem; margin-top:4px;">{inst["india_impact"]}</div>
            </div>
            ''', unsafe_allow_html=True)


def render_sentiment_panel(payload: dict):
    """Render VADER NLP sentiment gauge and headline ticker."""
    news = payload.get("news", {})
    score = news.get("score", 0.0)
    bias = news.get("bias", "NEUTRAL")
    headlines = news.get("headlines", [])
    cat_scores = news.get("category_scores", {})
    source_count = news.get("source_count", 0)

    gauge_color = "#39d353" if bias == "BULLISH" else "#ff3333" if bias == "BEARISH" else "#e3b341"
    bar_pct = int((score + 1) / 2 * 100)   # map -1..1 → 0..100%

    st.markdown('<div class="section-head" style="margin: 30px 0 16px 0;">News Sentiment Intelligence</div>', unsafe_allow_html=True)

    col_gauge, col_cats = st.columns([1, 2])

    with col_gauge:
        st.markdown(f'''
        <div class="glass-card" style="text-align:center; padding:20px;">
            <div style="font-size:2.4rem; font-weight:900; color:{gauge_color};">{score:+.3f}</div>
            <div style="color:{gauge_color}; font-weight:700; font-size:1rem; margin:6px 0;">{bias}</div>
            <div style="width:100%; background:rgba(255,255,255,0.06); border-radius:4px; height:8px; overflow:hidden; margin:10px 0;">
                <div style="width:{bar_pct}%; background:linear-gradient(90deg,#ff3333,#e3b341,#39d353); height:100%;"></div>
            </div>
            <div style="color:#484f58; font-size:0.75rem;">VADER NLP · {source_count} sources</div>
        </div>
        ''', unsafe_allow_html=True)

    with col_cats:
        if cat_scores:
            rows_html = "".join(
                f'''<div style="display:flex; justify-content:space-between; align-items:center;
                               padding:6px 0; border-bottom:1px solid rgba(255,255,255,0.04);">
                    <span style="color:#8b949e; font-size:0.8rem;">{cat}</span>
                    <span style="color:{"#39d353" if v > 0.05 else "#ff3333" if v < -0.05 else "#e3b341"};
                                font-weight:700; font-size:0.8rem;">{v:+.3f}</span>
                </div>'''
                for cat, v in list(cat_scores.items())[:8]
            )
            st.markdown(f'<div class="glass-card" style="padding:16px;">{rows_html}</div>', unsafe_allow_html=True)

    # Headlines ticker
    if headlines:
        st.markdown('<div style="margin-top:16px;">', unsafe_allow_html=True)
        for h in headlines[:10]:
            s = h.get("score", 0.0)
            hcolor = "#39d353" if s > 0.05 else "#ff3333" if s < -0.05 else "#8b949e"
            st.markdown(f'''
            <div style="display:flex; align-items:flex-start; gap:12px; padding:10px 14px;
                        background:rgba(255,255,255,0.02); border-left:3px solid {hcolor};
                        border-radius:0 8px 8px 0; margin-bottom:6px;">
                <span style="color:{hcolor}; font-weight:800; font-size:0.9rem; min-width:48px;">{s:+.2f}</span>
                <div>
                    <div style="color:#e6edf3; font-size:0.85rem; line-height:1.4;">{h["title"][:120]}</div>
                    <div style="color:#484f58; font-size:0.7rem; margin-top:3px;">{h.get("source","")}&nbsp;·&nbsp;{h.get("category","")}</div>
                </div>
            </div>
            ''', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)


def render_ai_ensemble_panel(payload: dict):
    """Show RF, MLP neural net, and RL agent predictions side-by-side."""
    rf  = payload.get("prediction", {})
    dl  = payload.get("dl_prediction", {})
    rl  = payload.get("rl_prediction", {})
    ms  = payload.get("model_status", {})

    st.markdown('<div class="section-head" style="margin: 30px 0 16px 0;">AI Ensemble — 3 Model Comparison</div>', unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)

    # ── Random Forest ──
    with col1:
        rf_bull = rf.get("bullish_probability", 0.5)
        rf_ready = rf.get("model_ready", False)
        rf_signal = "🟢 BUY CALL" if rf_bull > 0.55 else "🔴 BUY PUT" if rf_bull < 0.45 else "🟡 NO TRADE"
        rf_color = "#39d353" if rf_bull > 0.55 else "#ff3333" if rf_bull < 0.45 else "#e3b341"
        status_badge = f'<span style="background:#39d35322; color:#39d353; border-radius:4px; padding:2px 8px; font-size:0.7rem;">TRAINED · {ms.get("samples",0)} samples</span>' if rf_ready else '<span style="background:#e3b34122; color:#e3b341; border-radius:4px; padding:2px 8px; font-size:0.7rem;">ACCUMULATING</span>'
        st.markdown(f'''
        <div class="glass-card" style="text-align:center; padding:20px;">
            <div style="font-size:0.75rem; color:#8b949e; font-weight:700; text-transform:uppercase; letter-spacing:0.08em; margin-bottom:12px;">Random Forest</div>
            {status_badge}
            <div style="font-size:1.6rem; font-weight:900; color:{rf_color}; margin:14px 0 6px 0;">{rf_signal}</div>
            <div style="color:#8b949e; font-size:0.85rem; margin-bottom:12px;">Bullish: {rf_bull*100:.1f}%</div>
            <div style="width:100%; background:rgba(255,255,255,0.06); border-radius:4px; height:6px; overflow:hidden;">
                <div style="width:{int(rf_bull*100)}%; background:{rf_color}; height:100%;"></div>
            </div>
            <div style="color:#484f58; font-size:0.7rem; margin-top:8px;">200 trees · 17 features</div>
        </div>
        ''', unsafe_allow_html=True)

    # ── Neural Network (MLP) ──
    with col2:
        dl_ready = dl.get("dl_ready", False)
        dl_signal = dl.get("dl_signal", "Accumulating data...")
        dl_conf = dl.get("dl_confidence", 0.0)
        dl_call = dl.get("dl_call", 0.333)
        dl_put  = dl.get("dl_put", 0.333)
        dl_color = "#39d353" if "CALL" in dl_signal else "#ff3333" if "PUT" in dl_signal else "#e3b341"
        dl_badge = f'<span style="background:#39d35322; color:#39d353; border-radius:4px; padding:2px 8px; font-size:0.7rem;">TRAINED · {ms.get("dl_samples",0)} samples</span>' if dl_ready else '<span style="background:#e3b34122; color:#e3b341; border-radius:4px; padding:2px 8px; font-size:0.7rem;">ACCUMULATING</span>'
        st.markdown(f'''
        <div class="glass-card" style="text-align:center; padding:20px;">
            <div style="font-size:0.75rem; color:#8b949e; font-weight:700; text-transform:uppercase; letter-spacing:0.08em; margin-bottom:12px;">Deep Learning · MLP</div>
            {dl_badge}
            <div style="font-size:1.6rem; font-weight:900; color:{dl_color}; margin:14px 0 6px 0;">{"🟢 " if "CALL" in dl_signal else "🔴 " if "PUT" in dl_signal else "🟡 "}{dl_signal}</div>
            <div style="color:#8b949e; font-size:0.85rem; margin-bottom:12px;">Confidence: {dl_conf:.1f}%</div>
            <div style="display:flex; gap:4px; margin-top:8px;">
                <div style="flex:1; background:rgba(255,255,255,0.06); border-radius:4px; height:6px; overflow:hidden;">
                    <div style="width:{int(dl_put*100)}%; background:#ff3333; height:100%;"></div>
                </div>
                <div style="flex:1; background:rgba(255,255,255,0.06); border-radius:4px; height:6px; overflow:hidden;">
                    <div style="width:{int(dl_call*100)}%; background:#39d353; height:100%;"></div>
                </div>
            </div>
            <div style="color:#484f58; font-size:0.7rem; margin-top:8px;">PyTorch · 4-layer BatchNorm</div>
        </div>
        ''', unsafe_allow_html=True)

    # ── Reinforcement Learning (DQN) ──
    with col3:
        rl_ready = rl.get("rl_ready", False)
        rl_signal = rl.get("rl_signal", "NO TRADE")
        rl_eps = rl.get("rl_epsilon", 1.0)
        rl_steps = rl.get("rl_steps", 0)
        rl_exploring = rl.get("rl_exploring", True)
        rl_color = "#39d353" if "CALL" in rl_signal else "#ff3333" if "PUT" in rl_signal else "#e3b341"
        rl_status = f"EXPLORING ε={rl_eps:.2f}" if rl_exploring else f"EXPLOITING ε={rl_eps:.2f}"
        rl_badge_color = "#e3b341" if rl_exploring else "#39d353"
        st.markdown(f'''
        <div class="glass-card" style="text-align:center; padding:20px;">
            <div style="font-size:0.75rem; color:#8b949e; font-weight:700; text-transform:uppercase; letter-spacing:0.08em; margin-bottom:12px;">Reinforcement Learning · DQN</div>
            <span style="background:{rl_badge_color}22; color:{rl_badge_color}; border-radius:4px; padding:2px 8px; font-size:0.7rem;">{rl_status}</span>
            <div style="font-size:1.6rem; font-weight:900; color:{rl_color}; margin:14px 0 6px 0;">{"🟢 " if "CALL" in rl_signal else "🔴 " if "PUT" in rl_signal else "🟡 "}{rl_signal}</div>
            <div style="color:#8b949e; font-size:0.85rem; margin-bottom:12px;">Q-steps: {rl_steps:,}</div>
            <div style="width:100%; background:rgba(255,255,255,0.06); border-radius:4px; height:6px; overflow:hidden; margin-bottom:4px;">
                <div style="width:{int((1-rl_eps)*100)}%; background:#58a6ff; height:100%;"></div>
            </div>
            <div style="color:#484f58; font-size:0.7rem;">Experience: {int((1-rl_eps)*100)}% · Online Q-learning</div>
        </div>
        ''', unsafe_allow_html=True)


def render_dashboard_app():
    payload = load_data()
    render_market_tape(payload)
    render_global_intelligence_panel(payload)
    render_sentiment_panel(payload)
    render_ai_ensemble_panel(payload)

    tabs = st.tabs(["Dashboard", "Signals", "Options", "AI Model", "Performance", "Assistant"])
    with tabs[0]: render_dashboard_tab(payload)
    with tabs[1]: render_signals_tab(payload)
    with tabs[2]: render_options_tab(payload)
    with tabs[3]: render_ai_model_tab(payload)
    with tabs[4]: render_performance_tab(payload)
    with tabs[5]: render_assistant_tab(payload)



def main():
    apply_theme()
    auto_refresh(REFRESH_INTERVAL)
    
    with st.sidebar:
        st.markdown('''
        <div class="sidebar-brand">
            <div class="sidebar-logo">Trading Terminal</div>
            <div class="sidebar-title">NIFTY AI</div>
        </div>
        ''', unsafe_allow_html=True)
        
        selected_page = option_menu(
            menu_title=None,
            options=["Home", "Dashboard", "Lessons", "Settings"],
            icons=['house-door', 'graph-up-arrow', 'book', 'gear'],
            menu_icon="cast",
            default_index=["Home", "Dashboard", "Lessons", "Settings"].index(st.session_state.current_page) if st.session_state.current_page in ["Home", "Dashboard", "Lessons", "Settings"] else 0,
            styles={
                "container": {"padding": "0!important", "background-color": "transparent"},
                "icon": {"color": "#8b949e", "font-size": "1.2rem"}, 
                "nav-link": {"font-size": "0.95rem", "text-align": "left", "margin":"0px", "--hover-color": "#30363d", "color": "#c9d1d9"},
                "nav-link-selected": {"background-color": "rgba(88, 166, 255, 0.1)", "color": "#58a6ff", "border-left": "3px solid #58a6ff"},
            }
        )
        
        if selected_page != st.session_state.current_page:
            st.session_state.current_page = selected_page
            st.rerun()
            
        st.markdown("<br><br>", unsafe_allow_html=True)
        st.markdown('<div class="sidebar-footer">© 2026 QUANTUM AI</div>', unsafe_allow_html=True)

    if st.session_state.current_page == "Home":
        render_home_page()
    elif st.session_state.current_page == "Dashboard":
        render_dashboard_app()
    elif st.session_state.current_page == "Lessons":
        render_lessons_page()
    elif st.session_state.current_page == "Settings":
        st.warning("Settings module under construction.")

if __name__ == "__main__":
    main()
