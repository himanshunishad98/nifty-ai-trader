# NIFTY AI Trader Architecture

This document describes the internal architecture of the NIFTY AI Trader terminal, detailing the layers, asynchronous data ingestion pipeline, and centralized UI configuration constraints.

## Overview

The terminal is structured into four primary layers:
1. **Data Ingestion** (`scripts/data_engine.py`, `scripts/market_data.py`, etc.)
2. **Analysis Engine** (`scripts/analysis_engine.py`)
3. **Prediction & Signals** (`scripts/signal_engine.py`, `scripts/predictive_model.py`)
4. **Presentation (Terminal UI)** (`dashboard/dashboard.py`)

## 1. Asynchronous Data Pipeline
Fetching real-time data sequentially (market candles, VIX, sentiment, and the intensive NSE options chain) creates significant I/O latency. To resolve this, **`data_engine.py`** leverages Python's `asyncio` (and specifically `asyncio.to_thread()` and `ThreadPoolExecutor`) to achieve Concurrent I/O execution. 

This means `fetch_market_data`, `fetch_india_vix_data`, `fetch_option_chain`, and `fetch_news_sentiment` all execute simultaneously. 
The GUI latency drops to the speed of the slowest individual HTTP request rather than the sum of all requests.

## 2. Global Configuration (`config.yaml`)
To adapt the indicator weights, UI themes, or symbol inputs dynamically without rewriting python logic, we extract hardcoded magic numbers into `config.yaml`.
The file `scripts/config.py` acts as the parser, exposing constants like `UI_CONFIG` and `ANALYSIS_CONFIG`.
- **UI Refresh & Styling**: `dashboard.py` reads `REFRESH_INTERVAL` and dynamic `THEME` colors (e.g., specific HEX codes for metrics and `.terminal-card` backgrounds) directly from config.
- **Contract / Signal Weighting**: Variable sizes (like the NIFTY `CONTRACT_SIZE=75`) or AI module weights are injected into the calculations at runtime.

## 3. UI Terminal Rendering
Built on top of Streamlit, the dashboard replaces standard browser-reloads with the seamless `streamlit_autorefresh` plugin or native loops, preventing the interface from flashing. Custom HTML/CSS injection gives the `dashboard.py` rigid flexbox and css-grid `.terminal-card` components, effectively mimicking a proprietary institutional trading desk.

## Summary of Changes
- **`config.yaml`**: The new central brain for parameters.
- **`scripts/config.py`**: Loader module.
- **`scripts/data_engine.py`**: Rewritten `build_snapshot` for asynchronous, non-blocking I/O.
- **`dashboard/dashboard.py`**: Streamlined CSS blocks, removed JS layout flashing, and adopted `st-autorefresh`.
