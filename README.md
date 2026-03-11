# NIFTY AI Trader

AI-powered NIFTY 50 options intelligence dashboard built with Streamlit.

## Features
- Live NIFTY 50 signal engine (BUY CALL / NO TRADE / BUY PUT)
- 3-model AI Ensemble: Random Forest + Deep Learning MLP + RL DQN
- Global market intelligence (S&P 500, Nikkei, Crude, Gold, USD/INR, DXY, US10Y)
- News sentiment analysis (live web + RSS)
- Option chain analysis (PCR, GEX, Gamma Flip)
- Backtesting engine — self-trains on 10 years of NIFTY data

## Quick Deploy (Oracle Cloud)

```bash
git clone https://github.com/himanshunishad98/nifty-ai-trader.git
cd nifty-ai-trader
bash deploy/setup_oracle.sh
```

## Local Run

```bash
python -m venv .venv
.venv\Scripts\activate   # Windows
pip install -r requirements.txt
streamlit run dashboard/dashboard.py
```
