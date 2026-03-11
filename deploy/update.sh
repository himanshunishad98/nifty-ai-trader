#!/bin/bash
# ─────────────────────────────────────────────────────────────
#  NIFTY AI Trader — Maintenance & Update Script
#  Usage:
#    bash deploy/update.sh           # Pull latest code + restart
#    bash deploy/update.sh retrain   # Also re-run backtest
#    bash deploy/update.sh logs      # View live logs
# ─────────────────────────────────────────────────────────────
APP_DIR="$HOME/nifty-ai-trader"
cd "$APP_DIR"

case "${1:-update}" in
    update)
        echo "[update] Pulling latest code..."
        git pull
        source .venv/bin/activate
        pip install -r requirements.txt -q
        sudo systemctl restart nifty-trader
        echo "[update] Done. Status:"
        sudo systemctl status nifty-trader --no-pager -l
        ;;
    retrain)
        echo "[retrain] Re-running backtest to refresh models..."
        source .venv/bin/activate
        .venv/bin/python -X utf8 scripts/backtest_engine.py --years 10 --no-cache
        sudo systemctl restart nifty-trader
        echo "[retrain] Models retrained and service restarted."
        ;;
    logs)
        journalctl -u nifty-trader -f --no-pager
        ;;
    status)
        sudo systemctl status nifty-trader --no-pager -l
        ;;
    stop)
        sudo systemctl stop nifty-trader
        echo "Service stopped."
        ;;
    start)
        sudo systemctl start nifty-trader
        echo "Service started."
        ;;
    *)
        echo "Usage: $0 [update|retrain|logs|status|stop|start]"
        ;;
esac
