#!/bin/bash
# ─────────────────────────────────────────────────────────────
#  NIFTY AI Trader — Oracle Cloud Setup Script
#  Run as: bash deploy/setup_oracle.sh
#  Ubuntu 22.04 ARM (A1.Flex — 4 OCPU, 24 GB RAM)
# ─────────────────────────────────────────────────────────────
set -e

APP_DIR="$HOME/nifty-ai-trader"
PYTHON="python3.11"

echo "=============================================="
echo "  NIFTY AI Trader -- Oracle Cloud Setup"
echo "=============================================="

# ── 1. System packages ────────────────────────────────────────
echo "[1/7] Installing system packages..."
sudo apt-get update -qq
sudo apt-get install -y --no-install-recommends \
    python3.11 python3.11-venv python3.11-dev python3-pip \
    git curl wget nginx supervisor \
    build-essential libffi-dev libssl-dev \
    libxml2-dev libxslt1-dev \
    unzip htop

echo "  Python: $($PYTHON --version)"

# ── 2. Clone / update repo ────────────────────────────────────
echo "[2/7] Setting up app directory..."
if [ -d "$APP_DIR" ]; then
    echo "  Directory exists — pulling latest..."
    cd "$APP_DIR" && git pull
else
    echo "  Cloning repository..."
    # Replace with your actual GitHub repo URL:
    git clone https://github.com/YOUR_USERNAME/nifty-ai-trader.git "$APP_DIR"
    cd "$APP_DIR"
fi

# ── 3. Python virtual environment ────────────────────────────
echo "[3/7] Creating Python virtual environment..."
$PYTHON -m venv "$APP_DIR/.venv"
source "$APP_DIR/.venv/bin/activate"
pip install --upgrade pip -q

# ── 4. Install dependencies ───────────────────────────────────
echo "[4/7] Installing Python dependencies (may take 5-10 min)..."
pip install -r requirements.txt -q
echo "  Done."

# ── 5. Create data directories ───────────────────────────────
echo "[5/7] Creating required directories..."
mkdir -p "$APP_DIR/data/historical_cache"
mkdir -p "$APP_DIR/models"
mkdir -p "$APP_DIR/logs"
echo "  Directories ready."

# ── 6. Run backtest to pre-train models ──────────────────────
echo "[6/7] Running initial backtest to train AI models..."
echo "  This fetches 10 years of NIFTY data and trains RF + MLP + RL."
echo "  (Takes ~3-5 minutes)"
cd "$APP_DIR"
.venv/bin/python -X utf8 scripts/backtest_engine.py --years 10 --quiet
echo "  Models trained successfully."

# ── 7. Install systemd service ───────────────────────────────
echo "[7/7] Installing systemd service..."
sudo cp deploy/nifty-trader.service /etc/systemd/system/
sudo sed -i "s|__HOME__|$HOME|g" /etc/systemd/system/nifty-trader.service
sudo systemctl daemon-reload
sudo systemctl enable nifty-trader
sudo systemctl start nifty-trader
sleep 2
sudo systemctl status nifty-trader --no-pager

echo ""
echo "=============================================="
echo "  Setup Complete!"
echo "  App running at: http://$(curl -s ifconfig.me):8501"
echo "  Logs: journalctl -u nifty-trader -f"
echo "=============================================="
