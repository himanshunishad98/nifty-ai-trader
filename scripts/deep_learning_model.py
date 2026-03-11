"""deep_learning_model.py
4-layer PyTorch MLP for NIFTY direction prediction.
Trains on the same feature_snapshots table as the Random Forest.
Architecture: Input(17) → 128 → 64 → 32 → 3 classes (CALL / NO TRADE / PUT)
Features: Dropout, BatchNorm, ReLU. Saves to models/mlp.pt
"""
from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from scripts.predictive_model import FEATURE_COLUMNS

ROOT      = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT / "models"
MLP_PATH  = MODEL_DIR / "mlp.pt"
SCALER_PATH = MODEL_DIR / "mlp_scaler.pkl"

# Map signal labels to class indices
SIGNAL_MAP = {"BUY CALL": 2, "NO TRADE": 1, "BUY PUT": 0}
IDX_TO_LABEL = {0: "BUY PUT", 1: "NO TRADE", 2: "BUY CALL"}

N_FEATURES = len(FEATURE_COLUMNS)


# ─── Model Architecture ──────────────────────────────────────────────────────

class MarketMLP(nn.Module):
    """Multi-Layer Perceptron for 3-class market signal prediction."""

    def __init__(self, n_features: int = N_FEATURES, dropout: float = 0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_features, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(dropout / 2),

            nn.Linear(32, 3),   # BUY PUT / NO TRADE / BUY CALL
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ─── Helpers ─────────────────────────────────────────────────────────────────

def _signal_to_class(signal: str) -> int:
    return SIGNAL_MAP.get(signal, 1)


def _load_mlp() -> MarketMLP | None:
    if not MLP_PATH.exists():
        return None
    model = MarketMLP()
    model.load_state_dict(torch.load(MLP_PATH, map_location="cpu", weights_only=True))
    model.eval()
    return model


def _load_scaler():
    if not SCALER_PATH.exists():
        return None
    with SCALER_PATH.open("rb") as f:
        return pickle.load(f)


def _save_scaler(scaler) -> None:
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    with SCALER_PATH.open("wb") as f:
        pickle.dump(scaler, f)


# ─── Training ────────────────────────────────────────────────────────────────

def maybe_retrain_mlp(conn, min_rows: int = 40) -> dict:
    """
    Train / retrain the MLP on resolved feature snapshots.
    Returns a status dict compatible with the existing model_status payload.
    """
    from sklearn.preprocessing import StandardScaler

    query = """
        SELECT * FROM feature_snapshots
        WHERE signal IS NOT NULL
          AND signal != 'NO TRADE'
          AND target IS NOT NULL
    """
    frame = pd.read_sql_query(query, conn)

    if len(frame) < min_rows:
        return {"dl_trained": False, "dl_samples": len(frame)}

    X = frame[FEATURE_COLUMNS].fillna(0.0).values.astype(np.float32)
    y = frame["signal"].map(SIGNAL_MAP).fillna(1).values.astype(np.int64)

    # Normalise features
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    _save_scaler(scaler)

    X_t = torch.tensor(X, dtype=torch.float32)
    y_t = torch.tensor(y, dtype=torch.long)

    dataset  = TensorDataset(X_t, y_t)
    loader   = DataLoader(dataset, batch_size=min(32, len(dataset)), shuffle=True, drop_last=False)

    model     = MarketMLP()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()

    model.train()
    for _ in range(50):          # 50 epochs — fast on CPU for small datasets
        for xb, yb in loader:
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), MLP_PATH)

    return {"dl_trained": True, "dl_samples": len(frame)}


# ─── Inference ───────────────────────────────────────────────────────────────

def predict_dl(feature_row: dict) -> dict:
    """
    Returns DL model probabilities for BUY CALL / NO TRADE / BUY PUT.
    Falls back gracefully if model hasn't been trained yet.
    """
    model  = _load_mlp()
    scaler = _load_scaler()

    if model is None or scaler is None:
        return {
            "dl_call":     0.333,
            "dl_no_trade": 0.334,
            "dl_put":      0.333,
            "dl_signal":   "Accumulating data...",
            "dl_ready":    False,
        }

    x = np.array([[feature_row.get(c, 0.0) for c in FEATURE_COLUMNS]], dtype=np.float32)
    x = scaler.transform(x)
    x_t = torch.tensor(x, dtype=torch.float32)

    model.eval()
    with torch.no_grad():
        logits = model(x_t)
        probs  = torch.softmax(logits, dim=1)[0].numpy()

    predicted_class = int(np.argmax(probs))
    return {
        "dl_put":      round(float(probs[0]), 3),
        "dl_no_trade": round(float(probs[1]), 3),
        "dl_call":     round(float(probs[2]), 3),
        "dl_signal":   IDX_TO_LABEL[predicted_class],
        "dl_ready":    True,
        "dl_confidence": round(float(np.max(probs)) * 100, 1),
    }
