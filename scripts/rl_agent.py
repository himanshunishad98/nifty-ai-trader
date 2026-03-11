"""rl_agent.py
Real-time Online Reinforcement Learning Agent (Deep Q-Network variant).

Architecture:
  - State:   17-dim feature vector (same as MLP/RF)
  - Actions: 0=BUY PUT, 1=NO TRADE, 2=BUY CALL
  - Reward:  +1 for correct direction prediction, -1 for wrong, 0 for NO TRADE
  - Method:  Online Q-learning with experience replay (small buffer)
             Updates happen each time a prediction resolves (every ~15 min)

The agent uses a simple 2-layer Q-network and epsilon-greedy exploration
that decays over time as the agent accumulates experience.
"""
from __future__ import annotations

import pickle
import random
from collections import deque
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim

from scripts.predictive_model import FEATURE_COLUMNS

ROOT      = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT / "models"
RL_PATH   = MODEL_DIR / "rl_agent.pt"
RL_META_PATH = MODEL_DIR / "rl_meta.pkl"
SCALER_PATH  = MODEL_DIR / "mlp_scaler.pkl"   # share scaler with MLP

N_FEATURES = len(FEATURE_COLUMNS)
N_ACTIONS  = 3          # BUY PUT, NO TRADE, BUY CALL
BUFFER_SIZE   = 1000    # experience replay buffer
BATCH_SIZE    = 32
GAMMA         = 0.95    # discount factor
LR            = 5e-4
EPS_START     = 1.0
EPS_MIN       = 0.10
EPS_DECAY     = 0.995   # per update step

ACTION_LABELS = {0: "BUY PUT", 1: "NO TRADE", 2: "BUY CALL"}
SIGNAL_TO_ACTION = {"BUY PUT": 0, "NO TRADE": 1, "BUY CALL": 2}


# ─── Q-Network ───────────────────────────────────────────────────────────────

class QNetwork(nn.Module):
    def __init__(self, n_features: int = N_FEATURES, n_actions: int = N_ACTIONS):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_features, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, n_actions),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ─── Meta (epsilon, step count) ──────────────────────────────────────────────

def _load_meta() -> dict:
    if RL_META_PATH.exists():
        with RL_META_PATH.open("rb") as f:
            return pickle.load(f)
    return {"epsilon": EPS_START, "steps": 0, "total_reward": 0.0, "total_trades": 0}


def _save_meta(meta: dict) -> None:
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    with RL_META_PATH.open("wb") as f:
        pickle.dump(meta, f)


def _load_q_net() -> QNetwork:
    net = QNetwork()
    if RL_PATH.exists():
        net.load_state_dict(torch.load(RL_PATH, map_location="cpu", weights_only=True))
    net.eval()
    return net


def _save_q_net(net: QNetwork) -> None:
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    torch.save(net.state_dict(), RL_PATH)


def _load_scaler():
    if not SCALER_PATH.exists():
        return None
    with SCALER_PATH.open("rb") as f:
        return pickle.load(f)


# ─── Reward function ─────────────────────────────────────────────────────────

def _compute_reward(action: int, signal: str, correct: int | None) -> float:
    """
    +1.0  correct directional call
    -1.0  wrong directional call
     0.0  NO TRADE action (safe fallback)
    -0.3  chose action but outcome unknown
    """
    if correct is None:
        return -0.3
    action_label = ACTION_LABELS[action]
    if action_label == "NO TRADE":
        return 0.0
    if correct == 1:
        return 1.0
    return -1.0


# ─── Online Update ───────────────────────────────────────────────────────────

def update_rl_agent(conn) -> dict:
    """
    Pull recently resolved predictions, build (s, a, r, s') transitions,
    run one round of Q-learning, and save the updated network.
    Returns stats dict.
    """
    # Load resolved rows that haven't been learned from yet
    # We use actual_price_after IS NOT NULL as proxy for resolved
    query = """
        SELECT fs.*, p.correct_prediction, p.signal as predicted_signal
        FROM feature_snapshots fs
        JOIN predictions p ON fs.timestamp = p.timestamp
        WHERE p.correct_prediction IS NOT NULL
        ORDER BY fs.timestamp DESC
        LIMIT 200
    """
    try:
        frame = pd.read_sql_query(query, conn)
    except Exception:
        return {"rl_updated": False, "rl_steps": 0}

    if len(frame) < 10:
        meta = _load_meta()
        return {"rl_updated": False, "rl_steps": meta["steps"], "rl_epsilon": round(meta["epsilon"], 3)}

    scaler = _load_scaler()
    q_net  = _load_q_net()
    target_net = QNetwork()
    target_net.load_state_dict(q_net.state_dict())
    target_net.eval()

    optimizer = optim.Adam(q_net.parameters(), lr=LR)
    meta = _load_meta()

    # Build replay buffer from DB rows
    buffer: deque = deque(maxlen=BUFFER_SIZE)
    for _, row in frame.iterrows():
        try:
            state = np.array([row.get(c, 0.0) for c in FEATURE_COLUMNS], dtype=np.float32)
            if scaler:
                state = scaler.transform(state.reshape(1, -1))[0]
            action = SIGNAL_TO_ACTION.get(str(row.get("predicted_signal", "NO TRADE")), 1)
            reward = _compute_reward(action, str(row.get("predicted_signal", "")), row.get("correct_prediction"))
            # For online: next_state ~ current state (single-step episodes)
            buffer.append((state, action, reward, state.copy()))
        except Exception:
            continue

    if len(buffer) < BATCH_SIZE:
        return {"rl_updated": False, "rl_steps": meta["steps"], "rl_epsilon": round(meta["epsilon"], 3)}

    # Mini-batch Q-learning update (5 gradient steps per call)
    q_net.train()
    criterion = nn.SmoothL1Loss()
    total_loss = 0.0

    for _ in range(5):
        batch = random.sample(list(buffer), min(BATCH_SIZE, len(buffer)))
        states   = torch.tensor(np.array([b[0] for b in batch]), dtype=torch.float32)
        actions  = torch.tensor([b[1] for b in batch], dtype=torch.long)
        rewards  = torch.tensor([b[2] for b in batch], dtype=torch.float32)
        n_states = torch.tensor(np.array([b[3] for b in batch]), dtype=torch.float32)

        with torch.no_grad():
            next_q = target_net(n_states).max(1).values
        target_q = rewards + GAMMA * next_q

        current_q = q_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        loss = criterion(current_q, target_q)

        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(q_net.parameters(), 1.0)
        optimizer.step()
        total_loss += loss.item()

    # Decay epsilon
    meta["epsilon"] = max(EPS_MIN, meta["epsilon"] * EPS_DECAY)
    meta["steps"]  += 5
    meta["total_trades"] = len(frame)

    _save_q_net(q_net)
    _save_meta(meta)

    return {
        "rl_updated":     True,
        "rl_steps":       meta["steps"],
        "rl_epsilon":     round(meta["epsilon"], 3),
        "rl_avg_loss":    round(total_loss / 5, 4),
        "rl_experience":  len(buffer),
    }


# ─── Inference ───────────────────────────────────────────────────────────────

def predict_rl(feature_row: dict) -> dict:
    """
    Returns Q-values and the epsilon-greedy action chosen by the RL agent.
    """
    meta   = _load_meta()
    scaler = _load_scaler()
    q_net  = _load_q_net()

    x = np.array([feature_row.get(c, 0.0) for c in FEATURE_COLUMNS], dtype=np.float32)
    if scaler:
        x = scaler.transform(x.reshape(1, -1))[0]
    x_t = torch.tensor(x, dtype=torch.float32).unsqueeze(0)

    q_net.eval()
    with torch.no_grad():
        q_values = q_net(x_t)[0].numpy()

    greedy_action = int(np.argmax(q_values))
    epsilon = meta["epsilon"]
    exploring = epsilon > EPS_MIN + 0.05   # still in learning phase

    return {
        "rl_signal":     ACTION_LABELS[greedy_action],
        "rl_q_call":     round(float(q_values[2]), 3),
        "rl_q_no_trade": round(float(q_values[1]), 3),
        "rl_q_put":      round(float(q_values[0]), 3),
        "rl_epsilon":    round(epsilon, 3),
        "rl_steps":      meta["steps"],
        "rl_exploring":  exploring,
        "rl_ready":      meta["steps"] >= 10,
    }
