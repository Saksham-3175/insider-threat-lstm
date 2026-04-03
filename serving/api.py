"""
FastAPI serving layer for InsiderThreatLSTM.

Run from project root:
    uvicorn serving.api:app --reload --port 8000

Startup precomputes risk scores for all test-set users.
"""

import json
import sys
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from torch.utils.data import DataLoader, Dataset

# ── Source path ───────────────────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from model import InsiderThreatLSTM  # noqa: E402

# ── Paths ─────────────────────────────────────────────────────────────────────
_ROOT       = Path(__file__).parent.parent
_DATA_DIR   = _ROOT / "data" / "r4.2"
_OUTPUT_DIR = _ROOT / "outputs"

# ── Pipeline constants (must match pipeline.py) ───────────────────────────────
_NS_PER_DAY = np.int64(86_400 * 1_000_000_000)
_WINDOW_DAYS = 30
_STEP_DAYS   = 1

# ── Alert thresholds ──────────────────────────────────────────────────────────
HIGH_THRESH   = 0.7
MEDIUM_THRESH = 0.4


def _alert_level(score: float) -> str:
    if score > HIGH_THRESH:   return "HIGH"
    if score > MEDIUM_THRESH: return "MEDIUM"
    return "LOW"


# ─────────────────────────────────────────────────────────────────────────────
# Startup helpers — window metadata reconstruction
# ─────────────────────────────────────────────────────────────────────────────
def _user_date_ranges() -> dict:
    """Return {user: (min_ns, max_ns)} across all 5 event sources."""
    parts = []
    for src in ("logon", "device", "file", "email"):
        df = pd.read_csv(_DATA_DIR / f"{src}.csv", usecols=["user", "date"])
        df["date"] = pd.to_datetime(df["date"])
        parts.append(df[["user", "date"]])
    http = pd.read_parquet(_DATA_DIR / "http.parquet", columns=["user", "date"])
    http["date"] = pd.to_datetime(http["date"])
    parts.append(http[["user", "date"]])
    all_ev = pd.concat(parts, ignore_index=True)
    grp = all_ev.groupby("user")["date"]
    return {u: (np.int64(d.min().value), np.int64(d.max().value)) for u, d in grp}


def _window_starts(min_ns: np.int64, max_ns: np.int64) -> np.ndarray:
    first = (min_ns // _NS_PER_DAY) * _NS_PER_DAY
    last  = (max_ns // _NS_PER_DAY) * _NS_PER_DAY
    step  = np.int64(_STEP_DAYS)   * _NS_PER_DAY
    starts, ws = [], first
    while ws <= last:
        starts.append(ws); ws += step
    return np.array(starts, dtype=np.int64)


def _build_user_index(test_users: list, date_ranges: dict) -> dict:
    """
    Returns {user: (lo, hi)} aligned with test .npy arrays.
    Alphabetical iteration matches pipeline.py's groupby order.
    """
    idx, user_index = 0, {}
    for user in sorted(test_users):
        if user not in date_ranges:
            continue
        n = len(_window_starts(*date_ranges[user]))
        user_index[user] = (idx, idx + n)
        idx += n
    return user_index


# ─────────────────────────────────────────────────────────────────────────────
# Inference
# ─────────────────────────────────────────────────────────────────────────────
class _ArrayDataset(Dataset):
    def __init__(self, seq, feat):
        self.seq  = torch.from_numpy(seq).long()
        self.feat = torch.from_numpy(feat).float()
    def __len__(self): return len(self.seq)
    def __getitem__(self, i): return self.seq[i], self.feat[i]


@torch.no_grad()
def _infer(model: InsiderThreatLSTM, seq: np.ndarray, feat: np.ndarray,
           batch: int = 2048) -> np.ndarray:
    loader = DataLoader(_ArrayDataset(seq, feat), batch_size=batch)
    logits = np.concatenate([model(s, f).numpy() for s, f in loader])
    return 1.0 / (1.0 + np.exp(-logits))


# ─────────────────────────────────────────────────────────────────────────────
# App state + lifespan
# ─────────────────────────────────────────────────────────────────────────────
_state: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    print("[startup] Loading model and precomputing risk scores...")

    # Model
    ck    = torch.load(_OUTPUT_DIR / "best_model.pt", map_location="cpu",
                       weights_only=False)
    model = InsiderThreatLSTM(**ck["config"])
    model.load_state_dict(ck["model_state"])
    model.eval()
    _state["model"]      = model
    _state["checkpoint"] = {"epoch": ck["epoch"], "val_prauc": round(ck["val_prauc"], 4)}
    print(f"[startup] Model loaded — epoch {ck['epoch']}, val PR-AUC={ck['val_prauc']:.4f}")

    # Test arrays
    X_seq  = np.load(_OUTPUT_DIR / "X_test_seq.npy")
    X_feat = np.load(_OUTPUT_DIR / "X_test_feat.npy")
    y_test = np.load(_OUTPUT_DIR / "y_test.npy").astype(np.int32)
    splits = json.loads((_OUTPUT_DIR / "user_splits.json").read_text())
    test_users = sorted(splits["test"])

    # Inference
    print(f"[startup] Running inference on {len(y_test):,} test windows...")
    probs = _infer(model, X_seq, X_feat)

    # Window metadata
    print("[startup] Reconstructing window metadata...")
    date_ranges  = _user_date_ranges()
    user_index   = _build_user_index(test_users, date_ranges)

    # Ground truth (optional)
    gt_path = _DATA_DIR / "answers" / "insiders.csv"
    threat_map: dict = {}
    if gt_path.exists():
        gt = pd.read_csv(gt_path)
        gt = gt[gt["dataset"].astype(str) == "4.2"]
        gt["start"] = pd.to_datetime(gt["start"])
        gt["end"]   = pd.to_datetime(gt["end"])
        for _, row in gt.iterrows():
            threat_map[row["user"]] = (str(row["start"].date()), str(row["end"].date()))

    # Per-user score objects
    user_scores: dict = {}
    for user in sorted(test_users):
        if user not in user_index:
            continue
        lo, hi    = user_index[user]
        u_probs   = probs[lo:hi]
        u_labels  = y_test[lo:hi]
        risk      = float(u_probs.max()) if len(u_probs) else 0.0
        n_alerts  = int((u_probs > HIGH_THRESH).sum())

        # Window dates
        starts_ns = _window_starts(*date_ranges[user])
        win_dates = [str(pd.Timestamp(int(ns)).date()) for ns in starts_ns]

        user_scores[user] = {
            "user_id":        user,
            "risk_score":     round(risk, 4),
            "alert_level":    _alert_level(risk),
            "n_windows":      int(len(u_probs)),
            "n_alerts":       n_alerts,
            "n_positive_truth": int(u_labels.sum()),
            "is_threat_user": user in threat_map,
            "threat_start":   threat_map.get(user, (None, None))[0],
            "threat_end":     threat_map.get(user, (None, None))[1],
            "window_scores":  [
                {"date": d, "score": round(float(s), 4)}
                for d, s in zip(win_dates, u_probs.tolist())
            ],
        }

    users_sorted = sorted(user_scores.values(), key=lambda x: x["risk_score"], reverse=True)
    _state["user_scores"]  = user_scores
    _state["users_sorted"] = users_sorted
    _state["probs"]        = probs
    _state["y_test"]       = y_test

    print(f"[startup] Ready — {len(user_scores)} users scored.")
    yield
    _state.clear()


# ─────────────────────────────────────────────────────────────────────────────
# Pydantic models
# ─────────────────────────────────────────────────────────────────────────────
class WindowScore(BaseModel):
    date:  str
    score: float


class UserSummary(BaseModel):
    user_id:           str
    risk_score:        float
    alert_level:       str
    n_windows:         int
    n_alerts:          int
    n_positive_truth:  int
    is_threat_user:    bool
    threat_start:      Optional[str]
    threat_end:        Optional[str]


class UserDetail(UserSummary):
    window_scores: list[WindowScore]


class PredictRequest(BaseModel):
    tokens:   list[int]    # up to 200 event token ints (padded/truncated)
    features: list[float]  # exactly 6 floats


class PredictResponse(BaseModel):
    probability:  float
    alert_level:  str
    raw_logit:    float


# ─────────────────────────────────────────────────────────────────────────────
# Routes
# ─────────────────────────────────────────────────────────────────────────────
app = FastAPI(title="Insider Threat Detection API", version="1.0.0", lifespan=lifespan)


@app.get("/health")
def health():
    return {
        "status":       "ok",
        "model_loaded": "model" in _state,
        "n_users_scored": len(_state.get("user_scores", {})),
        "checkpoint":   _state.get("checkpoint"),
    }


@app.get("/users/scores", response_model=list[UserSummary])
def all_scores():
    if not _state:
        raise HTTPException(503, "Model not loaded")
    return [
        {k: v for k, v in u.items() if k != "window_scores"}
        for u in _state["users_sorted"]
    ]


@app.get("/users/high-risk", response_model=list[UserSummary])
def high_risk():
    if not _state:
        raise HTTPException(503, "Model not loaded")
    return [
        {k: v for k, v in u.items() if k != "window_scores"}
        for u in _state["users_sorted"]
        if u["risk_score"] > HIGH_THRESH
    ]


@app.get("/users/{user_id}", response_model=UserDetail)
def user_detail(user_id: str):
    scores = _state.get("user_scores", {})
    if user_id not in scores:
        raise HTTPException(404, f"User '{user_id}' not found in test set")
    return scores[user_id]


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    if "model" not in _state:
        raise HTTPException(503, "Model not loaded")
    if len(req.features) != 6:
        raise HTTPException(422, "features must have exactly 6 values")

    # Pad / truncate tokens to 200
    tokens = req.tokens[-200:] if len(req.tokens) > 200 else req.tokens
    seq = np.zeros((1, 200), dtype=np.int32)
    seq[0, 200 - len(tokens):] = tokens

    feat = np.array(req.features, dtype=np.float32).reshape(1, 6)
    model = _state["model"]

    with torch.no_grad():
        logit = model(
            torch.from_numpy(seq).long(),
            torch.from_numpy(feat).float(),
        ).item()

    prob = 1.0 / (1.0 + np.exp(-logit))
    return PredictResponse(
        probability=round(prob, 4),
        alert_level=_alert_level(prob),
        raw_logit=round(logit, 4),
    )
