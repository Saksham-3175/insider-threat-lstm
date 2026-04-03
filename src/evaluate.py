"""
evaluate.py — Full post-training evaluation of InsiderThreatLSTM.

Usage (from project root):
    python src/evaluate.py

Requires:
    outputs/best_model.pt
    outputs/X_test_seq.npy, X_test_feat.npy, y_test.npy
    outputs/user_splits.json
    data/r4.2/answers/insiders.csv

Produces:
    outputs/confusion_matrix.png
    outputs/pr_curve.png
    outputs/roc_curve.png
    outputs/threshold_sweep.png
    outputs/time_to_detection.png
    outputs/evaluation_results.json
"""

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score, average_precision_score, classification_report,
    confusion_matrix, f1_score, precision_recall_curve,
    precision_score, recall_score, roc_auc_score, roc_curve, auc,
)
from torch.utils.data import DataLoader, Dataset

# ── Paths ─────────────────────────────────────────────────────────────────────
DATA_DIR   = Path("data/r4.2")
OUTPUT_DIR = Path("outputs")

# ── Pipeline constants (must match pipeline.py exactly) ───────────────────────
_NS_PER_DAY  = np.int64(86_400 * 1_000_000_000)
WINDOW_DAYS  = 30
STEP_DAYS    = 1
THRESHOLD    = 0.5


# ─────────────────────────────────────────────────────────────────────────────
# Model (self-contained so evaluate.py has no src/ import dependency)
# ─────────────────────────────────────────────────────────────────────────────
class InsiderThreatLSTM(nn.Module):
    def __init__(self, vocab_size=16, embed_dim=32, lstm1_hidden=128,
                 lstm2_hidden=64, num_features=6, dropout=0.3):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm1 = nn.LSTM(embed_dim, lstm1_hidden, batch_first=True, dropout=dropout)
        self.lstm2 = nn.LSTM(lstm1_hidden, lstm2_hidden, batch_first=True, dropout=dropout)
        self.classifier = nn.Sequential(
            nn.Linear(lstm2_hidden + num_features, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 1),
        )

    def forward(self, seq_input, feat_input):
        x = self.embedding(seq_input)
        x, _ = self.lstm1(x)
        x, _ = self.lstm2(x)
        x = x[:, -1, :]
        x = torch.cat([x, feat_input], dim=1)
        return self.classifier(x).squeeze(1)


class ThreatDataset(Dataset):
    def __init__(self, seq_array, feat_array, label_array):
        self.seq   = torch.from_numpy(seq_array).long()
        self.feat  = torch.from_numpy(feat_array).float()
        self.label = torch.from_numpy(label_array.astype(np.float32))

    def __len__(self):
        return len(self.label)

    def __getitem__(self, idx):
        return self.seq[idx], self.feat[idx], self.label[idx]


# ─────────────────────────────────────────────────────────────────────────────
# 1. Load model + test data
# ─────────────────────────────────────────────────────────────────────────────
def load_model(path: Path):
    ck  = torch.load(path, map_location="cpu", weights_only=False)
    cfg = ck.get("config", {})
    model = InsiderThreatLSTM(**cfg)
    model.load_state_dict(ck["model_state"])
    model.eval()
    print(f"Loaded checkpoint — epoch {ck['epoch']}  val PR-AUC={ck['val_prauc']:.4f}")
    return model, ck


@torch.no_grad()
def run_inference(model: InsiderThreatLSTM, X_seq, X_feat, batch_size=1024) -> np.ndarray:
    ds     = ThreatDataset(X_seq, X_feat, np.zeros(len(X_seq), dtype=np.float32))
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False)
    logits = np.concatenate([model(s, f).numpy() for s, f, _ in loader])
    return 1.0 / (1.0 + np.exp(-logits))   # sigmoid — inference only


# ─────────────────────────────────────────────────────────────────────────────
# 2. Core metrics
# ─────────────────────────────────────────────────────────────────────────────
def compute_metrics(y_true: np.ndarray, probs: np.ndarray, threshold: float = 0.5) -> dict:
    preds        = (probs >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, preds).ravel()
    return {
        "threshold": threshold,
        "TP": int(tp), "TN": int(tn), "FP": int(fp), "FN": int(fn),
        "accuracy":  round(float(accuracy_score(y_true, preds)),        4),
        "precision": round(float(precision_score(y_true, preds, zero_division=0)), 4),
        "recall":    round(float(recall_score(y_true, preds, zero_division=0)),    4),
        "f1":        round(float(f1_score(y_true, preds, zero_division=0)),        4),
        "roc_auc":   round(float(roc_auc_score(y_true, probs)),                    4),
        "pr_auc":    round(float(average_precision_score(y_true, probs)),          4),
    }


# ─────────────────────────────────────────────────────────────────────────────
# 3. Window metadata reconstruction
#    Replicates pipeline.py's build_windows order so indices align 1-to-1
#    with the saved .npy arrays.
# ─────────────────────────────────────────────────────────────────────────────
def _get_user_date_ranges() -> dict:
    """Return {user: (min_ns, max_ns)} from all 5 event sources."""
    print("  Loading raw event dates to reconstruct window metadata...")
    parts = []
    for src in ("logon", "device", "file", "email"):
        df = pd.read_csv(DATA_DIR / f"{src}.csv", usecols=["user", "date"])
        df["date"] = pd.to_datetime(df["date"])
        parts.append(df[["user", "date"]])

    http = pd.read_parquet(DATA_DIR / "http.parquet", columns=["user", "date"])
    http["date"] = pd.to_datetime(http["date"])
    parts.append(http[["user", "date"]])

    all_ev = pd.concat(parts, ignore_index=True)
    grp    = all_ev.groupby("user")["date"]
    return {
        user: (np.int64(dates.min().value), np.int64(dates.max().value))
        for user, dates in grp
    }


def _window_starts(min_ns: np.int64, max_ns: np.int64) -> np.ndarray:
    """Chronological window-start timestamps (int64 ns) for one user."""
    first_day = (min_ns // _NS_PER_DAY) * _NS_PER_DAY
    last_day  = (max_ns // _NS_PER_DAY) * _NS_PER_DAY
    step_ns   = np.int64(STEP_DAYS)   * _NS_PER_DAY
    win_ns    = np.int64(WINDOW_DAYS) * _NS_PER_DAY
    starts = []
    ws = first_day
    while ws <= last_day:
        starts.append(ws)
        ws += step_ns
    return np.array(starts, dtype=np.int64)


def build_test_metadata(test_users: list, date_ranges: dict) -> tuple:
    """
    Returns:
        user_windows : list of (user, window_start_ns) aligned with test arrays
        user_index   : {user: (lo, hi)} slice into test arrays
    """
    user_windows = []
    user_index   = {}
    for user in sorted(test_users):           # alphabetical = pipeline order
        if user not in date_ranges:
            continue
        min_ns, max_ns = date_ranges[user]
        starts = _window_starts(min_ns, max_ns)
        lo = len(user_windows)
        for s in starts:
            user_windows.append((user, s))
        user_index[user] = (lo, lo + len(starts))
    return user_windows, user_index


# ─────────────────────────────────────────────────────────────────────────────
# 4. Ground truth
# ─────────────────────────────────────────────────────────────────────────────
def load_ground_truth() -> dict:
    gt = pd.read_csv(DATA_DIR / "answers" / "insiders.csv")
    gt = gt[gt["dataset"].astype(str) == "4.2"].copy()
    gt["start"] = pd.to_datetime(gt["start"])
    gt["end"]   = pd.to_datetime(gt["end"])
    return {row["user"]: (row["start"], row["end"]) for _, row in gt.iterrows()}


# ─────────────────────────────────────────────────────────────────────────────
# 5. Per-user threat analysis + time-to-detection
# ─────────────────────────────────────────────────────────────────────────────
def per_user_analysis(
    probs: np.ndarray,
    y_test: np.ndarray,
    user_index: dict,
    threat_users: dict,
    test_users: list,
    threshold: float = THRESHOLD,
) -> dict:
    threat_in_test  = {u for u in test_users if u in threat_users}
    results         = {}

    for user in sorted(threat_in_test):
        lo, hi = user_index[user]
        u_probs  = probs[lo:hi]
        u_labels = y_test[lo:hi]
        t_start, t_end = threat_users[user]

        # Positive windows the model scored >= threshold
        alert_mask = u_probs >= threshold
        n_windows  = len(u_probs)
        n_positive_truth = int(u_labels.sum())

        if alert_mask.any():
            first_alert_idx = int(alert_mask.argmax())
            # window_start for that index
            min_ns, max_ns = _GLOBAL_DATE_RANGES[user]
            starts_ns      = _window_starts(min_ns, max_ns)
            first_alert_ts = pd.Timestamp(int(starts_ns[first_alert_idx]))
            days_before    = (t_start - first_alert_ts).days   # positive = early
            detected       = True
        else:
            first_alert_ts = None
            days_before    = None
            detected       = False

        results[user] = {
            "detected":          detected,
            "n_windows":         n_windows,
            "n_positive_truth":  n_positive_truth,
            "n_alerts":          int(alert_mask.sum()),
            "threat_start":      str(t_start.date()),
            "threat_end":        str(t_end.date()),
            "first_alert_date":  str(first_alert_ts.date()) if first_alert_ts else None,
            "days_before_threat_start": days_before,
        }

    detected   = [u for u, r in results.items() if r["detected"]]
    missed     = [u for u, r in results.items() if not r["detected"]]
    print(f"\n  Threat users in test set : {len(results)}")
    print(f"  Detected (≥1 alert)      : {len(detected)}")
    print(f"  Missed (0 alerts)        : {len(missed)}")
    if detected:
        early = [r["days_before_threat_start"] for u, r in results.items()
                 if r["detected"] and r["days_before_threat_start"] is not None
                 and r["days_before_threat_start"] > 0]
        if early:
            print(f"  Avg days before threat   : {np.mean(early):.1f}")
    return results


# ─────────────────────────────────────────────────────────────────────────────
# 6. Plots
# ─────────────────────────────────────────────────────────────────────────────
def plot_confusion_matrix(y_true, probs, threshold=THRESHOLD):
    preds = (probs >= threshold).astype(int)
    cm    = confusion_matrix(y_true, preds)
    labels = np.array([["TN", "FP"], ["FN", "TP"]])
    annot  = np.array([[f"{lbl}\n{val:,}" for lbl, val in zip(row_lbl, row_val)]
                        for row_lbl, row_val in zip(labels, cm)])

    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(cm, annot=annot, fmt="", cmap="Blues", ax=ax,
                xticklabels=["Pred 0", "Pred 1"],
                yticklabels=["True 0", "True 1"], linewidths=0.5)
    ax.set_title(f"Confusion Matrix  (threshold={threshold})", fontsize=13)
    ax.set_ylabel("Actual"); ax.set_xlabel("Predicted")
    fig.tight_layout()
    path = OUTPUT_DIR / "confusion_matrix.png"
    fig.savefig(path, dpi=130)
    plt.close(fig)
    print(f"  Saved {path}")


def plot_pr_curve(y_true, probs):
    prec, rec, _ = precision_recall_curve(y_true, probs)
    pr_auc       = average_precision_score(y_true, probs)
    baseline     = y_true.mean()

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(rec, prec, lw=2, color="steelblue", label=f"PR-AUC = {pr_auc:.4f}")
    ax.axhline(baseline, color="gray", lw=1, linestyle="--",
               label=f"Random baseline ({baseline:.4f})")
    ax.set_xlabel("Recall"); ax.set_ylabel("Precision")
    ax.set_xlim([0, 1]); ax.set_ylim([0, 1])
    ax.set_title("Precision-Recall Curve", fontsize=13)
    ax.legend(); fig.tight_layout()
    path = OUTPUT_DIR / "pr_curve.png"
    fig.savefig(path, dpi=130)
    plt.close(fig)
    print(f"  Saved {path}")


def plot_roc_curve(y_true, probs):
    fpr, tpr, _ = roc_curve(y_true, probs)
    roc_auc     = auc(fpr, tpr)

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(fpr, tpr, lw=2, color="darkorange", label=f"ROC-AUC = {roc_auc:.4f}")
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="Random")
    ax.set_xlabel("False Positive Rate"); ax.set_ylabel("True Positive Rate")
    ax.set_xlim([0, 1]); ax.set_ylim([0, 1])
    ax.set_title("ROC Curve", fontsize=13)
    ax.legend(); fig.tight_layout()
    path = OUTPUT_DIR / "roc_curve.png"
    fig.savefig(path, dpi=130)
    plt.close(fig)
    print(f"  Saved {path}")


def plot_threshold_sweep(y_true, probs) -> list:
    thresholds = [round(t, 1) for t in np.arange(0.1, 1.0, 0.1)]
    rows = []
    for t in thresholds:
        preds = (probs >= t).astype(int)
        tp = int(((preds == 1) & (y_true == 1)).sum())
        fp = int(((preds == 1) & (y_true == 0)).sum())
        fn = int(((preds == 0) & (y_true == 1)).sum())
        p  = tp / max(tp + fp, 1)
        r  = tp / max(tp + fn, 1)
        f1 = 2 * p * r / max(p + r, 1e-9)
        rows.append({"threshold": t, "precision": p, "recall": r, "f1": f1,
                     "tp": tp, "fp": fp, "fn": fn})

    thresholds_arr = [r["threshold"]  for r in rows]
    prec_arr       = [r["precision"]  for r in rows]
    rec_arr        = [r["recall"]     for r in rows]
    f1_arr         = [r["f1"]         for r in rows]

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(thresholds_arr, prec_arr, "o-", label="Precision", color="steelblue")
    ax.plot(thresholds_arr, rec_arr,  "s-", label="Recall",    color="coral")
    ax.plot(thresholds_arr, f1_arr,   "^-", label="F1",        color="seagreen")
    ax.set_xlabel("Decision Threshold"); ax.set_ylabel("Score")
    ax.set_xlim([0.05, 0.95]); ax.set_ylim([0, 1])
    ax.xaxis.set_major_formatter(mticker.FormatStrFormatter("%.1f"))
    ax.set_title("Precision / Recall / F1 vs Threshold", fontsize=13)
    ax.legend(); fig.tight_layout()
    path = OUTPUT_DIR / "threshold_sweep.png"
    fig.savefig(path, dpi=130)
    plt.close(fig)
    print(f"  Saved {path}")
    return rows


def plot_time_to_detection(user_results: dict):
    users   = sorted(user_results.keys())
    days    = [user_results[u]["days_before_threat_start"] for u in users]
    colors  = []
    for d in days:
        if d is None:
            colors.append("lightgray")
        elif d > 0:
            colors.append("seagreen")
        else:
            colors.append("coral")

    # Replace None with 0 for plotting, label separately
    plot_days = [d if d is not None else 0 for d in days]

    fig, ax = plt.subplots(figsize=(10, max(4, len(users) * 0.55)))
    bars = ax.barh(users, plot_days, color=colors, edgecolor="white", height=0.6)

    ax.axvline(0, color="black", lw=1.2, linestyle="--")
    ax.set_xlabel("Days before threat start date\n(positive = detected early, negative = detected late)")
    ax.set_title("Time-to-Detection per Threat User", fontsize=13)

    # Annotate missed users
    for i, (d, user) in enumerate(zip(days, users)):
        if d is None:
            ax.text(0.5, i, "MISSED", va="center", ha="left",
                    fontsize=8, color="gray", style="italic")

    from matplotlib.patches import Patch
    legend_els = [
        Patch(facecolor="seagreen",  label="Detected early"),
        Patch(facecolor="coral",     label="Detected late"),
        Patch(facecolor="lightgray", label="Missed"),
    ]
    ax.legend(handles=legend_els, loc="lower right")
    fig.tight_layout()
    path = OUTPUT_DIR / "time_to_detection.png"
    fig.savefig(path, dpi=130)
    plt.close(fig)
    print(f"  Saved {path}")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
_GLOBAL_DATE_RANGES: dict = {}   # populated in main, used by per_user_analysis


def main():
    global _GLOBAL_DATE_RANGES

    # ── 1. Load ──────────────────────────────────────────────────────────────
    print("=" * 60)
    print("Step 1 — Loading model and test data")
    print("=" * 60)
    model, ck = load_model(OUTPUT_DIR / "best_model.pt")

    X_test_seq  = np.load(OUTPUT_DIR / "X_test_seq.npy")
    X_test_feat = np.load(OUTPUT_DIR / "X_test_feat.npy")
    y_test      = np.load(OUTPUT_DIR / "y_test.npy").astype(np.int32)

    splits     = json.load(open(OUTPUT_DIR / "user_splits.json"))
    test_users = sorted(splits["test"])

    print(f"Test set: {len(y_test):,} windows | "
          f"{int(y_test.sum()):,} positive ({100*y_test.mean():.3f}%)")

    # ── 2. Inference ─────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("Step 2 — Running inference")
    print("=" * 60)
    probs = run_inference(model, X_test_seq, X_test_feat)
    print(f"Probabilities — min={probs.min():.4f}  max={probs.max():.4f}  "
          f"mean={probs.mean():.4f}  median={np.median(probs):.4f}")

    # ── 3. Core metrics ──────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print(f"Step 3 — Core metrics (threshold={THRESHOLD})")
    print("=" * 60)
    metrics = compute_metrics(y_test, probs, THRESHOLD)
    for k, v in metrics.items():
        print(f"  {k:12s}: {v}")

    print("\nClassification report:")
    preds = (probs >= THRESHOLD).astype(int)
    print(classification_report(y_test, preds, digits=4,
                                target_names=["benign", "threat"]))

    # ── 4. Window metadata ───────────────────────────────────────────────────
    print("=" * 60)
    print("Step 4 — Reconstructing window metadata for per-user analysis")
    print("=" * 60)
    _GLOBAL_DATE_RANGES = _get_user_date_ranges()
    _, user_index = build_test_metadata(test_users, _GLOBAL_DATE_RANGES)

    total_meta = sum(hi - lo for lo, hi in user_index.values())
    if total_meta != len(y_test):
        print(f"  WARNING: metadata windows ({total_meta}) != test array ({len(y_test)}). "
              "Per-user analysis may be inaccurate.")
    else:
        print(f"  Metadata aligned: {total_meta:,} windows ✓")

    # ── 5. Ground truth + per-user analysis ──────────────────────────────────
    print("\n" + "=" * 60)
    print("Step 5 — Per-user threat detection analysis")
    print("=" * 60)
    threat_users = load_ground_truth()
    user_results = per_user_analysis(
        probs, y_test, user_index, threat_users, test_users, THRESHOLD
    )

    print(f"\n  {'User':12s}  {'Detected':8s}  {'Alerts':7s}  {'Days before':12s}  "
          f"{'Threat start':12s}")
    print("  " + "-" * 60)
    for user, r in sorted(user_results.items()):
        d = r["days_before_threat_start"]
        d_str = f"{d:+d}" if d is not None else "—"
        det = "YES" if r["detected"] else "NO"
        print(f"  {user:12s}  {det:8s}  {r['n_alerts']:7d}  {d_str:12s}  "
              f"{r['threat_start']}")

    # ── 6. Threshold sweep ───────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("Step 6 — Threshold sweep")
    print("=" * 60)
    print(f"  {'Threshold':>9}  {'Precision':>10}  {'Recall':>8}  "
          f"{'F1':>8}  {'TP':>6}  {'FP':>7}  {'FN':>6}")
    print("  " + "-" * 60)
    sweep_rows = plot_threshold_sweep(y_test, probs)
    for r in sweep_rows:
        print(f"  {r['threshold']:>9.1f}  {r['precision']:>10.4f}  "
              f"{r['recall']:>8.4f}  {r['f1']:>8.4f}  "
              f"{r['tp']:>6}  {r['fp']:>7}  {r['fn']:>6}")

    # ── 7. Plots ─────────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("Step 7 — Saving plots")
    print("=" * 60)
    plot_confusion_matrix(y_test, probs, THRESHOLD)
    plot_pr_curve(y_test, probs)
    plot_roc_curve(y_test, probs)
    plot_time_to_detection(user_results)

    # ── 8. Save JSON ─────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("Step 8 — Saving evaluation_results.json")
    print("=" * 60)
    results = {
        "checkpoint": {
            "epoch":      ck["epoch"],
            "val_prauc":  round(ck["val_prauc"], 4),
            "val_rocauc": round(ck["val_rocauc"], 4),
        },
        "test_set": {
            "n_windows":  int(len(y_test)),
            "n_positive": int(y_test.sum()),
            "n_negative": int((y_test == 0).sum()),
        },
        "metrics_at_default_threshold": metrics,
        "threshold_sweep": sweep_rows,
        "per_user_threat_analysis": user_results,
    }
    out_path = OUTPUT_DIR / "evaluation_results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"  Saved {out_path}")
    print("\nEvaluation complete.")


if __name__ == "__main__":
    main()
