# %% [markdown]
# # Insider Threat LSTM — Colab Training
# Dual-input LSTM on CERT r4.2. Primary metric: PR-AUC.
# Run time-to-runtime: Cell 1 → 2 → 3 → 4 → 5 → 6.

# %% ── Cell 1: Setup ─────────────────────────────────────────────────────────
from google.colab import drive
drive.mount("/content/drive")

import subprocess
subprocess.run(["pip", "install", "-q", "wandb", "mlflow"], check=True)

import os
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")
if device.type == "cuda":
    print(f"GPU   : {torch.cuda.get_device_name(0)}")
    print(f"VRAM  : {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

import wandb
wandb.login()   # prompts for API key

# %% ── Cell 2: Load data ─────────────────────────────────────────────────────
import numpy as np

# ── USER: set this to your Drive folder containing the .npy outputs ──────────
DRIVE_DIR = "/content/drive/MyDrive/insider-threat-lstm/outputs"
# ─────────────────────────────────────────────────────────────────────────────

def load_split(split: str):
    seq   = np.load(os.path.join(DRIVE_DIR, f"X_{split}_seq.npy"))
    feat  = np.load(os.path.join(DRIVE_DIR, f"X_{split}_feat.npy"))
    label = np.load(os.path.join(DRIVE_DIR, f"y_{split}.npy"))
    return seq, feat, label

X_train_seq,  X_train_feat,  y_train = load_split("train")
X_val_seq,    X_val_feat,    y_val   = load_split("val")
X_test_seq,   X_test_feat,   y_test  = load_split("test")

for name, seq, feat, label in [
    ("train", X_train_seq, X_train_feat, y_train),
    ("val",   X_val_seq,   X_val_feat,   y_val),
    ("test",  X_test_seq,  X_test_feat,  y_test),
]:
    n_pos = int(label.sum())
    n_tot = len(label)
    print(
        f"{name:5s}  seq={seq.shape}  feat={feat.shape}  labels={label.shape}"
        f"  pos={n_pos:,} ({100*n_pos/n_tot:.2f}%)"
    )

# %% ── Cell 3: Model + Dataset ───────────────────────────────────────────────
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


class InsiderThreatLSTM(nn.Module):
    def __init__(
        self,
        vocab_size: int = 16,
        embed_dim: int = 32,
        lstm1_hidden: int = 128,
        lstm2_hidden: int = 64,
        num_features: int = 6,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm1 = nn.LSTM(
            embed_dim, lstm1_hidden,
            batch_first=True, dropout=dropout,
        )
        self.lstm2 = nn.LSTM(
            lstm1_hidden, lstm2_hidden,
            batch_first=True, dropout=dropout,
        )
        self.classifier = nn.Sequential(
            nn.Linear(lstm2_hidden + num_features, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 1),
        )

    def forward(self, seq_input: torch.Tensor, feat_input: torch.Tensor) -> torch.Tensor:
        # seq_input : (batch, 200) long
        # feat_input: (batch, 6)  float32
        # No sigmoid here — BCEWithLogitsLoss expects raw logits
        x = self.embedding(seq_input)           # (batch, 200, 32)
        x, _ = self.lstm1(x)                    # (batch, 200, 128)
        x, _ = self.lstm2(x)                    # (batch, 200, 64)
        x = x[:, -1, :]                         # (batch, 64)
        x = torch.cat([x, feat_input], dim=1)   # (batch, 70)
        x = self.classifier(x)                  # (batch, 1)
        return x.squeeze(1)                     # (batch,) raw logits


class ThreatDataset(Dataset):
    def __init__(self, seq_array, feat_array, label_array):
        self.seq   = torch.from_numpy(seq_array).long()
        self.feat  = torch.from_numpy(feat_array).float()
        self.label = torch.from_numpy(label_array.astype(np.float32))

    def __len__(self):
        return len(self.label)

    def __getitem__(self, idx):
        return self.seq[idx], self.feat[idx], self.label[idx]


model = InsiderThreatLSTM().to(device)
total_params = sum(p.numel() for p in model.parameters())
print(f"Model params: {total_params:,}")

# %% ── Cell 4: Training setup ─────────────────────────────────────────────────
import mlflow
from torch.optim.lr_scheduler import ReduceLROnPlateau

# ── Hyperparams ───────────────────────────────────────────────────────────────
BATCH_SIZE    = 512
LR            = 0.001
MAX_EPOCHS    = 50
ES_PATIENCE   = 7    # early stopping on val PR-AUC
LR_PATIENCE   = 3    # ReduceLROnPlateau patience
LR_FACTOR     = 0.5
POS_WEIGHT_CAP = 20.0

# ── DataLoaders ───────────────────────────────────────────────────────────────
train_ds = ThreatDataset(X_train_seq, X_train_feat, y_train)
val_ds   = ThreatDataset(X_val_seq,   X_val_feat,   y_val)
test_ds  = ThreatDataset(X_test_seq,  X_test_feat,  y_test)

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                          pin_memory=True, num_workers=2)
val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False,
                          pin_memory=True, num_workers=2)
test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False,
                          pin_memory=True, num_workers=2)

# ── pos_weight ────────────────────────────────────────────────────────────────
n_pos  = int(y_train.sum())
n_neg  = len(y_train) - n_pos
pos_weight_val = min(n_neg / max(n_pos, 1), POS_WEIGHT_CAP)
print(f"Train pos={n_pos:,}  neg={n_neg:,}  pos_weight={pos_weight_val:.2f}")

criterion = nn.BCEWithLogitsLoss(
    pos_weight=torch.tensor([pos_weight_val], device=device)
)
optimizer = torch.optim.Adam(model.parameters(), lr=LR)
scheduler = ReduceLROnPlateau(
    optimizer, mode="max", factor=LR_FACTOR, patience=LR_PATIENCE
)

# ── W&B run ───────────────────────────────────────────────────────────────────
run = wandb.init(
    project="insider-threat-lstm",
    config={
        "batch_size":      BATCH_SIZE,
        "lr":              LR,
        "max_epochs":      MAX_EPOCHS,
        "es_patience":     ES_PATIENCE,
        "lr_patience":     LR_PATIENCE,
        "lr_factor":       LR_FACTOR,
        "pos_weight":      pos_weight_val,
        "pos_weight_cap":  POS_WEIGHT_CAP,
        "vocab_size":      16,
        "embed_dim":       32,
        "lstm1_hidden":    128,
        "lstm2_hidden":    64,
        "num_features":    6,
        "dropout":         0.3,
        "total_params":    total_params,
    },
)

# ── MLflow run ────────────────────────────────────────────────────────────────
mlflow.set_experiment("insider-threat-lstm")
mlflow_run = mlflow.start_run()
mlflow.log_params({
    "batch_size":   BATCH_SIZE,
    "lr":           LR,
    "max_epochs":   MAX_EPOCHS,
    "es_patience":  ES_PATIENCE,
    "pos_weight":   round(pos_weight_val, 4),
    "total_params": total_params,
})
print(f"W&B run  : {run.url}")
print(f"MLflow ID: {mlflow_run.info.run_id}")

# %% ── Cell 5: Training loop ──────────────────────────────────────────────────
from sklearn.metrics import average_precision_score, roc_auc_score

BEST_MODEL_PATH = os.path.join(DRIVE_DIR, "best_model.pt")

best_val_prauc  = -1.0
epochs_no_improve = 0
history = {"train_loss": [], "val_loss": [], "val_prauc": [], "val_rocauc": []}


def run_epoch(loader, training: bool):
    model.train(training)
    total_loss = 0.0
    all_logits, all_labels = [], []
    ctx = torch.enable_grad() if training else torch.no_grad()
    with ctx:
        for seq, feat, label in loader:
            seq, feat, label = seq.to(device), feat.to(device), label.to(device)
            logits = model(seq, feat)
            loss   = criterion(logits, label)
            if training:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            total_loss += loss.item() * len(label)
            all_logits.append(logits.detach().cpu())
            all_labels.append(label.cpu())
    all_logits = torch.cat(all_logits).numpy()
    all_labels = torch.cat(all_labels).numpy()
    avg_loss   = total_loss / len(loader.dataset)
    probs      = 1 / (1 + np.exp(-all_logits))   # sigmoid
    prauc      = average_precision_score(all_labels, probs)
    rocauc     = roc_auc_score(all_labels, probs) if all_labels.sum() > 0 else 0.0
    return avg_loss, prauc, rocauc, probs, all_labels


for epoch in range(1, MAX_EPOCHS + 1):
    train_loss, train_prauc, train_rocauc, _, _ = run_epoch(train_loader, training=True)
    val_loss,   val_prauc,   val_rocauc,   val_probs, val_labels = run_epoch(val_loader, training=False)

    scheduler.step(val_prauc)
    current_lr = optimizer.param_groups[0]["lr"]

    # ── Metrics dict ──────────────────────────────────────────────────────────
    metrics = {
        "epoch":        epoch,
        "train/loss":   train_loss,
        "train/prauc":  train_prauc,
        "train/rocauc": train_rocauc,
        "val/loss":     val_loss,
        "val/prauc":    val_prauc,
        "val/rocauc":   val_rocauc,
        "lr":           current_lr,
    }

    wandb.log(metrics)
    mlflow.log_metrics(
        {k.replace("/", "_"): v for k, v in metrics.items()},
        step=epoch,
    )

    history["train_loss"].append(train_loss)
    history["val_loss"].append(val_loss)
    history["val_prauc"].append(val_prauc)
    history["val_rocauc"].append(val_rocauc)

    print(
        f"Epoch {epoch:3d}/{MAX_EPOCHS} | "
        f"train loss={train_loss:.4f} PR-AUC={train_prauc:.4f} | "
        f"val loss={val_loss:.4f} PR-AUC={val_prauc:.4f} ROC-AUC={val_rocauc:.4f} | "
        f"lr={current_lr:.6f}"
    )

    # ── Best model checkpoint ─────────────────────────────────────────────────
    if val_prauc > best_val_prauc:
        best_val_prauc = val_prauc
        epochs_no_improve = 0
        torch.save({
            "epoch":       epoch,
            "model_state": model.state_dict(),
            "val_prauc":   val_prauc,
            "val_rocauc":  val_rocauc,
            "config": {
                "vocab_size": 16, "embed_dim": 32,
                "lstm1_hidden": 128, "lstm2_hidden": 64,
                "num_features": 6,  "dropout": 0.3,
            },
        }, BEST_MODEL_PATH)
        print(f"  ✓ Saved best model (val PR-AUC={val_prauc:.4f})")
    else:
        epochs_no_improve += 1
        if epochs_no_improve >= ES_PATIENCE:
            print(f"Early stopping at epoch {epoch} (no improvement for {ES_PATIENCE} epochs).")
            break

# ── End-of-training summary ───────────────────────────────────────────────────
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

fig, axes = plt.subplots(1, 2, figsize=(12, 4))
epochs_ran = range(1, len(history["train_loss"]) + 1)

axes[0].plot(epochs_ran, history["train_loss"], label="train")
axes[0].plot(epochs_ran, history["val_loss"],   label="val")
axes[0].set_title("Loss"); axes[0].set_xlabel("Epoch"); axes[0].legend()

axes[1].plot(epochs_ran, history["val_prauc"],  label="val PR-AUC")
axes[1].plot(epochs_ran, history["val_rocauc"], label="val ROC-AUC")
axes[1].set_title("Metrics"); axes[1].set_xlabel("Epoch"); axes[1].legend()

curves_path = os.path.join(DRIVE_DIR, "training_curves.png")
fig.savefig(curves_path, dpi=120, bbox_inches="tight")
plt.close(fig)

wandb.log({"training_curves": wandb.Image(curves_path)})
mlflow.log_artifact(curves_path)
mlflow.log_artifact(BEST_MODEL_PATH)
mlflow.log_metric("best_val_prauc", best_val_prauc)
wandb.log({"best_val_prauc": best_val_prauc})
print(f"\nTraining complete. Best val PR-AUC: {best_val_prauc:.4f}")

# %% ── Cell 6: Test evaluation ───────────────────────────────────────────────
from sklearn.metrics import (
    confusion_matrix, classification_report,
    precision_recall_curve, roc_curve, auc,
)

# ── Load best checkpoint ──────────────────────────────────────────────────────
checkpoint = torch.load(BEST_MODEL_PATH, map_location=device)
model.load_state_dict(checkpoint["model_state"])
print(f"Loaded best model from epoch {checkpoint['epoch']} "
      f"(val PR-AUC={checkpoint['val_prauc']:.4f})")

# ── Run test inference ────────────────────────────────────────────────────────
model.eval()
test_logits_list, test_labels_list = [], []
with torch.no_grad():
    for seq, feat, label in test_loader:
        logits = model(seq.to(device), feat.to(device))
        test_logits_list.append(logits.cpu())
        test_labels_list.append(label)

test_logits = torch.cat(test_logits_list).numpy()
test_labels = torch.cat(test_labels_list).numpy()
test_probs  = 1 / (1 + np.exp(-test_logits))   # sigmoid at inference

test_prauc  = average_precision_score(test_labels, test_probs)
test_rocauc = roc_auc_score(test_labels, test_probs)
print(f"\nTest PR-AUC : {test_prauc:.4f}")
print(f"Test ROC-AUC: {test_rocauc:.4f}")

# ── Confusion matrix + report at threshold=0.5 ───────────────────────────────
test_preds = (test_probs >= 0.5).astype(int)
print("\nConfusion matrix (threshold=0.5):")
print(confusion_matrix(test_labels, test_preds))
print("\nClassification report:")
print(classification_report(test_labels, test_preds, digits=4))

# ── PR curve + ROC curve ──────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

prec, rec, _ = precision_recall_curve(test_labels, test_probs)
axes[0].plot(rec, prec, lw=2)
axes[0].set_xlabel("Recall"); axes[0].set_ylabel("Precision")
axes[0].set_title(f"PR Curve  (AUC={test_prauc:.4f})")
axes[0].set_xlim([0, 1]); axes[0].set_ylim([0, 1])

fpr, tpr, _ = roc_curve(test_labels, test_probs)
roc_auc_val = auc(fpr, tpr)
axes[1].plot(fpr, tpr, lw=2)
axes[1].plot([0, 1], [0, 1], "k--", lw=1)
axes[1].set_xlabel("FPR"); axes[1].set_ylabel("TPR")
axes[1].set_title(f"ROC Curve  (AUC={roc_auc_val:.4f})")
axes[1].set_xlim([0, 1]); axes[1].set_ylim([0, 1])

curves_test_path = os.path.join(DRIVE_DIR, "test_curves.png")
fig.savefig(curves_test_path, dpi=120, bbox_inches="tight")
plt.close(fig)
wandb.log({"test/pr_roc_curves": wandb.Image(curves_test_path)})
mlflow.log_artifact(curves_test_path)

# ── Threshold sweep ───────────────────────────────────────────────────────────
print(f"\n{'Threshold':>10}  {'Precision':>10}  {'Recall':>10}  {'F1':>10}  {'TP':>6}  {'FP':>6}")
print("-" * 62)
sweep_results = []
for thresh in [round(t, 1) for t in np.arange(0.1, 1.0, 0.1)]:
    preds  = (test_probs >= thresh).astype(int)
    tp = int(((preds == 1) & (test_labels == 1)).sum())
    fp = int(((preds == 1) & (test_labels == 0)).sum())
    fn = int(((preds == 0) & (test_labels == 1)).sum())
    prec_t = tp / max(tp + fp, 1)
    rec_t  = tp / max(tp + fn, 1)
    f1_t   = 2 * prec_t * rec_t / max(prec_t + rec_t, 1e-9)
    print(f"{thresh:>10.1f}  {prec_t:>10.4f}  {rec_t:>10.4f}  {f1_t:>10.4f}  {tp:>6}  {fp:>6}")
    sweep_results.append({"threshold": thresh, "precision": prec_t,
                          "recall": rec_t, "f1": f1_t})

# ── Log final test metrics ────────────────────────────────────────────────────
wandb.log({
    "test/prauc":  test_prauc,
    "test/rocauc": test_rocauc,
})
mlflow.log_metrics({
    "test_prauc":  test_prauc,
    "test_rocauc": test_rocauc,
})
mlflow.end_run()
wandb.finish()
print("\nDone.")
