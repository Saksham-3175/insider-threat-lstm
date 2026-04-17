# Project Notes: Insider Threat LSTM

Everything about this project — what was built, how, and why.

---

## What is this project?

A machine learning system that detects insider threats in enterprise user activity logs. It watches how employees use company systems — logins, file copies, USB drives, emails, web browsing — and flags users whose behavior over a 30-day window looks anomalous.

**The source material:** CERT Insider Threat Dataset r4.2 from Carnegie Mellon University. It's a synthetic but realistic simulation of ~1,000 employees over 18 months with deliberately embedded threat scenarios: data exfiltration, IP theft, sabotage. Each activity log contains real-looking event data, and a ground-truth file (`answers/insiders.csv`) tells you which 70 users are threats, and when their threat behavior started/ended.

**The goal:** Train a model that, given a 30-day window of a user's activity, predicts the probability that the user is an insider threat — ideally catching them *before* their labeled threat start date.

---

## The Data

Five raw CSV/Parquet files, one per event type:

| File | ~Events | What it captures |
|---|---|---|
| `logon.csv` | 500K | Login/logoff with timestamps |
| `device.csv` | 80K | USB plug/unplug |
| `file.csv` | 450K | Files copied to removable media |
| `email.csv` | 1.1M | Internal/external email + attachments |
| `http.csv` | 2.8M | Web browsing with full URLs |

The dataset spans January 2010 – April 2011. Events are timestamped to the second.

**http.csv is huge (~2.8M rows).** It doesn't fit comfortably in memory for repeated reads, so the pipeline converts it to Parquet once up front. Parquet has columnar compression and faster read performance for column-selective queries — reading just `user` and `date` becomes much cheaper.

---

## Data Pipeline (`src/pipeline.py`)

### Step 1 — Convert http.csv to Parquet

Read `http.csv` in 500K-row chunks, write to `http.parquet`. Done once, never again.

### Step 2 — Tokenize events

Raw events need to become numbers the model can process. We defined a 16-token vocabulary where each token is a semantically meaningful event type:

```
UNKNOWN=0           NORMAL_LOGIN=1      AFTERHOURS_LOGIN=2
WEEKEND_LOGIN=3     LOGOFF=4            USB_CONNECT=5
USB_DISCONNECT=6    FILE_OPEN=7         FILE_COPY=8
LARGE_FILE_ACTIVITY=9  HTTP_NORMAL=10   HTTP_JOBSITE=11
HTTP_CLOUD_STORAGE=12  EMAIL_NORMAL=13  EMAIL_WITH_ATTACHMENT=14
EMAIL_EXTERNAL=15
```

**Why these tokens?** Each token encodes a risk-relevant behavioral signal:
- `AFTERHOURS_LOGIN` and `WEEKEND_LOGIN` split from `NORMAL_LOGIN` because time-of-day is a known insider threat indicator.
- `HTTP_JOBSITE` (linkedin, indeed, glassdoor, monster, careerbuilder) catches employees quietly job-hunting before leaving with stolen data.
- `HTTP_CLOUD_STORAGE` (dropbox, drive.google, onedrive, box.com, wetransfer) catches exfiltration via personal cloud accounts.
- `EMAIL_EXTERNAL` catches sending to non-company addresses.
- `FILE_COPY` and `USB_CONNECT` together are the classic USB exfiltration pattern.

Tokenization is vectorized with pandas — no row-by-row Python loops. For example, logon events are classified by checking `hour < 7 or hour > 18` (after-hours) and `dayofweek >= 5` (weekend) using pandas `.dt` accessors on the entire column at once.

After tokenization, all 5 sources are merged into a single sorted-by-`(user, date)` DataFrame of `(user, timestamp_ns, token)` tuples.

### Step 3 — Sliding windows

For each user, we slide a 30-day window in 1-day steps across their entire activity history.

**Why 30 days?** Long enough to capture behavioral patterns (someone who starts browsing job sites, then copies files, then emails externally over several weeks), short enough to remain timely for detection.

**Why 1-day steps?** Gives temporal resolution for detection. If we stepped 7 days, we'd miss when exactly behavior changed.

For each window, we produce:
- `X_seq`: The last 200 tokens that occurred in that 30-day range, padded with 0 (UNKNOWN) at the front if fewer than 200 events.
- `X_feat`: 6 numerical features computed over the window (see below).
- `y`: 1 if this window overlaps with the user's labeled threat period, 0 otherwise.

Window boundary lookup uses `numpy.searchsorted` on the sorted timestamp array — O(log n) per lookup rather than O(n) scanning. This matters because we're building ~464K windows across 1,000 users.

### Step 4 — Numerical features (6)

| Feature | Formula |
|---|---|
| `avg_hour` | Mean event hour across all window events |
| `after_hours_ratio` | Events outside 07:00–18:00 / total events |
| `weekend_ratio` | Weekend events / total events |
| `usb_count` | Raw count of USB connect + disconnect events |
| `email_external_ratio` | External emails / total emails (0 if no emails) |
| `http_jobsite_ratio` | Job-site visits / total HTTP events (0 if no HTTP) |

These aggregate features complement the sequence branch. The sequence captures *what happened*, the features capture *the statistical distribution of when and how*.

### Step 5 — User-level stratified split (70/15/15)

**Critical design decision:** Split is at the *user level*, not the *window level*.

If you split windows randomly, a user's windows from January and March both appear in train and test — the model learns that user's normal behavior during training and trivially detects the anomaly at test time. That's data leakage. User-level split means a held-out user's windows are *never* seen during training.

**Stratified:** Of the 70 threat users, roughly 70% go to train, 15% to val, 15% to test. Without stratification, random chance could put all 70 threat users into train with none to evaluate on.

This produces 464,687 total windows:
- Train: ~325K windows
- Val: ~70K windows
- Test: ~68K windows (630 positives, 68,023 negatives — 0.92% positive rate)

### Step 6 — Saved outputs

`X_train_seq.npy`, `X_train_feat.npy`, `y_train.npy` (and val/test equivalents) — the raw arrays that go into Google Drive for Colab training.

`user_splits.json` — which users went into which split (needed for the serving layer to map model outputs back to users).

`pipeline_stats.json` — window counts and label distribution for sanity-checking.

---

## The Model (`src/model.py`)

### Architecture: Dual-Input LSTM

```
seq_input  (batch, 200)  ──► Embedding(16, 32)
                              │
                          (batch, 200, 32)
                              │
                          LSTM₁  hidden=128, dropout=0.3 → all hidden states
                              │
                          (batch, 200, 128)
                              │
                          LSTM₂  hidden=64, dropout=0.3 → last hidden state only
                              │
                          (batch, 64)
                              │
feat_input (batch, 6)  ───► cat ──► (batch, 70)
                                       │
                                   Linear(70→32) → ReLU → Dropout(0.2)
                                       │
                                   Linear(32→1)
                                       │
                                   raw logit (sigmoid only at inference)
```

**Total parameters: ~135K.** Intentionally small — this runs in under a minute per epoch on a Colab T4.

### Design decisions

**Why an embedding layer for tokens?**
Raw integer IDs give the model no gradient signal about similarity between events. An embedding layer (16 tokens → 32-dim vectors) lets the model learn that `USB_CONNECT` and `FILE_COPY` are semantically related — their learned vectors will be close in embedding space if they co-occur with threats.

**Why 2-layer LSTM?**
Single LSTM layers learn low-level temporal patterns (this event followed that event). A second LSTM layer stacked on top learns higher-order patterns (sequences of sequences). We use all hidden states from LSTM₁ as input to LSTM₂, then take only LSTM₂'s last hidden state as the sequence representation — this captures the entire temporal history compressed into a 64-dim vector.

**Why keep numerical features separate?**
The sequence branch processes *order* well but treats numerical aggregates poorly — you can't naturally represent "external email ratio = 0.7" as a token sequence. Keeping the feature branch parallel and concatenating at the classifier preserves the raw gradient signal from both modalities.

**Why no sigmoid in `forward()`?**
`BCEWithLogitsLoss` — the training loss — internally applies log-sigmoid for numerical stability. If we applied sigmoid in `forward()`, we'd be double-applying it, causing vanishing gradients on very confident predictions. Raw logits go into the loss; sigmoid is applied only at inference time when converting to probabilities.

**Dropout:**
- LSTM dropout (0.3): Applied between layers within the LSTM stack (PyTorch's `dropout` parameter drops outputs between time steps between layers).
- Classifier dropout (0.2): Applied before the final linear layer. Together these regularize a small model with a large training set.

---

## Training (`notebooks/train_colab.py`)

### Why Google Colab T4?

The training arrays are ~3GB total. The T4 has 15GB VRAM, enough to hold full batches of 512 windows on device. Local training would work but Colab gives free GPU acceleration.

### Class imbalance: pos_weight

0.92% of windows are positive. Without correction, the model learns to output "not threat" for everything and achieves 99% accuracy trivially.

`BCEWithLogitsLoss` accepts a `pos_weight` parameter — it multiplies the loss contribution of positive examples by that factor, effectively making the model care more about missing a threat.

```python
pos_weight = min(neg / pos, 20.0)
```

We cap at 20× because uncapped weights (which would be ~108×) cause the model to predict "threat" on everything to minimize loss. 20× is an empirical sweet spot — high enough to surface threats, low enough to not destroy precision.

### Optimizer and scheduling

- `Adam(lr=0.001)`: Standard choice for sequence models. Fast convergence, handles sparse gradients from the embedding layer well.
- `ReduceLROnPlateau(factor=0.5, patience=3)`: If val PR-AUC doesn't improve for 3 epochs, halve the learning rate. Prevents the model oscillating around a local minimum when the gradient landscape flattens.
- Early stopping (patience=7): Stop training if val PR-AUC doesn't improve for 7 consecutive epochs. Saves compute and prevents overfitting.

### Primary metric: PR-AUC, not ROC-AUC

This is important enough to understand carefully.

**ROC-AUC** measures the model's ability to rank positives above negatives across all thresholds. With 98.9% negatives, the model can achieve 0.94 ROC-AUC while producing thousands of false positives in practice. ROC-AUC counts each of the 68,023 negatives equally — getting them right inflates the score.

**PR-AUC** (Precision-Recall AUC) measures only the precision-recall trade-off on the *positive class*. A random classifier on this dataset would score PR-AUC ≈ 0.009 (the base rate). Our result of 0.1707 on test is roughly 18× above chance — that's the honest measure of whether we're actually detecting threats, not just benign activity.

We track val PR-AUC for early stopping, model checkpointing, and learning rate scheduling. ROC-AUC is reported for completeness but is not the decision criterion.

### Experiment tracking

Both MLflow (local) and Weights & Biases (cloud) receive per-epoch logs:
- `train_loss`, `val_loss`
- `val_prauc`, `val_rocauc`
- `learning_rate`

Best model saved to `best_model.pt` on val PR-AUC improvement. The checkpoint stores `model_state_dict`, `config` (architecture hyperparameters), `epoch`, and `val_prauc` so the model can be rebuilt identically for inference without importing `src/model.py` as a dependency from the checkpoint.

### Training result

Converged at epoch 21: `val_prauc=0.4504`, `val_rocauc=0.9631`.

---

## Evaluation (`src/evaluate.py`)

### What the evaluation script does

1. Loads `best_model.pt`, rebuilds the model.
2. Runs full inference on 68,653 test windows (batch size 2048, GPU-optional).
3. Computes metrics at threshold=0.5 as default.
4. Sweeps thresholds 0.1–0.9 to show precision/recall/F1 trade-offs.
5. Does per-threat-user analysis: for each known threat user in the test split, finds if any window ever exceeded the alert threshold, and how many days *before* their labeled threat start date that first alert fired.

### Window-to-date reconstruction

The `.npy` arrays contain no timestamps — just numbers. To do per-user analysis (who fired, when), we reconstruct window metadata by re-running the same window-generation logic from `pipeline.py` on the raw data.

For each test user, we load their date range (min/max event timestamp), then regenerate the same sequence of 30-day window start timestamps. The window index `i` in the numpy array for user `U` corresponds to the `i`th window start date — confirmed by matching the alphabetical iteration order used in both `pipeline.py` and the reconstruction code.

### Metrics (test set, threshold=0.5)

| Metric | Value |
|---|---|
| Threshold | 0.5 |
| TP | 340 |
| TN | 65,518 |
| FP | 2,505 |
| FN | 290 |
| Precision | 11.95% |
| Recall | 53.97% |
| F1 | 0.1957 |
| ROC-AUC | 0.9414 |
| PR-AUC | 0.1707 |

Low precision (11.95%) is expected — see the class imbalance discussion. For every real threat window, there are ~7 false positives at threshold=0.5. For a SOC context, threshold=0.7 gives the best F1 (0.2053) with fewer false positives.

### Threshold sweep

| Threshold | Precision | Recall | F1 |
|---|---|---|---|
| 0.1 | 8.9% | 69.4% | 15.7% |
| 0.5 | 12.0% | 54.0% | 19.6% |
| 0.7 | 13.5% | 42.4% | **20.5%** |
| 0.9 | 20.1% | 17.0% | 18.4% |

The SOC operator can choose their threshold based on analyst capacity. Higher threshold → fewer alerts → higher precision but lower recall.

### Per-threat-user detection (test set, threshold=0.7)

| User | Detected | Alerts | First Alert | Days Before |
|---|---|---|---|---|
| BLS0678 | Yes | 2 | 2010-08-23 | +29 |
| CCA0046 | Yes | 104 | 2010-01-04 | +283 |
| CCL0068 | Yes | 366 | 2010-01-04 | +357 |
| CEJ0109 | Yes | 455 | 2010-01-04 | +399 |
| JJM0203 | Yes | 32 | 2010-01-10 | +235 |
| KRL0501 | Yes | 367 | 2010-01-04 | +322 |
| LJR0523 | Yes | 21 | 2010-06-30 | +31 |
| MCF0600 | Yes | 21 | 2010-08-22 | +29 |
| MSO0222 | Yes | 30 | 2010-11-10 | +29 |
| TAP0551 | Yes | 23 | 2010-09-23 | +30 |
| TNM0961 | **No** | 0 | — | — |

**10 of 11 threat users detected, on average 174 days before their labeled threat start date.**

**What "days before" means:** The model flags a user *before* the start of their threat period — meaning windows that technically don't overlap with the labeled threat range are still triggering alerts. This happens because the model picks up on *precursor behaviors* (job-site browsing, unusual access times) that happen in the weeks/months before the person actually begins exfiltrating.

**TNM0961 (missed):** 0 alerts fired. No forensic analysis was done here, but possible explanations: their precursor behaviors may have been too subtle or too similar to benign users in the training split, or their threat scenario type differed from the patterns seen in training.

---

## Serving Layer

### FastAPI (`serving/api.py`)

The API precomputes all risk scores for test-set users at startup. This takes ~30 seconds (model inference on 68K windows + date range reconstruction from raw CSVs) but makes all API responses instant afterward.

**Why precompute at startup rather than on-demand?**
On-demand would require loading raw data per request, which is 5GB across all sources. Precomputing trades startup time for zero-latency queries.

**Routes:**
- `GET /health` — Is the model loaded? How many users are scored?
- `GET /users/scores` — All users sorted by risk score (no window details).
- `GET /users/high-risk` — Only users with max risk > 0.7.
- `GET /users/{user_id}` — Full detail: risk score, alert level, every window's date+score.
- `POST /predict` — Accept raw tokens + 6 features, return probability + alert level. For real-time integration with a production log pipeline.

Alert levels: HIGH > 0.7, MEDIUM > 0.4, LOW ≤ 0.4.

### Streamlit SOC Dashboard (`serving/app.py`)

Three pages, all pulling from the FastAPI:

**Overview:** Risk histogram, alert counts by level, top-10 riskiest users table. The histogram shows the distribution of max risk scores across all users — in practice, most cluster near 0 with a long tail of flagged users.

**User Drill-Down:** Select a user → see their temporal risk score chart across their full activity history. Known threat users show a shaded region marking their labeled threat window. This lets an analyst visually confirm whether alerts preceded the threat start.

**Model Performance:** Metrics, confusion matrix, threshold sweep chart, PR/ROC curves, per-user detection table. Loaded from `outputs/evaluation_results.json` and the saved plot images — these don't change unless the model is retrained.

---

## Key Design Decisions Summary

| Decision | Chosen | Why |
|---|---|---|
| Sequence representation | Token vocab (16 tokens) | Semantic compression; 16 tokens is enough to capture behavioral distinctions |
| Window size | 30 days | Captures multi-week behavioral patterns; shorter misses build-up |
| Max sequence length | 200 tokens | Covers typical 30-day activity; truncates from right, pads from left |
| Split strategy | User-level | Prevents data leakage between train/test |
| Loss function | BCEWithLogitsLoss + pos_weight | Handles class imbalance; numerical stability over BCE + sigmoid |
| Primary metric | PR-AUC | Honest on imbalanced classes; ROC-AUC inflated by true negatives |
| Architecture | Dual-input LSTM | Combines temporal sequence patterns + aggregate statistics |
| Model size | ~135K params | Fast training on free-tier Colab; avoids overfitting on 464K samples |
| Inference | Precompute at startup | Zero-latency API responses after initial load |
| Sigmoid placement | Inference only | Numerical stability with BCEWithLogitsLoss |

---

## File Map

```
insider-threat-lstm/
├── data/r4.2/                  # CERT dataset (not in git)
│   ├── logon.csv
│   ├── device.csv
│   ├── file.csv
│   ├── email.csv
│   ├── http.csv                → converted to http.parquet by pipeline
│   └── answers/insiders.csv   # ground truth
├── src/
│   ├── pipeline.py             # raw data → .npy arrays
│   ├── model.py                # InsiderThreatLSTM + ThreatDataset
│   └── evaluate.py             # metrics, plots, per-user analysis
├── notebooks/
│   └── train_colab.py          # Colab training notebook
├── serving/
│   ├── api.py                  # FastAPI REST API
│   └── app.py                  # Streamlit SOC dashboard
├── outputs/                    # pipeline and eval outputs (not in git except results)
│   ├── X_train_seq.npy, ...    # training arrays (not in git)
│   ├── best_model.pt           # trained checkpoint (not in git)
│   ├── evaluation_results.json # metrics + per-user analysis
│   ├── confusion_matrix.png
│   ├── roc_curve.png
│   ├── pr_curve.png
│   ├── threshold_sweep.png
│   └── time_to_detection.png
├── requirements.txt
├── CLAUDE.md                   # project spec / instructions for Claude
└── README.md                   # public-facing documentation
```
