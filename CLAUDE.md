**Project:** LSTM-Based Insider Threat Detection (PyTorch)
**Dataset:** CERT Insider Threat Dataset r4.2 (at `data/r4.2/`)
**Venv:** `source ~/Code/ml-venv/bin/activate`

**Goal:** Dual-input LSTM insider threat detector in PyTorch, train on full dataset using Colab T4, serve via FastAPI + Streamlit, track experiments with MLflow + Weights & Biases.

**Architecture:**

- Dual-input LSTM: sequence branch (token embeddings → 2-layer LSTM) + feature branch (numerical features)
- Sequence input: (batch, 200) int tokens from EVENT_VOCAB (16 tokens)
- Feature input: (batch, 6) float32 — avg_hour, after_hours_ratio, weekend_ratio, usb_count, email_external_ratio, http_jobsite_ratio
- LSTM Layer 1: input=embed_dim(32), hidden=128, batch_first=True, dropout=0.3, returns all hidden states
- LSTM Layer 2: input=128, hidden=64, batch_first=True, dropout=0.3, returns last hidden state only
- Classifier: Linear(64+6, 32) → ReLU → Dropout(0.2) → Linear(32, 1)
- Use BCEWithLogitsLoss (no sigmoid in forward, apply sigmoid only at inference)
- Total params: ~137K

**Data pipeline constants:**

- Window size: 30 days, step: 1 day
- Max sequence length: 200 tokens per window
- User-level train/val/test split: 70/15/15
- http.csv: convert to Parquet first (chunked read), then process full file

**EVENT_VOCAB (16 tokens):**
UNKNOWN=0, NORMAL_LOGIN=1, AFTERHOURS_LOGIN=2, WEEKEND_LOGIN=3, LOGOFF=4, USB_CONNECT=5, USB_DISCONNECT=6, FILE_OPEN=7, FILE_COPY=8, LARGE_FILE_ACTIVITY=9, HTTP_NORMAL=10, HTTP_JOBSITE=11, HTTP_CLOUD_STORAGE=12, EMAIL_NORMAL=13, EMAIL_WITH_ATTACHMENT=14, EMAIL_EXTERNAL=15

**Key domains:**

- Job sites: linkedin, indeed, glassdoor, monster, careerbuilder
- Cloud storage: dropbox, drive.google, onedrive, box.com, wetransfer
- Org email domain: @dtaa.com

**Ground truth:** data/r4.2/answers/insiders.csv → filter dataset==4.2 → 70 threat users with start/end dates

**Training:**

- Environment: Google Colab T4 (15GB VRAM)
- Loss: BCEWithLogitsLoss with pos_weight (capped at 20x)
- Optimizer: Adam(lr=0.001)
- EarlyStopping on val PR-AUC (patience=7), ReduceLROnPlateau (factor=0.5, patience=3)
- Primary metric: PR-AUC (not ROC-AUC)
- Tracking: MLflow (local) + Weights & Biases (cloud dashboard)

**Current phase:** Phase 0 — Scaffold
**Stack:** PyTorch · FastAPI · Streamlit · MLflow · wandb

## **Commit convention:** Use conventional commits (feat, fix, docs, refactor, etc). No tool or AI attribution in commit messages. Example: `feat: data pipeline with parquet conversion`
