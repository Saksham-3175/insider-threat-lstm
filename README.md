# Insider Threat LSTM

LSTM-based insider threat detector trained on CERT Insider Threat Dataset r4.2.

## Architecture

Dual-input LSTM: sequence branch (token embeddings → 2-layer LSTM) + numerical feature branch, fused at the classifier head.

- ~137K parameters
- Primary metric: PR-AUC
- Training target: Google Colab T4

## Stack

PyTorch · FastAPI · Streamlit · MLflow · Weights & Biases

## Setup

```bash
source ~/Code/ml-venv/bin/activate
pip install -r requirements.txt
```
