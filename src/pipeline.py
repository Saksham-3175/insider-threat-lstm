"""
Data pipeline: CERT r4.2 → windowed sequence + feature arrays.

Steps:
  1. Convert http.csv → http.parquet (one-time, snappy)
  2. Load all sources, merge, sort by (user, date)
  3. Vectorized event tokenization
  4. Load ground truth (insiders.csv → threat users dict)
  5. Sliding 30-day windows per user (numpy.searchsorted)
  6. User-level stratified 70/15/15 split
  7. Save .npy arrays + metadata JSON to outputs/
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

# ── Paths ─────────────────────────────────────────────────────────────────────
DATA_DIR   = Path("data/r4.2")
OUTPUT_DIR = Path("outputs")

# ── EVENT_VOCAB ───────────────────────────────────────────────────────────────
UNKNOWN              = 0
NORMAL_LOGIN         = 1
AFTERHOURS_LOGIN     = 2
WEEKEND_LOGIN        = 3
LOGOFF               = 4
USB_CONNECT          = 5
USB_DISCONNECT       = 6
FILE_OPEN            = 7
FILE_COPY            = 8
LARGE_FILE_ACTIVITY  = 9
HTTP_NORMAL          = 10
HTTP_JOBSITE         = 11
HTTP_CLOUD_STORAGE   = 12
EMAIL_NORMAL         = 13
EMAIL_WITH_ATTACHMENT = 14
EMAIL_EXTERNAL       = 15

VOCAB = {
    "UNKNOWN": UNKNOWN, "NORMAL_LOGIN": NORMAL_LOGIN,
    "AFTERHOURS_LOGIN": AFTERHOURS_LOGIN, "WEEKEND_LOGIN": WEEKEND_LOGIN,
    "LOGOFF": LOGOFF, "USB_CONNECT": USB_CONNECT, "USB_DISCONNECT": USB_DISCONNECT,
    "FILE_OPEN": FILE_OPEN, "FILE_COPY": FILE_COPY,
    "LARGE_FILE_ACTIVITY": LARGE_FILE_ACTIVITY, "HTTP_NORMAL": HTTP_NORMAL,
    "HTTP_JOBSITE": HTTP_JOBSITE, "HTTP_CLOUD_STORAGE": HTTP_CLOUD_STORAGE,
    "EMAIL_NORMAL": EMAIL_NORMAL, "EMAIL_WITH_ATTACHMENT": EMAIL_WITH_ATTACHMENT,
    "EMAIL_EXTERNAL": EMAIL_EXTERNAL,
}

# ── Domain lists ──────────────────────────────────────────────────────────────
JOBSITE_DOMAINS = ["linkedin", "indeed", "glassdoor", "monster", "careerbuilder"]
CLOUD_DOMAINS   = ["dropbox", "drive.google", "onedrive", "box.com", "wetransfer"]
ORG_DOMAIN      = "@dtaa.com"

# ── Pipeline constants ────────────────────────────────────────────────────────
WINDOW_DAYS  = 30
STEP_DAYS    = 1
MAX_SEQ_LEN  = 200
TRAIN_FRAC   = 0.70
VAL_FRAC     = 0.15
# TEST_FRAC = 1 - TRAIN_FRAC - VAL_FRAC = 0.15


# ─────────────────────────────────────────────────────────────────────────────
# Step 1: Convert http.csv → http.parquet
# ─────────────────────────────────────────────────────────────────────────────
def convert_http_to_parquet() -> None:
    parquet_path = DATA_DIR / "http.parquet"
    if parquet_path.exists():
        print("[Step 1] http.parquet already exists — skipping conversion.")
        return

    csv_path = DATA_DIR / "http.csv"
    print(f"[Step 1] Converting {csv_path} → {parquet_path} ...")
    chunks = []
    for i, chunk in enumerate(pd.read_csv(csv_path, chunksize=500_000)):
        chunks.append(chunk)
        cumulative = sum(len(c) for c in chunks)
        print(f"  chunk {i + 1}: {len(chunk):,} rows  (cumulative {cumulative:,})")

    df = pd.concat(chunks, ignore_index=True)
    df.to_parquet(parquet_path, compression="snappy", index=False)
    print(f"  Saved {len(df):,} rows → {parquet_path}")


# ─────────────────────────────────────────────────────────────────────────────
# Step 2: Load all sources and merge
# ─────────────────────────────────────────────────────────────────────────────
def _load_logon() -> pd.DataFrame:
    df = pd.read_csv(DATA_DIR / "logon.csv", usecols=["date", "user", "activity"])
    df["date"]   = pd.to_datetime(df["date"])
    df["source"] = "logon"
    return df[["user", "date", "source", "activity"]]


def _load_device() -> pd.DataFrame:
    df = pd.read_csv(DATA_DIR / "device.csv", usecols=["date", "user", "activity"])
    df["date"]   = pd.to_datetime(df["date"])
    df["source"] = "device"
    return df[["user", "date", "source", "activity"]]


def _load_file() -> pd.DataFrame:
    df = pd.read_csv(DATA_DIR / "file.csv", usecols=["date", "user"])
    df["date"]     = pd.to_datetime(df["date"])
    df["source"]   = "file"
    df["activity"] = pd.NA
    return df[["user", "date", "source", "activity"]]


def _load_http() -> pd.DataFrame:
    df = pd.read_parquet(DATA_DIR / "http.parquet", columns=["date", "user", "url"])
    df["date"]     = pd.to_datetime(df["date"])
    df["source"]   = "http"
    df["activity"] = pd.NA
    return df[["user", "date", "source", "activity", "url"]]


def _load_email() -> pd.DataFrame:
    df = pd.read_csv(
        DATA_DIR / "email.csv",
        usecols=["date", "user", "to", "attachments"],
    )
    df["date"]     = pd.to_datetime(df["date"])
    df["source"]   = "email"
    df["activity"] = pd.NA
    return df[["user", "date", "source", "activity", "to", "attachments"]]


def load_and_merge() -> pd.DataFrame:
    print("[Step 2] Loading data sources...")
    loaders = [
        ("logon",  _load_logon),
        ("device", _load_device),
        ("file",   _load_file),
        ("http",   _load_http),
        ("email",  _load_email),
    ]
    parts = []
    for name, fn in loaders:
        df = fn()
        print(f"  {name:8s}: {len(df):>10,} rows")
        parts.append(df)

    merged = pd.concat(parts, ignore_index=True, sort=False)
    merged.sort_values(["user", "date"], inplace=True, kind="stable")
    merged.reset_index(drop=True, inplace=True)
    print(f"  Merged : {len(merged):,} rows | {merged['user'].nunique():,} users")
    return merged


# ─────────────────────────────────────────────────────────────────────────────
# Step 3: Vectorized tokenization
# ─────────────────────────────────────────────────────────────────────────────
def _tok_logon(df: pd.DataFrame) -> np.ndarray:
    tokens   = np.full(len(df), NORMAL_LOGIN, dtype=np.int8)
    activity = df["activity"].fillna("").str.lower()
    hour     = df["date"].dt.hour.values
    dow      = df["date"].dt.dayofweek.values  # 0=Mon … 6=Sun
    is_logoff = activity.str.contains("logoff").values
    is_logon  = ~is_logoff
    # Priority: weekend > after-hours > normal (weekend label wins)
    tokens[is_logon & ((hour < 7) | (hour > 18))] = AFTERHOURS_LOGIN
    tokens[is_logon & (dow >= 5)]                  = WEEKEND_LOGIN
    tokens[is_logoff]                              = LOGOFF
    return tokens


def _tok_device(df: pd.DataFrame) -> np.ndarray:
    tokens     = np.full(len(df), USB_DISCONNECT, dtype=np.int8)
    is_connect = df["activity"].fillna("").str.lower().str.contains("connect").values
    tokens[is_connect] = USB_CONNECT
    return tokens


def _tok_file(df: pd.DataFrame) -> np.ndarray:
    # file.csv entries are all copies to removable media; no size column available
    return np.full(len(df), FILE_COPY, dtype=np.int8)


def _tok_http(df: pd.DataFrame) -> np.ndarray:
    tokens = np.full(len(df), HTTP_NORMAL, dtype=np.int8)
    url = df["url"].fillna("").str.lower()
    jobsite_pat = "|".join(JOBSITE_DOMAINS)
    cloud_pat   = "|".join(d.replace(".", r"\.") for d in CLOUD_DOMAINS)
    is_cloud    = url.str.contains(cloud_pat,   regex=True).values
    is_jobsite  = url.str.contains(jobsite_pat, regex=False).values
    tokens[is_cloud]   = HTTP_CLOUD_STORAGE
    tokens[is_jobsite] = HTTP_JOBSITE   # jobsite overrides cloud if both match
    return tokens


def _tok_email(df: pd.DataFrame) -> np.ndarray:
    tokens      = np.full(len(df), EMAIL_NORMAL, dtype=np.int8)
    to_col      = df["to"].fillna("")
    attachments = pd.to_numeric(df["attachments"], errors="coerce").fillna(0).values
    has_attach  = attachments > 0
    # External = any recipient address not at @dtaa.com
    # Regex: @ NOT followed by dtaa.com captures any non-org address
    is_external = to_col.str.contains(r"@(?!dtaa\.com\b)", regex=True).values
    tokens[has_attach]  = EMAIL_WITH_ATTACHMENT
    tokens[is_external] = EMAIL_EXTERNAL   # external overrides attachment label
    return tokens


_TOKENIZERS = {
    "logon":  _tok_logon,
    "device": _tok_device,
    "file":   _tok_file,
    "http":   _tok_http,
    "email":  _tok_email,
}


def tokenize(merged: pd.DataFrame) -> pd.DataFrame:
    print("[Step 3] Tokenizing events...")
    tokens = np.zeros(len(merged), dtype=np.int8)

    for source, fn in _TOKENIZERS.items():
        idx = np.where((merged["source"] == source).values)[0]
        if len(idx) == 0:
            continue
        source_tokens = fn(merged.iloc[idx].reset_index(drop=True))
        tokens[idx]   = source_tokens
        unique, counts = np.unique(source_tokens, return_counts=True)
        breakdown = {int(k): int(v) for k, v in zip(unique, counts)}
        print(f"  {source:8s}: {len(idx):>10,} events → {breakdown}")

    # Drop source-specific columns; keep only what windowing needs
    events = merged[["user", "date"]].copy()
    events["token"] = tokens
    return events


# ─────────────────────────────────────────────────────────────────────────────
# Step 4: Load ground truth
# ─────────────────────────────────────────────────────────────────────────────
def load_ground_truth() -> dict:
    """Return {user_id: (start_date, end_date)} for dataset-4.2 threat users."""
    gt_path = DATA_DIR / "answers" / "insiders.csv"
    if not gt_path.exists():
        print(f"[Step 4] WARNING: {gt_path} not found — all labels will be 0.")
        return {}
    df = pd.read_csv(gt_path)
    df = df[df["dataset"].astype(str) == "4.2"].copy()
    df["start"] = pd.to_datetime(df["start"])
    df["end"]   = pd.to_datetime(df["end"])
    threat_users = {
        row["user"]: (row["start"], row["end"]) for _, row in df.iterrows()
    }
    print(f"[Step 4] {len(threat_users)} threat users loaded.")
    return threat_users


# ─────────────────────────────────────────────────────────────────────────────
# Step 5: Sliding windows (per user, numpy.searchsorted on int64 ns timestamps)
# ─────────────────────────────────────────────────────────────────────────────
_NS_PER_DAY = np.int64(86_400 * 1_000_000_000)


def _windows_for_user(
    dates_ns: np.ndarray,   # sorted int64 nanosecond timestamps
    tokens: np.ndarray,     # int8, same length
    threat_range,           # None or (start_ns, end_ns) as int64
) -> tuple:
    window_ns = np.int64(WINDOW_DAYS) * _NS_PER_DAY
    step_ns   = np.int64(STEP_DAYS)   * _NS_PER_DAY

    # Align to midnight of first / last event day
    first_day = (dates_ns[0]  // _NS_PER_DAY) * _NS_PER_DAY
    last_day  = (dates_ns[-1] // _NS_PER_DAY) * _NS_PER_DAY

    seqs, feats, labels = [], [], []
    win_start = first_day

    while win_start <= last_day:
        win_end = win_start + window_ns
        lo = int(np.searchsorted(dates_ns, win_start, side="left"))
        hi = int(np.searchsorted(dates_ns, win_end,   side="left"))

        w_tokens = tokens[lo:hi]
        n = len(w_tokens)

        # Sequence: right-aligned (most recent events at end), pad left with 0
        seq = np.zeros(MAX_SEQ_LEN, dtype=np.int32)
        if n > 0:
            take = min(n, MAX_SEQ_LEN)
            seq[MAX_SEQ_LEN - take:] = w_tokens[-take:]

        # Numerical features
        if n == 0:
            feat = np.zeros(6, dtype=np.float32)
        else:
            w_dates = dates_ns[lo:hi]
            # Hour from nanoseconds (no pandas overhead)
            hours = ((w_dates % _NS_PER_DAY) // (3_600 * 10**9)).astype(np.float32)
            # Day-of-week: epoch (1970-01-01) was Thursday (3 in Mon=0 scheme)
            days_epoch = w_dates // _NS_PER_DAY
            dow = ((days_epoch + 3) % 7).astype(np.int32)  # 0=Mon … 6=Sun

            avg_hour          = float(hours.mean())
            after_hours_ratio = float(np.mean((hours < 7) | (hours > 18)))
            weekend_ratio     = float(np.mean(dow >= 5))
            usb_count         = float(np.sum(
                (w_tokens == USB_CONNECT) | (w_tokens == USB_DISCONNECT)
            ))
            n_email = int(np.sum(
                (w_tokens == EMAIL_NORMAL) |
                (w_tokens == EMAIL_WITH_ATTACHMENT) |
                (w_tokens == EMAIL_EXTERNAL)
            ))
            email_external_ratio = (
                float(np.sum(w_tokens == EMAIL_EXTERNAL) / n_email) if n_email > 0 else 0.0
            )
            n_http = int(np.sum(
                (w_tokens == HTTP_NORMAL) |
                (w_tokens == HTTP_JOBSITE) |
                (w_tokens == HTTP_CLOUD_STORAGE)
            ))
            http_jobsite_ratio = (
                float(np.sum(w_tokens == HTTP_JOBSITE) / n_http) if n_http > 0 else 0.0
            )
            feat = np.array(
                [avg_hour, after_hours_ratio, weekend_ratio, usb_count,
                 email_external_ratio, http_jobsite_ratio],
                dtype=np.float32,
            )

        # Label: 1 if window overlaps the user's threat date range
        label = 0
        if threat_range is not None:
            t_start, t_end = threat_range
            if win_start < t_end and win_end > t_start:
                label = 1

        seqs.append(seq)
        feats.append(feat)
        labels.append(label)
        win_start += step_ns

    return seqs, feats, labels


def build_windows(
    events: pd.DataFrame, threat_users: dict
) -> tuple:
    print("[Step 5] Building sliding windows...")

    # Pre-convert threat date ranges to int64 nanoseconds
    threat_ns = {
        user: (np.int64(start.value), np.int64(end.value))
        for user, (start, end) in threat_users.items()
    }

    all_seq, all_feat, all_labels, all_users = [], [], [], []

    groups = events.groupby("user", sort=False)
    for user, grp in tqdm(groups, total=groups.ngroups, desc="Building windows"):
        grp      = grp.sort_values("date")
        dates_ns = grp["date"].values.astype(np.int64)
        tokens   = grp["token"].values.astype(np.int8)

        seqs, feats, labels = _windows_for_user(
            dates_ns, tokens, threat_ns.get(user)
        )
        all_seq.extend(seqs)
        all_feat.extend(feats)
        all_labels.extend(labels)
        all_users.extend([user] * len(labels))

    X_seq     = np.array(all_seq,    dtype=np.int32)
    X_feat    = np.array(all_feat,   dtype=np.float32)
    y         = np.array(all_labels, dtype=np.int8)
    users_arr = np.array(all_users)

    n_pos = int(y.sum())
    n_tot = len(y)
    print(f"  Total windows : {n_tot:,}")
    print(f"  Positive      : {n_pos:,}  ({100 * n_pos / n_tot:.3f}%)")
    return X_seq, X_feat, y, users_arr


# ─────────────────────────────────────────────────────────────────────────────
# Step 6: User-level stratified 70/15/15 split
# ─────────────────────────────────────────────────────────────────────────────
def split_users(users_arr: np.ndarray, threat_users: dict) -> dict:
    print("[Step 6] Splitting users (stratified, seed=42)...")
    unique_users = list(dict.fromkeys(users_arr.tolist()))  # dedup, stable order

    rng    = np.random.default_rng(42)
    threat = np.array([u for u in unique_users if u in threat_users])
    benign = np.array([u for u in unique_users if u not in threat_users])
    rng.shuffle(threat)
    rng.shuffle(benign)

    def _split(arr: np.ndarray):
        n       = len(arr)
        n_train = int(np.floor(n * TRAIN_FRAC))
        n_val   = int(np.floor(n * VAL_FRAC))
        return arr[:n_train], arr[n_train:n_train + n_val], arr[n_train + n_val:]

    t_tr, t_va, t_te = _split(threat)
    b_tr, b_va, b_te = _split(benign)

    splits = {
        "train": sorted(np.concatenate([t_tr, b_tr]).tolist()),
        "val":   sorted(np.concatenate([t_va, b_va]).tolist()),
        "test":  sorted(np.concatenate([t_te, b_te]).tolist()),
    }

    print(f"  train : {len(splits['train']):4d} users  ({len(t_tr)} threat, {len(b_tr)} benign)")
    print(f"  val   : {len(splits['val']):4d} users  ({len(t_va)} threat, {len(b_va)} benign)")
    print(f"  test  : {len(splits['test']):4d} users  ({len(t_te)} threat, {len(b_te)} benign)")
    return splits


# ─────────────────────────────────────────────────────────────────────────────
# Step 7: Save outputs
# ─────────────────────────────────────────────────────────────────────────────
def save_outputs(
    X_seq: np.ndarray,
    X_feat: np.ndarray,
    y: np.ndarray,
    users_arr: np.ndarray,
    splits: dict,
    threat_users: dict,
) -> None:
    print("[Step 7] Saving outputs...")
    OUTPUT_DIR.mkdir(exist_ok=True)

    stats: dict = {
        "total_windows":   int(len(y)),
        "total_positive":  int(y.sum()),
        "total_negative":  int((y == 0).sum()),
        "class_balance_pct": round(100 * float(y.sum()) / max(len(y), 1), 4),
        "n_threat_users":  len(threat_users),
        "splits": {},
    }

    for split_name, split_users_list in splits.items():
        split_set = set(split_users_list)
        mask = np.array([u in split_set for u in users_arr], dtype=bool)

        np.save(OUTPUT_DIR / f"X_{split_name}_seq.npy",  X_seq[mask])
        np.save(OUTPUT_DIR / f"X_{split_name}_feat.npy", X_feat[mask])
        np.save(OUTPUT_DIR / f"y_{split_name}.npy",      y[mask])

        n_pos = int(y[mask].sum())
        n_tot = int(mask.sum())
        print(
            f"  {split_name:5s}: {n_tot:>8,} windows | "
            f"{n_pos:>6,} positive ({100 * n_pos / max(n_tot, 1):.2f}%)"
        )
        stats["splits"][split_name] = {
            "n_windows":  n_tot,
            "n_positive": n_pos,
            "n_users":    len(split_users_list),
        }

    with open(OUTPUT_DIR / "vocab.json", "w") as f:
        json.dump(VOCAB, f, indent=2)
    with open(OUTPUT_DIR / "user_splits.json", "w") as f:
        json.dump(splits, f, indent=2)
    with open(OUTPUT_DIR / "pipeline_stats.json", "w") as f:
        json.dump(stats, f, indent=2)

    print(f"  All outputs written to {OUTPUT_DIR}/")


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────
def run() -> None:
    convert_http_to_parquet()
    merged = load_and_merge()
    events = tokenize(merged)
    del merged          # free memory before the windowing allocation
    threat_users = load_ground_truth()
    X_seq, X_feat, y, users_arr = build_windows(events, threat_users)
    del events          # done with source data
    splits = split_users(users_arr, threat_users)
    save_outputs(X_seq, X_feat, y, users_arr, splits, threat_users)
    print("\nPipeline complete.")


if __name__ == "__main__":
    run()
