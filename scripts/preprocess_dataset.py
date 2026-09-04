"""
scripts/preprocess_dataset.py
Phase 4 — Preprocessing the 300k working sample.

Reads: data/interim/working_sample.parquet
Produces:
    data/processed/X_train.parquet       — Scaled training features
    data/processed/metadata_train.parquet— Metadata for training partition
    data/processed/X_eval.parquet        — Scaled evaluation features
    data/processed/metadata_eval.parquet — Metadata for evaluation partition
    data/processed/X_calibration.parquet — Training-only calibration partition features
    data/processed/metadata_calibration.parquet — Metadata for calibration partition
    artifacts/preprocessors/standard_scaler.joblib
    artifacts/preprocessors/feature_names.json
    artifacts/preprocessors/feature_mask.json   — Which features AFP/RS/FS may modify
    artifacts/preprocessors/afp_benign_profile.parquet  — mu, sigma per feature
    artifacts/preprocessors/training_bounds.parquet     — min/max per feature
    data/manifests/split_manifest.json
    artifacts/reports/preprocessing_report.md
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path
import hashlib
import resource
import os

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from recall_aware_ids.utils.seeds import set_global_seeds
from recall_aware_ids.utils.config import get_experiment_config

cfg = get_experiment_config()

SPLIT_SEED    = cfg["split"]["split_seed"]            # 42
TRAIN_FRAC    = cfg["split"]["train_fraction"]         # 0.70
CAL_FRAC      = cfg["calibration"]["fraction_of_train"] # 0.15
CAL_SEED      = cfg["calibration"]["calibration_seed"]  # 42
SAMPLE_SEED   = cfg["dataset"]["sampling_seed"]        # 42

PROCESSED_DIR   = PROJECT_ROOT / "data" / "processed"
MANIFESTS_DIR   = PROJECT_ROOT / "data" / "manifests"
PREPROCESSORS_DIR = PROJECT_ROOT / "artifacts" / "preprocessors"
INTERIM_DIR     = PROJECT_ROOT / "data" / "interim"
REPORTS_DIR     = PROJECT_ROOT / "artifacts" / "reports"

for d in [PROCESSED_DIR, MANIFESTS_DIR, PREPROCESSORS_DIR, REPORTS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# Columns that will NEVER be used as model features
NON_FEATURE_COLS = {
    "Label", "label", "attack_family", "y_binary",
    "Timestamp", "timestamp", "Flow ID", "Src IP", "Dst IP",
    "_raw_row_idx", "_orig_row_idx", "_source_file",
    "Src Port",  # Excluded due to proven leakage (missing in 9 of 10 files)
}

# Metadata columns to preserve for pairing
METADATA_COLS = ["_source_file", "_raw_row_idx", "y_binary", "attack_family"]

# Columns that defense mechanisms may NOT modify
PROTECTED_FEATURE_PATTERNS = [
    # Binary flag columns (0 or 1 only)
    "FIN Flag Cnt", "SYN Flag Cnt", "RST Flag Cnt", "PSH Flag Cnt",
    "ACK Flag Cnt", "URG Flag Cnt", "CWE Flag Count", "ECE Flag Cnt",
    # Protocol (categorical integer)
    "Protocol",
    # Ports — kept as features, but not perturbed
    "Dst Port", "Src Port",
    "Fwd PSH Flags", "Bwd PSH Flags", "Fwd URG Flags", "Bwd URG Flags",
    "Down/Up Ratio",
]

def sha256_file(path: Path, buf_size: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(buf_size):
            h.update(chunk)
    return h.hexdigest()

def build_feature_mask(feature_columns: list[str]) -> list[bool]:
    """True = AFP/RS/FS may perturb this feature; False = preserve exactly."""
    mask = []
    for col in feature_columns:
        protected = any(pat.lower() in col.lower() for pat in PROTECTED_FEATURE_PATTERNS)
        mask.append(not protected)
    return mask


def preprocess() -> None:
    t0 = time.time()
    set_global_seeds(SAMPLE_SEED)

    print("=" * 70)
    print("PHASE 4 — Preprocessing Working Sample")
    print("=" * 70)

    ws_path = INTERIM_DIR / "working_sample.parquet"
    if not ws_path.exists():
        raise FileNotFoundError(f"Working sample not found: {ws_path}\nRun audit_dataset.py first.")

    ws_hash = sha256_file(ws_path)
    print(f"\nLoading {ws_path} (SHA-256: {ws_hash[:12]})...")
    df = pd.read_parquet(ws_path)
    print(f"  Loaded: {len(df):,} rows, {len(df.columns)} columns")

    # Safety: identify metadata columns actually present
    meta_cols = [c for c in METADATA_COLS if c in df.columns]
    
    # Check for NaNs or infinites
    has_nans = df.isna().any().any()
    if has_nans:
        print("  WARNING: NaNs found in dataframe!")
    
    # Identify feature columns
    feature_columns = [
        c for c in df.columns
        if c not in NON_FEATURE_COLS and c not in {"Label", "label"}
    ]
    # Ensure stable ordering
    feature_columns = sorted(feature_columns)
    print(f"  Feature columns: {len(feature_columns)}")

    # Ensure all feature columns are numeric
    for c in feature_columns:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    
    pre_drop_len = len(df)
    df = df.dropna(subset=feature_columns)
    post_drop_len = len(df)
    print(f"  After dropna (NaN/Inf check): {len(df):,} rows (Dropped {pre_drop_len - post_drop_len})")

    X = df[feature_columns].values.astype(np.float32)
    y = df["y_binary"].values.astype(np.int8)

    print(f"\n  X shape: {X.shape} | y distribution: "
          f"{dict(zip(*np.unique(y, return_counts=True)))}")

    # ── Deterministic constrained stratified split ─────────────────────────
    print("\nSplitting exact 210,000/90,000 stratified...")
    
    # Split the indices rather than the array to easily extract metadata too
    indices = np.arange(len(df))
    # EXACT 90,000 evaluation rows
    idx_train_full, idx_eval = train_test_split(
        indices, test_size=90000, stratify=y, random_state=SPLIT_SEED
    )
    
    # ── Calibration partition (from training only) ─────────────────────────
    print(f"\nCarving exact 31,500 calibration partition from training...")
    y_train_full = y[idx_train_full]
    # EXACT 31,500 calibration rows
    idx_train, idx_cal = train_test_split(
        idx_train_full, test_size=31500, stratify=y_train_full, random_state=CAL_SEED
    )
    
    print(f"  Train: {len(idx_train):,} | Eval: {len(idx_eval):,} | Calibration: {len(idx_cal):,}")
    
    X_train = X[idx_train]
    y_train = y[idx_train]
    df_meta_train = df.iloc[idx_train][meta_cols].copy()
    
    X_eval_part = X[idx_eval]
    y_eval_part = y[idx_eval]
    df_meta_eval = df.iloc[idx_eval][meta_cols].copy()
    
    X_cal = X[idx_cal]
    y_cal = y[idx_cal]
    df_meta_cal = df.iloc[idx_cal][meta_cols].copy()

    # Disjointness check using composite stable identity
    train_identities = set(zip(df_meta_train["_source_file"], df_meta_train["_raw_row_idx"]))
    eval_identities = set(zip(df_meta_eval["_source_file"], df_meta_eval["_raw_row_idx"]))
    cal_identities = set(zip(df_meta_cal["_source_file"], df_meta_cal["_raw_row_idx"]))
    assert len(train_identities.intersection(eval_identities)) == 0, "Composite identity overlap: Train/Eval"
    assert len(train_identities.intersection(cal_identities)) == 0, "Composite identity overlap: Train/Cal"
    assert len(eval_identities.intersection(cal_identities)) == 0, "Composite identity overlap: Eval/Cal"
    
    # ── StandardScaler (fit on X_train ONLY) ──────────────────────────────
    print("\nFitting StandardScaler on training partition...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train).astype(np.float32)
    X_eval_scaled  = scaler.transform(X_eval_part).astype(np.float32)
    X_cal_scaled   = scaler.transform(X_cal).astype(np.float32)

    print(f"  Scaler mean range: [{scaler.mean_.min():.4f}, {scaler.mean_.max():.4f}]")
    print(f"  Scaler std range:  [{scaler.scale_.min():.4f}, {scaler.scale_.max():.4f}]")

    # ── AFP benign profile (mu, sigma from benign training records only) ───
    print("\nComputing AFP benign profile from training set...")
    benign_mask_train = y_train == 0
    if benign_mask_train.sum() < 10:
        raise RuntimeError("Fewer than 10 benign training samples — cannot compute AFP profile.")
    X_benign_scaled = X_train_scaled[benign_mask_train]
    benign_mean = X_benign_scaled.mean(axis=0)
    benign_std  = np.maximum(X_benign_scaled.std(axis=0, ddof=1), 1e-6)
    
    # ── Training bounds (for semantic clipping in defenses/attacks) ────────
    train_min = X_train_scaled.min(axis=0)
    train_max = X_train_scaled.max(axis=0)

    # ── Feature mask (which features defenses may modify) ─────────────────
    feature_mask = build_feature_mask(feature_columns)
    n_modifiable = sum(feature_mask)
    print(f"\nFeature mask: {n_modifiable} modifiable, {len(feature_mask) - n_modifiable} protected")

    # ── Save everything ────────────────────────────────────────────────────
    print("\nSaving processed data...")

    # Training data
    pd.DataFrame(X_train_scaled, columns=feature_columns).to_parquet(PROCESSED_DIR / "X_train.parquet", index=False)
    df_meta_train.to_parquet(PROCESSED_DIR / "metadata_train.parquet", index=False)
    
    # Eval data
    pd.DataFrame(X_eval_scaled, columns=feature_columns).to_parquet(PROCESSED_DIR / "X_eval.parquet", index=False)
    df_meta_eval.to_parquet(PROCESSED_DIR / "metadata_eval.parquet", index=False)
    
    # Calibration data
    pd.DataFrame(X_cal_scaled, columns=feature_columns).to_parquet(PROCESSED_DIR / "X_calibration.parquet", index=False)
    df_meta_cal.to_parquet(PROCESSED_DIR / "metadata_calibration.parquet", index=False)

    # Preprocessors
    joblib.dump(scaler, PREPROCESSORS_DIR / "standard_scaler.joblib")
    with open(PREPROCESSORS_DIR / "feature_names.json", "w") as f:
        json.dump(feature_columns, f, indent=2)
    with open(PREPROCESSORS_DIR / "feature_mask.json", "w") as f:
        json.dump({
            "feature_mask": feature_mask,
            "feature_columns": feature_columns,
            "n_modifiable": n_modifiable,
            "n_protected": len(feature_mask) - n_modifiable,
            "protected_patterns": PROTECTED_FEATURE_PATTERNS,
        }, f, indent=2)

    pd.DataFrame({
        "feature": feature_columns,
        "benign_mean": benign_mean,
        "benign_std": benign_std,
    }).to_parquet(PREPROCESSORS_DIR / "afp_benign_profile.parquet", index=False)

    pd.DataFrame({
        "feature": feature_columns,
        "train_min": train_min,
        "train_max": train_max,
    }).to_parquet(PREPROCESSORS_DIR / "training_bounds.parquet", index=False)

    # Checksums & metadata
    split_meta = {
        "input_file": str(ws_path),
        "input_sha256": ws_hash,
        "train_rows": int(len(X_train)),
        "eval_rows": int(len(X_eval_part)),
        "calibration_rows": int(len(X_cal)),
        "n_features": len(feature_columns),
        "n_modifiable_features": int(n_modifiable),
        "split_seed": SPLIT_SEED,
        "cal_seed": CAL_SEED,
        "train_class_dist": {int(k): int(v) for k, v in zip(*np.unique(y_train, return_counts=True))},
        "eval_class_dist": {int(k): int(v) for k, v in zip(*np.unique(y_eval_part, return_counts=True))},
        "cal_class_dist": {int(k): int(v) for k, v in zip(*np.unique(y_cal, return_counts=True))},
    }
    with open(MANIFESTS_DIR / "split_manifest.json", "w") as f:
        json.dump(split_meta, f, indent=2)

    # Report
    peak_ram = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1048576 if sys.platform == "darwin" else 1024)  # MB
    elapsed = time.time() - t0
    
    report = f"""# Phase 4 Preprocessing Report
    
**Input File:** {ws_path.name} (SHA-256: {ws_hash})
**Input Rows:** {len(df):,}
**NaN/Inf Drops:** {pre_drop_len - post_drop_len}

## Split Strategy
* **Train Fraction:** {TRAIN_FRAC} (210,000 total train/cal)
* **Eval Fraction:** {1 - TRAIN_FRAC} (90,000)
* **Calibration:** {CAL_FRAC} of Train
* **Split Seed:** {SPLIT_SEED}
* **Disjointness:** Confirmed (Train, Eval, and Calibration share no indices)
* **Transformations:** `StandardScaler` fitted on Train only, applied to all.

## Partition Counts
* **Train:** {len(X_train):,}
  * Classes: {split_meta["train_class_dist"]}
* **Evaluation:** {len(X_eval_part):,}
  * Classes: {split_meta["eval_class_dist"]}
* **Calibration:** {len(X_cal):,}
  * Classes: {split_meta["cal_class_dist"]}

## Features
* **Total Included Features:** {len(feature_columns)}
* **Modifiable Features:** {n_modifiable}
* **Protected Features:** {len(feature_mask) - n_modifiable}
* **Excluded Cols:** {NON_FEATURE_COLS}
* **Metadata Saved:** {METADATA_COLS}

## Performance
* **Runtime:** {elapsed:.1f}s
* **Peak RAM:** {peak_ram:.1f} MB

_Generated automatically by scripts/preprocess_dataset.py_
"""
    with open(REPORTS_DIR / "preprocessing_report.md", "w") as f:
        f.write(report)

    print(f"\n✓ Phase 4 complete in {elapsed:.1f}s")
    print(f"  Peak RAM: {peak_ram:.1f} MB")
    print("\n  Next: python scripts/train_model.py")


if __name__ == "__main__":
    preprocess()
