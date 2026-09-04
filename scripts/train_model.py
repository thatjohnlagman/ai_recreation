"""
scripts/train_model.py
Phase 5 — Model Training (Frozen Random Forest)

Produces:
    artifacts/models/frozen_rf.joblib
    artifacts/models/frozen_rf.sha256
    artifacts/reports/learning_curve.md
    artifacts/reports/learning_curve.png
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
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import balanced_accuracy_score, recall_score, precision_score, f1_score, average_precision_score

PROJECT_ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from recall_aware_ids.utils.seeds import set_global_seeds
from recall_aware_ids.utils.config import _resolve_config_dir, load_yaml

cfg = load_yaml(_resolve_config_dir() / "model.yaml")
rf_cfg = cfg["random_forest"]
ser_cfg = cfg["serialization"]

PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
MODELS_DIR = PROJECT_ROOT / "artifacts" / "models"
REPORTS_DIR = PROJECT_ROOT / "artifacts" / "reports"

for d in [MODELS_DIR, REPORTS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

def get_oob_metrics(rf: RandomForestClassifier, y_true: np.ndarray) -> dict:
    # OOB predictions (probabilities)
    oob_proba = rf.oob_decision_function_
    # OOB hard predictions
    oob_pred = np.argmax(oob_proba, axis=1)
    
    # Calculate coverage
    coverage = np.mean(~np.isnan(oob_proba).any(axis=1))
    
    # Only compute metrics for rows that have valid OOB predictions
    valid = ~np.isnan(oob_proba).any(axis=1)
    y_val = y_true[valid]
    y_pred_val = oob_pred[valid]
    y_prob_val = oob_proba[valid, 1]
    
    return {
        "coverage": coverage,
        "balanced_accuracy": balanced_accuracy_score(y_val, y_pred_val),
        "recall": recall_score(y_val, y_pred_val, pos_label=1),
        "precision": precision_score(y_val, y_pred_val, pos_label=1),
        "f1": f1_score(y_val, y_pred_val, pos_label=1),
        "pr_auc": average_precision_score(y_val, y_prob_val)
    }

def sha256_file(path: Path, buf_size: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(buf_size):
            h.update(chunk)
    return h.hexdigest()

def plot_learning_curve(sizes: list[int], metrics: list[dict], out_path: Path):
    plt.figure(figsize=(10, 6))
    
    # Extract lists
    ba = [m["balanced_accuracy"] for m in metrics]
    rec = [m["recall"] for m in metrics]
    prec = [m["precision"] for m in metrics]
    f1 = [m["f1"] for m in metrics]
    prauc = [m["pr_auc"] for m in metrics]
    
    plt.plot(sizes, ba, marker='o', label="Balanced Accuracy")
    plt.plot(sizes, rec, marker='s', label="Recall")
    plt.plot(sizes, prec, marker='^', label="Precision")
    plt.plot(sizes, f1, marker='d', label="F1-Score")
    plt.plot(sizes, prauc, marker='x', label="PR-AUC")
    
    plt.title("Random Forest OOB Performance by Training Set Size (Phase 5)")
    plt.xlabel("Number of Training Records")
    plt.ylabel("Score")
    plt.ylim([0.8, 1.0])
    plt.grid(True, linestyle="--", alpha=0.7)
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()

def train() -> None:
    t0 = time.time()
    set_global_seeds(rf_cfg["random_state"])

    print("=" * 70)
    print("PHASE 5 — Training Frozen Random Forest & Learning Curve")
    print("=" * 70)

    # 1. Load the sub-training partition ONLY
    x_path = PROCESSED_DIR / "X_train.parquet"
    y_path = PROCESSED_DIR / "metadata_train.parquet"
    if not x_path.exists() or not y_path.exists():
        raise FileNotFoundError(f"Training data not found. Run preprocess_dataset.py first.")

    print(f"Loading {x_path.name}...")
    X_train_df = pd.read_parquet(x_path)
    y_train_df = pd.read_parquet(y_path)
    
    X_full = X_train_df.values.astype(np.float32)
    y_full = y_train_df["y_binary"].values.astype(np.int8)
    
    total_records = len(X_full)
    print(f"  Loaded: {total_records:,} training records")
    if total_records != 178500:
        raise ValueError(f"Expected exactly 178,500 training records, found {total_records}")

    # 2. Construct deterministic nested subsets
    print("\nRunning Learning Curve (25%, 50%, 75%, 100%)...")
    np.random.seed(rf_cfg["random_state"])
    
    # Shuffle indices once
    shuffled_indices = np.random.permutation(total_records)
    
    fractions = [0.25, 0.50, 0.75, 1.0]
    sizes = [int(total_records * f) for f in fractions]
    
    metrics_log = []
    models = []
    
    # Ensure bootstrap is active (it's True by default, but verify we don't set it to False)
    # We will use oob_score=True to get out-of-bag estimates.
    for size in sizes:
        print(f"  Training on {size:,} records...")
        idx = shuffled_indices[:size]
        X_sub = X_full[idx]
        y_sub = y_full[idx]
        
        # Enforce exact frozen configuration, plus oob_score=True
        rf = RandomForestClassifier(
            n_estimators=rf_cfg["n_estimators"],
            max_depth=rf_cfg["max_depth"],
            criterion=rf_cfg["criterion"],
            n_jobs=rf_cfg["n_jobs"],
            random_state=rf_cfg["random_state"],
            class_weight=rf_cfg.get("class_weight", None),
            oob_score=True,
            bootstrap=True
        )
        
        t_start = time.time()
        rf.fit(X_sub, y_sub)
        t_fit = time.time() - t_start
        
        metrics = get_oob_metrics(rf, y_sub)
        metrics["size"] = size
        metrics["fit_time"] = t_fit
        metrics_log.append(metrics)
        models.append(rf)
        
        print(f"    OOB F1-Score: {metrics['f1']:.4f} | Recall: {metrics['recall']:.4f} | Time: {t_fit:.1f}s")
        
    # 3. Final model selection
    # Since the 100% model used the exact frozen configuration (with oob_score=True enabled), 
    # and oob_score does not affect the tree structure, we can safely use models[-1] as the official model.
    print("\nSelecting 100% model as official frozen model.")
    official_rf = models[-1]
    
    model_path = PROJECT_ROOT / ser_cfg["model_path"]
    joblib.dump(official_rf, model_path)
    model_hash = sha256_file(model_path)
    with open(PROJECT_ROOT / ser_cfg["hash_path"], "w") as f:
        f.write(model_hash)
        
    print(f"  Serialized model to {ser_cfg['model_path']}")
    print(f"  SHA-256: {model_hash}")

    # 4. Generate Reports and Plots
    print("\nGenerating Learning Curve Report...")
    fig_path = REPORTS_DIR / "learning_curve.png"
    plot_learning_curve(sizes, metrics_log, fig_path)
    
    report_md = f"""# Phase 5 Learning Curve Report
    
## Frozen Configuration
* `n_estimators`: {rf_cfg['n_estimators']}
* `max_depth`: {rf_cfg['max_depth']}
* `criterion`: {rf_cfg['criterion']}
* `class_weight`: {rf_cfg.get('class_weight')}
* `random_state`: {rf_cfg['random_state']}
* `oob_score`: True (for metrics extraction)

## Secondary Robustness Analysis (Learning Curve)
The following metrics are computed exclusively on the **Out-Of-Bag (OOB)** samples of the `{total_records:,}` record `X_train` partition. No evaluation or calibration data was exposed during this process.

| Train Size | Coverage | Balanced Acc | Recall | Precision | F1-Score | PR-AUC | Fit Time |
|------------|----------|--------------|--------|-----------|----------|--------|----------|
"""
    for m in metrics_log:
        report_md += f"| {m['size']:,} | {m['coverage']:.2%} | {m['balanced_accuracy']:.4f} | {m['recall']:.4f} | {m['precision']:.4f} | {m['f1']:.4f} | {m['pr_auc']:.4f} | {m['fit_time']:.1f}s |\n"

    report_md += f"""
## Descriptive Interpretation
The model achieves near-perfect coverage of the out-of-bag samples. 
As training size increases from 25% ({sizes[0]:,}) to 100% ({sizes[-1]:,}), we observe the OOB metrics. If the curve plateaus significantly before 100%, it indicates diminishing returns from additional data for the core attack classes. However, this does not imply equivalence to the full 16.1M dataset, particularly for extremely rare attack families that may not be well-represented in smaller subsamples.

## Official Model Artifact
* **Path:** `{ser_cfg['model_path']}`
* **SHA-256:** `{model_hash}`
"""
    with open(REPORTS_DIR / "learning_curve.md", "w") as f:
        f.write(report_md)
        
    peak_ram = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1048576 if sys.platform == "darwin" else 1024)
    elapsed = time.time() - t0
    print(f"\n✓ Phase 5 complete in {elapsed:.1f}s")
    print(f"  Peak RAM: {peak_ram:.1f} MB")
    print("\n  Next: Validate reload consistency via tests.")

if __name__ == "__main__":
    train()
