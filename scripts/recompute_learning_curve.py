"""
scripts/recompute_learning_curve.py
Re-runs the Phase 5 learning curve subsets to extract detailed metrics (TP, TN, FP, FN, specificities)
without overwriting the official 100% frozen model.
Updates the learning_curve.md and learning_curve.png.
"""

import sys
import time
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, average_precision_score

PROJECT_ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from recall_aware_ids.utils.config import _resolve_config_dir, load_yaml

cfg = load_yaml(_resolve_config_dir() / "model.yaml")
rf_cfg = cfg["random_forest"]

PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
REPORTS_DIR = PROJECT_ROOT / "artifacts" / "reports"

def plot_learning_curve(sizes, metrics, out_path):
    plt.figure(figsize=(10, 6))
    
    ba = [m["balanced_accuracy"] for m in metrics]
    rec = [m["attack_recall"] for m in metrics]
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

def run():
    x_path = PROCESSED_DIR / "X_train.parquet"
    y_path = PROCESSED_DIR / "metadata_train.parquet"
    
    X_train_df = pd.read_parquet(x_path)
    y_train_df = pd.read_parquet(y_path)
    
    X_full = X_train_df.values.astype(np.float32)
    y_full = y_train_df["y_binary"].values.astype(np.int8)
    total_records = len(X_full)
    
    np.random.seed(rf_cfg["random_state"])
    shuffled_indices = np.random.permutation(total_records)
    
    fractions = [0.25, 0.50, 0.75, 1.0]
    sizes = [int(total_records * f) for f in fractions]
    
    metrics_log = []
    
    total_t0 = time.time()
    
    for size in sizes:
        print(f"Training on {size:,} records...")
        idx = shuffled_indices[:size]
        X_sub = X_full[idx]
        y_sub = y_full[idx]
        
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
        
        oob_proba = rf.oob_decision_function_
        oob_pred = np.argmax(oob_proba, axis=1)
        valid = ~np.isnan(oob_proba).any(axis=1)
        
        y_val = y_sub[valid]
        y_pred_val = oob_pred[valid]
        y_prob_val = oob_proba[valid, 1]
        
        tn, fp, fn, tp = confusion_matrix(y_val, y_pred_val).ravel()
        
        attack_recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        benign_recall = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        f1 = 2 * (precision * attack_recall) / (precision + attack_recall) if (precision + attack_recall) > 0 else 0.0
        balanced_accuracy = (attack_recall + benign_recall) / 2
        pr_auc = average_precision_score(y_val, y_prob_val)
        coverage = np.mean(valid)
        
        metrics = {
            "size": size,
            "benign_count": int(np.sum(y_val == 0)),
            "attack_count": int(np.sum(y_val == 1)),
            "tp": int(tp), "tn": int(tn), "fp": int(fp), "fn": int(fn),
            "attack_recall": attack_recall,
            "benign_recall": benign_recall,
            "balanced_accuracy": balanced_accuracy,
            "precision": precision,
            "f1": f1,
            "pr_auc": pr_auc,
            "coverage": coverage,
            "fit_time": t_fit
        }
        metrics_log.append(metrics)
        print(f"  F1: {f1:.4f} | BalAcc: {balanced_accuracy:.4f} | AtkRec: {attack_recall:.4f} | BenRec: {benign_recall:.4f}")
    
    total_time = time.time() - total_t0
    
    # Generate Markdown Report
    report_md = f"""# Phase 5 Learning Curve Report

## Frozen Configuration
* `n_estimators`: {rf_cfg['n_estimators']}
* `max_depth`: {rf_cfg['max_depth']}
* `criterion`: {rf_cfg['criterion']}
* `class_weight`: {rf_cfg.get('class_weight')}
* `random_state`: {rf_cfg['random_state']}
* `bootstrap`: True
* `oob_score`: True (for metrics extraction, does not alter internal splits)

## Runtimes
* **Official 100% Model Fit Time:** {metrics_log[-1]['fit_time']:.1f}s
* **Cumulative Learning Curve Runtime:** {total_time:.1f}s

## Secondary Robustness Analysis (Learning Curve)
The following metrics are computed exclusively on the **Out-Of-Bag (OOB)** samples of the `{total_records:,}` record `X_train` partition. No evaluation or calibration data was accessed. OOB metrics represent internal training-only estimates, not guaranteed unbiased generalization to external datasets.

### Class Counts & Confusion Matrices
| Train Size | Benign Count | Attack Count | TN | FP | FN | TP |
|------------|--------------|--------------|----|----|----|----|
"""
    for m in metrics_log:
        report_md += f"| {m['size']:,} | {m['benign_count']:,} | {m['attack_count']:,} | {m['tn']:,} | {m['fp']:,} | {m['fn']:,} | {m['tp']:,} |\n"

    report_md += """
### Evaluated Metrics
| Train Size | Coverage | Bal Acc | Attack Recall | Benign Recall | Precision | F1-Score | PR-AUC |
|------------|----------|---------|---------------|---------------|-----------|----------|--------|
"""
    for m in metrics_log:
        report_md += f"| {m['size']:,} | {m['coverage']:.2%} | {m['balanced_accuracy']:.4f} | {m['attack_recall']:.4f} | {m['benign_recall']:.4f} | {m['precision']:.4f} | {m['f1']:.4f} | {m['pr_auc']:.4f} |\n"

    report_md += """
## Descriptive Interpretation
The OOB metrics demonstrate whether the RF effectively utilizes increasing amounts of training data within the sampled distribution. We observe high coverage out-of-the-box. If the curve plateaus significantly before 100%, it indicates diminishing gains for the specific attack classes represented in this working sample. It does not definitively prove equivalence to the 16.1M full dataset, nor does it guarantee robust detection for extremely rare attack families not adequately dense in smaller subsets.
"""
    with open(REPORTS_DIR / "learning_curve.md", "w") as f:
        f.write(report_md)
        
    plot_learning_curve(sizes, metrics_log, REPORTS_DIR / "learning_curve.png")
    print("Corrected report and plot generated.")

if __name__ == "__main__":
    run()
