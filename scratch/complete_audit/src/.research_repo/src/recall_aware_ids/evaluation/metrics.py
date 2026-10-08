"""
src/recall_aware_ids/evaluation/metrics.py
Batch-level metric computation.

Implements exact thesis definitions (Chapter 3):
    Precision  = TP / (TP + FP)   [attack = positive class]
    Recall     = TP / (TP + FN)
    F1-Score   = 2 * Precision * Recall / (Precision + Recall)

Zero-division handling (from IMPLEMENTATION_DECISIONS.md):
    If TP+FP=0: Precision = 0.0 (not NaN)
    If TP+FN=0: Recall    = 0.0 (not NaN)
    If P+R=0:   F1        = 0.0 (not NaN)

Raw confusion counts (TP, FP, TN, FN) are always recorded alongside
derived metrics. No values are hidden or suppressed.
"""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np


@dataclass
class BatchMetrics:
    """Complete metrics for one evaluation batch — maps directly to Thesis Table 7 fields."""
    batch_id: int
    sample_count: int
    tp: int
    fp: int
    tn: int
    fn: int
    precision: float
    recall: float
    f1_score: float
    precision_defined: bool   # False if TP+FP=0
    recall_defined: bool      # False if TP+FN=0


def compute_batch_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    batch_id: int,
) -> BatchMetrics:
    """
    Compute TP, FP, TN, FN and derived metrics for one batch.

    y_true, y_pred: binary arrays where 1 = attack (positive class).
    """
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)

    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))

    # Precision
    if tp + fp > 0:
        precision = tp / (tp + fp)
        precision_defined = True
    else:
        precision = 0.0
        precision_defined = False

    # Recall
    if tp + fn > 0:
        recall = tp / (tp + fn)
        recall_defined = True
    else:
        recall = 0.0
        recall_defined = False

    # F1
    if precision + recall > 0:
        f1 = 2.0 * precision * recall / (precision + recall)
    else:
        f1 = 0.0

    return BatchMetrics(
        batch_id=batch_id,
        sample_count=len(y_true),
        tp=tp,
        fp=fp,
        tn=tn,
        fn=fn,
        precision=precision,
        recall=recall,
        f1_score=f1,
        precision_defined=precision_defined,
        recall_defined=recall_defined,
    )


def aggregate_metrics(metrics_list: list[BatchMetrics]) -> dict:
    """
    Compute aggregate statistics across batches for reporting.
    Returns means, stds, and min/max for Precision, Recall, F1.
    """
    if not metrics_list:
        return {}

    precisions = np.array([m.precision for m in metrics_list])
    recalls    = np.array([m.recall    for m in metrics_list])
    f1s        = np.array([m.f1_score  for m in metrics_list])

    return {
        "n_batches": len(metrics_list),
        "total_samples": sum(m.sample_count for m in metrics_list),
        "total_tp": sum(m.tp for m in metrics_list),
        "total_fp": sum(m.fp for m in metrics_list),
        "total_tn": sum(m.tn for m in metrics_list),
        "total_fn": sum(m.fn for m in metrics_list),
        "precision_mean": float(precisions.mean()),
        "precision_std":  float(precisions.std(ddof=1)) if len(precisions) > 1 else 0.0,
        "precision_min":  float(precisions.min()),
        "precision_max":  float(precisions.max()),
        "recall_mean":    float(recalls.mean()),
        "recall_std":     float(recalls.std(ddof=1)) if len(recalls) > 1 else 0.0,
        "recall_min":     float(recalls.min()),
        "recall_max":     float(recalls.max()),
        "f1_mean":        float(f1s.mean()),
        "f1_std":         float(f1s.std(ddof=1)) if len(f1s) > 1 else 0.0,
        "f1_min":         float(f1s.min()),
        "f1_max":         float(f1s.max()),
        "n_undefined_precision": int(sum(not m.precision_defined for m in metrics_list)),
        "n_undefined_recall":    int(sum(not m.recall_defined    for m in metrics_list)),
    }
