import numpy as np
from sklearn.metrics import average_precision_score

def calculate_metrics(y_true, y_pred, positive_scores, eligible_mask=None, attempted_mask=None, successful_mask=None, zero_division_value=0.0):
    # Enforce strict binary labels
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)
    
    tp = np.sum((y_true == 1) & (y_pred == 1))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    tn = np.sum((y_true == 0) & (y_pred == 0))
    fn = np.sum((y_true == 1) & (y_pred == 0))
    
    total = tp + fp + tn + fn
    accuracy = (tp + tn) / total if total > 0 else zero_division_value
    
    recall = tp / (tp + fn) if (tp + fn) > 0 else zero_division_value
    precision = tp / (tp + fp) if (tp + fp) > 0 else zero_division_value
    
    if (precision + recall) > 0:
        f1 = 2 * (precision * recall) / (precision + recall)
    else:
        f1 = zero_division_value
        
    specificity = tn / (tn + fp) if (tn + fp) > 0 else zero_division_value
    balanced_accuracy = (recall + specificity) / 2.0
    
    # PR-AUC via average_precision_score
    if len(np.unique(y_true)) > 1:
        pr_auc = average_precision_score(y_true, positive_scores)
    else:
        pr_auc = zero_division_value
        
    eligible_count = np.sum(eligible_mask) if eligible_mask is not None else 0
    attempted_count = np.sum(attempted_mask) if attempted_mask is not None else 0
    successful_count = np.sum(successful_mask) if successful_mask is not None else 0
    
    asr = successful_count / attempted_count if attempted_count > 0 else 0.0 # Silent probing will have attempted=0, meaning ASR=0.0 which maps to not applicable in logs
    
    return {
        "tp": int(tp),
        "fp": int(fp),
        "tn": int(tn),
        "fn": int(fn),
        "accuracy": float(accuracy),
        "recall": float(recall),
        "precision": float(precision),
        "f1": float(f1),
        "balanced_accuracy": float(balanced_accuracy),
        "pr_auc_average_precision": float(pr_auc),
        "eligible_count": int(eligible_count),
        "attempted_count": int(attempted_count),
        "successful_count": int(successful_count),
        "asr": float(asr)
    }
