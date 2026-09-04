import numpy as np
from sklearn.metrics import average_precision_score

def _validate_bool_mask(mask, name):
    if mask is None:
        return None
    if isinstance(mask, bool) or not isinstance(mask, np.ndarray):
        raise TypeError(f"{name} must be a numpy array")
    if mask.ndim != 1:
        raise ValueError(f"{name} must be 1D")
    if mask.dtype != bool:
        raise TypeError(f"{name} must have dtype bool")
    return mask

def calculate_metrics(y_true, y_pred, positive_scores, eligible_mask=None, attempted_mask=None, successful_mask=None, zero_division_value=0.0, asr_applicable=True):
    # Enforce strict binary labels and 1D arrays
    if not isinstance(y_true, np.ndarray) or not isinstance(y_pred, np.ndarray) or not isinstance(positive_scores, np.ndarray):
        raise TypeError("y_true, y_pred, and positive_scores must be numpy arrays")
        
    if y_true.ndim != 1 or y_pred.ndim != 1 or positive_scores.ndim != 1:
        raise ValueError("Inputs must be 1D arrays")
        
    if len(y_true) != len(y_pred) or len(y_true) != len(positive_scores):
        raise ValueError("Arrays must have equal lengths")
        
    if not np.all(np.isin(y_true, [0, 1])) or not np.all(np.isin(y_pred, [0, 1])):
        raise ValueError("Labels and predictions must be strictly binary [0, 1]")
        
    if not np.all(np.isfinite(positive_scores)):
        raise ValueError("Scores must be finite")
        
    if np.any(positive_scores < 0.0) or np.any(positive_scores > 1.0):
        raise ValueError("Scores must be within [0, 1]")
        
    if not np.isfinite(zero_division_value) or zero_division_value < 0.0 or zero_division_value > 1.0:
        raise ValueError("zero_division_value must be finite and within [0, 1]")
        
    eligible_mask = _validate_bool_mask(eligible_mask, "eligible_mask")
    attempted_mask = _validate_bool_mask(attempted_mask, "attempted_mask")
    successful_mask = _validate_bool_mask(successful_mask, "successful_mask")
    
    if eligible_mask is not None and len(eligible_mask) != len(y_true):
        raise ValueError("Mask length mismatch")
    if attempted_mask is not None and len(attempted_mask) != len(y_true):
        raise ValueError("Mask length mismatch")
    if successful_mask is not None and len(successful_mask) != len(y_true):
        raise ValueError("Mask length mismatch")
        
    if asr_applicable:
        if eligible_mask is None or attempted_mask is None or successful_mask is None:
            raise ValueError("asr_applicable=True requires eligible, attempted, and successful masks")
    
    if asr_applicable is False:
        if attempted_mask is not None and np.any(attempted_mask):
            raise ValueError("non-applicable ASR requires zero attempted samples")
        if successful_mask is not None and np.any(successful_mask):
            raise ValueError("non-applicable ASR requires zero successful samples")
            
    if attempted_mask is not None and eligible_mask is not None:
        if not np.all(eligible_mask[attempted_mask]):
            raise ValueError("Attempted mask must be a subset of eligible mask")
            
    if successful_mask is not None and attempted_mask is not None:
        if not np.all(attempted_mask[successful_mask]):
            raise ValueError("Successful mask must be a subset of attempted mask")

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
    
    # PR-AUC via average_precision_score exactly
    if len(np.unique(y_true)) > 1:
        pr_auc = average_precision_score(y_true, positive_scores)
    else:
        pr_auc = zero_division_value
        
    eligible_count = np.sum(eligible_mask) if eligible_mask is not None else 0
    attempted_count = np.sum(attempted_mask) if attempted_mask is not None else 0
    successful_count = np.sum(successful_mask) if successful_mask is not None else 0
    
    if asr_applicable:
        asr = successful_count / attempted_count if attempted_count > 0 else zero_division_value
    else:
        asr = None
        
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
        "asr": float(asr) if asr is not None else None,
        "asr_applicable": bool(asr_applicable)
    }
