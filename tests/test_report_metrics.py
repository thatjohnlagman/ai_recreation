import os
import sys
from pathlib import Path
import numpy as np
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score, balanced_accuracy_score

def test_metric_relationships():
    # Synthetic test to verify that the math relationships asserted in the script are valid
    # and that balanced accuracy != F1 score for imbalanced data
    
    y_true = np.array([0]*800 + [1]*200) # 800 benign, 200 attack
    # Simulate a model with 95% attack recall and 99% benign recall
    # TP = 190, FN = 10, TN = 792, FP = 8
    
    y_pred = np.zeros(1000)
    y_pred[0:8] = 1 # 8 false positives
    y_pred[800:990] = 1 # 190 true positives
    # 10 false negatives (left as 0)
    
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    assert tn == 792
    assert fp == 8
    assert fn == 10
    assert tp == 190
    
    attack_recall = tp / (tp + fn)
    benign_recall = tn / (tn + fp)
    
    assert np.isclose(attack_recall, 0.95)
    assert np.isclose(benign_recall, 0.99)
    
    # Balanced Accuracy: arithmetic mean of recalls
    bal_acc = (attack_recall + benign_recall) / 2
    sklearn_bal_acc = balanced_accuracy_score(y_true, y_pred)
    assert np.isclose(bal_acc, sklearn_bal_acc), f"Expected {sklearn_bal_acc}, got {bal_acc}"
    
    # Precision
    precision = tp / (tp + fp)
    sklearn_precision = precision_score(y_true, y_pred, pos_label=1)
    assert np.isclose(precision, sklearn_precision)
    
    # F1 Score: harmonic mean of precision and recall
    f1 = 2 * (precision * attack_recall) / (precision + attack_recall)
    sklearn_f1 = f1_score(y_true, y_pred, pos_label=1)
    assert np.isclose(f1, sklearn_f1)
    
    # Crucial assertion: Balanced accuracy should NOT equal F1 in this imbalanced case
    assert not np.isclose(bal_acc, f1), "Balanced Accuracy and F1 Score should not be equal!"
    
    print("Metric math validation tests passed successfully.")

if __name__ == "__main__":
    test_metric_relationships()
