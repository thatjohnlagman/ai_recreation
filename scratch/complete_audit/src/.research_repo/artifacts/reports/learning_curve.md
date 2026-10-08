# Phase 5 Learning Curve Report

## Frozen Configuration
* `n_estimators`: 200
* `max_depth`: 20
* `criterion`: gini
* `class_weight`: None
* `random_state`: 42
* `bootstrap`: True
* `oob_score`: True (for metrics extraction, does not alter internal splits)

## Runtimes
* **Official 100% Model Fit Time:** 33.5s
* **Cumulative Learning Curve Runtime:** 80.1s

## Secondary Robustness Analysis (Learning Curve)
The following metrics are computed exclusively on the **Out-Of-Bag (OOB)** samples of the `178,500` record `X_train` partition. No evaluation or calibration data was accessed. OOB metrics represent internal training-only estimates, not guaranteed unbiased generalization to external datasets.

### Class Counts & Confusion Matrices
| Train Size | Benign Count | Attack Count | TN | FP | FN | TP |
|------------|--------------|--------------|----|----|----|----|
| 44,625 | 36,973 | 7,652 | 36,869 | 104 | 444 | 7,208 |
| 89,250 | 74,029 | 15,221 | 73,870 | 159 | 850 | 14,371 |
| 133,875 | 111,137 | 22,738 | 110,913 | 224 | 1,232 | 21,506 |
| 178,500 | 148,114 | 30,386 | 147,876 | 238 | 1,674 | 28,712 |

### Evaluated Metrics
| Train Size | Coverage | Bal Acc | Attack Recall | Benign Recall | Precision | F1-Score | PR-AUC |
|------------|----------|---------|---------------|---------------|-----------|----------|--------|
| 44,625 | 100.00% | 0.9696 | 0.9420 | 0.9972 | 0.9858 | 0.9634 | 0.9746 |
| 89,250 | 100.00% | 0.9710 | 0.9442 | 0.9979 | 0.9891 | 0.9661 | 0.9759 |
| 133,875 | 100.00% | 0.9719 | 0.9458 | 0.9980 | 0.9897 | 0.9673 | 0.9774 |
| 178,500 | 100.00% | 0.9717 | 0.9449 | 0.9984 | 0.9918 | 0.9678 | 0.9783 |

## Descriptive Interpretation
The OOB metrics demonstrate whether the RF effectively utilizes increasing amounts of training data within the sampled distribution. We observe high coverage out-of-the-box. If the curve plateaus significantly before 100%, it indicates diminishing gains for the specific attack classes represented in this working sample. It does not definitively prove equivalence to the 16.1M full dataset, nor does it guarantee robust detection for extremely rare attack families not adequately dense in smaller subsets.
