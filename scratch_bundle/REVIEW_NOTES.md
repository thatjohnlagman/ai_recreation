# Phase 7 Review Notes

## Environment
- Python: 3.9.6
- NumPy: 2.0.2
- scikit-learn: 1.6.1
- pandas: 2.3.3
- joblib: 1.5.3

## Interfaces
- `BaseAttack(feature_names, modifiable_mask, training_bounds)`
  - `project_and_clip(X_adv, X_orig)`
  - `calculate_magnitudes(X_adv, X_orig)`
- `BlackBoxOracle(predict_func, max_queries_per_sample=None)`
  - `predict(X, sample_ids, stage='unknown')`
  - Returns `BudgetExhaustedFailure`, `BatchBudgetFailure`, or `DuplicateSampleIDsFailure` structurally.

## Boundary Query Accounting
- **Ineligible inputs:** 1 query (eligibility check). Attempted=False.
- **Exhausted reference searches:** 1 eligibility query + N screening queries. Aborts search at exactly 39 queries to preserve exactly 11 minimum budget required for binary search.
- **Successful endpoint discovery:** 1 eligibility + M screening queries (M >= 1). Leaves exactly 12 budget for binary search.
- **Complete valid attempt:** Eligibility + Endpoint Screenings + exactly 10 binary search steps + 1 Final verification.
## Metric Definitions
- **Eligibility:** True label is Attack AND Target model predicts Attack initially.
- **Success:** Modified sample predicted as Benign, within query budget, adhering to all bounds and masks.
- **L0:** Number of exact float32 changed features.
- **L1, L2, L∞:** Computed in float64 space based on exact distance between projected output and original.

## Test Mapping & Phase 7 Safety
- All tests rewritten to strictly use synthetic matrices mapping 1:1 to every branching case required.
- `predict_proba` and `estimators_` mentions in `test_attacks.py` are intentional negative assertions verifying that the restricted API successfully hides model internals.
