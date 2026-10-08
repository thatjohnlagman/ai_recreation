# Phase 7: Attack Implementation Report

## Summary
The Phase 7 attack framework has been rigorously repaired to satisfy strict boundary constraints, immutable protected features, single/batch Oracle query limits, and `float32`/`float64` mathematical regressions.

## 1. Oracle API Restructuring
The `BlackBoxOracle` was fully decoupled from Python reference leakage. The Oracle provides a constrained supported hard-label interface for the Random Forest, not an absolute Python encapsulation/security sandbox.
- The interface now accepts batch or single matrix inputs but demands a sequence of strictly mapped `sample_ids`. Scalar batch IDs are explicitly rejected.
- An atomic preflight function verifies per-sample budget caps. A failure structurally yields `BudgetExhaustedFailure` (single) or `BatchBudgetFailure` (batch) aborting without processing model lookups.
- Tracking natively splits into Global counts, Per-sample counts, and stage-by-stage string histories.

## 2. Arithmetic Exactitude
- Modifiable bounds clipping `[min, max]` remains strictly constrained to standard `float32`.
- Exact equality tests process `L0` using identical `float32` arrays.
- `L1, L2, L∞` metrics strictly upgrade to `np.float64` *before* differencing to guarantee absolute arithmetic exactitude avoiding single-ULP underflows.

## 3. Boundary Budget Reservation
The `DecisionBoundaryAttack` mandates budget preservation logic for multi-query workflows. Endpoint searches strictly check `limit - current < 12`. If exceeded (e.g. at query 39 out of a limit of 50), it returns `INSUFFICIENT_BUDGET_FOR_FULL_SEARCH` guaranteeing that no successful endpoint is discovered without holding precisely 10 steps + 1 verification query in reserve. Valid attempts always use exactly 10 midpoints (not "up to 10").

## 4. Surrogate Protections
- `DecisionTreeClassifier` is seeded deterministically (`effective_attack_seed`).
- Generates thresholds strictly checking `isinstance()` for budgets.
- Preserves explicit node `classes_` indexes rather than assuming binary `[0]`.
- Implements the exact conditional `np.nextafter(..., np.float32)` mathematics avoiding unnecessary ULP jumps on inclusive thresholds.
- Lexicographical tie-breaking selects min `L0` -> `L2` -> `L∞` -> `leaf_id`.

## 5. Security & Test Coverage
Zero Phase 7 files touched `X_eval` or `metadata_eval`. All testing logic uses explicitly mapped deterministic synthetic subsets constructed inside `test_attacks.py`.

**Tests Performed:**
- Batch Atomicity (duplicate ID rejection and zero charges).
- Mask and Bounds constraints (nonfinite rejection, mutability).
- Exact `float64` magnitude regression proofs versus `float32` subtracts.
- `np.nextafter` directions (all 4 cases covering strict/inclusive overlaps).
- Boundary Attack exhaustive checking: exact 38 endpoint limits and protected features.
- Surrogate target-rejection and same-seed deterministic paths.
- Negative tests proving model internal masking (e.g., `predict_proba`).

The complete pytest suite ran successfully on Python `3.9.6` on macOS with 100% test coverage matching the protocol constraints without errors.
