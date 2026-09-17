# Phase 10D: Official Evaluation Orchestration Specification

## 1. Overview and Core Invariants
The Phase 10 evaluation matrix comprises 279 reference configurations evaluated across 144 sequential batches of 500 network flows (72,000 total flows per run), deriving from the frozen configuration and dataset partitions established in Phase 10A and Phase 10B.

This document establishes the architecture and execution contract of the evaluation orchestrator (`scripts/run_evaluation.py`), guaranteeing:
1. **Zero Unintentional Execution**: Default execution mode performs non-mutating preflight only. Full evaluation execution requires explicit `--execute`.
2. **Deterministic Whole-Run Restart**: Runs are never resumed halfway through a controller state sequence. Any failure, crash, or incomplete run restarts cleanly from batch 0.
3. **Run Immutability & Reuse**: Completed runs with valid `completion.json` and validated serialized outputs are never recomputed.
4. **Quarantine of Corrupted / Incomplete Runs**: Partial or invalid output directories are immediately quarantined to `<run_id>_quarantined_<timestamp>` with detailed failure diagnostic records.
5. **Exact Pairing & Common Randomness**: Base and RA configurations share identical stochastic seeds and batch-level pseudorandom state sequences `rng = np.random.RandomState(int(seed) * 10000 + int(batch_id))`. Under equal defense intensities, Base and RA generate bit-identical defended inputs.
6. **Strict Timing Isolation**: At batch $t$, defense intensity is determined prior to accessing batch-$t$ labels. Prediction and defense occur on defended outputs. Labels are accessed post-prediction to calculate TP/FN. Feedback is delivered strictly for batch $t+1$. Base receives no feedback.
7. **Two-Stage Atomic Publication**: All execution writes to a staging directory `.staging_<run_id>_<timestamp>`. Serialized outputs (`config.json`, `confusion.json`, `scores.json`, `run_summary.json`) are reopened and validated via `_validate_run_outputs()` before `completion.json` is written. Upon validation, the directory is atomically renamed to its final target path `<output_dir>/<run_id>`.

---

## 2. Dynamic Matrix Derivation
The evaluation matrix is derived dynamically from `configs/experiment.yaml`, `configs/attacks.yaml`, `configs/defenses.yaml`, and `configs/controllers.yaml`:
- **Primary References**: 5 seeds (42, 43, 44, 45, 46) × 3 attacks (`SilentProbing`, `SurrogateTransfer`, `DecisionBoundary`) × 3 defenses (`afp`, `feature_squeezing`, `randomized_smoothing`) × 2 controllers (`Base`, `C1`) = **90 references**.
- **Sensitivity References**: 3 seeds (42, 43, 44) × 3 attacks × 3 defenses × 7 controllers (`C1`..`C7`) = **189 references**.
- **Total Matrix References**: $90 + 189 =$ **279 references**.
- **C1 Aliases**: Sensitivity runs for C1 (3 seeds × 3 attacks × 3 defenses = 27 runs) are exact mathematical duplicates of primary C1 runs. These **27 runs** are marked with `is_alias = True` and point to `alias_for_run_id`.
- **Unique Executions**: $279 - 27 =$ **252 unique executions**.
- **Batches per Execution**: **144 batches** of 500 samples each.
- **Unique Batch Evaluations**: $252 \times 144 =$ **36,288 batch evaluations**.

---

## 3. Defense Adapters and Base Parameters
1. **Adaptive Feature Poisoning (AFP)**:
   - Fixed base intensity: $\epsilon = 0.0003$.
   - Calibrated constant: $\alpha = 0.5$.
   - Controller intensity range: $[0.0, 0.005]$.
   - Model inference: `model.predict(X_def)` and `model.predict_proba(X_def)[:, 1]` on defended output.
2. **Feature Squeezing (FS)**:
   - Fixed base continuous intensity: $2.0$.
   - Effective bit depth logged: $d = \max(0, 6 - \lfloor\text{intensity}\rfloor)$.
   - Controller intensity range: $[1.0, 5.0]$.
   - Model inference: `model.predict(X_def)` and `model.predict_proba(X_def)[:, 1]` on defended output.
3. **Randomized Smoothing (RS)**:
   - Fixed base intensity: $\sigma = 0.0002$.
   - Controller intensity range: $[0.0, 0.001]$.
   - 11-member ensemble inference with chunk size 100 to remain within M4/16-GB memory limits.
   - Positive score: positive vote fraction $\text{votes} / 11$. Hard prediction: majority vote.
4. **Protected Features**:
   - All 15 protected feature columns remain strictly immutable. `DefenseResult.protected_feature_modification_count == 0` is asserted on every batch.
   - All values remain strictly within training bounds: `final_bounds_violation_count == 0`, `final_nan_count == 0`, `final_inf_count == 0`.

---

## 4. Run Lifecycle and State Machine
Each unique execution transitions through the following formal state machine:
```
               [Select Run]
                    |
          Does <run_id> exist?
             /             \
          [YES]            [NO]
           /                 \
  Validate Run Outputs        \
      /         \              \
  [VALID]     [INVALID]         \
    |             |              \
 [REUSE]     Quarantine Dir       \
 (Skip)      to <run>_quarantined  \
                  \                 /
                   \               /
                    v             v
             Create Staging Directory
           .staging_<run_id>_<timestamp>
                          |
             Reset Controller State (batch 0)
                          |
             Execute 144 Batches in Order
                          |
             Write config.json, confusion.json,
             scores.json, run_summary.json
                          |
             Reopen & Validate Serialized Outputs
                          |
             Write completion.json (LAST)
                          |
             Atomic Directory Rename to <output_dir>/<run_id>
```

---

## 5. Provenance Requirements
Every completed run verifies and records full provenance in `completion.json` and `run_summary.json`:
- `frozen_rf_hash`: `9608672c5d5e38a9272c560679cdd2291399de0e0917c8c9dff373bfd200f51d`
- `evaluation_roles_hash`: `cbf650879aa1369fa26b803777f30d4b0add9e7aff2c05d399ba7f42563f7b45`
- `evaluation_batches_hash`: `4084017e5593732e455763416f7fc38254fc9dab2466eca289b66e68dc0ff79a`
- `attacks_yaml_hash`, `defenses_yaml_hash`, `controllers_yaml_hash`, `experiment_yaml_hash`
- `cache_manifest_hash`: SHA-256 of the attack cache manifest used for the run.
- Zero placeholder or dummy strings permitted.
