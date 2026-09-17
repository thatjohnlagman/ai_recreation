# Phase 10D: Official Evaluation Orchestration Specification

## 1. Overview and Core Invariants
The Phase 10 evaluation matrix comprises 279 reference configurations evaluated across 144 sequential batches of 500 network flows (72,000 total flows per run), deriving from the frozen configuration and dataset partitions established in Phase 10A and Phase 10B.

This document establishes the architecture and execution contract of the evaluation orchestrator (`scripts/run_evaluation.py`), guaranteeing:
1. **Zero Unintentional Execution**: Default execution mode performs non-mutating preflight only. Full evaluation execution requires explicit `--execute`.
2. **Deterministic Whole-Run Restart**: Runs are never resumed halfway through a controller state sequence. Any failure, crash, or incomplete run restarts cleanly from batch 0.
3. **Run Immutability & Validated Reuse**: Completed runs with valid `completion.json` and validated serialized outputs are never recomputed. Reuse validates seed, scenario, defense, config, cache identity, expected provenance, and internal batch-record run IDs.
4. **Quarantine of Corrupted / Incomplete Runs**: Partial or invalid output directories are immediately quarantined to `<run_id>_quarantined_<timestamp>` with detailed failure diagnostic records.
5. **Exact Pairing & Common Randomness**: Base and RA configurations share identical stochastic seeds and batch-level pseudorandom state sequences `rng = np.random.RandomState(int(seed) * 10000 + int(batch_id))`. Under equal defense intensities, Base and RA generate bit-identical defended inputs.
6. **Strict Timing Isolation**: At batch $t$, defense intensity is determined prior to accessing batch-$t$ labels. Prediction and defense occur on defended outputs. Labels are accessed post-prediction to calculate TP/FN. Feedback is delivered strictly for batch $t+1$. Base receives no feedback.
7. **Two-Stage Atomic Publication**: All execution writes to a staging directory `.staging_<run_id>_<timestamp>`. Serialized outputs (`config.json`, `confusion.json`, `scores.json`, `run_summary.json`) are reopened and validated via `_validate_run_outputs()` before `completion.json` is written. Upon validation, the directory is atomically renamed to its final target path `<output_dir>/<run_id>`.
8. **Independent Cache Inventory Pinning**: Official attack caches are validated against `artifacts/reports/cache_inventory_v2.json` before any manifest or Parquet is trusted. Manifests are never trusted circularly.
9. **Pointer-Only Alias Contract**: Sensitivity C1 aliases are published as pointer-only directories containing `alias_pointer.json` and `completion.json`, never masquerading as independently executed runs.
10. **Dependency-Safe Filtering**: When filters select an alias, its required primary target execution is automatically resolved and scheduled ahead of the alias.

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
Defense fixed base intensities and dynamic bounds are derived directly from frozen `configs/defenses.yaml`:
1. **Adaptive Feature Poisoning (AFP)**:
   - Fixed base intensity: $\epsilon = 0.0003$.
   - Controller intensity range: $[0.0, 0.0003]$.
   - Calibrated constant: $\alpha = 0.5$.
   - Model inference: `model.predict(X_def)` and `model.predict_proba(X_def)[:, 1]` on defended output.
2. **Feature Squeezing (FS)**:
   - Fixed base continuous intensity: $2.0$.
   - Controller intensity range: $[0.0, 2.0]$.
   - Effective bit depth logged: $d = \max(0, 6 - \lfloor\text{intensity}\rfloor)$.
   - Model inference: `model.predict(X_def)` and `model.predict_proba(X_def)[:, 1]` on defended output.
3. **Randomized Smoothing (RS)**:
   - Fixed base intensity: $\sigma = 0.0002$.
   - Controller intensity range: $[0.0, 0.0002]$.
   - 11-member ensemble inference with chunk size 100 to remain within M4/16-GB memory limits.
   - Positive score: positive vote fraction $\text{votes} / 11$. Hard prediction: majority vote.
4. **Protected Features**:
   - All 15 protected feature columns remain strictly immutable. `DefenseResult.protected_feature_modification_count == 0` is asserted on every batch.
   - All values remain strictly within training bounds: `final_bounds_violation_count == 0`, `final_nan_count == 0`, `final_inf_count == 0`.

---

## 4. Complete Provenance Architecture (11 Fields)
Every completed run constructs and verifies all 11 required provenance keys against canonical on-disk artifacts:
1. `frozen_rf_hash`: `artifacts/models/frozen_rf.joblib`
2. `scaler_hash`: `artifacts/preprocessors/standard_scaler.joblib`
3. `feature_names_hash`: `artifacts/preprocessors/feature_names.json`
4. `feature_mask_hash`: `artifacts/preprocessors/feature_mask.json`
5. `training_bounds_hash`: `artifacts/preprocessors/training_bounds.json`
6. `evaluation_roles_hash`: `data/manifests/evaluation_roles.csv`
7. `evaluation_batches_hash`: `data/manifests/evaluation_batches.csv`
8. `attacks_yaml_hash`: `configs/attacks.yaml`
9. `controllers_yaml_hash`: `configs/controllers.yaml`
10. `defenses_yaml_hash`: `configs/defenses.yaml`
11. `experiment_yaml_hash`: `configs/experiment.yaml`

**Pre-Run Dry Validation**:
Before initiating batch 0 of any run, `CompletionMarker` is pre-instantiated and validated with the full 11-field provenance dictionary, ensuring provenance validity before expensive batch inference begins.

---

## 5. Non-Circular Cache Validation
Rather than circularly trusting a cache's own `manifest.json`, the orchestrator implements independent verification:
- Loads authoritative `artifacts/reports/cache_inventory_v2.json`.
- Locates the canonical scenario and seed entry.
- Compares SHA-256 of `X_attacked.parquet`, `status.parquet`, `manifest.json`, and `completion.json` against inventory records.
- Compares internal model, preprocessor, and config hashes against independently computed provenance.
- Only after all 4 files are independently pinned against the inventory is the manifest forwarded to `ConcreteAttackCacheProvider`.
- Self-consistent but inventory-divergent caches are strictly rejected.

---

## 6. Strict Git Execution Cleanliness
Implements the Phase 10B cleanliness policy:
- Validates that `phase10-protocol-freeze` tag is an ancestor of `HEAD`.
- Validates zero diff in `src/` against `phase10-protocol-freeze`.
- Rejects any staged or unstaged modifications to tracked files.
- Rejects untracked code, test, script, config, or documentation files.
- Allowlist strictly permits:
  - `artifacts/caches/`
  - configured evaluation output directory
  - `.gemini/` IDE workspace metadata
  - historical ZIP bundles (`phase10*.zip`)
  - execution logs (`*.log`)
- Fails closed on any Git parsing error.

---

## 7. Disk-Space Safety Gate
- Checks filesystem free space at the evaluation output destination.
- Catches only `OSError` during disk stat acquisition.
- If free space is below minimum threshold (10.0 GB), raises `RuntimeError` and aborts immediately. Warnings are never substituted for safety failures.

---

## 8. Measurement-Label Resolution & Preflight Integrity
`run_preflight()` directly executes manifest cross-validation without opening evaluation Parquets:
- `evaluation_roles.csv` and `evaluation_batches.csv` required columns verified.
- Unique positions asserted: exactly 72,000 measurement positions and 18,000 crafting positions.
- Crafting positions never overlap measurement positions.
- Evaluation batch positions equal exactly the measurement positions.
- Batches normalize strictly to integers $0 \dots 143$, each having exactly 500 rows.
- Binary labels (`y_binary \in \{0, 1\}`) and composite identities (`_source_file`, `_raw_row_idx`) agree wherever present.

---

## 9. Pointer-Only Alias Publication Contract
Aliases represent identical mathematical evaluations (sensitivity C1 runs aliasing primary C1 runs).
- Aliases never masquerade as independently executed runs.
- Published as pointer directories containing:
  - `alias_pointer.json`: immutable record containing `target_run_id`, `target_path`, `target_run_hash`, `alias_tuple`, `target_validation_summary`.
  - `completion.json`: alias-specific completion marker certifying publication.
- Target run outputs are validated before alias creation.
- Published atomically via same-filesystem staging and renaming.
- Corrupted or invalid existing alias directories are quarantined.
- Downstream analysis code resolves `alias_pointer.json` rather than double-counting executions.

---

## 10. Dependency-Safe Filter Planning
Execution planning guarantees safety before starting run 0:
- When filters (`--run-id`, `--config`, etc.) select an alias:
  - If target run is already completed and validated on disk, the alias is scheduled.
  - If target run is not yet completed, the target run is automatically resolved and prepended to the execution sequence ahead of the alias.
- The entire filtered execution plan is validated upfront.
- Non-positive `--max-runs` arguments are rejected with an explicit error.

---

## 11. Genuinely Synthetic Production-Wiring Canary
A dedicated end-to-end canary (`test_production_wiring_canary` in `tests/test_run_evaluation.py`) validates the complete production wiring:
- Genuinely synthetic 72,000-row cache generated entirely in `tmp_path` within training feature bounds, completely prohibiting access to official evaluation caches or evaluation Parquets.
- Production provenance construction (all 11 canonical on-disk artifacts).
- Independent cache validation against a dedicated canary inventory ledger.
- Production defense adapters, policies, and controller factories.
- Full 144-batch orchestration with sequential timing isolation.
- Output reopening, recalculation, and cross-metric verification.
- Completion marker publication, validated reuse, and quarantine behavior.
- Operates entirely in a synthetic temporary directory without opening evaluation Parquets or writing under `artifacts/evaluation_runs/`.
