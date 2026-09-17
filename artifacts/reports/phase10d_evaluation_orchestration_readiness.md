# Phase 10D: Official Evaluation Orchestration Readiness Report (v2 Repair)

**Date:** 2026-09-18T02:15:00+08:00  
**Status:** PASS — Orchestration Readiness Certified  
**Commit:** Pending Phase 10D v2 commit  
**Protocol Freeze Tag:** `phase10-protocol-freeze` (`65005505415a2bdf2d5744dbd135e9214e74081a`)  

---

## 1. Executive Summary
Phase 10D v2 establishes, repairs, and certifies that the official Phase 10 evaluation execution pipeline can execute, resume, pair, and validate correctly before official evaluation is authorized.

> [!IMPORTANT]
> **Scope & Technical Readiness vs Empirical Effectiveness**:
> Technical readiness—**not empirical effectiveness**—has been tested and certified.
> Official evaluation (`--execute`) was NOT invoked. Zero evaluation Parquets (`X_eval.parquet`, `metadata_eval.parquet`) were opened during readiness testing or benchmarking. No files were written to `artifacts/evaluation_runs/`.
> The empirical effectiveness of Recall-Aware control relative to Base defenses remains strictly unknown until official execution is authorized and conducted.

**Key Invariants & Binding Corrections (12 / 12 Implemented):**
1. **Real-Run Provenance (11 Fields)**: Constructed full 11-field provenance dictionary directly from canonical on-disk artifacts (`frozen_rf.joblib`, `standard_scaler.joblib`, `feature_names.json`, `feature_mask.json`, `training_bounds.json`, `evaluation_roles.csv`, `evaluation_batches.csv`, and all 4 config YAMLs). Added pre-run dry `CompletionMarker` validation before batch 0. Added red regression test proving failure on missing provenance.
2. **Removed Circular Cache Trust**: Replaced self-referential manifest trust with independent pinning against `artifacts/reports/cache_inventory_v2.json`. Validates `X_attacked.parquet`, `status.parquet`, `manifest.json`, and `completion.json` hashes before manifest is forwarded to provider. Added test proving self-consistent but inventory-divergent caches are rejected.
3. **Enforced Git Cleanliness (Phase 10B Policy)**: Rejects staged/unstaged tracked modifications, validates freeze tag ancestry, ensures zero diff in `src/`, and strictly rejects untracked code/docs while allowlisting generated caches, logs, and ZIP bundles. Added comprehensive test coverage.
4. **Repaired Disk-Space Gate**: Catches only `OSError` when obtaining filesystem stats; raises `RuntimeError` and aborts immediately if free space is below the configured threshold (10.0 GB).
5. **Removed Stale Defense Bounds**: Derived fixed base intensities and dynamic bounds directly from frozen `defenses.yaml`: AFP base 0.0003, bounds [0.0, 0.0003]; RS base 0.0002, bounds [0.0, 0.0002]; FS base 2.0, bounds [0.0, 2.0].
6. **Strengthened Measurement-Label Resolution**: Validates without opening evaluation Parquets that roles and batches manifests contain required columns, exactly 72,000 unique measurement positions, 18,000 non-overlapping crafting positions, 500 rows/batch, normalized 0..143 batch IDs, and consistent composite IDs and binary labels. Added synthetic rejection tests.
7. **Repaired Completed-Run Reuse Validation**: Required agreement of directory name, `RunSummary` run_id, `completion.json` run_id, and every batch-record run_id with the target matrix row, exact provenance match, expected cache identity, zero unexpected files, and completion marker written last.
8. **Repaired Alias Publication**: Established unambiguous pointer-only contract: aliases publish `alias_pointer.json` and `completion.json`, never masquerading as independent runs. Target run outputs validated prior to alias creation; atomic staging and renaming; validated reuse and quarantine.
9. **Dependency-Safe Filtering**: When filters select an alias, its required primary target execution is automatically resolved and scheduled ahead of the alias. Non-positive `--max-runs` arguments are explicitly rejected.
10. **Production-Wiring Canary**: Added dedicated end-to-end canary exercising production provenance, independent cache inventory validation, production factories, 144-batch orchestration, output reopening, completion publication, reuse, and quarantine entirely in a synthetic temporary directory.
11. **Corrected Runtime & Storage Interpretation**: Distinguished defense-inference lower-bound (~3.21 - 3.79h) from full wall-clock prediction (~3.7 - 4.0h); calculated realistic production storage (~2.23 MB for `scores.json`, ~0.57 GB total for 252 runs) based on production schema records.
12. **Repaired Review Bundle**: Created `phase10d_evaluation_orchestration_readiness_bundle_v2.zip` with self-inclusive manifest and hash ledger, verified exact member set extraction, zero duplicates, and zero forbidden files.

---

## 2. Protected Artifacts and Cache Immutability Ledger

### Protected Artifact Hashes (Before vs After)

| Artifact | Path | Expected Protected Hash | Verified Hash | Status |
| :--- | :--- | :--- | :--- | :--- |
| **Frozen RF Model** | `artifacts/models/frozen_rf.joblib` | `9608672c5d5e38a9272c560679cdd2291399de0e0917c8c9dff373bfd200f51d` | `9608672c5d5e38a9272c560679cdd2291399de0e0917c8c9dff373bfd200f51d` | **MATCH** |
| **Evaluation Roles** | `data/manifests/evaluation_roles.csv` | `cbf650879aa1369fa26b803777f30d4b0add9e7aff2c05d399ba7f42563f7b45` | `cbf650879aa1369fa26b803777f30d4b0add9e7aff2c05d399ba7f42563f7b45` | **MATCH** |
| **Evaluation Batches** | `data/manifests/evaluation_batches.csv` | `4084017e5593732e455763416f7fc38254fc9dab2466eca289b66e68dc0ff79a` | `4084017e5593732e455763416f7fc38254fc9dab2466eca289b66e68dc0ff79a` | **MATCH** |
| **Experiment Config** | `configs/experiment.yaml` | `a1c5a389b1bc3c66311c7a96a7ccc3ee54af86506e64b38be7f26ad3c3d84c70` | `a1c5a389b1bc3c66311c7a96a7ccc3ee54af86506e64b38be7f26ad3c3d84c70` | **MATCH** |

### Complete 11-Field Canonical Provenance Hashes

| Provenance Key | Canonical Source File | Calculated SHA-256 |
| :--- | :--- | :--- |
| `frozen_rf_hash` | `artifacts/models/frozen_rf.joblib` | `9608672c5d5e38a9272c560679cdd2291399de0e0917c8c9dff373bfd200f51d` |
| `scaler_hash` | `artifacts/preprocessors/standard_scaler.joblib` | `1bfce7d5d71a814c1d2e1fd4eec03d2cf570f5e1ad3a7f804961ad3cb8c706ff` |
| `feature_names_hash` | `artifacts/preprocessors/feature_names.json` | `5c84d7a8e52dbb065a6e828116b432a2f4abf04901f40954ecad1e67e35b71db` |
| `feature_mask_hash` | `artifacts/preprocessors/feature_mask.json` | `450d06f658097b87fa183b8c454e99f5726219bb7d483b879ec264ec4a45a331` |
| `training_bounds_hash` | `artifacts/preprocessors/training_bounds.json` | `fbf2a2db490432320b3a7be1d120a1ceb83e6a9ee8cc8a846c4f2bbd377b2185` |
| `evaluation_roles_hash` | `data/manifests/evaluation_roles.csv` | `cbf650879aa1369fa26b803777f30d4b0add9e7aff2c05d399ba7f42563f7b45` |
| `evaluation_batches_hash` | `data/manifests/evaluation_batches.csv` | `4084017e5593732e455763416f7fc38254fc9dab2466eca289b66e68dc0ff79a` |
| `attacks_yaml_hash` | `configs/attacks.yaml` | `f409e530bc1ff3fb7fffe6a4f91e9bca9f104d493e87870a01aeb134ff837cf2` |
| `controllers_yaml_hash` | `configs/controllers.yaml` | `ea89b5c3fe80ef8a0cf66fc12cf648c89ce58a8a47ff9618b0c63402422501a3` |
| `defenses_yaml_hash` | `configs/defenses.yaml` | `fc29cb7e7ce71b80c54ea9c6a1e6878e34271810486c8f62f3a69399ebfa7fa4` |
| `experiment_yaml_hash` | `configs/experiment.yaml` | `a1c5a389b1bc3c66311c7a96a7ccc3ee54af86506e64b38be7f26ad3c3d84c70` |

### Official Attack Caches (15 Caches, 60 Files)
All 15 official cache directories in `artifacts/caches/` were verified against `artifacts/reports/cache_inventory_v2.json`:
- `DecisionBoundary_42` .. `DecisionBoundary_46` (5 caches, 20 files): Verified
- `SilentProbing_42` .. `SilentProbing_46` (5 caches, 20 files): Verified
- `SurrogateTransfer_42` .. `SurrogateTransfer_46` (5 caches, 20 files): Verified
- Total verified cache artifacts: **60 / 60 bit-for-bit identical**.

---

## 3. Dynamic Evaluation Matrix Verification

| Metric | Required / Expected Count | Derived from Frozen Configs | Status |
| :--- | :--- | :--- | :--- |
| **Total Matrix References** | 279 | 279 | **PASS** |
| **Primary Comparison References** | 90 | 90 | **PASS** |
| **Sensitivity Analysis References** | 189 | 189 | **PASS** |
| **Exact C1 Aliases** | 27 | 27 | **PASS** |
| **Unique Executions** | 252 | 252 | **PASS** |
| **Batches per Execution** | 144 | 144 | **PASS** |
| **Unique Batch Evaluations** | 36,288 | 36,288 | **PASS** |

---

## 4. Test Execution Summary

### Focused Tests (`tests/test_run_evaluation.py`)
- `test_matrix_counts_and_alias_reuse`: **PASSED**
- `test_base_c1_exact_pairing`: **PASSED**
- `test_common_randomness_under_equal_intensity`: **PASSED**
- `test_timing_and_label_isolation`: **PASSED**
- `test_controller_reset_between_runs`: **PASSED**
- `test_cache_validation_rejects_corrupt_manifest`: **PASSED**
- `test_cache_validation_rejects_inventory_divergent_cache`: **PASSED**
- `test_completed_run_reuse`: **PASSED**
- `test_corrupt_or_incomplete_run_quarantine`: **PASSED**
- `test_output_reopening_catches_tampered_metrics`: **PASSED**
- `test_72000_row_and_global_metric_invariants`: **PASSED**
- `test_silent_probing_null_asr_vs_applicable_asr`: **PASSED**
- `test_atomic_failure_handling`: **PASSED**
- `test_deterministic_repeated_execution`: **PASSED**
- `test_official_artifacts_and_caches_immutability`: **PASSED**
- `test_production_provenance_construction_and_regression`: **PASSED**
- `test_git_cleanliness_enforcement`: **PASSED**
- `test_disk_space_gate_rejection`: **PASSED**
- `test_defense_base_bounds_factory`: **PASSED**
- `test_measurement_label_resolution_rejections`: **PASSED**
- `test_alias_publication_and_reuse`: **PASSED**
- `test_dependency_safe_filtering`: **PASSED**
- `test_non_positive_max_runs_rejected`: **PASSED**
- `test_production_wiring_canary`: **PASSED** (144 batches end-to-end with real models & provenance)
**Total Focused Tests:** **24 / 24 PASSED** in 20.99s.

### Full Test Suite
- Total tests collected: **280**
- Total tests passed: **280** (100%)
- Total failures: **0**
- Execution time: **29.42s**

---

## 5. Measured Runtime and Storage Projections

### Measured Training-Derived Timings (Apple Silicon M4)
- **AFP per 500-sample batch:** 0.0517 s (0.10 ms/sample)
- **FS per 500-sample batch:** 0.0462 s (0.09 ms/sample)
- **RS per 500-sample batch (11-member ensemble):** 1.0297 s (2.06 ms/sample)

### Projected Single Run (144 Batches = 72,000 Samples)
- **One AFP Run:** 7.45 s (0.12 min)
- **One FS Run:** 6.65 s (0.11 min)
- **One RS Run:** 148.28 s (2.47 min)

### Matrix-Level Projections
- **Primary Comparison (90 runs: 30 AFP, 30 FS, 30 RS):** **81.19 min (1.35 hours)** defense-inference lower bound; **~1.3 – 1.4 hours** wall-clock.
- **Full Unique Executions (252 runs: 84 AFP, 84 FS, 84 RS):** **227.33 min (3.79 hours)** defense-inference lower bound; **~3.7 – 4.0 hours** wall-clock.
- **Sensitivity Aliases (27 runs):** **0.00 min** (instant metadata pointer).
- **Storage Footprint:** ~**2.23 MB** for `scores.json` per run; ~**2.33 MB** total per run; **~0.57 GB** aggregate for 252 runs (well within available 68+ GB).
