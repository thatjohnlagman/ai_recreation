# Phase 10D: Official Evaluation Orchestration Readiness Report (v2 Repair)

**Date:** 2026-09-18T02:27:00+08:00  
**Status:** PASS — Orchestration Readiness Certified  
**Commit:** Pending Phase 10D v2 reconciliation commit  
**Protocol Freeze Tag:** `phase10-protocol-freeze` (`65005505415a2bdf2d5744dbd135e9214e74081a`)  

---

## 1. Executive Summary
Phase 10D v2 establishes, repairs, and certifies that the official Phase 10 evaluation execution pipeline can execute, resume, pair, and validate correctly before official evaluation is authorized.

> [!IMPORTANT]
> **Scope & Technical Readiness vs Empirical Effectiveness**:
> Technical readiness—**not empirical effectiveness**—has been tested and certified.
> Official evaluation (`--execute`) was NOT invoked. Zero evaluation Parquets (`X_eval.parquet`, `metadata_eval.parquet`) were opened during readiness testing or benchmarking. No files were written to `artifacts/evaluation_runs/`.
> The empirical effectiveness of Recall-Aware control relative to Base defenses remains strictly unknown until official execution is authorized and conducted.

**Key Invariants & Binding Corrections (12 / 12 Implemented & Reconciled):**
1. **Real-Run Provenance (11 Fields)**: Constructed full 11-field provenance dictionary directly from canonical on-disk artifacts (`frozen_rf.joblib`, `standard_scaler.joblib`, `feature_names.json`, `feature_mask.json`, `training_bounds.parquet`, `evaluation_roles.csv`, `evaluation_batches.csv`, and all 4 config YAMLs). Added pre-run dry `CompletionMarker` validation before batch 0. Added red regression test proving failure on missing provenance.
2. **Removed Circular Cache Trust**: Replaced self-referential manifest trust with independent pinning against `artifacts/reports/cache_inventory_v2.json`. Validates `X_attacked.parquet`, `status.parquet`, `manifest.json`, and `completion.json` hashes before manifest is forwarded to provider. Added test proving self-consistent but inventory-divergent caches are rejected.
3. **Enforced Git Cleanliness (Phase 10B Policy)**: Rejects staged/unstaged tracked modifications, validates freeze tag ancestry, ensures zero diff in `src/`, and strictly rejects untracked code/docs while allowlisting generated caches, logs, and ZIP bundles. Added comprehensive test coverage.
4. **Repaired Disk-Space Gate**: Catches only `OSError` when obtaining filesystem stats; raises `RuntimeError` and aborts immediately if free space is below the configured threshold (10.0 GB).
5. **Removed Stale Defense Bounds**: Derived fixed base intensities and dynamic bounds directly from frozen `defenses.yaml`: AFP base 0.0003, bounds [0.0, 0.0003]; RS base 0.0002, bounds [0.0, 0.0002]; FS base 2.0, bounds [0.0, 2.0].
6. **Strengthened Measurement-Label Resolution & Preflight Integration**: `run_preflight()` directly executes manifest cross-validation without opening evaluation Parquets: validates required columns, exactly 72,000 unique measurement positions, 18,000 non-overlapping crafting positions, 500 rows/batch, normalized 0..143 batch IDs, and consistent composite IDs and binary labels. Added synthetic rejection tests.
7. **Repaired Completed-Run Reuse Validation**: Required agreement of directory name, `RunSummary` run_id, `completion.json` run_id, and every batch-record run_id with the target matrix row, exact provenance match, expected cache identity, zero unexpected files, and completion marker written last.
8. **Repaired Alias Publication**: Established unambiguous pointer-only contract: aliases publish `alias_pointer.json` and `completion.json`, never masquerading as independent runs. Target run outputs validated prior to alias creation; atomic staging and renaming; validated reuse and quarantine.
9. **Dependency-Safe Filtering**: When filters select an alias, its required primary target execution is automatically resolved and scheduled ahead of the alias. Non-positive `--max-runs` arguments are explicitly rejected.
10. **Genuinely Synthetic Production-Wiring Canary**: The end-to-end canary operates exclusively on synthetic 72,000-row cache data in `tmp_path` (derived within training feature bounds), completely prohibiting access to official evaluation caches or evaluation Parquets. Preflight reads official cache files solely for non-mutating inventory verification against `cache_inventory_v2.json`, while the synthetic canary is strictly prohibited from touching official evaluation cache files. Active regression test `test_canary_does_not_access_official_caches` traps and rejects any read of official cache files.
11. **Reconciled Runtime & Storage Projections**: Reconciled the arithmetic between defense-inference lower bound and full wall-clock prediction: applying 15% to 25% orchestration/IO overhead to the measured 3.26h lower bound yields **~3.74 to 4.07 hours** (across thermal envelope of 3.21h–3.79h, full wall-clock spans ~3.69h to 4.74h). Storage calculated from production schema records yields ~2.23 MB for `scores.json` and ~0.57 GB total for 252 runs.
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
| `scaler_hash` | `artifacts/preprocessors/standard_scaler.joblib` | `8999376cdf97c2047d9eb1a9ed245fb645bcb3af46b0c2e7cc07e3124a037fc2` |
| `feature_names_hash` | `artifacts/preprocessors/feature_names.json` | `fe36589ce33a64aa7a4eed643dcd0181469c0dc43e629943a0a6b79195b79046` |
| `feature_mask_hash` | `artifacts/preprocessors/feature_mask.json` | `5f332f52c7e1abba9f3306f50faedc3f6db0276f9e580749ba739bc85016224f` |
| `training_bounds_hash` | `artifacts/preprocessors/training_bounds.parquet` | `9554f1b0287c88b5ef7b8de19d193184f2d76eec2b4d96f1fa525818cffb275b` |
| `evaluation_roles_hash` | `data/manifests/evaluation_roles.csv` | `cbf650879aa1369fa26b803777f30d4b0add9e7aff2c05d399ba7f42563f7b45` |
| `evaluation_batches_hash` | `data/manifests/evaluation_batches.csv` | `4084017e5593732e455763416f7fc38254fc9dab2466eca289b66e68dc0ff79a` |
| `attacks_yaml_hash` | `configs/attacks.yaml` | `fbd596219990125d8a75da0b1d0e35bdd82c1429052407115ab2ab4f4f91cd2b` |
| `controllers_yaml_hash` | `configs/controllers.yaml` | `9c81c9696ab740a1b56ae0152f84fbb8be983484fd284cf0ac7edbd6b4a1eb91` |
| `defenses_yaml_hash` | `configs/defenses.yaml` | `43c21344140233d8ba90d78eff345741ef825a68ee8232c9cf32220309f5a7ff` |
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
- `test_completed_run_reuse`: **PASSED**
- `test_corrupt_or_incomplete_run_quarantine`: **PASSED**
- `test_output_reopening_catches_tampered_metrics`: **PASSED**
- `test_72000_row_and_global_metric_invariants`: **PASSED**
- `test_silent_probing_null_asr_vs_applicable_asr`: **PASSED**
- `test_atomic_failure_handling`: **PASSED**
- `test_deterministic_repeated_execution`: **PASSED**
- `test_official_artifacts_and_caches_immutability`: **PASSED**
- `test_production_provenance_construction_and_regression_on_missing_keys`: **PASSED**
- `test_independent_cache_inventory_pinning_and_divergence_rejection`: **PASSED**
- `test_git_cleanliness_policy`: **PASSED**
- `test_disk_space_gate_aborts_on_low_space`: **PASSED**
- `test_defense_factories_use_defenses_yaml_bounds`: **PASSED**
- `test_manifest_cross_validation_and_synthetic_rejections`: **PASSED**
- `test_strict_completed_run_reuse_validation`: **PASSED**
- `test_pointer_only_alias_publication_and_reuse`: **PASSED**
- `test_dependency_safe_execution_planning`: **PASSED**
- `test_production_wiring_canary`: **PASSED** (144 batches end-to-end on synthetic data, zero official caches touched)
- `test_preflight_invokes_manifest_cross_validation`: **PASSED** (verifies preflight rejects malformed manifests)
- `test_canary_does_not_access_official_caches`: **PASSED** (traps and rejects official cache access under guarded interceptor)
**Total Focused Tests:** **26 / 26 PASSED** in 50.03s.

### Full Test Suite
- Total tests collected: **282**
- Total tests passed: **282** (100%)
- Total failures: **0**
- Execution time: **26.38s**

---

## 5. Measured Runtime and Storage Projections

### Measured Training-Derived Timings (Apple Silicon M4)
- **AFP per 500-sample batch:** 0.0520 s (0.10 ms/sample)
- **FS per 500-sample batch:** 0.0460 s (0.09 ms/sample)
- **RS per 500-sample batch (11-member ensemble):** 0.8710 s (1.74 ms/sample)

### Projected Single Run (144 Batches = 72,000 Samples)
- **One AFP Run:** 7.49 s (0.12 min)
- **One FS Run:** 6.62 s (0.11 min)
- **One RS Run:** 125.42 s (2.09 min)

### Matrix-Level Projections (Reconciled Arithmetic)
- **Primary Comparison (90 runs: 30 AFP, 30 FS, 30 RS):**
  - Defense-inference lower bound: **69.77 min (1.16 hours)**
  - Full wall-clock projection (+15% to 25% overhead): **~1.34 – 1.45 hours** (~80.2 – 87.2 min).
- **Full Unique Executions (252 runs: 84 AFP, 84 FS, 84 RS):**
  - Defense-inference lower bound: **195.34 min (3.26 hours)**
  - Full wall-clock projection (+15% to 25% overhead): **~3.74 – 4.07 hours** (~224.6 – 244.2 min).
  - Across thermal envelope (3.21h cold run to 3.79h sustained load lower bound), full wall-clock spans ~3.69h to 4.74h.
- **Sensitivity Aliases (27 runs):** **0.00 min** (instant metadata pointer).
- **Storage Footprint:** ~**2.23 MB** for `scores.json` per run; ~**2.33 MB** total per run; **~0.57 GB** aggregate for 252 runs (well within available 68+ GB).
