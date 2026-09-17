# Phase 10D: Official Evaluation Orchestration Readiness Report

**Date:** 2026-09-18T01:38:30+08:00  
**Status:** PASS — Orchestration Readiness Established  
**Commit:** Pending Phase 10D commit  
**Protocol Freeze Tag:** `phase10-protocol-freeze` (`65005505415a2bdf2d5744dbd135e9214e74081a`)  

---

## 1. Executive Summary
Phase 10D establishes and certifies that the official Phase 10 evaluation execution pipeline can execute, resume, pair, and validate correctly before any expensive evaluation is initiated.

**Key Invariants & Certifications:**
1. **Zero Premature Evaluation**: Official evaluation (`--execute`) was NOT invoked. Zero evaluation Parquets (`X_eval.parquet`, `metadata_eval.parquet`) were accessed during testing or benchmarking.
2. **Strict Immutability of Official Attack Caches**: All 15 official attack caches in `artifacts/caches/` (60 files: Parquets, status files, manifests, completions) were verified bit-for-bit unchanged before and after all operations.
3. **Protected Artifact Integrity**: The frozen Random Forest model (`frozen_rf.joblib`), evaluation roles manifest, evaluation batches manifest, and frozen configs match their authoritative SHA-256 hashes exactly.
4. **Dynamic Matrix Derivation**: Proved from frozen configs: 279 matrix references, 27 exact C1 aliases, 252 unique executions, 90 primary references, 189 sensitivity references, 144 batches per execution, and 36,288 unique batch evaluations.
5. **Safe Official Execution Entry Point**: Implemented `scripts/run_evaluation.py` featuring safe-by-default execution (non-mutating preflight), deterministic restart from batch 0, corrupt/incomplete run quarantine, whole-run reuse, and two-stage atomic publication.
6. **Exact Base/C1 Pairing & Common Randomness**: Base and RA configurations are paired on `(seed, attack_scenario, defense_mechanism, batch_id)`. For equal intensities, Base and RA receive bit-identical defense pseudorandomness.
7. **Timing Isolation**: For RA batch $t$, defense intensity is determined prior to accessing batch-$t$ labels; feedback is submitted strictly after batch completion for batch $t+1$; Base receives no feedback.
8. **Test Certification**: All 14 focused tests in `tests/test_run_evaluation.py` passed. All 270 tests across the entire repository test suite passed with 0 failures.

---

## 2. Protected Artifacts and Cache Immutability Ledger

### Protected Artifact Hashes (Before vs After)

| Artifact | Path | Expected Protected Hash | Verified Hash | Status |
| :--- | :--- | :--- | :--- | :--- |
| **Frozen RF Model** | `artifacts/models/frozen_rf.joblib` | `9608672c5d5e38a9272c560679cdd2291399de0e0917c8c9dff373bfd200f51d` | `9608672c5d5e38a9272c560679cdd2291399de0e0917c8c9dff373bfd200f51d` | **MATCH** |
| **Evaluation Roles** | `data/manifests/evaluation_roles.csv` | `cbf650879aa1369fa26b803777f30d4b0add9e7aff2c05d399ba7f42563f7b45` | `cbf650879aa1369fa26b803777f30d4b0add9e7aff2c05d399ba7f42563f7b45` | **MATCH** |
| **Evaluation Batches** | `data/manifests/evaluation_batches.csv` | `4084017e5593732e455763416f7fc38254fc9dab2466eca289b66e68dc0ff79a` | `4084017e5593732e455763416f7fc38254fc9dab2466eca289b66e68dc0ff79a` | **MATCH** |
| **Experiment Config** | `configs/experiment.yaml` | `a1c5a389b1bc3c66311c7a96a7ccc3ee54af86506e64b38be7f26ad3c3d84c70` | `a1c5a389b1bc3c66311c7a96a7ccc3ee54af86506e64b38be7f26ad3c3d84c70` | **MATCH** |

### Official Attack Caches (15 Caches, 60 Files)
All 15 official cache directories in `artifacts/caches/` were verified against `artifacts/reports/cache_inventory_v2.json`:
- `DecisionBoundary_42` .. `DecisionBoundary_46` (5 caches): Verified
- `SilentProbing_42` .. `SilentProbing_46` (5 caches): Verified
- `SurrogateTransfer_42` .. `SurrogateTransfer_46` (5 caches): Verified
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
**Total Focused Tests:** **14 / 14 PASSED** in 2.65s.

### Full Test Suite
- Total tests collected: **270**
- Total tests passed: **270** (100%)
- Total failures: **0**
- Execution time: **10.92s**

---

## 5. Measured Runtime and Storage Projections

### Measured Training-Derived Timings (Apple Silicon M4)
- **AFP per 500-sample batch:** 0.0519 s (0.10 ms/sample)
- **FS per 500-sample batch:** 0.0450 s (0.09 ms/sample)
- **RS per 500-sample batch (11-member ensemble):** 0.8861 s (1.77 ms/sample)

### Projected Single Run (144 Batches = 72,000 Samples)
- **One AFP Run:** 7.47 s (0.12 min)
- **One FS Run:** 6.48 s (0.11 min)
- **One RS Run:** 127.60 s (2.13 min)

### Matrix-Level Projections
- **Primary Comparison (90 runs: 30 AFP, 30 FS, 30 RS):** **70.77 min (1.18 hours)**
- **Full Unique Executions (252 runs: 84 AFP, 84 FS, 84 RS):** **198.17 min (3.30 hours)**
- **Sensitivity Aliases (27 runs):** **0.00 min** (instant metadata link)
- **Total Storage for 252 Runs:** ~**0.21 GB** (well within available 68+ GB)
