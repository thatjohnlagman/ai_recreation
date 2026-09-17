# Phase 10A Protocol Freeze Readiness Report (v5.1 Final Correction)

**Project**: Recall-Aware Intrusion Detection System (Thesis Implementation)  
**Host Machine**: MacBook Air (Mac16,12), Apple M4, 16 GB RAM, macOS 27.0.0, Python 3.9.6 (arm64)  
**Date**: September 2026  
**Status**: Ready for External Freeze Review (`experiment.date_frozen` remains `null`)  

---

## 1. Executive Summary

Phase 10A v5.1 final review corrections have been successfully completed, validated, and verified on the native Apple Silicon M4 platform. All 196 automated unit and integration tests across the repository pass with zero failures (including 22 new focused tests covering run-level ASR, strengthened `RunSummary` validations, Silent Probing metadata provenance, and output reopening/quarantine). A 5,000-row training-derived runtime and memory pilot was executed without errors, demonstrating stable memory utilization (peak RSS 472.23 MB) and confirming computational feasibility. All three protected artifacts remain bit-for-bit identical to their frozen authoritative hashes. No evaluation records were loaded or inspected, no official attack caches were generated, and the formal freeze date remains unset pending external review.

---

## 2. Platform and Environment Status

- **Environment**: `.venv-m4` initialized from macOS system Python 3.9.6 arm64.
- **Transferred `.venv`**: Preserved unchanged; native extension quarantine issues bypassed cleanly via native M4 environment.
- **Dependencies**: All packages installed from `requirements-lock.txt` with exact version parity (`numpy 2.0.2`, `pandas 2.3.3`, `pyarrow 21.0.0`, `scikit-learn 1.6.1`, `joblib 1.5.3`, `PyYAML 6.0.3`, `pytest 8.4.2`).
- **Path Configuration**: `src/` linked directly into site-packages via `recall_aware_ids.pth`.

---

## 3. Final Review Corrections (v5.1)

### 3.1 Run-Level Attack Success Rate (ASR) Semantics
- **Runner Correction**: In `src/recall_aware_ids/experiment/runner.py`, updated `global_asr` calculation:
  - Silent Probing: `global_asr = None` (attack not applicable).
  - Applicable attack with attempts: `global_asr = float(successful / attempted)`.
  - Applicable attack with zero attempts: `global_asr = 0.0`.
- **Validation**: Added focused test `test_runner_applicable_zero_attempts_asr_zero` verifying serialized `run_summary.json` contains `0.0`.

### 3.2 Strengthened `RunSummary` and `BatchConfusionLog` Contracts
- **Batch Confusion**: Enforced `eligible_count <= 500` in `BatchConfusionLog`.
- **RunSummary Contract**:
  - Scenario-specific deterministic ASR validation: Silent Probing requires `global_asr is None`; Surrogate Transfer and Decision Boundary require finite `[0,1]`; zero-attempt applicable runs require `global_asr == 0.0`.
  - Type strictness: `total_queries` and individual values in `status_code_counts` strictly reject `bool`.
  - Status counts sum: Total across all status codes in `status_code_counts` must sum to exactly 72,000.
  - Complete `cache_identity`: Must contain all 11 required provenance keys (`_REQUIRED_PROVENANCE_KEYS`) without missing, malformed, or dummy/placeholder hashes (`"0"*64`, `"a"*64`, `"dummy"`).
  - Magnitude summaries: Each summary (`l0`, `l1`, `l2`, `linf`) must contain exactly `{"mean", "min", "max", "std"}`, all non-negative finite numeric values, satisfying `min <= mean <= max` and `std >= 0`.
  - Count hierarchy: `total_eligible <= 72000`, `total_attempted <= total_eligible`, and `total_successful <= total_attempted`.

### 3.3 Elimination of Fabricated Silent Probing Provenance
- **AttackCacheBuilder**: Removed all fabricated `sha256(b"silent_probing")` and default metadata fallbacks from `_build_silent_probing()`.
- **Explicit Metadata Requirement**: All three scenarios unconditionally require explicit, non-empty `attack_script_hashes`, `attack_parameters`, and `query_budgets`.
- **Manifest Integrity**:
  - Every script hash validated as a lowercase 64-character SHA-256.
  - Placeholder, all-zero, or repeated dummy hashes rejected.
  - Caller-computed file hash enforcement: Hashes generated from labels or filenames (e.g. `b"silent_probing"`, `b"silent_probing.py"`, script names) are explicitly detected and rejected.
  - Parameter and budget keys must be non-empty strings, values must be JSON-serializable, numeric values must be finite.
  - Query budget values must be non-negative integers; `bool` values are rejected with `TypeError`.
  - Silent Probing explicitly requires zero query budgets and `modifies_samples=False`.
  - Decision Boundary explicitly records the 50-query attack budget.

### 3.4 Pre-Completion Output Reopening and Validation
- **Reopening Gate**: Before creating `completion.json`, `ExperimentRunner` reopens and inspects:
  - `config.json` (144 valid `BatchConfigLog` records, batch IDs `0..143`).
  - `confusion.json` (144 valid `BatchConfusionLog` records, batch IDs `0..143`).
  - `scores.json` (144 score records, batch IDs `0..143`, exactly 500 aligned elements per batch for `scores`, `predictions`, and `labels`, binary predictions and labels, finite scores in `[0,1]`, total elements exactly 72,000).
  - `run_summary.json` (reconstructs valid `RunSummary`).
- **Cross-Record Recomputation**:
  - Reconstructed totals from 144 confusion records verified to sum to 72,000 and match `RunSummary` fields (`tp, fp, tn, fn, total_eligible, total_attempted, total_successful`).
  - Global PR-AUC recomputed across all 72,000 concatenated scores and labels, verified to match `RunSummary.pr_auc_average_precision` within `1e-7`.
- **Quarantine Invariant**: Any corruption or mismatch aborts before `completion.json` is written, renaming staging directory to `_failed_quarantined_*` with a `quarantined_failure.json` record. Verified by `test_runner_quarantines_on_malformed_serialized_output`.

### 3.5 Bundle Gating and Accounting Repairs
- **Untracked `.DS_Store`**: Removed `src/recall_aware_ids/.DS_Store` from tracking and added `**/.DS_Store` to `.gitignore`.
- **Hard Bundle Gating**: The bundle builder immediately aborts before ZIP creation if `forbidden_errors` is non-empty, `dup_errors` is non-empty, any test/pilot exit code is non-zero, or required evidence files are missing.
- **Two-Phase Member Accounting**:
  - Final member-name list constructed first.
  - `BUNDLE_MANIFEST.txt` generated listing every final member (including `BUNDLE_MANIFEST.txt` and `FILE_HASHES.sha256`).
  - `FILE_HASHES.sha256` generated hashing every member except itself (including the hash of `BUNDLE_MANIFEST.txt`).
- **Independent Post-Extraction Verification**: Extracted archive tested with `zipfile.testzip()`, checked for zero duplicates, verified that ZIP member set equals manifest member set, verified every non-hash-manifest member against `FILE_HASHES.sha256`, and confirmed zero forbidden members.

---

## 4. Test Suite Execution Evidence

### 4.1 Full Repository Test Suite
```text
Total collected: 196 test items across 17 test modules
Status: 196 passed, 0 failed, 0 warnings in 5.55s
```

### 4.2 Key Test Module Results
- `tests/test_phase10a_v5.py`: 25 passed
- `tests/test_schemas.py`: 58 passed (all `RunSummary` and `BatchConfusionLog` validations)
- `tests/test_caching.py`: 20 passed (all builder/provider validations, omitted/fabricated metadata rejections)
- `tests/test_runner_validation.py`: 17 passed (all timing invariants, zero-attempt ASR, output reopening/quarantine)
- `tests/test_experiment_runner.py`: 9 passed
- `tests/test_matrix.py`: 4 passed
- `tests/test_controller.py`: 12 passed
- `tests/test_defenses.py`: 18 passed
- `tests/test_attacks.py`: 7 passed

---

## 5. M4 Training-Derived Pilot Results

Executed exclusively on 5,000 deterministic records from `X_train.parquet` / `metadata_train.parquet`:
- **AFP**: 0.1372s (final invalid cells: 0, protected modified: 0, projected cells: 102,445)
- **FS**: 0.0913s (effective bit-depth: 4, final invalid cells: 0, protected modified: 0)
- **RS**: 9.1437s (positive vote fraction range: [0.000, 1.000], protected modified: 0)
- **Surrogate Transfer**: 8.5713s (1,000 crafting queries, 652 eval queries, 76 eligible, 76 attempted, 1 success)
- **Decision Boundary**: 17.9244s (1,274 oracle queries, 98 eligible, 98 attempted, 98 successes)
- **Memory Profile**:
  - Absolute Peak RSS: 472.23 MB (well below 16 GB host RAM / 8 GB conservative ceiling)
  - Incremental RSS: 318.58 MB
- **Exit Code**: 0 (Clean termination)

---

## 6. Protected Artifacts Audit

| Artifact | Pinned SHA-256 | Verified Local SHA-256 | Integrity |
|---|---|---|---|
| `artifacts/models/frozen_rf.joblib` | `9608672c5d5e38a9272c560679cdd2291399de0e0917c8c9dff373bfd200f51d` | `9608672c5d5e38a9272c560679cdd2291399de0e0917c8c9dff373bfd200f51d` | Exact Match |
| `data/manifests/evaluation_roles.csv` | `cbf650879aa1369fa26b803777f30d4b0add9e7aff2c05d399ba7f42563f7b45` | `cbf650879aa1369fa26b803777f30d4b0add9e7aff2c05d399ba7f42563f7b45` | Exact Match |
| `data/manifests/evaluation_batches.csv` | `4084017e5593732e455763416f7fc38254fc9dab2466eca289b66e68dc0ff79a` | `4084017e5593732e455763416f7fc38254fc9dab2466eca289b66e68dc0ff79a` | Exact Match |

---

## 7. Working Tree and Bundle Status

- **Tracked Git Status**: Clean (all changes committed in `Fix final Phase 10A v5 review blockers`).
- **Untracked / Ignored Artifacts**:
  - `RED_TEST_OUTPUT.txt` (local diagnostic log)
  - Prior review bundles (`phase10a_freeze_candidate_bundle_v2.zip`, `_v3.zip`, `_v4.zip`, `_v5.zip`)
- **Bundle File**: `phase10a_freeze_candidate_bundle_v5_1.zip`
- **Forbidden Files Scan**: 0 forbidden files found.
- **Duplicate Members Scan**: 0 duplicate members found.
- **Protocol Freeze Status**: `experiment.date_frozen` remains `null`. Official evaluation data (`X_eval.parquet`, `metadata_eval.parquet`) and official attack caches were NOT accessed.
