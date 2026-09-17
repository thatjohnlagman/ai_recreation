# Phase 10B: Attack-Cache Orchestration Readiness Report

## Executive Summary
Phase 10B implements and validates the missing attack-cache orchestration entry point (`scripts/build_evaluation_caches.py`) for the thesis evaluation protocol. All work is conducted on Apple Silicon (`MacBook Air Mac16,12`, Apple M4, 16 GB RAM, macOS 27.0.0, Python 3.9.6 arm64) on branch `phase10b-cache-orchestration` created from the frozen commit `65005505415a2bdf2d5744dbd135e9214e74081a` (tag: `phase10-protocol-freeze`).

**Key Phase 10B Achievements:**
- Implemented `scripts/build_evaluation_caches.py` as a thin orchestration layer over `AttackCacheBuilder`.
- CLI is safe by default: running without `--execute` or with `--preflight-only` performs non-mutating preflight without reading evaluation features or labels.
- Dynamically derived the exact 15 canonical `(scenario, seed)` pairs required by the frozen 252 unique runs in the evaluation matrix.
- Enforced complete provenance tracking, atomic staging, failure quarantine, and whole-cache reuse.
- Created 18 focused synthetic unit tests (`tests/test_build_evaluation_caches.py`), all passing (100%).
- Full project test suite passed cleanly: **218 passed** in 7.27s (0 failures).
- Zero official attack caches were generated; zero official evaluation runs were executed. Evaluation features and labels remain unread.

---

## 1. Frozen Protocol Verification

### 1.1 Git Ancestry & Tag Integrity
- **Freeze Commit**: `65005505415a2bdf2d5744dbd135e9214e74081a`
- **Annotated Tag**: `phase10-protocol-freeze`
- **Current Branch**: `phase10b-cache-orchestration`
- **Ancestry Verification**: `git merge-base --is-ancestor phase10-protocol-freeze HEAD` -> **CONFIRMED** (Exit Code 0).
- **Tag Immutability**: The tag was not moved, recreated, or modified.

### 1.2 Protected Hashes Comparison
All three protected artifacts were verified against their authoritative frozen SHA-256 digests:

| Artifact | File Path | Expected Frozen Hash | Actual Hash | Status |
|---|---|---|---|---|
| Frozen RF Model | `artifacts/models/frozen_rf.joblib` | `9608672c5d5e38a9272c560679cdd2291399de0e0917c8c9dff373bfd200f51d` | `9608672c5d5e38a9272c560679cdd2291399de0e0917c8c9dff373bfd200f51d` | **MATCH** |
| Evaluation Roles | `data/manifests/evaluation_roles.csv` | `cbf650879aa1369fa26b803777f30d4b0add9e7aff2c05d399ba7f42563f7b45` | `cbf650879aa1369fa26b803777f30d4b0add9e7aff2c05d399ba7f42563f7b45` | **MATCH** |
| Evaluation Batches | `data/manifests/evaluation_batches.csv` | `4084017e5593732e455763416f7fc38254fc9dab2466eca289b66e68dc0ff79a` | `4084017e5593732e455763416f7fc38254fc9dab2466eca289b66e68dc0ff79a` | **MATCH** |

### 1.3 Configuration Freeze Status
- `git diff phase10-protocol-freeze -- configs/` produced **0 differences** (clean).
- `configs/experiment.yaml` confirms `experiment.date_frozen`: `"2026-09-17T21:23:51+08:00"` (unaltered).
- No attacks, defenses, controller parameters, model settings, splits, seeds, evaluation assignments, feature mask, or calibrated intensities were modified.

---

## 2. Partition Allocations & Matrix Derivation

### 2.1 Partition Structure
- `data/manifests/evaluation_roles.csv`: Exactly **90,000 total records**.
  - Crafting Pool: **18,000 records** (`role == "crafting"`).
  - Measurement Pool: **72,000 records** (`role == "measurement"`).
- `data/manifests/evaluation_batches.csv`: Exactly **144 batches of 500 records** (72,000 records total).

### 2.2 Matrix Derivation & Canonical Cache Pairs
From `configs/`, the evaluation matrix defines 279 total rows (90 primary + 189 sensitivity). 27 sensitivity C1 runs are aliases reusing the corresponding primary C1 run, leaving exactly **252 unique execution runs** across **36,288 batches**.

From these 252 runs, exactly **15 distinct canonical `(attack_scenario, seed)` cache pairs** are derived:

| # | Attack Scenario | Seed | Canonical Slug | Associated Defenses | Associated Controllers |
|---|-----------------|------|----------------|---------------------|------------------------|
| 1 | Decision Boundary | 42 | `DecisionBoundary_42` | AFP, FS, RS | Base, C1..C7 |
| 2 | Decision Boundary | 43 | `DecisionBoundary_43` | AFP, FS, RS | Base, C1..C7 |
| 3 | Decision Boundary | 44 | `DecisionBoundary_44` | AFP, FS, RS | Base, C1..C7 |
| 4 | Decision Boundary | 45 | `DecisionBoundary_45` | AFP, FS, RS | Base, C1 |
| 5 | Decision Boundary | 46 | `DecisionBoundary_46` | AFP, FS, RS | Base, C1 |
| 6 | Silent Probing | 42 | `SilentProbing_42` | AFP, FS, RS | Base, C1..C7 |
| 7 | Silent Probing | 43 | `SilentProbing_43` | AFP, FS, RS | Base, C1..C7 |
| 8 | Silent Probing | 44 | `SilentProbing_44` | AFP, FS, RS | Base, C1..C7 |
| 9 | Silent Probing | 45 | `SilentProbing_45` | AFP, FS, RS | Base, C1 |
| 10 | Silent Probing | 46 | `SilentProbing_46` | AFP, FS, RS | Base, C1 |
| 11 | Surrogate Transfer | 42 | `SurrogateTransfer_42` | AFP, FS, RS | Base, C1..C7 |
| 12 | Surrogate Transfer | 43 | `SurrogateTransfer_43` | AFP, FS, RS | Base, C1..C7 |
| 13 | Surrogate Transfer | 44 | `SurrogateTransfer_44` | AFP, FS, RS | Base, C1..C7 |
| 14 | Surrogate Transfer | 45 | `SurrogateTransfer_45` | AFP, FS, RS | Base, C1 |
| 15 | Surrogate Transfer | 46 | `SurrogateTransfer_46` | AFP, FS, RS | Base, C1 |

---

## 3. Test Suite Verification

### 3.1 Focused Phase 10B Tests (`tests/test_build_evaluation_caches.py`)
18 focused tests were executed using synthetic fixtures and a small training-derived slice. Zero evaluation records were accessed:

| Test Case | Description | Result |
|---|---|---|
| `test_safe_preflight_default` | Preflight runs by default, generating 0 caches | **PASSED** |
| `test_execute_required_for_data_access` | Data access strictly requires `--execute` | **PASSED** |
| `test_exact_derivation_of_15_pairs` | Derives exactly 15 pairs from matrix | **PASSED** |
| `test_alias_reuse` | 27 aliases point to primary runs with identical scenario/seed | **PASSED** |
| `test_rejection_of_altered_freeze_ancestry` | Rejects non-ancestor or tampered freeze commit | **PASSED** |
| `test_rejection_of_altered_config_date_frozen` | Rejects altered `date_frozen` | **PASSED** |
| `test_protected_hash_mismatch_rejection` | Rejects any mismatch in RF, Roles, or Batches | **PASSED** |
| `test_no_defense_config_dimension_in_cache_identity` | Proves cache identity is independent of defenses/controllers | **PASSED** |
| `test_atomic_publication_and_failure_quarantine` | Staging is quarantined on error; destination remains clean | **PASSED** |
| `test_complete_cache_reuse` | Valid existing cache is reused without recomputing | **PASSED** |
| `test_invalid_cache_rejection` | Incomplete or corrupt cache is quarantined and rebuilt | **PASSED** |
| `test_labels_unavailable_during_candidate_generation` | Candidate generator operates without label access | **PASSED** |
| `test_correct_scenario_routing` | Routes all 3 scenarios to canonical builders | **PASSED** |
| `test_exact_global_200_target_boundary_selection` | Exactly 200 global boundary targets selected per seed | **PASSED** |
| `test_silent_probing_zero_query_and_identity_behavior` | Identity transform, 0 queries, all `NOT_APPLICABLE` | **PASSED** |
| `test_no_partial_cache_resumption` | Stale staging/tmp directories are quarantined; restarts clean | **PASSED** |
| `test_deterministic_pair_ordering` | Pair sequence is strictly deterministic | **PASSED** |
| `test_training_derived_smoke` | End-to-end smoke test using training records slice | **PASSED** |

**Focused Results**: 18 passed in 2.24s (100%).

### 3.2 Full Project Regression Suite
- Total Tests: **218 passed** in 7.27s (0 failed, 0 errors).
- Clean regression across runner validation, schemas, caching, controller, defenses, attacks, and matrix modules.

---

## 4. Evaluation Data Isolation Confirmation

- **Preflight Isolation**: `scripts/build_evaluation_caches.py` was executed with no arguments and with `--preflight-only`. In both executions:
  - `data/processed/X_eval.parquet` was NOT opened or read.
  - `data/processed/metadata_eval.parquet` was NOT opened or read.
  - Evaluation ground-truth labels were NOT inspected.
- **Cache Inventory**:
  - `artifacts/caches/` contains **0 attack caches** (0 / 15 present).
  - `artifacts/results/` contains **0 evaluation run results**.

---

## 5. Execution Command for Later Official Phase

When authorized to generate official attack caches in a subsequent phase, the exact command to execute is:

```bash
python scripts/build_evaluation_caches.py --execute
```

To generate a single controlled cache during testing:
```bash
python scripts/build_evaluation_caches.py --execute --scenario "Silent Probing" --seed 42
```
