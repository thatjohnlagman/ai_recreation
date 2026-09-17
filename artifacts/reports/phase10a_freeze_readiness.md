# Phase 10A Protocol Freeze Readiness Report

**Project**: Recall-Aware Intrusion Detection System (Thesis Implementation)  
**Host Machine**: MacBook Air (Mac16,12), Apple M4, 16 GB RAM, macOS 27.0.0, Python 3.9.6 (arm64)  
**Date**: September 2026  
**Status**: Ready for External Freeze Review (`experiment.date_frozen` remains `null`)  

---

## 1. Executive Summary

Phase 10A red-green integration repair has been successfully executed, validated, and verified on the native Apple Silicon M4 platform. All 174 automated unit and integration tests across the repository pass with zero failures. A 5,000-row training-derived runtime and memory pilot was executed without errors, demonstrating stable memory utilization (peak RSS 469.19 MB) and confirming computational feasibility. All three protected artifacts remain bit-for-bit identical to their frozen authoritative hashes. No evaluation records were loaded or inspected, no official attack caches were generated, and the formal freeze date remains unset pending external review.

---

## 2. Platform and Environment Status

- **Environment**: `.venv-m4` initialized from macOS system Python 3.9.6 arm64.
- **Transferred `.venv`**: Preserved unchanged; native extension quarantine issues bypassed cleanly via native M4 environment.
- **Dependencies**: All packages installed from `requirements-lock.txt` with exact version parity (`numpy 2.0.2`, `pandas 2.3.3`, `pyarrow 21.0.0`, `scikit-learn 1.6.1`, `joblib 1.5.3`, `PyYAML 6.0.3`, `pytest 8.4.2`).
- **Path Configuration**: `src/` linked directly into site-packages via `recall_aware_ids.pth`.

---

## 3. Component Repairs and Validation Summary

### 3.1 Attack Cache Validation and Building
- **Schema & Manifest Integrity**: Enforced strict 64-character lowercase hex string validation for all provenance and file hashes. Prohibited placeholder strings (`"0"*64`, `"a"*64`, `"dummy"`). Required non-empty dictionary structures for `attack_script_hashes`, `attack_parameters`, and `query_budgets`.
- **Type Rigor**: Unconditional enforcement of strict boolean types for status flags (`eligible`, `attempted`, `successful`), rejecting integer coercion. Enforced integer dtypes for `queries_used`, rejecting boolean values. Enforced floating-point types for all 78 feature columns.
- **Semantic Constraints**: Silent Probing strictly validated to have zero queries, zero attempted samples, and uniform `NOT_APPLICABLE` status codes.
- **Oracle Routing**: Routed screening queries in both `AttackCacheBuilder` and pilot scripts through `BlackBoxOracle` to prevent uncounted target model predictions.
- **Atomic Operations & Quarantine**: Cache builder writes to `.tmp` staging directories, re-validates written Parquets and manifests through the strict provider, and quarantines incomplete or failed artifacts.

### 3.2 Decision Boundary Target Selection
- Reordered selection logic to prioritize integer type validation, followed by checking candidate eligibility counts, and enforcing exactly 200 attack targets globally across the 72,000 measurement pool in official mode.

### 3.3 Evaluation Matrix and Controller Constraints
- Enforced strict whitelist for controller configurations (`C1` through `C7`). Explicitly reject unauthorized configurations (`C8`).
- Confirmed experimental matrix dimensions: 252 unique executions, 36,288 batch records, and 84 unique RS runs across primary and sensitivity references.

### 3.4 Schemas and Serialization
- Enforced ISO 8601 timestamp validation in `CompletionMarker` and `FailureRecord`.
- Re-ordered `CompletionMarker` post-init validation to verify provenance hashes prior to parsing timestamp strings.
- Enforced exact confusion matrix totals: exactly 500 per batch and 72,000 per official run.

### 3.5 Test Suite Isolation
- Removed legacy dependencies on official evaluation data (`X_eval.parquet`, `metadata_eval.parquet`) from all tests (`test_preprocess_dataset.py`, `test_evaluation_batches.py`). Replaced them with synthetic or training-derived fixtures.

---

## 4. Test Suite Execution Evidence

### 4.1 Focused Phase 10A Suite (`tests/test_phase10a_v5.py`)
```text
tests/test_phase10a_v5.py::test_v5_provider_rejects_malformed_identity[] PASSED
tests/test_phase10a_v5.py::test_v5_provider_rejects_malformed_identity[unknown] PASSED
tests/test_phase10a_v5.py::test_v5_provider_rejects_malformed_identity[aaaaaaaa...] PASSED
tests/test_phase10a_v5.py::test_v5_provider_rejects_malformed_identity[00000000...] PASSED
tests/test_phase10a_v5.py::test_v5_provider_rejects_malformed_identity[not-a-hash] PASSED
tests/test_phase10a_v5.py::test_v5_provider_rejects_malformed_identity[None] PASSED
tests/test_phase10a_v5.py::test_v5_provider_rejects_empty_nested_identity[attack_script_hashes] PASSED
tests/test_phase10a_v5.py::test_v5_provider_rejects_empty_nested_identity[attack_parameters] PASSED
tests/test_phase10a_v5.py::test_v5_provider_rejects_empty_nested_identity[query_budgets] PASSED
tests/test_phase10a_v5.py::test_v5_provider_rejects_query_types[True] PASSED
tests/test_phase10a_v5.py::test_v5_provider_rejects_query_types[0.5] PASSED
tests/test_phase10a_v5.py::test_v5_provider_rejects_query_types[nan] PASSED
tests/test_phase10a_v5.py::test_v5_provider_rejects_query_types[1] PASSED
tests/test_phase10a_v5.py::test_v5_provider_rejects_scrambled_identity PASSED
tests/test_phase10a_v5.py::test_v5_provider_rejects_string_features PASSED
tests/test_phase10a_v5.py::test_v5_provider_rejects_silent_success PASSED
tests/test_phase10a_v5.py::test_v5_provider_cannot_disable_boolean_validation PASSED
tests/test_phase10a_v5.py::test_v5_builder_rejects_incomplete_provenance PASSED
tests/test_phase10a_v5.py::test_v5_boundary_official_target_count PASSED
tests/test_phase10a_v5.py::test_v5_completion_timestamp[] PASSED
tests/test_phase10a_v5.py::test_v5_completion_timestamp[yesterday] PASSED
tests/test_phase10a_v5.py::test_v5_completion_timestamp[2026-99-99T25:00:00Z] PASSED
tests/test_phase10a_v5.py::test_v5_matrix_rejects_c8_replacing_c7 PASSED
tests/test_phase10a_v5.py::test_v5_no_untracked_target_calls[src/recall_aware_ids/experiment/caching.py] PASSED
tests/test_phase10a_v5.py::test_v5_no_untracked_target_calls[scripts/run_m2_pilot.py] PASSED
Results: 25 passed in 0.61s
```

### 4.2 Full Repository Test Suite
```text
Total collected: 174 test items across 17 test modules
Status: 174 passed, 0 failed, 0 warnings in 4.48s
```

---

## 5. M4 Training-Derived Pilot Results

Executed exclusively on 5,000 deterministic records from `X_train.parquet` / `metadata_train.parquet`:
- **AFP**: 0.0984s (final invalid cells: 0, protected modified: 0, projected cells: 102,445)
- **FS**: 0.0806s (effective bit-depth: 4, final invalid cells: 0, protected modified: 0)
- **RS**: 8.3327s (positive vote fraction range: [0.000, 1.000], protected modified: 0)
- **Surrogate Transfer**: 8.5486s (1,000 crafting queries, 652 eval queries, 76 eligible, 76 attempted, 1 success)
- **Decision Boundary**: 17.9009s (1,274 oracle queries, 98 eligible, 98 attempted, 98 successes)
- **Memory Profile**:
  - Absolute Peak RSS: 469.19 MB (well below 16 GB host RAM / 8 GB conservative ceiling)
  - Incremental RSS: 315.88 MB
- **Scaling Projection**: Randomized Smoothing represents the primary runtime bottleneck ($11 \text{ RF inferences} \times 72,000 \text{ samples} \times 84 \text{ runs} \approx 66.5\text{M RF calls}$).

---

## 6. Protected Artifacts Audit

| Artifact | Pinned SHA-256 | Verified Local SHA-256 | Integrity |
|---|---|---|---|
| `artifacts/models/frozen_rf.joblib` | `9608672c5d5e38a9272c560679cdd2291399de0e0917c8c9dff373bfd200f51d` | `9608672c5d5e38a9272c560679cdd2291399de0e0917c8c9dff373bfd200f51d` | Exact Match |
| `data/manifests/evaluation_roles.csv` | `cbf650879aa1369fa26b803777f30d4b0add9e7aff2c05d399ba7f42563f7b45` | `cbf650879aa1369fa26b803777f30d4b0add9e7aff2c05d399ba7f42563f7b45` | Exact Match |
| `data/manifests/evaluation_batches.csv` | `4084017e5593732e455763416f7fc38254fc9dab2466eca289b66e68dc0ff79a` | `4084017e5593732e455763416f7fc38254fc9dab2466eca289b66e68dc0ff79a` | Exact Match |

---

## 7. Protocol Freeze Recommendation

1. **Readiness**: Phase 10A implementation repairs are complete. All contracts, timing invariants, and schema validations are fully satisfied.
2. **Next Steps**: Package the v5 freeze candidate bundle for external stakeholder sign-off. Upon approval, formal timestamping of `experiment.date_frozen` and execution of official Phase 10 evaluation may commence.
