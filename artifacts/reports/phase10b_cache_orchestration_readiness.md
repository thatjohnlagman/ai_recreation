# Phase 10B: Attack-Cache Orchestration Readiness Report

## Executive Summary
Phase 10B implements and validates the attack-cache orchestration entry point (`scripts/build_evaluation_caches.py`) for the thesis evaluation protocol, incorporating all 10 binding corrections from external review. All operations were conducted on Apple Silicon (`MacBook Air Mac16,12`, Apple M4, 16 GB RAM, macOS 27.0.0, Python 3.9.6 arm64) on branch `phase10b-cache-orchestration` created from freeze commit `65005505415a2bdf2d5744dbd135e9214e74081a` (annotated tag: `phase10-protocol-freeze`).

**Key Phase 10B Achievements & Enforced Corrections:**
1. **Frozen Code Untouched**: `src/recall_aware_ids/` is 100% untouched relative to `phase10-protocol-freeze` (`git diff phase10-protocol-freeze -- src/` is empty).
2. **Strengthened Freeze Check**: Bit-level byte-for-byte and SHA-256 comparison of every frozen config and protocol file against the tagged version confirmed exact equality.
3. **Label Isolation Rule**: True labels are never supplied to attack generators, surrogate fitting procedures, endpoint search, or reference selection (benign reference pool selected via `predict_fn(X_craft) == 0`).
4. **Canonicalized Matrix Pairs**: Derived exactly 15 ordered `(scenario, seed)` 2-tuples across `SilentProbing`, `SurrogateTransfer`, `DecisionBoundary` and seeds `42..46` with deterministic ordering and verified sensitivity-alias reuse.
5. **Strict Cache Completion Validation**: Reuse requires a valid completion marker, manifest, provenance, schemas, exact row count, attacked-feature hash, and status hash.
6. **Safe Staging and Quarantine**: Staging and quarantine directories are strictly confined beneath the configured cache root, with complete path traversal rejection.
7. **Separated Training Smoke Test**: Dedicated training-derived smoke test in `tests/test_training_smoke.py` uses only `X_train.parquet` and `metadata_train.parquet`, writing to a temporary directory outside `artifacts/caches`.
8. **Complete Real Provenance Forwarding**: Complete provenance hashes forwarded to `AttackCacheBuilder`, rejecting any placeholder or malformed values.
9. **Zero Evaluation Data Access**: Preflight and test runs never opened or read `X_eval.parquet` or `metadata_eval.parquet`. Official `artifacts/caches` contains 0 caches.
10. **Review Bundle Exclusions**: Review ZIP strictly excludes all datasets, Parquet files, CSV files, serialized models, credentials, Git internals, and virtual environments.

---

## 1. Frozen Protocol Verification

### 1.1 Git Ancestry & Tag Integrity
- **Freeze Commit**: `65005505415a2bdf2d5744dbd135e9214e74081a`
- **Annotated Tag**: `phase10-protocol-freeze`
- **Current Branch**: `phase10b-cache-orchestration`
- **Ancestry Verification**: `git merge-base --is-ancestor phase10-protocol-freeze HEAD` -> **CONFIRMED** (Exit Code 0).

### 1.2 Strengthened Frozen Configuration and Protocol File Verification
Every frozen file on disk was compared byte-for-byte and hash-for-hash against `git show phase10-protocol-freeze:<file>`:

| File Path | SHA-256 Digest | Byte-for-Byte Match vs Tag | Status |
|---|---|---|---|
| `configs/attacks.yaml` | `fbd596219990125d8a75da0b1d0e35bdd82c1429052407115ab2ab4f4f91cd2b` | **EXACT MATCH** | **VERIFIED** |
| `configs/controllers.yaml` | `9c81c9696ab740a1b56ae0152f84fbb8be983484fd284cf0ac7edbd6b4a1eb91` | **EXACT MATCH** | **VERIFIED** |
| `configs/defenses.yaml` | `43c21344140233d8ba90d78eff345741ef825a68ee8232c9cf32220309f5a7ff` | **EXACT MATCH** | **VERIFIED** |
| `configs/experiment.yaml` | `a1c5a389b1bc3c66311c7a96a7ccc3ee54af86506e64b38be7f26ad3c3d84c70` | **EXACT MATCH** | **VERIFIED** |
| `configs/model.yaml` | `5cc87521b7d194ca747c3f0280c21a4f20e5b2aa3ab930580b653d95ce2c34cd` | **EXACT MATCH** | **VERIFIED** |
| `docs/EXPERIMENT_PROTOCOL.md` | `775fd2c5d409e9337aeb31251e111363d8b9f3b06148fce54bce5bc96bb59f1a` | **EXACT MATCH** | **VERIFIED** |

- Tree verification: Zero added or deleted files in `configs/` relative to `phase10-protocol-freeze`.
- `experiment.date_frozen`: Confirmed `"2026-09-17T21:23:51+08:00"`.

### 1.3 Authoritative Protected Hashes Comparison
All three protected artifacts were verified against their authoritative frozen SHA-256 digests:

| Artifact | File Path | Authoritative Frozen Hash | Actual Hash | Status |
|---|---|---|---|---|
| Frozen RF Model | `artifacts/models/frozen_rf.joblib` | `9608672c5d5e38a9272c560679cdd2291399de0e0917c8c9dff373bfd200f51d` | `9608672c5d5e38a9272c560679cdd2291399de0e0917c8c9dff373bfd200f51d` | **MATCH** |
| Evaluation Roles | `data/manifests/evaluation_roles.csv` | `cbf650879aa1369fa26b803777f30d4b0add9e7aff2c05d399ba7f42563f7b45` | `cbf650879aa1369fa26b803777f30d4b0add9e7aff2c05d399ba7f42563f7b45` | **MATCH** |
| Evaluation Batches | `data/manifests/evaluation_batches.csv` | `4084017e5593732e455763416f7fc38254fc9dab2466eca289b66e68dc0ff79a` | `4084017e5593732e455763416f7fc38254fc9dab2466eca289b66e68dc0ff79a` | **MATCH** |

---

## 2. Partition Allocations & Matrix Derivation

### 2.1 Partition Manifest Structure
- `data/manifests/evaluation_roles.csv`: Exactly **90,000 total records**.
  - Crafting Pool: **18,000 records** (`role == "crafting"`).
  - Measurement Pool: **72,000 records** (`role == "measurement"`).
- `data/manifests/evaluation_batches.csv`: Exactly **144 batches of 500 records** (72,000 records total).

### 2.2 Canonical Derived Cache Pairs
From `configs/`, the evaluation matrix defines 279 total rows. Exactly 27 sensitivity C1 runs are aliases reusing the corresponding primary C1 run, leaving exactly **252 unique executions** across **36,288 batches**.

From these 252 runs, exactly **15 distinct canonical `(scenario, seed)` pairs** are derived, ordered deterministically:

| # | Canonical Scenario | Seed | Canonical Slug | Associated Defenses | Associated Controllers |
|---|---|---|---|---|---|
| 1 | `SilentProbing` | 42 | `SilentProbing_42` | AFP, FS, RS | Base, C1..C7 |
| 2 | `SilentProbing` | 43 | `SilentProbing_43` | AFP, FS, RS | Base, C1..C7 |
| 3 | `SilentProbing` | 44 | `SilentProbing_44` | AFP, FS, RS | Base, C1..C7 |
| 4 | `SilentProbing` | 45 | `SilentProbing_45` | AFP, FS, RS | Base, C1 |
| 5 | `SilentProbing` | 46 | `SilentProbing_46` | AFP, FS, RS | Base, C1 |
| 6 | `SurrogateTransfer` | 42 | `SurrogateTransfer_42` | AFP, FS, RS | Base, C1..C7 |
| 7 | `SurrogateTransfer` | 43 | `SurrogateTransfer_43` | AFP, FS, RS | Base, C1..C7 |
| 8 | `SurrogateTransfer` | 44 | `SurrogateTransfer_44` | AFP, FS, RS | Base, C1..C7 |
| 9 | `SurrogateTransfer` | 45 | `SurrogateTransfer_45` | AFP, FS, RS | Base, C1 |
| 10 | `SurrogateTransfer` | 46 | `SurrogateTransfer_46` | AFP, FS, RS | Base, C1 |
| 11 | `DecisionBoundary` | 42 | `DecisionBoundary_42` | AFP, FS, RS | Base, C1..C7 |
| 12 | `DecisionBoundary` | 43 | `DecisionBoundary_43` | AFP, FS, RS | Base, C1..C7 |
| 13 | `DecisionBoundary` | 44 | `DecisionBoundary_44` | AFP, FS, RS | Base, C1..C7 |
| 14 | `DecisionBoundary` | 45 | `DecisionBoundary_45` | AFP, FS, RS | Base, C1 |
| 15 | `DecisionBoundary` | 46 | `DecisionBoundary_46` | AFP, FS, RS | Base, C1 |

## 3. Test Suite Verification

### 3.1 Focused Phase 10B Tests (46 Passed)
The focused suite comprises 45 tests in `tests/test_build_evaluation_caches.py` and 1 dedicated training-derived smoke test in `tests/test_training_smoke.py`:

| Test Module & Case | Description | Result |
|---|---|---|
| `test_safe_preflight_default` | Preflight runs by default, generating 0 caches | **PASSED** |
| `test_execute_required_for_data_access` | Data access strictly requires `--execute` | **PASSED** |
| `test_preflight_evaluation_file_blocking` | Preflight never accesses evaluation data files | **PASSED** |
| `test_exact_derivation_of_15_pairs` | Derives exactly 15 canonical pairs from matrix | **PASSED** |
| `test_alias_reuse` | 27 aliases target primary runs with identical scenario/seed | **PASSED** |
| `test_frozen_file_byte_comparison` | Byte-for-byte and SHA-256 comparison vs freeze tag | **PASSED** |
| `test_frozen_file_tampering_detected` | Tampered frozen file causes immediate preflight failure | **PASSED** |
| `test_rejection_of_altered_freeze_ancestry` | Rejects non-ancestor or tampered freeze commit | **PASSED** |
| `test_protected_hash_mismatch_rejection` | Rejects any mismatch in RF, Roles, or Batches | **PASSED** |
| `test_no_defense_config_dimension_in_cache_identity` | Proves cache identity is independent of defenses/controllers | **PASSED** |
| `test_atomic_publication_and_failure_quarantine` | Staging is quarantined on error; destination remains clean | **PASSED** |
| `test_complete_cache_reuse` | Valid existing cache is reused without recomputing | **PASSED** |
| `test_invalid_cache_rejection` | Incomplete or corrupt cache is quarantined and rebuilt | **PASSED** |
| `test_path_containment_and_traversal_rejection` | Quarantine strictly rejects paths outside cache root | **PASSED** |
| `test_independent_cache_reuse_rejects_wrong_scenario` | Independent validation rejects self-consistent cache with wrong scenario | **PASSED** |
| `test_independent_cache_reuse_rejects_wrong_seed` | Independent validation rejects self-consistent cache with wrong seed | **PASSED** |
| `test_independent_cache_reuse_rejects_stale_provenance` | Independent validation rejects cache with one stale provenance hash | **PASSED** |
| `test_independent_cache_reuse_rejects_altered_manifest_and_completion` | Independent validation rejects cache whose manifest and completion altered together | **PASSED** |
| `test_independent_cache_reuse_rejects_earlier_orchestration_commit` | Independent validation rejects cache from earlier orchestration commit | **PASSED** |
| `test_reproducible_state_enforcement_passes_clean` | Enforces reproducible execution state when working tree is clean | **PASSED** |
| `test_reproducible_state_rejects_unstaged_tracked_modification` | Refuses execution if tracked files have unstaged modifications (' M') | **PASSED** |
| `test_reproducible_state_rejects_staged_modification` | Refuses execution if tracked files have staged modifications ('M ') | **PASSED** |
| `test_reproducible_state_rejects_staged_and_unstaged_modification` | Refuses execution if tracked files have combined staged/unstaged changes ('MM') | **PASSED** |
| `test_reproducible_state_rejects_staged_addition` | Refuses execution if tracked files have staged additions ('A ') | **PASSED** |
| `test_reproducible_state_rejects_untracked_code_file` | Refuses execution if untracked code/test/config files exist ('??') | **PASSED** |
| `test_reproducible_state_allows_historical_untracked_zip_archive` | Allows historical untracked ZIP archives and logs | **PASSED** |
| `test_partition_alignment_validation_passes_valid` | Validates 90k aligned rows, 18k craft, 72k meas, 144 batches of 500 | **PASSED** |
| `test_partition_alignment_rejects_shuffled_metadata_composite_identities` | Rejects shuffled metadata composite identities vs roles | **PASSED** |
| `test_partition_alignment_rejects_modified_raw_row_idx` | Rejects even a single modified `_raw_row_idx` | **PASSED** |
| `test_partition_alignment_rejects_modified_source_file` | Rejects even a single modified `_source_file` | **PASSED** |
| `test_partition_alignment_rejects_duplicated_evaluation_position` | Rejects duplicated eval_position values in X, meta, or roles | **PASSED** |
| `test_partition_alignment_rejects_missing_or_out_of_range_position` | Rejects out-of-range positions or gaps vs 0..89999 | **PASSED** |
| `test_partition_alignment_rejects_mismatched_y_binary` | Rejects mismatched y_binary between roles and metadata | **PASSED** |
| `test_partition_alignment_rejects_overlapping_composite_identities` | Rejects overlapping crafting and measurement composite identities | **PASSED** |
| `test_partition_alignment_rejects_extra_trailing_feature_column` | Rejects extra trailing feature columns (exact schema equality) | **PASSED** |
| `test_partition_alignment_rejects_reordered_feature_columns` | Rejects reordered feature columns | **PASSED** |
| `test_partition_alignment_rejects_incomplete_measurement_to_batch_coverage` | Rejects non-72000 measurement rows or incomplete batch mapping | **PASSED** |
| `test_partition_alignment_rejects_nan_or_inf` | Rejects non-finite (NaN or Inf) feature values | **PASSED** |
| `test_surrogate_fitting_labels_originate_from_oracle_not_ground_truth` | Proves surrogate training labels come from oracle, not ground truth | **PASSED** |
| `test_surrogate_candidate_generation_does_not_receive_labels` | Proves surrogate candidate generation takes no labels | **PASSED** |
| `test_decision_boundary_reference_selection_uses_model_predictions` | Proves DB reference selection uses model predictions on crafting rows, not true labels | **PASSED** |
| `test_boundary_perturbation_search_does_not_use_labels_for_path` | Proves boundary binary search interpolation path uses oracle decisions, not labels | **PASSED** |
| `test_label_change_affects_clean_tp_eligibility_without_altering_attack_logic` | Proves label changes affect clean-TP eligibility/status without altering attack logic | **PASSED** |
| `test_scenario_seed_filtering` | Validates CLI parsing and scenario/seed filters | **PASSED** |
| `test_deterministic_pair_ordering` | Verifies deterministic ordering of 15 pairs | **PASSED** |
| `test_training_smoke_uses_only_training_partition_and_tmp_dir` | Separated smoke test uses only training slice in temp dir | **PASSED** |

**Focused Results**: 46 passed (100%).

### 3.2 Full Project Regression Suite
- Total Tests: **246 passed** (0 failed, 0 errors across all test modules).
- Clean regression across all project modules.

---

## 4. Review Bundle Verification

The review bundle `phase10b_cache_orchestration_review_bundle_v2_1.zip` is constructed by `scripts/build_phase10b_review_bundle.py` strictly after committing all tracked Phase 10B source, test, configuration, and documentation changes:

- **Target Bundle Filename**: `phase10b_cache_orchestration_review_bundle_v2_1.zip`
- **Integrity Guarantee**:
  - The final member list is constructed exactly once.
  - `MANIFEST.txt` lists every final member exactly once.
  - `FILE_HASHES.sha256` records SHA-256 hashes for all members except itself.
  - Extraction verification requires `hash-manifest names == ZIP members - {"FILE_HASHES.sha256"}`.
  - Every listed hash is independently recomputed and verified byte-for-byte upon extraction.
  - Duplicate and forbidden scans operate over the final archive member set.
  - No tracked checksum file is rewritten after the final commit.
  - The final ZIP SHA-256 digest and size are calculated and reported externally upon archive completion.

---

## 5. Execution Command for Later Official Phase

When authorized to generate official attack caches in a subsequent phase, the exact command to execute is:

```bash
python scripts/build_evaluation_caches.py --execute
```
