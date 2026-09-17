# Phase 10B: Attack-Cache Orchestration Architecture & Semantics

## 1. Scope and Formal Boundaries

This document defines the architecture, operational semantics, and validation rules for the Phase 10 evaluation attack-cache orchestration layer implemented in `scripts/build_evaluation_caches.py`.

### 1.1 Scope Boundaries
It is critical to distinguish between the distinct stages of the thesis evaluation pipeline:
1. **Phase 10B (Current Phase — Orchestration Readiness)**:
   - Implements and verifies the orchestration script and CLI entry point subject to 10 binding corrections.
   - Proves matrix derivation, byte-for-byte freeze verification, label isolation, complete real provenance, safe staging/quarantine, and whole-cache reuse.
   - Operates strictly **without accessing evaluation features (`X_eval.parquet`) or evaluation labels (`metadata_eval.parquet`)**.
   - Generates **zero official caches**; conducts **zero official runs**.
2. **Later Official Cache Construction**:
   - Authorized exclusively by future execution with `--execute`.
   - Accesses evaluation records through exact `eval_position` joins.
   - Generates the 15 canonical attack caches.
3. **Later Defense Experiments**:
   - Executes Base and RA controllers across all 252 unique runs using pre-generated attack caches.
4. **Empirical Recall-Aware (RA) Effectiveness**:
   - **Remains strictly unknown**. Statistical significance, recall stability, evasion resistance, and trade-off metrics will only be determined after official runs complete.

---

## 2. Orchestration Architecture (`scripts/build_evaluation_caches.py`)

`scripts/build_evaluation_caches.py` is designed as a **thin orchestration layer** over the frozen `AttackCacheBuilder`. It does not duplicate or alter frozen experiment logic in `src/`; instead, it coordinates run derivation, staging safety, byte-level freeze validation, and provenance binding.

### 2.1 CLI Interface and Safety Guarantees
The CLI is **safe by default**:
- Running without flags (`python scripts/build_evaluation_caches.py`) executes non-mutating preflight only.
- Passing `--preflight-only` explicitly executes the identical non-mutating preflight.
- Official cache construction requires the explicit `--execute` authorization flag.
- Provides optional selection filters: `--scenario` and `--seed` for controlled individual execution.
- Rejects any unknown scenarios, unauthorized seeds, corrupted configurations, or tampered hashes.

```text
build_evaluation_caches.py
  ├── parse_args()                    -> validates CLI arguments, scenarios, and seeds
  ├── verify_frozen_configurations()  -> byte-for-byte and SHA-256 comparison vs freeze tag
  ├── run_preflight()                 -> non-mutating check of git, 3 protected hashes, configs, manifests
  ├── print_preflight_report()        -> formats and displays 15 canonical pairs and cache readiness
  └── build_evaluation_caches()       -> [requires --execute]
        ├── derive_cache_pairs()
        ├── validate_completed_cache() [strict whole-cache reuse]
        ├── safe staging directory beneath cache root
        ├── AttackCacheBuilder.build()
        ├── completion.json emission
        └── atomic rename & safe quarantine beneath cache root
```

### 2.2 Strengthened Freeze Verification
Preflight verifies that the protocol files on disk are **identical to the byte** with their tagged versions at `phase10-protocol-freeze`:
- `configs/attacks.yaml`
- `configs/controllers.yaml`
- `configs/defenses.yaml`
- `configs/experiment.yaml`
- `configs/model.yaml`
- `docs/EXPERIMENT_PROTOCOL.md`

For every file, `git show phase10-protocol-freeze:<path>` is retrieved, and both the raw byte streams and the SHA-256 digests are asserted to match exactly. Additionally, `git ls-tree` verifies that no extra or missing files exist in `configs/`.

---

## 3. Canonical Matrix Derivation

`derive_cache_pairs()` dynamically inspects the frozen matrix (`generate_evaluation_matrix(configs_dir)`), normalizes scenario names, and validates strict cache identity constraints:

### 3.1 Canonical Cache Pairs
- **Scenarios**: `SilentProbing`, `SurrogateTransfer`, `DecisionBoundary`.
- **Seeds**: `42`, `43`, `44`, `45`, `46`.
- **Count**: Exactly **15 ordered `(scenario, seed)` pairs**.
- **No Defense or Controller Fields**: Cache identity comprises strictly `(scenario, seed)`.
- **Deterministic Pair Ordering**: Ordered by canonical scenario order, then seed ascending.
- **Sensitivity-Alias Reuse**: All 27 sensitivity C1 runs map to primary runs with the exact same `(scenario, seed)` identity.

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

---

## 4. Accurate Label Isolation Semantics

Per the frozen evaluation protocol and thesis threat model:
1. **Adversary Knowledge Isolation**:
   - **Surrogate Fitting**: True labels are never provided to surrogate training; surrogate fitting labels originate strictly from `BlackBoxOracle(predict_fn)` hard-label decisions.
   - **Benign Reference Pool**: True labels do not choose the benign reference pool. In Decision Boundary, references are selected strictly by querying target model predictions on crafting rows (`predict_fn(X_craft) == 0`), not from ground-truth `y_craft`.
   - **Perturbation Search**: True labels do not guide threshold or path construction, interpolation direction, or binary search midpoint evaluation.
   - **Candidate Construction**: Candidate perturbation generators take only feature vectors `X_orig` and interact with the model strictly via the black-box oracle.
2. **Harness Label Usage**:
   - The evaluation harness may use ground-truth labels for clean-TP eligibility (`orig_pred == 1 & y_meas == 1`), deterministic selection of the 200 Decision-Boundary targets from eligible clean TPs, and post-generation status/ASR accounting.

---

## 5. Independent Cache-Reuse Validation

A cache directory is considered complete and reusable **only** if validated against independently computed expectations derived from current inputs—never by circular comparison against the cache's own manifest:

1. **Independent Identity Derivation**:
   - Before checking cache reuse, the orchestrator computes `expected_cache_identity` from canonical scenario, seed, frozen configurations, preprocessor hashes, and manifest identities.
   - The cache's own manifest is never passed back as its own expected identity.
2. **Strict Verification Checks**:
   - Completion and manifest scenarios equal the requested canonical scenario.
   - Completion and manifest seeds equal the requested seed.
   - Every required manifest identity/provenance field equals the independently computed expected value.
   - Exact matching: no missing or additional identity fields.
   - Completion provenance matches the independently computed identity.
   - Attacked-feature (`X_attacked.parquet`) and status (`status.parquet`) file hashes match both manifest and completion records.
   - `ConcreteAttackCacheProvider` parses the directory, verifying:
     - Exactly 72,000 measurement rows (or N during synthetic tests).
     - 78 finite float32 feature columns in exact frozen order.
     - Alignment of `eval_position` with `evaluation_batches.csv`.
     - Status table boolean and numeric constraints.
3. **Reproducible Execution State**:
   - Official execution (`--execute`) verifies that the freeze tag is an ancestor of HEAD, frozen configurations and protocol match byte-for-byte, protected hashes match, `src/` is unmodified relative to the freeze tag, and the Git working tree has zero dirty or untracked source/test/config files.

File existence alone is **strictly insufficient** for reuse. Any incomplete, altered, or stale cache is quarantined and rebuilt.

---

## 6. Safe Staging and Quarantine

1. **Containment Beneath Cache Root**:
   - All staging directories (`{target}.staging_{pid}_{uuid}`) and quarantine directories (`{prefix}_quarantined_{ts}`) are created strictly beneath the configured cache root (`output_dir`).
   - `quarantine_directory()` asserts that the directory to be quarantined is strictly relative to `cache_root` and is not `cache_root` itself.
   - Any attempt to move, quarantine, or write to a path outside `cache_root` is immediately rejected with `ValueError`.
2. **Atomic Publication**:
   - Staging directories are located on the same filesystem as `output_dir`.
   - `completion.json` is written LAST into the staging directory.
   - Only after full re-validation of the staging directory is it atomically renamed to the target cache directory.

---

## 7. Complete Real Provenance Forwarding

`AttackCacheBuilder` receives and validates complete, uncompromised provenance hashes:
- Freeze Commit (`65005505415a2bdf2d5744dbd135e9214e74081a`) & Tag (`phase10-protocol-freeze`)
- Current Orchestration Commit
- Implementation script hashes (`base.py`, `silent_probing.py`, `surrogate_transfer.py`, `boundary_attack.py`, `oracle.py`, `caching.py`, `build_evaluation_caches.py`)
- Configuration hashes (`attacks.yaml`, `controllers.yaml`, `defenses.yaml`, `experiment.yaml`, `model.yaml`)
- Feature artifacts (`scaler.joblib`, `feature_names.json`, `feature_mask.json`, `training_bounds.parquet`)
- Role and batch hashes (`evaluation_roles.csv`, `evaluation_batches.csv`, `crafting_identity_hash`, `measurement_identity_hash`)
- Model hash (`frozen_rf.joblib`)
- Attack parameters and query budgets

Every provenance hash must be a valid 64-character lowercase hex string. Missing, placeholder, fabricated, or malformed hashes are rejected with an immediate error.

---

## 8. Separation of Training Smoke Test

- **Synthetic Unit Suite** (`tests/test_build_evaluation_caches.py`): Contains 45 focused tests using purely synthetic fixtures. Does not load any parquet dataset files.
- **Training-Derived Smoke Test** (`tests/test_training_smoke.py`): Contains 1 end-to-end smoke test using a small deterministic 60-row slice of `data/processed/X_train.parquet` and `metadata_train.parquet`. Writes to a temporary directory in `tmp_path` (completely outside `artifacts/caches`). Strictly asserts that `X_eval.parquet` and `metadata_eval.parquet` are never opened.
