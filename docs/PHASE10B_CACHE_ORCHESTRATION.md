# Phase 10B: Attack-Cache Orchestration Architecture & Semantics

## 1. Scope and Formal Boundaries

This document defines the architecture, operational semantics, and validation rules for the Phase 10 evaluation attack-cache orchestration layer implemented in `scripts/build_evaluation_caches.py`.

### 1.1 Scope Boundaries
It is critical to distinguish between the distinct stages of the thesis evaluation pipeline:
1. **Phase 10B (Current Phase — Orchestration Readiness)**:
   - Implements and verifies the orchestration script and CLI entry point.
   - Proves matrix derivation, provenance generation, atomic publication, failure quarantine, and whole-cache reuse.
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

`scripts/build_evaluation_caches.py` is designed as a **thin orchestration layer** over the frozen `AttackCacheBuilder`. It does not duplicate attack algorithms or cache verification logic; instead, it coordinates the derivation of runs, management of staging environments, atomic publication, and provenance binding.

### 2.1 CLI Interface and Safety Guarantees
The CLI is **safe by default**:
- Running without flags (`python scripts/build_evaluation_caches.py`) executes non-mutating preflight only.
- Passing `--preflight-only` explicitly executes the identical non-mutating preflight.
- Official cache construction requires the explicit `--execute` authorization flag.
- Provides optional selection filters: `--scenario` and `--seed` for controlled individual execution.
- Rejects any unknown scenarios, unauthorized seeds, corrupted configurations, or tampered hashes.

```text
build_evaluation_caches.py
  ├── parse_args()             -> validates CLI arguments, scenarios, and seeds
  ├── run_preflight()          -> verifies git ancestry, 3 protected hashes, configs, manifests
  ├── print_preflight_report() -> formats and displays 15 derived pairs and cache readiness
  └── build_caches()           -> [requires --execute]
        ├── derive_cache_pairs()
        ├── validate_completed_cache() [whole-cache reuse]
        ├── staging directory setup
        ├── AttackCacheBuilder.build()
        ├── completion.json emission
        └── atomic rename & quarantine
```

### 2.2 Preflight Mode Guarantees
Preflight mode performs exhaustive integrity verification without touching evaluation data:
- **Zero data access**: Never loads `data/processed/X_eval.parquet` or `data/processed/metadata_eval.parquet`.
- **Zero label leakage**: Does not read evaluation ground-truth labels.
- **Ancestry verification**: Proves that `phase10-protocol-freeze` (`65005505415a2bdf2d5744dbd135e9214e74081a`) is an ancestor of the working HEAD.
- **Protected hash check**: Asserts bit-level equality of the 3 protected SHA-256 hashes (RF, Roles, Batches).
- **Configuration freeze check**: Verifies `git diff phase10-protocol-freeze -- configs/` is empty and `experiment.date_frozen` matches `"2026-09-17T21:23:51+08:00"`.
- **Partition allocation check**: Inspects `evaluation_roles.csv` to confirm 18,000 crafting and 72,000 measurement records, and `evaluation_batches.csv` to confirm 144 batches of 500.

---

## 3. Dynamic Matrix Derivation

Rather than manually hardcoding cache targets, `derive_cache_pairs()` dynamically inspects the frozen matrix (`generate_evaluation_matrix(configs_dir)`):

### 3.1 Derived Combinations
- **Total Matrix Rows**: 279 (90 primary comparison runs + 189 sensitivity runs).
- **Aliases**: Exactly 27 sensitivity C1 runs are designated as aliases. Each alias points to an existing primary run with the **exact same attack scenario and seed**.
- **Unique Executions**: Exactly 252 runs requiring evaluation data.
- **Batches**: $252 \times 144 = 36,288$ batches.
- **Derived Distinct `(attack_scenario, seed)` Pairs**: Exactly **15 canonical pairs**.

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

### 3.2 Independence from Defense and Controller Configurations
Attack caches simulate adversary actions on network traffic **prior to the deployment of host defenses or adaptive controllers**. Therefore:
- The cache identity comprises strictly `(attack_scenario, effective_seed)`.
- All three perturbation defenses (Adaptive Feature Poisoning, Feature Squeezing, Randomized Smoothing) and all eight controller configurations (Base, C1..C7) share the exact same attack cache for a given scenario and seed.
- Generating caches before defenses ensures empirical comparison across defenses and controllers evaluates responses to identical adversary inputs.

---

## 4. Later Official Execution Behavior

When `--execute` is authorized in a future evaluation phase, cache generation follows strict data isolation rules:

### 4.1 Disjoint Partition Isolation
1. **Join by `eval_position`**: `X_eval.parquet` and `metadata_eval.parquet` are joined against `data/manifests/evaluation_roles.csv` using strict `eval_position` indexing.
2. **Crafting Pool (18,000 records)**:
   - Reserved strictly for adversary training/querying (Surrogate Transfer) and benign reference points (Decision Boundary).
   - Never included in measurement batches, runner evaluation, or statistical metrics.
3. **Measurement Pool (72,000 records)**:
   - Contains exactly 144 sequential batches of 500 records.
   - All 72,000 rows are preserved in every cache. Ineligible or unattempted rows retain their original feature values.
   - Every column must remain finite float32, with columns exactly matching `feature_columns` in order.

### 4.2 Candidate Generation Label Blindness
- **Evaluation ground-truth labels are never supplied to an attack generator**.
- In **Surrogate Transfer**: A Decision Tree surrogate is fitted on `X_crafting` using labels queried from the BlackBoxOracle (`y_pool_oracle`). During measurement candidate generation, `generate_candidate(x)` operates strictly on feature vector `x` with zero access to true labels.
- In **Decision Boundary**: Screening queries identify eligible attack targets using target model predictions. Benign references are drawn exclusively from crafting benign samples.
- True labels are consulted **only post-hoc** in `evaluate_transfer` or status logging to categorize eligibility (`INELIGIBLE_TRUE_BENIGN`, `INELIGIBLE_FALSE_NEGATIVE`) and evasion success.

### 4.3 Scenario Protocols
1. **Silent Probing**:
   - Identity transformation ($X_{\text{adv}} = X_{\text{meas}}$).
   - Zero oracle queries ($Q = 0$).
   - ASR is not applicable (`global_asr = None`).
   - All sample statuses are strictly `NOT_APPLICABLE`.
2. **Surrogate Transfer**:
   - Surrogate Decision Tree fitted strictly on the 18,000 crafting records.
   - Target queries for candidate evaluation tracked separately through `BlackBoxOracle`.
   - All 72,000 measurement positions retained.
3. **Decision Boundary**:
   - Hard-label binary search interpolation using `BlackBoxOracle` with `max_queries_per_sample = 50`.
   - Exactly 200 globally selected measurement targets per seed via `select_boundary_targets`.
   - Unselected and unattempted samples retain original feature values with zero queries used.

---

## 5. Atomic Output, Resumption, and Quarantine

To prevent corrupted or half-written caches from ever being consumed by the experiment runner:

### 5.1 Staging and Atomic Publication Lifecycle
1. **Staging Directory**: Every cache is initially built into a separate staging path:
   `artifacts/caches/{scenario}_{seed}.staging`
2. **Builder Isolation**: `AttackCacheBuilder` internally builds into `.tmp`, verifies schema through `ConcreteAttackCacheProvider`, and writes into `.staging`.
3. **Completion Marker Written Last**:
   - After all parquet files and `manifest.json` are written, `completion.json` is generated.
   - Contains start/end timestamps, artifact checksums, git commits, and completion state (`COMPLETED`).
4. **Pre-Publication Re-Validation**:
   - `validate_completed_cache()` verifies that `manifest.json` matches all expected hashes and that `ConcreteAttackCacheProvider` can parse all 72,000 rows.
5. **Atomic Rename**:
   - Only after successful validation is `.staging` renamed to `artifacts/caches/{scenario}_{seed}`.

### 5.2 Resumption and Whole-Cache Reuse
- If a target cache directory exists:
  - It is inspected with `validate_completed_cache()`.
  - If `completion.json` exists, state is `COMPLETED`, checksums match, and provider validation succeeds: the cache is **reused without recomputation**.
  - A valid completed cache is **never overwritten**.
- If a target cache directory exists but is incomplete, missing files, or corrupted:
  - It is immediately quarantined to `{scenario}_{seed}_quarantined_{timestamp}`.
  - Generation starts from the beginning.
  - **No partial-cache resumption is permitted.**

### 5.3 Failure Quarantine
- If any exception occurs during candidate generation, serialization, or validation:
  - The `.staging` directory and any intermediate `.tmp` directories are immediately renamed to `{scenario}_{seed}_failed_quarantined_{timestamp}`.
  - No incomplete directory is left at the canonical destination path.

---

## 6. Provenance Requirements

Every cache directory contains two companion provenance artifacts:

1. **`manifest.json`** (Adheres to frozen `_MANIFEST_REQUIRED_KEYS` & `AttackCacheManifest`):
   - Model hash (`frozen_rf_hash`)
   - Split & assignment hashes (`evaluation_roles_hash`, `evaluation_batches_hash`, `crafting_identity_hash`, `measurement_identity_hash`)
   - Preprocessor hashes (`feature_names_hash`, `feature_mask_hash`, `training_bounds_hash`, `scaler_hash`)
   - Configuration hashes (`attacks_yaml_hash`, `controllers_yaml_hash`, `defenses_yaml_hash`, `experiment_yaml_hash`)
   - Evaluation data hashes (`X_eval_hash`, `metadata_eval_hash`)
   - Script hashes (`attack_script_hashes`)
   - Attack parameters and query budgets
   - Artifact checksums (`X_attacked_sha256`, `status_sha256`, `output_sha256`)
   - Row count (exactly 72,000)

2. **`completion.json`** (Extended Orchestration Completion Marker):
   - Freeze commit: `65005505415a2bdf2d5744dbd135e9214e74081a`
   - Freeze tag: `phase10-protocol-freeze`
   - Execution script commit: Current HEAD
   - Start and end ISO 8601 timestamps
   - Completion state: `"COMPLETED"`
   - Artifact checksum mapping for `X_attacked.parquet`, `status.parquet`, and `manifest.json`
