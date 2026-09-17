# Model Handoff Checkpoint — Recall-Aware IDS

**Written:** September 2026
**Project Root (Canonical):** `/Users/johnferrylagman/Desktop/thesissep20276/september 2026/recall-aware-ids/`
**Dataset (Read-Only):** `/Users/johnferrylagman/Desktop/thesissep20276/datasets/cic-ids2018/`

---

## 1. Current Project Status by Phase

*   **Phase 0 (Protocol Freeze & Docs):** ✅ **Completed**. Architecture, configs, and constraints documented. Canonical project root established.
*   **Phase 1 (Environment):** ✅ **Completed**. Virtual environment, seeds, and initial scaffolding exist.
*   **Phase 2 (Dataset Validation):** ✅ **Completed**. CSVs verified in the raw dataset directory.
*   **Phase 3 (Dataset Audit & Extraction):** ✅ **Completed**. See bugs resolved below. Exactly 300,000 records sampled cleanly. 
*   **Phase 4 (Preprocessing):** ✅ **Completed**. Exact 90,000 Eval, 178,500 Train, 31,500 Cal subsets partitioned cleanly with zero composite-identifier overlaps, zero NaNs, and excluded `Src Port` due to documented missingness in 9 out of 10 source files.
*   **Phase 5 (Training Frozen RF):** ✅ **Completed**. RF trained exclusively on `X_train` (178,500). Secondary OOB learning curve recomputed to correct metric reporting artifacts (F1 != Balanced Accuracy); tests verify strict math definitions. Official `frozen_rf.joblib` remains entirely intact and valid.
*   **Phase 6 (Evaluation Roles & Batches):** ✅ **Completed**. 90,000-record `X_eval` securely partitioned into an 18k Crafting Pool and a 72k Measurement Pool. Exactly 144 immutable measurement batches created.
*   **Phase 7 (Attack Implementation & Repair):** ✅ **Completed**. `BlackBoxOracle` strictly forces batch duplicate limits. Repaired all bounds math and rewrote complete synthetic suite. Tests passed perfectly. No final caches created. Zero evaluation data loaded.
*   **Phase 8 (Defense Implementation & Calibration):** ✅ **Completed**. AFP calibrated at `epsilon_base=0.0003`, `alpha=0.5` (recall `0.8122`); RS calibrated at `sigma=0.0002` (recall `0.8217`); FS calibrated at `intensity=2` (recall `0.9403`).
*   **Phase 10A (Integration, Execution-Path Validation, & Protocol Freeze Readiness):** ✅ **Completed**. Red-green integration repair and final review corrections validated on native M4 hardware (`.venv-m4`). 196/196 tests passing. 5,000-sample training-derived pilot validated (peak RSS 469.19 MB). Strict schema, cache provider/builder, output reopening/validation, and controller contracts verified. Protected artifacts remain bit-for-bit identical. `experiment.date_frozen` remains `null`.
*   **Phase 10B+ (Official Evaluation & Statistical Analysis):** ⏳ **Pending External Review**. Official runs (252 executions) will commence once freeze candidate bundle v5.1 is approved.

---

## 2. Phase 5 & 6 Methodological Clarifications

1. **Learning Curve Stratification:** The subsets used in the Phase 5 learning curve were constructed via deterministic nested random selection. While their observed class proportions remained highly representative of the full training distribution, they were not strictly enforced to be perfectly stratified down to the integer.
2. **Evaluation Batch Counts:** Any earlier references to "180 batches" only described the theoretical mathematical divisibility of the *unsplit* 90,000-row evaluation partition. After enforcing the frozen 20% crafting reservation (18,000 records), the final experimental measurement pool is 72,000 records, yielding exactly **144 measurement batches of 500**.
3. **Configuration Freeze Date:** The `experiment.date_frozen` field has deliberately not been set yet. The final configuration-freeze date will be established after defense calibration and controller parameter selection, immediately before Phase 10 final runs.

## 3. Phase 3 Bugs Discovered and Resolved

During Phase 3, two critical data processing bugs were uncovered and resolved:

### Bug 1: Numeric Feature Coercion and NaN Dropping
*   **Issue:** Many columns in the original dataset had `inf` or `nan` values, or mixed strings disguised as numeric features.
*   **Resolution:** Enforced `pd.to_numeric(errors="coerce")` on all feature columns during the audit pass. `inf` values were replaced with `nan`, and any row containing `nan` in its feature set was explicitly dropped using `.dropna(subset=numeric_feature_cols)`.

### Bug 2: Index-Matching Extraction Failure
*   **Issue:** The Pass 2 extraction script previously attempted to extract the working sample by storing the exact `pandas DataFrame index` of valid rows during Pass 1, which reset/shifted when rows were dropped, causing a misalignment.
*   **Resolution:** We completely eliminated the need for index matching by implementing a **single-pass true streaming hypergeometric sampler**. For each chunk read, we allocate exact strata quotas directly based on remaining population and quota, sampling inline exactly the required number of records using `numpy.random.hypergeometric` and `numpy.random.choice` without replacement.

---

## 4. Inventory of Generated Artifacts (Phase 3)

The dataset audit generated the following final artifacts in the canonical root:

*   **`data/manifests/raw_files.csv`**: Contains file sizes, total rows, valid rows, dropped rows, and duplicated headers for the 10 source files.
*   **`data/manifests/schema.json`**: Data types (schema) for all columns extracted during the audit.
*   **`data/manifests/class_distribution.csv`**: Global distribution of labels across all 10 files.
*   **`data/manifests/invalid_rows_by_file.csv`**: Log of invalid/dropped rows per source file.
*   **`data/manifests/working_sample_manifest.json`**: Exact stratum allocations, SHA-256 hash, and metadata verifying exactly 300,000 working sample rows extracted.
*   **`data/interim/working_sample.parquet`** (42.7 MB): The finalized 300,000-row stratified dataset, clean and ready for Phase 4 preprocessing.
*   **`artifacts/reports/data_audit.md`**: The human-readable markdown report of the dataset status.

---

## 5. Instructions for the Next Agent

1. **Phase 10A is Complete**: Integration semantics, cache builder/provider contracts, runner atomic invariants, and matrix validation are fully repaired and verified (174/174 tests passing, training-derived pilot validated).
2. **Canonical Root**: Always use the canonical project root at `recall-aware-ids/`.
3. **Execution**: The next task is Phase 10B (Official Evaluation Matrix Execution) upon formal external sign-off on the v5.1 freeze candidate bundle. The official frozen RF model is serialized in `artifacts/models/frozen_rf.joblib` and must not be retrained.
4. **Configuration Freeze Date**: Keep `experiment.date_frozen` as `null` until formal external review approves protocol freeze.
5. **Constraints**: Maintain hardware memory constraints (8 GB RAM limit). Keep the data chunked if processing entire ensembles. Do not inspect evaluation records prematurely.
