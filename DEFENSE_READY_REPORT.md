# THESIS TOOL DEFENSE READINESS REPORT
**Standalone ML-IDS Demonstration Platform (`ai_recreation`)**  
**Audit & Defense Preparation Date:** October 6, 2026  
**Auditor / Implementation Agent:** Antigravity (Google DeepMind)

---

## 1. Executive Summary & Git Repository State

This report certifies the defense readiness and audit verification of the standalone IDS demonstration platform (`ai_recreation`), incorporating full support for the **90,000-row Phase 10 frozen research evaluation dataset** alongside the preserved **20-flow fixture regression profile**.

The demonstration platform strictly enforces the research foundation: the frozen Random Forest model, training bounds, standard scaler, feature mask, core attack and defense mathematics, and the 5-decision Recall-Aware controller cadence are preserved byte-for-byte. The platform cleanly decouples query telemetry (surrogate fitting and decision boundary bisection) from target metric accounting, ensuring that live dashboard confusion metrics, recall gauges, and controller adaptation are driven solely by measured target flows.

### Git State Metadata
- **Repository Root**: `c:\Users\reddr\ai_recreation`
- **Active Git Branch**: `feature/expanded-simulation-data` (isolated reversible branch branched from `integrate-runtime-demo`)
- **Starting Git Commit**: `a240ecc3d7934a229ee0fdb72a97d04615de06a0`
- **Working Tree State**: Clean and isolated modifications across data loading, server telemetry, attacker sampling, test suites, and documentation.
- **Packaging Standard**: Per security best practices, the deliverable archive includes a verified `FILE_MANIFEST.csv` certifying the cryptographic SHA-256 hash of every included file. The archive's outer SHA-256 checksum is published outside the archive to prevent circular hash dependencies. The previous validated 20-flow fallback archive is safely preserved as `ids_standalone_tool_audit_fixture20_fallback.zip`.

---

## 2. Source-of-Truth Invariants Verified

| Invariant | Specification / Expectation | Verified State |
| :--- | :--- | :--- |
| **Model Checksum** | SHA-256 `9608672c5d5e38a9272c560679cdd2291399de0e0917c8c9dff373bfd200f51d` | **EXACT MATCH** (Verified via hashlib SHA-256) |
| **Model Architecture** | 200-tree binary RandomForest (`[0, 1]`), 78 numeric features | **EXACT MATCH** (`n_estimators=200`, `n_features_in_=78`, `classes_=[0, 1]`) |
| **Feature Mask** | Exactly 63 modifiable and 15 protected features | **EXACT MATCH** (`sum(mask)==63`, `sum(~mask)==15`) |
| **Standard Scaler** | Training-fitted StandardScaler joblib | **EXACT MATCH** (`e4604b39bbf8479e0a05b3e218206d2038740c3ff4b58e7b952f4eb27d532b2f`) |
| **Training Bounds** | Per-feature min/max bounds parquet | **EXACT MATCH** (`9554f1b0287c88b5ef7b8de19d193184f2d76eec2b4d96f1fa525818cffb275b`) |
| **AFP Benign Profile** | Centroid & standard deviation profile | **EXACT MATCH** (`cba824db6b78083c50937c569f257d0794957e8498f399f57d6b38c26bb5e5ea`) |
| **Core Algorithms & Configs** | 18 core Python and YAML files across `attacks/`, `defenses/`, `controller/`, `configs/` | **EXACT MATCH** (All 18 files byte-identical) |
| **Fixture20 Parquets** | Byte-identical regression profile (`X_demo.parquet`, `metadata_demo.parquet`) | **PRESERVED BYTE-FOR-BYTE** (`a39f6ed...`, `593479e...`) |
| **Expanded Data Archive** | `ids_expanded_simulation_data.zip` (11,404,876 bytes, SHA-256 `35b2d3ed...`) | **VERIFIED & EXTRACTED** to `runtime_package/expanded_data/` |
| **Expanded Eval Data** | `X_eval.parquet` (90,000 x 78 float32), `metadata_eval.parquet` (90,000 rows) | **EXACT MATCH** (`cd509adc...`, `b213bb0f...`) |
| **Evaluation Roles** | `evaluation_roles.csv`: 72,000 measurement, 18,000 crafting | **EXACT MATCH** (`cbf65087...`, 0 overlap, full coverage) |
| **Evaluation Batches** | `evaluation_batches.csv`: 144 batches of 500 rows | **EXACT MATCH** (`4084017e...`, exact alignment) |

---

## 3. Architecture & Integration Details

### A. Shared Dataset Loader (`runtime_package/data_loader.py`)
- Provides a clean, single-point interface: `get_dataset(profile: str) -> LoadedDataset`.
- Caches loaded datasets in memory once per process, eliminating disk rereads across HTTP requests.
- Validates 78 finite numerical float32 columns, pre-indexes crafting and measurement roles, and validates 0 role overlap.
- Pre-indexes class indices:
  - `measurement_indices`: 72,000 rows (59,743 Benign, 12,257 Attack)
  - `crafting_indices`: 18,000 rows (14,936 Benign, 3,064 Attack)
- Derives canonical evaluation position (`0` through `89,999`) in memory from original row indices.

### B. Server Data Routing & Telemetry (`server.py`)
- Governed by `IDS_DATA_PROFILE` environment variable (defaults to `expanded`, supports explicit `fixture20`).
- Exposes data telemetry in `/api/dashboard/stats`: `data_profile`, `data_fingerprint`, `pool_stats` (measurement rows, crafting rows, available targets, available references).
- Strict profile & bounds validation:
  - Validates `data_profile` in incoming requests against the server's active profile; rejects mismatches with `400 Bad Request` prior to modifying state or counters.
  - Rejects unknown profiles and out-of-range `sample_id` values with `400 Bad Request`.
- Strict metric separation:
  - Crafting queries (`is_query=True`, `query_stage` present, or crafting-role row) increment `query_count` telemetry only.
  - Target flows (`is_query=False`) update `total_traffic`, `detected_attacks`, confusion matrix (`tp`, `fn`, `fp`, `tn`), and advance the 5-decision Recall-Aware controller cadence.

### C. Attacker Console Sampling & Workflows (`attacker_sim.py`)
- Implements seeded queues without replacement:
  - `measurement_benign_queue`, `measurement_attack_queue`, and `measurement_ddos_queue`.
  - Guarantees targets are not reused within a session across menu actions while queues still have unused rows.
  - Re-seeds with explicit notification upon complete pool exhaustion.
- Silent Probing: draws fresh attack target from 72,000 measurement pool; transmits unchanged vector once with 0 crafting queries.
- Surrogate Transfer: draws 20 crafting references (10 benign, 10 attack) to fit local Decision Tree surrogate; queries oracle with `is_query=True`; draws fresh measurement attack target, crafts candidate, and submits final candidate once as measured target flow.
- Decision Boundary: draws 50 benign references from crafting pool; performs 1D bisection search with `is_query=True` (up to 50 queries); submits final evasion candidate once as measured target flow.
- Comparative Benchmark: evaluates selected defense under Base vs. Recall-Aware on identical 5-flow sequence.
- Cross-Defense Benchmark: replays bounded 5-flow measurement sequence across AFP, RS, FS, and None in Base mode, resetting metrics before each arm and reporting confusion matrices, recall, FPR, and confidence semantics (RS vote fractions vs RF probability estimates).

### D. Frontend Presentation (`frontend/`)
- Header Dataset Badge: dynamically displays `● Expanded (72k targets / 18k crafting)` or `● Fixture20 (20 targets)` based on active server telemetry.
- Feed Event Labeling: distinctly tags query flows (`[QUERY: surrogate_fitting]` / `[QUERY: boundary_search]`) to prevent visual confusion with measured target flows.

---

## 4. Verification & Test Evidence

### Test Suite Execution Summary

| Test Suite / Script | Command Executed | Tests Run | Result | Evidence / Details |
| :--- | :--- | :---: | :---: | :--- |
| **Expanded Data Integration Suite** | `python tests/test_expanded_data_integration.py` | 6 / 6 | **100% PASS** | 90,000 data contract, roles, schemas, seeded sampling without replacement, queue exhaustion, profile mismatch rejection, live HTTP measurement submissions, query vs target metric separation, cross-defense benchmark across AFP/RS/FS/None. |
| **Comprehensive Readiness Suite** | `python tests/verify_defense_readiness.py` (fixture20) | 12 / 12 | **100% PASS** | Model SHA-256, feature mask, offline CLI `run.py`, oracle error propagation, cold start & reset truthfulness, input validation, genuine evasion candidate replay adaptation proof (0.00030 -> 0.00012), seed independence, provenance, surrogate fitting HTTP path accounting, HTML escaping, dynamic target resolution. |
| **Audit Test Runner (A–G)** | `python tests/run_audit_tests.py` | 7 / 7 | **100% PASS** | No attacker baseline, silent probing, genuine boundary attack evasion, dataset family validation, unknown synthetic handling, IP geolocation, mode toggles. |
| **Pytest Full Suite** | `pytest -v` | 22 tests | **18 PASS / 4 SKIP** | All unit, contract, and fixture tests pass; live expanded tests skip cleanly when server is in fixture20 mode. |

### Expanded Integration Verification Log

```text
=================================================================
  EXPANDED SIMULATION DATA INTEGRATION VERIFICATION
=================================================================

>> [Check 1] Verifying 90,000-row Data Contract, Roles, and Schemas...
   [PASS] 90,000 rows validated: 72k measurement (59,743 benign, 12,257 attack), 18k crafting (14,936 benign, 3,064 attack).
   [PASS] 78 float32 finite features, 63 modifiable, 15 protected. Complete coverage, 0 overlap.

>> [Check 2] Verifying Seeded Sampling Without Replacement & Queue Exhaustion...
   [PASS] 100 benign & 50 attack measurement draws verified with 0 duplicates before exhaustion.
   [PASS] Seed reproducibility and pool exhaustion cycle mechanics verified.

>> [Check 3] Verifying Backend Profile Mismatch & Input Validation Rejection...
   [PASS] Profile mismatch rejected with 400 before counter modification.
   [PASS] Unknown profile and out-of-bounds sample_ids rejected with 400 before counter modification.

>> [Check 4] Verifying Live HTTP Submissions of Measurement Flows...
   [PASS] Benign (ID 0) and Attack (ID 7) successfully evaluated over HTTP.
   [PASS] Target flow counters correctly updated: total_traffic=2, decisions=1.

>> [Check 5] Verifying Query Telemetry vs. Target Decision Metric Separation...
   [PASS] 10 crafting queries incremented query_count to 10 with 0 pollution of target metrics.
   [PASS] Controller batch_id remained 0 during queries; target flow incremented traffic to 1.

>> [Check 6] Verifying Cross-Defense Comparison Workflow...
========================================================================================
  CROSS-DEFENSE EVALUATION SUMMARY (IDENTICAL 5-FLOW MEASUREMENT SEQUENCE)
========================================================================================
Defense    Mode     TP    FN    FP    TN    Recall     FPR        Intensity    Confidence Semantics    
------------------------------------------------------------------------------------------------
AFP        Base     3     0     0     2     1.000      0.000      0.00030      RF probability          
RS         Base     3     0     0     2     1.000      0.000      0.00020      Vote fraction (ensemble)
FS         Base     3     0     0     2     1.000      0.000      2.00000      RF probability          
NONE       Base     3     0     0     2     1.000      0.000      0.00000      RF probability          
------------------------------------------------------------------------------------------------
   [PASS] Cross-defense comparison completed across AFP, RS, FS, and None.

=================================================================
  ALL EXPANDED DATA INTEGRATION CHECKS PASSED SUCCESSFULLY!
=================================================================
```

---

## 5. Thesis Defense Rehearsal Script

### Step 1: Server Launch in Normal Expanded Mode (Terminal 1)
```powershell
cd c:\Users\reddr\ai_recreation
.\.venv\Scripts\python.exe server.py
```
- Open browser to `http://localhost:8000`.
- **Points to Highlight to Committee**:
  1. **Dataset Profile Pill**: Displays `● Expanded (72k targets / 18k crafting)`. Point out that the tool is connected to the full research measurement pool.
  2. **Truthful Cold Start**: Total Traffic is 0, Detected Attacks is 0, Recall and FPR display `—`, and Threat Locations shows an empty map.
  3. **Offline Integrity**: Scripts, styles, Chart.js, and Leaflet load locally from `frontend/vendor/` with zero external CDN dependencies.

### Step 2: Sampling Without Replacement (Terminal 2)
```powershell
cd c:\Users\reddr\ai_recreation
# Send 10 benign measurement flows sampled without replacement
.\.venv\Scripts\python.exe attacker_sim.py --mode benign --count 10
```
- Observe in console that each draw reports distinct original measurement IDs (e.g., IDs from 0 to 89,999 within the 72,000 measurement pool).
- Dashboard updates: Total Traffic = 10, Detected Attacks = 0, FPR = 0.000.

### Step 3: Silent Probing & Adversarial Attacks
```powershell
# 1. Silent Probing: unperturbed measurement attack flow
.\.venv\Scripts\python.exe attacker_sim.py --mode silent

# 2. Surrogate Transfer: crafting references -> surrogate fit -> candidate submission
.\.venv\Scripts\python.exe attacker_sim.py --mode surrogate

# 3. Decision Boundary: bisection search queries isolated in query telemetry
.\.venv\Scripts\python.exe attacker_sim.py --mode boundary
```
- Point out to committee: during surrogate fitting and boundary bisection, query calls are logged with `is_query=True`, incrementing `query_count` telemetry without corrupting live target traffic counts or advancing the controller cadence prematurely.

### Step 4: Cross-Defense Benchmark Replay
```powershell
.\.venv\Scripts\python.exe attacker_sim.py --mode compare-all
```
- Evaluates the identical 5-flow saved measurement sequence across AFP, RS, FS, and None under Base mode.
- Committee point: emphasize that RS confidence represents vote fractions across 11 sub-models, whereas AFP, FS, and None report Random Forest probability estimates. Equal verdicts do not imply identical internal representations.

### Step 5: Regression Fallback (Fixture20 Rehearsal)
To demonstrate backward compatibility with the original 20-flow rehearsal:
```powershell
# Stop server (Ctrl+C), then restart in fixture20 mode:
$env:IDS_DATA_PROFILE="fixture20"
.\.venv\Scripts\python.exe server.py

# In Terminal 2, run readiness suite to demonstrate genuine sample 10 evasion and controller adaptation:
.\.venv\Scripts\python.exe tests/verify_defense_readiness.py
```
- All 12 checks pass cleanly, verifying the genuine sample 10 evasion candidate and controller adaptation ($0.00030 \to 0.00012$).

---

## 6. Explicit Presentation Boundaries

The presenter must clearly communicate these technical boundaries during the examination:

1. **Preprocessed Feature Vectors**:
   - The platform evaluates 78-feature normalized float32 vectors (CSE-CIC-IDS2018 schema).
   - It **does not** sniff raw PCAP network packets or extract online flow features from network hardware interfaces.
2. **Local Demonstration vs. Official Thesis Evaluation**:
   - The standalone tool is an interactive, real-time HTTP demonstration of inline defenses and adaptive feedback control.
   - It is not a substitute for the official Phase 10/11 research benchmark (which evaluated all 144 batches of 500 flows). Thesis statistical performance tables cited in the defense originate from the approved research data package.
3. **Simulation Feedback vs. Autonomous Label Discovery**:
   - The IDS relies on post-decision simulation feedback from an authorized simulation session to close the controller loop. Authoritative dataset labels strictly take precedence for exact dataset rows. The platform does not possess autonomous label discovery for uninspected production traffic.

---

## 7. Deliverable Archive & Manifest Verification

A clean, self-contained standalone audit ZIP has been assembled containing the working runtime, frozen Random Forest model, both data profiles (`expanded` and `fixture20`), vendored frontend assets, test suites, documentation, and a fresh `FILE_MANIFEST.csv`.

- **Exclusions**: The original root import archive (`ids_expanded_simulation_data.zip`), older audit archives, virtual environments (`.venv`), git repositories (`.git`), bytecode caches (`__pycache__`), and inactive legacy files (`models/`, `datasets/demo/`, `archive/`) are excluded to avoid duplication and maintain a minimal footprint.
- **Manifest Integrity**: `FILE_MANIFEST.csv` lists every packaged file along with its exact byte count and SHA-256 hash.
- **External Archive Hash**: The outer SHA-256 checksum and file size of the package are reported outside the archive in the final handoff statement.

### Key Included Artifacts & Verified Hashes
- `runtime_package/model/frozen_rf.joblib`: `9608672c5d5e38a9272c560679cdd2291399de0e0917c8c9dff373bfd200f51d` (55,251,670 B)
- `runtime_package/model/feature_mask.json`: `ee9bae9fbd552d5c92a43b9475ebaee79356f0d8b74de2d04c1c10689c158a55` (3,033 B)
- `runtime_package/model/feature_names.json`: `4ee6b2d9127a423eb3f7da3cb1ab6d644b3a90a4a0e378fe780528b2b009148c` (1,541 B)
- `runtime_package/model/training_bounds.parquet`: `9554f1b0287c88b5ef7b8de19d193184f2d76eec2b4d96f1fa525818cffb275b` (3,552 B)
- `runtime_package/demo_data/X_demo.parquet`: `a39f6ed3de09ae11dae5c271df515abb440a839567e2f8fa5394764e1e6dbada` (47,911 B)
- `runtime_package/demo_data/metadata_demo.parquet`: `593479e0275fc715d7534ee8d9a67805a47cd0c9ec9ed7d2d53b02beaff0a5db` (4,394 B)
- `runtime_package/expanded_data/data/processed/X_eval.parquet`: `cd509adc96828fc9fbcf3d51546b394194a5f2b8bd71a9375395bf2ab28c3241` (13,103,425 B)
- `runtime_package/expanded_data/data/processed/metadata_eval.parquet`: `b213bb0f110b2beeccbd0b2c17618e2a3b7ec6437ac3a78f25a3e4cb94726fca` (1,234,320 B)
- `runtime_package/expanded_data/data/manifests/evaluation_roles.csv`: `cbf650879aa1369fa26b803777f30d4b0add9e7aff2c05d399ba7f42563f7b45` (1,586,839 B)
- `runtime_package/expanded_data/data/manifests/evaluation_batches.csv`: `4084017e5593732e455763416f7fc38254fc9dab2466eca289b66e68dc0ff79a` (4,498,409 B)

---
*Report certified defense-ready for thesis examination on October 8, 2026.*
