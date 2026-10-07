# ML-IDS + Recall-Aware Defense Demonstration Platform

## Research Demonstration Runtime & Rehearsal Tool

This repository contains the standalone thesis demonstration platform for:
**Recall-Aware Adaptive Feature Perturbation for Machine Learning-Based Intrusion Detection Systems**.

The platform couples an inline Machine Learning Intrusion Detection System (ML-IDS) guarded by active adversarial defenses (**Adaptive Feature Poisoning [AFP]**, **Randomized Smoothing [RS]**, and **Feature Squeezing [FS]**) with an adaptive **Recall-Aware Feedback Controller**. A companion console simulates black-box adversarial attacks and benign background traffic in real time.

---

## 1. Quick Start

### Prerequisites
- Python 3.12 or 3.13 (64-bit)
- Dependencies pinned in `requirements.txt` (including `scikit-learn==1.6.1`)

### Step 1: Install Dependencies
```powershell
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
```

### Step 2: Start Protected Server (Terminal 1)
```powershell
python server.py
```
By default, the presentation server binds loopback-only to `127.0.0.1:8000`.

### Step 3: Launch Authorized Dashboard (Terminal 2 or Auto-Launch)
```powershell
python launch_dashboard.py
```
This local launcher requests a one-time launch ticket, opens your default browser, and establishes a server-validated `HttpOnly` operator session cookie (`ids_operator_session`). Dashboard users can operate defense selection, Recall-Aware switching, and baseline resets without copying, typing, or managing tokens in the UI.

### Step 4: Run Attacker & Simulation Console (Terminal 3)
```powershell
python attacker_sim.py
```

The console menu exposes the canonical study procedures:
- **[1]** Send Legitimate Benign Traffic Flow (`Request allowed (HTTP 200)`)
- **[2]** Launch Silent Probing (Sequential Unchanged Baseline Flows, 0 Preliminary Crafting Queries)
- **[3]** Launch Surrogate Transferability Attack (Decision Tree Surrogate)
- **[4]** Launch Decision Boundary Attack (1D Bisection Search)
- **[5]** Continuous Real-time Traffic Stream
- **[0]** Exit

Direct CLI modes are available for headless execution:
```powershell
# Canonical attack workflows against 72,000 measurement pool
python attacker_sim.py --mode silent
python attacker_sim.py --mode surrogate
python attacker_sim.py --mode boundary

# Traffic generation with optional seed
python attacker_sim.py --dataset expanded --seed 42 --mode benign
python attacker_sim.py --mode stream
```

### Step 5: Operator Benchmark & Control CLI (Separate Operator Workflow)
Operator configurations and comparative evaluations are isolated in `operator_benchmarks.py`:
```powershell
# Set defense or controller mode (authorized via local .operator_token)
python operator_benchmarks.py --defense afp
python operator_benchmarks.py --mode recall-aware
python operator_benchmarks.py --reset

# 5-Flow Control-Path Demonstrations (Base vs. Recall-Aware on identical target sequence)
python operator_benchmarks.py --compare --defense afp
python operator_benchmarks.py --compare-all
```
*Note*: The 5-flow comparative benchmark serves as a control-path demonstration. Because Recall-Aware feedback after decision 5 dynamically affects subsequent decisions, a 5-flow sequence illustrates controller mechanics rather than population statistical superiority.

---

## 2. System Architecture & Model Specification

| Component | Specification |
| :--- | :--- |
| **Model Checksum** | SHA-256 `9608672c5d5e38a9272c560679cdd2291399de0e0917c8c9dff373bfd200f51d` |
| **Model File** | `runtime_package/model/frozen_rf.joblib` |
| **Model Type** | 200-tree binary RandomForest (`classes_=[0, 1]`) |
| **Input Schema** | Exactly 78 numerical features (CSE-CIC-IDS2018 standard schema) |
| **Feature Mask** | Exactly 63 modifiable features and 15 protected network-layer features |
| **Active Defense (AFP)** | Adaptive Feature Poisoning: Study-defined bounded noise scaled by standard deviation: $\varepsilon = 0.0003$, $\alpha = 0.5$ (not centroid projection) |
| **Active Defense (RS)** | Gaussian noisy copies with majority voting (11 noise members) |
| **Active Defense (FS)** | Decimal precision reduction: $d = \max(0, 6 - \text{int}(\text{squeezing\_intensity}))$ |
| **Recall-Aware Controller** | Rolling evaluation window ($W=5$), $R_{\text{crit}}=0.85$, $R_{\text{min}}=0.95$, fast decay $\times 0.40$, slow decay $\times 0.90$, growth $\times 1.05$ |
| **Expanded Data Profile** | **90,000 rows** in `runtime_package/expanded_data/` (72,000 measurement targets, 18,000 crafting references; disjoint, full coverage) |
| **Fixture20 Profile** | Preserved 20 rows in `runtime_package/demo_data/` (`X_demo.parquet`, `metadata_demo.parquet`) |

---

## 3. Automated Verification Tests

The platform includes exhaustive verification test suites:

```powershell
# 1. Expanded data integration suite (contract, roles, sampling, metrics, comparisons)
python tests/test_expanded_data_integration.py

# 2. Authoritative 12-point defense readiness verification suite (fixture20 regression)
$env:IDS_DATA_PROFILE="fixture20"
python -u tests/verify_defense_readiness.py

# 3. Audit test runner (Tests A-G)
python -u tests/run_audit_tests.py

# 4. Pytest suite
pytest -v
```

---

## 4. Note on Inactive Historical / Legacy Artifacts

- **`models/rf_ids_cic.pkl.xz`**, **`models/X_ref_cic.json`**, **`models/X_bounds_cic.json`**:
  These files are legacy 77-feature checkpoints from early Phase 1–3 exploratory work. They are **inactive** and superseded by the authoritative 78-feature frozen model in `runtime_package/model/frozen_rf.joblib`.
- **`datasets/demo/`**:
  Contains legacy CSV evaluation slices from early exploration. Active runtime evaluation uses `runtime_package/demo_data/*.parquet`.
- **`archive/`**:
  Contains historical thesis progress presentation scripts, exploratory notebooks, and prototype Streamlit dashboards from early development. Inactive and excluded from the runtime standalone defense package.
- These legacy files are preserved in the git tree for archival history but are excluded from the standalone audit package (`ids_standalone_tool_audit.zip`).

---

## 5. Scope & Academic Presentation Boundaries

1. **Demonstration Tool vs. Research Protocol**:
   This demonstration tool illustrates live HTTP feature-flow classification, inline perturbation, and real-time controller adaptation. It does **not** substitute for the Phase 10/11 research evaluation (which evaluated 144 measurement batches of 500 records each against frozen test caches). Official thesis benchmark tables derive from the frozen research package.
2. **Recorded Features and Autonomous Label Discovery**:
   The demonstration operates on recorded/preprocessed 78-feature flow vectors (CSE-CIC-IDS2018). It does **not** perform live packet capture (pcap) or live network feature extraction. Reference labels are required for recall/FPR evaluation and Recall-Aware controller feedback. In this demonstration, ground truth is post-decision simulation feedback provided strictly during server-validated simulation sessions to close the controller feedback loop. Unlabeled flows do not update confusion matrix metrics or controller state.
3. **Map Telemetry & Local GeoIP Provenance**:
   - The threat map uses bundled Natural Earth 110m low-resolution GeoJSON outline vectors (`frontend/vendor/ne_110m_admin_0_countries.geojson`), rendering an offline world basemap without external tile dependencies. OpenStreetMap tiles load dynamically as an optional enhancement when internet access is available.
   - Geolocation enrichment uses the bundled local DB-IP City Lite MMDB database (`runtime_package/geoip/dbip-city-lite.mmdb`) with an LRU lookup cache, requiring zero external runtime API calls.
   - **Provenance Notice**: IP addresses submitted during traffic simulation are client-submitted demonstration values, not captured peer addresses or verified attacker identities. Geolocation pins visualize the approximate location of these demo IPs and must not be interpreted as physical attribution of actual threat actors. RFC 1918 private ranges and RFC 5737 documentation test networks receive truthful non-geographic status and produce no map markers.
