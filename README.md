# ML-IDS + Recall-Aware Defense Demonstration Platform

## Research Demonstration Runtime & Rehearsal Tool

This repository contains the standalone thesis demonstration platform for:
**Recall-Aware Adaptive Feature Perturbation for Machine Learning-Based Intrusion Detection Systems**.

The platform couples an inline Machine Learning Intrusion Detection System (ML-IDS) guarded by active adversarial defenses (**Adaptive Feature Perturbation [AFP]**, **Randomized Smoothing [RS]**, and **Feature Squeezing [FS]**) with an adaptive **Recall-Aware Feedback Controller**. A companion console simulates black-box adversarial attacks and benign background traffic in real time.

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

### Step 2: Start Protected Server & SOC Dashboard (Terminal 1)
```powershell
python server.py
```
Open your browser to: **`http://localhost:8000`**

### Step 3: Run Attacker & Simulation Console (Terminal 2)
```powershell
python attacker_sim.py
```

The console menu presents:
- **[1]** Send Legitimate Benign Traffic Flow (`200 OK`)
- **[2]** Launch DDoS Volumetric Flood Burst (`403 Forbidden`)
- **[3]** Launch Silent Probing (Sequential Unchanged Baseline Flows, 0 Queries)
- **[4]** Launch Surrogate Transferability Attack (Decision Tree Surrogate)
- **[5]** Launch Decision Boundary Attack (1D Bisection Search)
- **[6]** Continuous Real-time Traffic Stream
- **[7]** Local Control-Path Demonstration: Base vs. Recall-Aware (Identical Flows)
- **[8]** Switch Active Defense (`[AFP -> RS -> FS -> None]`)
- **[9]** Toggle Controller Mode (`[Recall-Aware <-> Base]`)
- **[0]** Exit

Direct CLI modes are also available:
```powershell
# Attack workflows against 72,000 measurement pool
python attacker_sim.py --mode silent
python attacker_sim.py --mode surrogate
python attacker_sim.py --mode boundary

# Control-path and cross-defense comparisons
python attacker_sim.py --mode compare --defense afp
python attacker_sim.py --mode compare-all

# Traffic generation with optional seed
python attacker_sim.py --dataset expanded --seed 42 --mode benign
python attacker_sim.py --mode ddos --count 10
```

---

## 2. System Architecture & Model Specification

| Component | Specification |
| :--- | :--- |
| **Model Checksum** | SHA-256 `9608672c5d5e38a9272c560679cdd2291399de0e0917c8c9dff373bfd200f51d` |
| **Model File** | `runtime_package/model/frozen_rf.joblib` |
| **Model Type** | 200-tree binary RandomForest (`classes_=[0, 1]`) |
| **Input Schema** | Exactly 78 numerical features (CSE-CIC-IDS2018 standard schema) |
| **Feature Mask** | Exactly 63 modifiable features and 15 protected network-layer features |
| **Active Defense (AFP)** | Adaptive Feature Poisoning (thesis title: Adaptive Feature Perturbation): Bounded noise scaled by standard deviation: $\varepsilon = 0.0003$, $\alpha = 0.5$ (no centroid projection) |
| **Active Defense (RS)** | 11 noise members, Gaussian smoothing |
| **Active Defense (FS)** | Feature bit squeezing |
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
2. **Autonomous Label Discovery**:
   A production IDS operating in real time does not have access to ground truth labels. In this demonstration, ground truth is post-decision simulation feedback provided strictly during server-validated simulation sessions to close the controller feedback loop. Unlabeled flows do not update confusion matrix metrics or controller state.
3. **Map Tiles**:
   Vendored frontend dependencies (`leaflet.js`, `leaflet.css`, `chart.umd.min.js`) ensure that all dashboards, metrics, graphs, tables, and forensic inspection work completely offline. Map basemap tiles (ArcGIS/OpenStreetMap) load dynamically when internet connectivity is present; in fully offline presentation environments, the map renders with a clean dark/light canvas background without breaking or raising exceptions.
