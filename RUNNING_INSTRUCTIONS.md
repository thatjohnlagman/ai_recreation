# Running Instructions — IDS + Recall-Aware Research Platform

Complete operational guide to start, configure, and evaluate the **ML-IDS + Recall-Aware Defense Platform**, the real-time SOC web dashboard, and the research attacker simulation console in both **Expanded (90k)** and **Fixture20 (20-flow)** modes.

---

## 1. Environment & Prerequisites

Ensure the frozen virtual environment and pinned dependencies (`scikit-learn==1.6.1`) are active:

```powershell
# Windows PowerShell
cd c:\Users\reddr\ai_recreation
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

---

## 2. Dataset Architecture & Metric Scopes

The platform supports two distinct data profiles via a unified shared loader (`runtime_package/data_loader.py`):

| Property | `expanded` (Default Research Profile) | `fixture20` (Regression / Fallback Profile) |
| :--- | :--- | :--- |
| **Total Rows** | **90,000** rows (78 float32 features) | **20** rows (78 float32 features) |
| **Measurement Pool** | **72,000** target rows (59,743 Benign, 12,257 Attack) | **20** rows (10 Benign, 10 Attack) |
| **Crafting Pool** | **18,000** reference rows (14,936 Benign, 3,064 Attack) | Reused 20 rows for legacy compatibility |
| **Role Overlap** | **0** overlap (disjoint, full coverage) | N/A |
| **Batches** | 144 batches of 500 flows | Single batch of 20 flows |
| **Prevalence** | CSE-CIC-IDS2018 natural distribution | Balanced 50% Benign / 50% Attack |
| **Source Data** | `runtime_package/expanded_data/data/processed/` | `runtime_package/demo_data/` (byte-preserved) |

### Important Operational Invariants:
1. **Preprocessed Feature Vectors**: Features are already transformed by the frozen training-fitted `StandardScaler`. The standalone IDS evaluates 78-feature numerical vectors; it **does not** sniff raw PCAP packets or extract online flow features from network interfaces.
2. **Strict Metric Separation (Expanded Mode)**:
   - **Target Flows (`is_query=False`)**: Ordinary benign/attack traffic, bursts, streams, silent probes, and the final evaluated adversarial candidates draw from the 72,000 measurement pool. Only target flows update live SOC dashboard metrics (`total_traffic`, `detected_attacks`, `tp`, `fn`, `fp`, `tn`, `recall`, `fpr`) and advance the 5-decision Recall-Aware controller cadence.
   - **Crafting Queries (`is_query=True`)**: Black-box surrogate fitting queries and decision-boundary bisection queries draw exclusively from the 18,000 crafting reference pool. These requests are tagged with their query stage and increment an isolated `query_count` telemetry counter. Crafting queries **never** modify the target confusion matrix or advance controller adaptation batches.

---

## 3. Starting the Protected Server & SOC Dashboard

### Normal Launch: Expanded Simulation Mode (Default)
In your first terminal, launch the server. It automatically defaults to the `expanded` profile:

```powershell
# PowerShell (Default expanded profile)
.\.venv\Scripts\python.exe server.py
```
*(Or explicitly specify: `$env:IDS_DATA_PROFILE="expanded"; .\.venv\Scripts\python.exe server.py`)*

### Fallback Launch: Fixture20 Regression Profile
To run existing regression checks or legacy tests against the byte-identical 20-flow demo fixture:

```powershell
# PowerShell (Explicit fixture20 profile)
$env:IDS_DATA_PROFILE="fixture20"
.\.venv\Scripts\python.exe server.py
```

- **SOC Web Dashboard**: Open [http://localhost:8000](http://localhost:8000) in your browser.
- **Header Profile Badge**:
  - In expanded mode: displays `● Expanded (72k targets / 18k crafting)`.
  - In fixture20 mode: displays `● Fixture20 (20 targets)`.
- **Dashboard Controls**:
  - **Defense Selector**: Adaptive Feature Poisoning (**AFP**), Randomized Smoothing (**RS**), Feature Squeezing (**FS**), or Bypassed (**None**).
  - **Controller Mode**: **Recall-Aware** (Dynamic 5-decision feedback window) vs. **Base** (Static calibrated intensity).
  - **Controller State**: **Green** (healthy, $\varepsilon=0.00030$), **Yellow** (warning decay), **Red** (critical fast decay, $\varepsilon=0.00012$).
  - **Live Telemetry**: Real-time WebSocket connection streaming traffic metrics, recall gauge, confusion matrix, query counter, and threat feed.

---

## 4. Running the Attacker Simulation Console

Open a **second terminal** to launch the attacker simulation.

### Normal Launch (Interactive Menu)
```powershell
.\.venv\Scripts\python.exe attacker_sim.py
```
*(By default, connects to the expanded backend, verifies data profile match, and provides interactive attack selection)*

### CLI Arguments Reference

| Argument | Description | Default |
| :--- | :--- | :--- |
| `--dataset` | Active data profile (`expanded` or `fixture20`) | `expanded` |
| `--seed` | Random seed for sampling queues without replacement | `None` (pseudo-random per session) |
| `--mode` | Execution mode (see table below) | Interactive menu |
| `--defense` | Target defense for single/benchmark evaluation (`afp`, `rs`, `fs`, `none`) | `afp` |
| `--controller` | Target controller mode (`recall-aware` or `base`) | `recall-aware` |
| `--target` | Target backend URL | `http://localhost:8000` |
| `--count` | Number of flows for burst/stream actions | `5` (or mode default) |

---

## 5. Direct Command-Line Modes & Workflows

### A. Research Attack Workflows (Expanded Mode)

| Workflow | PowerShell Command | Behavior & Verification |
| :--- | :--- | :--- |
| **Silent Probing** | `.\.venv\Scripts\python.exe attacker_sim.py --mode silent` | Draws a fresh attack target from the 72,000 measurement pool without replacement. Submits unchanged flow once (0 queries during synthesis). |
| **Surrogate Transfer** | `.\.venv\Scripts\python.exe attacker_sim.py --mode surrogate` | Draws 20 crafting references (10 benign, 10 attack) to fit local Decision Tree surrogate. Queries oracle (`is_query=True`, zero target pollution). Draws measurement attack target, crafts perturbation, and submits final candidate once as measured target flow. |
| **Decision Boundary** | `.\.venv\Scripts\python.exe attacker_sim.py --mode boundary` | Draws 50 benign references from crafting pool. Runs 1D bisection search (`is_query=True`, up to 50 queries) between attack target and benign reference. Submits final evasion candidate once as measured target flow. |

### B. Seeded Sampling & Exact Replay

To execute reproducible runs where queues draw the exact same sequence of measurement targets:

```powershell
# Replay identical session sequence using fixed seed 42
.\.venv\Scripts\python.exe attacker_sim.py --dataset expanded --seed 42 --mode silent
```

### C. Cross-Defense & Control-Path Comparisons

1. **Compare Base vs. Recall-Aware for Selected Defense**:
   Evaluates an identical 5-flow sequence under static Base mode and dynamic Recall-Aware mode for the selected defense (AFP, RS, FS, or None):
   ```powershell
   .\.venv\Scripts\python.exe attacker_sim.py --mode compare --defense afp
   ```

2. **Replay Identical Sequence Across All 4 Defenses (Compare All)**:
   Replays a bounded 5-flow measurement target sequence across AFP, RS, FS, and None in Base mode, resetting metrics before each arm and reporting confusion matrices and confidence semantics:
   ```powershell
   .\.venv\Scripts\python.exe attacker_sim.py --mode compare-all
   ```

### D. Volumetric, Stream & Benign Flows

```powershell
# Send 10 benign measurement flows (verifies resource granted, 200 OK)
.\.venv\Scripts\python.exe attacker_sim.py --mode benign --count 10

# Send 10 real DDoS measurement flows (filters actual DDoS families)
.\.venv\Scripts\python.exe attacker_sim.py --mode ddos --count 10

# Continuous mixed traffic stream at 1 flow/second
.\.venv\Scripts\python.exe attacker_sim.py --mode stream
```

---

## 6. Verification & Test Suites

The test suite validates both profiles and the frozen invariants:

```powershell
# 1. Expanded Data Integration Suite (6 checks: contracts, sampling, rejection, telemetry, comparison)
.\.venv\Scripts\python.exe tests/test_expanded_data_integration.py

# 2. Defense Readiness Verification Suite (12 checks: model hash, adaptation proof, cold start, etc.)
$env:IDS_DATA_PROFILE="fixture20"
.\.venv\Scripts\python.exe tests/verify_defense_readiness.py

# 3. Comprehensive Audit Suite (Tests A through G)
.\.venv\Scripts\python.exe tests/run_audit_tests.py

# 4. Pytest Suite (Unit, integration, and contract tests)
pytest -v
```

---

## 7. Useful REST Endpoints

- `GET  /api/dashboard/stats`: Returns telemetry counters, `data_profile`, `data_fingerprint`, `pool_stats`, `afp` controller state, and recent alerts.
- `POST /api/server/data`: Main inspection endpoint; accepts `sample_id`, `data_profile`, `is_query`, `query_stage`.
- `POST /api/dashboard/set-defense`: Switches defense (`{"defense": "afp" | "rs" | "fs" | "none"}`).
- `POST /api/dashboard/set-mode`: Switches controller mode (`{"mode": "recall-aware" | "base"}`).
- `POST /api/dashboard/reset`: Resets metrics, controller state, and feeds.
- `WS   /ws`: Real-time WebSocket telemetry stream.
