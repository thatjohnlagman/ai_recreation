# Running Instructions — IDS + Recall-Aware Research Platform

Complete operational guide to start, configure, and evaluate the **ML-IDS + Recall-Aware Defense Platform**, the real-time SOC web dashboard, and the research attacker simulation console in both **Expanded (90k)** and **Fixture20 (20-flow)** modes.

---

## 1. Environment & Prerequisites

Ensure the virtual environment and pinned dependencies are active in your working terminal:

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
| **Combined Fingerprint** | SHA-256 over `[X_eval, metadata_eval, evaluation_roles]` | SHA-256 over `[X_demo, metadata_demo]` |
| **Role Overlap** | **0** overlap (disjoint, full coverage) | N/A |
| **Research Batches** | 144 formal evaluation batches of 500 flows | Single batch of 20 flows |
| **Live Controller Cadence** | Atomic updates every **5 attack targets** | Atomic updates every **5 attack targets** |
| **Prevalence** | CSE-CIC-IDS2018 natural evaluation distribution | Balanced 50% Benign / 50% Attack |
| **Source Data** | `runtime_package/expanded_data/data/processed/` | `runtime_package/demo_data/` (byte-preserved) |

### Critical Operational Invariants:
1. **Preprocessed Feature Vectors**: All input rows are already scaled by the training-fitted `StandardScaler`. The demonstration platform operates on standardized 78-feature numerical vectors; it **does not** sniff raw PCAP packets or extract online flow features from network interfaces.
2. **Environment Variable Scope**: Setting `--dataset fixture20` on an attacker command only configures that specific attacker process. It **does NOT** change the server environment or persistent terminal state. `IDS_DATA_PROFILE` must be set explicitly in each terminal for server and test commands.
3. **Session-Bound Query Accounting (Expanded Mode)**:
   - **Target Flows (`is_query=False`)**: Ordinary benign/attack traffic, bursts, streams, silent probes, and final adversarial candidates draw from the 72,000 measurement pool. Only target flows update live SOC dashboard metrics (`total_traffic`, `detected_attacks`, `tp`, `fn`, `fp`, `tn`, `recall`, `fpr`) and advance the 5-attack-target Recall-Aware controller cadence.
   - **Crafting Queries (`is_query=True`)**: Black-box surrogate fitting queries and decision-boundary bisection queries draw exclusively from the 18,000 crafting reference pool. These requests require an active server-confirmed simulation session, are tagged with their query stage, and increment an isolated `query_count` counter. They **never** modify the target confusion matrix or advance controller adaptation batches.
4. **Live Cadence vs. Research Batches**: In formal research evaluation (Phase 1–11), batches contain 500 flows. In this live demonstration platform, the controller updates every **5 attack targets** to provide immediate, observable SOC dashboard feedback.

---

## 3. Starting the Protected Server & SOC Dashboard

### Terminal 1: Expanded Simulation Mode (Default)
In your first terminal, set the environment variable and launch the server:

```powershell
# Windows PowerShell (Expanded Mode)
$env:IDS_DATA_PROFILE = "expanded"
.\.venv\Scripts\python.exe server.py
```

### Terminal 1 (Alternative): Fixture20 Regression Profile
To run regression checks or legacy tests against the byte-identical 20-flow demo fixture:

```powershell
# Windows PowerShell (Fixture20 Mode)
$env:IDS_DATA_PROFILE = "fixture20"
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
# Windows PowerShell (Terminal 2)
$env:IDS_DATA_PROFILE = "expanded"
.\.venv\Scripts\python.exe attacker_sim.py
```
*(Connects to the expanded backend, verifies data profile and combined fingerprint match, and presents interactive options)*

### CLI Arguments Reference

| Argument | Description | Default |
| :--- | :--- | :--- |
| `--dataset` | Active data profile (`expanded` or `fixture20`) | `expanded` |
| `--seed` | Random seed for sampling queues without replacement | `42` (deterministic new-process replay; specify different integer for varied draw sequence) |
| `--mode` | Execution mode (`menu`, `silent`, `surrogate`, `boundary`, `compare`, `compare-all`, `ddos`, `benign`, `stream`) | `menu` (Interactive menu) |
| `--defense` | Explicit target defense (`afp`, `rs`, `fs`, `none`). If omitted in compare mode, reads active defense from server telemetry. | Omitted (inherits server defense) |
| `--controller` | Target controller mode (`recall-aware` or `base`) | `recall-aware` |
| `--target` | Target backend URL | `http://localhost:8000` |
| `--count` | Number of flows for burst/stream actions | `5` (or mode default) |

---

## 5. Direct Command-Line Modes & Workflows

### A. Research Attack Workflows (Expanded Mode)

| Workflow | PowerShell Command | Behavior & Verification |
| :--- | :--- | :--- |
| **Silent Probing** | `.\.venv\Scripts\python.exe attacker_sim.py --mode silent` | Draws a fresh attack target from the 72,000 measurement pool without replacement. Submits unchanged flow once (0 queries during synthesis). |
| **Surrogate Transfer** | `.\.venv\Scripts\python.exe attacker_sim.py --mode surrogate` | Selects 20 bounded reference flows (10 benign, 10 attack) from crafting pool using session RNG. Queries oracle with `is_query=True` (zero target metrics). Draws measurement attack target, crafts perturbation, and submits final candidate once as measured target flow. |
| **Decision Boundary** | `.\.venv\Scripts\python.exe attacker_sim.py --mode boundary` | Selects up to 50 benign references from crafting pool using session RNG. Runs 1D bisection search (`is_query=True`, candidate queries retain target origin ID). If bisection fails, submits unchanged original flow labeled as baseline fallback. |

### B. Seeded Sampling & Exact Replay

The CLI defaults to `--seed 42` for deterministic process replay. To vary the sequence, provide an explicit seed:

```powershell
# Replay deterministic default sequence (seed 42)
.\.venv\Scripts\python.exe attacker_sim.py --dataset expanded --seed 42 --mode silent

# Draw varied sequence using seed 999
.\.venv\Scripts\python.exe attacker_sim.py --dataset expanded --seed 999 --mode silent
```

### C. Cross-Defense & Control-Path Comparisons

1. **Compare Base vs. Recall-Aware for Active Defense (Menu Option 7)**:
   When no `--defense` argument is supplied, the tool queries validated server telemetry (`payload["afp"]["defense_name"]`) and preserves that defense across both arms. Provide `--defense` to override:
   ```powershell
   # Uses active server defense (e.g. RS if RS was selected)
   .\.venv\Scripts\python.exe attacker_sim.py --mode compare

   # Explicit override to AFP
   .\.venv\Scripts\python.exe attacker_sim.py --mode compare --defense afp
   ```

2. **Replay Identical Sequence Across All 4 Defenses (Compare All)**:
   Replays a bounded 5-flow measurement target sequence across AFP, RS, FS, and None in Base mode, resetting metrics before each arm and reporting confusion matrices and defense-specific confidence semantics:
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

### Suite 1: Expanded Data Integration Suite (10 Checks)
Requires server running in `expanded` mode on port 8000:

```powershell
# Terminal 1: Launch expanded server
$env:IDS_DATA_PROFILE = "expanded"
.\.venv\Scripts\python.exe server.py

# Terminal 2: Run expanded verification suite
$env:IDS_DATA_PROFILE = "expanded"
.\.venv\Scripts\python.exe tests/test_expanded_data_integration.py
```

### Suite 2: Fixture20 Defense Readiness & Audit Suites
Requires server running in `fixture20` mode on port 8000:

```powershell
# Terminal 1: Launch fixture20 server
$env:IDS_DATA_PROFILE = "fixture20"
.\.venv\Scripts\python.exe server.py

# Terminal 2: Run 12 defense readiness checks
$env:IDS_DATA_PROFILE = "fixture20"
.\.venv\Scripts\python.exe tests/verify_defense_readiness.py

# Terminal 2: Run 7 master audit checks (Tests A through G)
$env:IDS_DATA_PROFILE = "fixture20"
.\.venv\Scripts\python.exe tests/run_audit_tests.py

# Terminal 2: Run full pytest suite (26 tests)
$env:IDS_DATA_PROFILE = "fixture20"
pytest -v
```

---

## 7. Useful REST Endpoints

- `GET  /api/dashboard/stats`: Returns telemetry counters, `data_profile`, `data_fingerprint`, `data_stats`, `afp` controller state, and recent alerts.
- `POST /api/server/data`: Main inspection endpoint; accepts `sample_id`, `data_profile`, `is_query`, `query_stage`.
- `POST /api/dashboard/set-defense`: Switches defense (`{"defense": "afp" | "rs" | "fs" | "none"}`).
- `POST /api/dashboard/set-mode`: Switches controller mode (`{"mode": "recall-aware" | "base"}`).
- `POST /api/dashboard/reset`: Resets metrics, controller state, and feeds.
- `POST /api/simulation/start`: Starts simulation session and returns server-confirmed `session_id`.
- `POST /api/simulation/stop`: Concludes simulation session.
- `WS   /ws`: Real-time WebSocket telemetry stream.
