# Running Instructions — IDS + Recall-Aware Research Platform

Complete operational guide to start, configure, and evaluate the **ML-IDS + Recall-Aware Defense Platform**, the real-time SOC web dashboard, the research attacker simulation console, and the authorized operator benchmark utility in both **Expanded (90k)** and **Fixture20 (20-flow)** modes.

---

## 1. Environment & Prerequisites

Ensure the virtual environment and pinned dependencies are active in your working terminal:

```powershell
# Windows PowerShell
cd c:\Users\reddr\ai_recreation
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### Operator Authorization Setup
Server management routes (`/api/dashboard/set-defense`, `/api/dashboard/set-mode`, `/api/dashboard/toggle-afp`, `/api/dashboard/reset`) require operator authorization.

To set up the operator token for local development and benchmarking:
```powershell
# Copy template to active local operator token file
Copy-Item operator_token.txt.template operator_token.txt
```
The default demo token is `ids-operator-secret-2026`. Alternatively, set the environment variable `$env:IDS_OPERATOR_TOKEN = "ids-operator-secret-2026"`.

---

## 2. Dataset Architecture & Role Separation

### Dataset Profiles via Shared Loader (`runtime_package/data_loader.py`)

| Property | `expanded` (Default Research Profile) | `fixture20` (Regression / Fallback Profile) |
| :--- | :--- | :--- |
| **Total Rows** | **90,000** rows (78 float32 features) | **20** rows (78 float32 features) |
| **Measurement Pool** | **72,000** target rows (59,743 Benign, 12,257 Attack) | **20** rows (10 Benign, 10 Attack) |
| **Crafting Pool** | **18,000** reference rows (14,936 Benign, 3,064 Attack) | Reused 20 rows for legacy compatibility |
| **Combined Fingerprint** | SHA-256 over `[X_eval, metadata_eval, evaluation_roles]` | SHA-256 over `[X_demo, metadata_demo]` |
| **Role Overlap** | **0** overlap (disjoint, full coverage) | N/A |
| **Live Controller Cadence** | Atomic updates every **5 attack targets** | Atomic updates every **5 attack targets** |
| **Source Data** | `runtime_package/expanded_data/data/processed/` | `runtime_package/demo_data/` (byte-preserved) |

### Role Separation: Attacker vs. Defender vs. Operator
1. **Attacker Console (`attacker_sim.py`)**: Submits recorded traffic-flow features and executes the three black-box study procedures. It has **zero management authority**: it cannot switch defense, change controller mode, or reset defender counters.
2. **Defender Server & Dashboard (`server.py` + `frontend/`)**: Displays real ML predictions, confidence, defense settings, HTTP actions, and reference-label metrics. All procedure names, claimed scenario badges, and dataset family labels are completely removed from defender telemetry and UI.
3. **Operator Utilities (`operator_benchmarks.py`)**: Dedicated script for operators to perform Base vs. Recall-Aware comparisons and cross-defense benchmarks, authenticated via `X-Operator-Token`.

---

## 3. Starting the Protected Server & SOC Dashboard

### Terminal 1: Launch Protected Server (Default: Expanded Mode)
```powershell
# Windows PowerShell (Expanded Mode)
$env:IDS_DATA_PROFILE = "expanded"
.\.venv\Scripts\python.exe server.py
```

*(For fixture20 fallback/regression mode, set `$env:IDS_DATA_PROFILE = "fixture20"` before launching).*

- **SOC Web Dashboard**: Open [http://localhost:8000](http://localhost:8000) in your browser.
- **Top Navigation Bar**:
  - `IDS Active` status badge.
  - Active defense pill (e.g. `AFP Defense AFP (RA)`).
  - Operator Auth Status button (`Operator Auth Configured` / `Operator Auth Required`). Clicking opens a modal to configure or view the operator token in browser session storage.
  - Data profile status: `Data Expanded (72k) | 0q`.
- **Selected Flow Forensics (4 Clean Inspector Cards)**:
  - Source Origin: IP address and network classification (Private vs. Public).
  - Target Endpoint: Protected destination IP and service name.
  - IDS Classification & Confidence: Pure ML-derived binary verdict (`Attack` or `Benign`) with exact inference probability.
  - Defense & Action: Active defense mechanism, perturbation epsilon, and HTTP action (`200 FORWARDED` or `403 BLOCKED`).

---

## 4. Running the Attacker Simulation Console

Open **Terminal 2** to launch the attacker simulation.

### Interactive Menu
```powershell
# Windows PowerShell (Terminal 2)
$env:IDS_DATA_PROFILE = "expanded"
.\.venv\Scripts\python.exe attacker_sim.py
```

The menu strictly exposes the 3 canonical study procedures plus benign/stream traffic utilities:
```text
[1] Send recorded benign flow
[2] Silent Probing: submit unchanged malicious flow, zero preliminary queries
[3] Surrogate Transferability: craft using a local surrogate and crafting references
[4] Decision-Boundary Attack: search using server responses and crafting references
[5] Continuous recorded-flow stream
[0] Exit
```

### Direct CLI Commands

```powershell
# 1. Send recorded benign flow
.\.venv\Scripts\python.exe attacker_sim.py --mode benign --count 5

# 2. Silent Probing (unchanged attack flow, 0 crafting queries)
.\.venv\Scripts\python.exe attacker_sim.py --mode silent

# 3. Surrogate Transferability (crafts perturbation using local surrogate + 20 crafting references)
.\.venv\Scripts\python.exe attacker_sim.py --mode surrogate

# 4. Decision-Boundary Attack (1D bisection search using server query responses)
.\.venv\Scripts\python.exe attacker_sim.py --mode boundary

# 5. Continuous traffic stream (mixed benign / attack at 1 flow/second)
.\.venv\Scripts\python.exe attacker_sim.py --mode stream

# Deterministic sequence replay with explicit seed
.\.venv\Scripts\python.exe attacker_sim.py --mode silent --seed 42
```

---

## 5. Running Operator Benchmarks

Automated cross-defense comparisons and control-path evaluation have been relocated to `operator_benchmarks.py`.

```powershell
# Compare Base vs. Recall-Aware on active defense (or specify --defense)
.\.venv\Scripts\python.exe operator_benchmarks.py --mode compare

# Explicitly test Base vs. Recall-Aware on AFP
.\.venv\Scripts\python.exe operator_benchmarks.py --mode compare --defense afp

# Run cross-defense comparison across AFP, RS, FS, and None
.\.venv\Scripts\python.exe operator_benchmarks.py --mode compare-all

# Reset defender evaluation metrics and controller state
.\.venv\Scripts\python.exe operator_benchmarks.py --mode reset

# Switch defense and controller mode via operator CLI
.\.venv\Scripts\python.exe operator_benchmarks.py --mode set-defense --defense rs
.\.venv\Scripts\python.exe operator_benchmarks.py --mode set-mode --controller base
```

The operator script loads `operator_token.txt` automatically, or accepts `--token <TOKEN>` / `$env:IDS_OPERATOR_TOKEN`.

---

## 6. Verification & Test Suites

### Suite 1: Expanded Data Integration Suite (11 Checks)
Verifies expanded dataset loading, non-replacement sampling, role separation, 401 unauthorized rejection, and 400 unsupported procedure rejection:
```powershell
# Requires server running with IDS_DATA_PROFILE=expanded
$env:IDS_DATA_PROFILE = "expanded"
.\.venv\Scripts\python.exe tests/test_expanded_data_integration.py
```

### Suite 2: Defense Readiness Suite (12 Checks)
Verifies frozen RF hash, 18 core algorithm files, authorized management calls, and canonical procedures:
```powershell
.\.venv\Scripts\python.exe tests/verify_defense_readiness.py
```

### Suite 3: Master Runtime Audit Suite (Tests A through G)
Verifies no background leakages, silent probing, genuine evasion, reference family handling, and defense switching:
```powershell
.\.venv\Scripts\python.exe tests/run_audit_tests.py
```

### Suite 4: Full Pytest Suite (26 tests)
```powershell
pytest -v
```

---

## 7. Useful REST Endpoints

### Public Endpoints (No Operator Token Required)
- `GET  /api/dashboard/stats`: Returns telemetry counters, `data_profile`, `data_fingerprint`, `data_stats`, and `afp` controller state.
- `POST /api/server/data`: Main inspection endpoint; accepts standardized 78-feature vector or `sample_id`.
- `POST /api/simulation/start`: Starts simulation session and returns server-confirmed `session_id`.
- `POST /api/simulation/stop`: Concludes simulation session.
- `WS   /ws`: Real-time WebSocket telemetry stream.

### Operator-Protected Management Endpoints (`X-Operator-Token` Required)
- `POST /api/dashboard/set-defense`: Switches defense (`{"defense": "afp" | "rs" | "fs" | "none"}`). Returns 401 if unauthenticated.
- `POST /api/dashboard/set-mode`: Switches controller mode (`{"mode": "recall-aware" | "base"}`). Returns 401 if unauthenticated.
- `POST /api/dashboard/toggle-afp`: Toggles AFP active/bypass. Returns 401 if unauthenticated.
- `POST /api/dashboard/reset`: Resets metrics, controller state, and feeds. Returns 401 if unauthenticated.
