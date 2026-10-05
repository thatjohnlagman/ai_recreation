# Quick Running Instructions — Research Runtime Platform

Quick guide to start and run the **IDS + Recall-Aware Research Platform**, the real-time SOC web dashboard, and the research attacker simulation console.

---

## 1. Prerequisites

Ensure dependencies are installed in your virtual environment:

```powershell
# Windows PowerShell
.\.venv\Scripts\Activate.ps1
pip install -r runtime_package/requirements.txt fastapi uvicorn websockets requests
```

---

## 2. Start the Protected Server & SOC Dashboard

Run the backend server in your first terminal:

```powershell
.\.venv\Scripts\python.exe server.py
```

- **SOC Web Dashboard**: Open [http://localhost:8000](http://localhost:8000) in your browser.
- **Features**:
  - Live full-width SOC dashboard with **Default Light Mode** and **Dark Mode** toggle (☀️/🌙).
  - Interactive Leaflet threat map with pulsating radar attack pins.
  - Research defense selector: **AFP** (Adaptive Feature Poisoning), **RS** (Randomized Smoothing), **FS** (Feature Squeezing), and **None** (Bypassed).
  - Controller Mode toggle: **Recall-Aware** (Dynamic Feedback Window) vs. **Base** (Static Calibrated Intensity).
  - Controller State indicator badge: **Green** (healthy), **Yellow** (warning decay), **Red** (critical fast decay).
  - Real-time WebSocket connection streaming traffic metrics, recall gauge, and detection feeds.

---

## 3. Run the Attacker Simulation Console

Open a **second terminal** and launch the attacker script:

### Interactive Menu Console
```powershell
.\.venv\Scripts\python.exe attacker_sim.py
```
*(Provides an interactive prompt to select Silent Probing, Surrogate Transfer, Decision Boundary, Volumetric Flood, Continuous Stream, or Benchmark)*

### Direct Command-Line Modes

| Scenario | Command | Description |
| :--- | :--- | :--- |
| **Decision Boundary** | `.\.venv\Scripts\python.exe attacker_sim.py --mode boundary` | **Research Attack**: Runs 1D bisection search between attack sample and benign pool to find decision boundary. |
| **Surrogate Transfer** | `.\.venv\Scripts\python.exe attacker_sim.py --mode surrogate` | **Research Attack**: Fits a local Decision Tree surrogate from oracle query labels and tests transfer evasion. |
| **Silent Probing** | `.\.venv\Scripts\python.exe attacker_sim.py --mode silent` | **Research Attack**: Generates offline feature-space perturbations without target queries and transmits to target server. |
| **Research Benchmark** | `.\.venv\Scripts\python.exe attacker_sim.py --mode compare` | **Key Thesis Demo**: Evaluates traffic bursts under Base Defense vs. Recall-Aware Defense to showcase dynamic intensity adaptation. |
| **Continuous Stream** | `.\.venv\Scripts\python.exe attacker_sim.py --mode stream` | Streams mixed benign & attack packets at 1 packet/sec to watch live dashboard updates. |
| **Volumetric DDoS** | `.\.venv\Scripts\python.exe attacker_sim.py --mode ddos --count 10` | Sends rapid bursts of high-volume malicious traffic to test server throughput and alerts. |
| **Legitimate Benign** | `.\.venv\Scripts\python.exe attacker_sim.py --mode benign` | Sends legitimate benign traffic flows to verify resource granting (`200 OK`). |

---

## 4. Useful API Endpoints

The server exposes REST endpoints for automated testing or integration:

- `GET  /api/dashboard/stats` — Retrieve current SOC metrics, defense status, controller state, and recent alerts.
- `POST /api/server/data` — Protected server endpoint; evaluated inline by IDS + Defense + Recall-Aware Controller.
- `POST /api/dashboard/set-defense` — Switch active defense (`{"defense": "afp" | "rs" | "fs" | "none"}`).
- `POST /api/dashboard/set-mode` — Switch mode (`{"mode": "recall-aware" | "base"}`).
- `POST /api/dashboard/reset` — Reset dashboard counters and threat logs.
- `WS   /ws` — Real-time telemetry WebSocket feed.
