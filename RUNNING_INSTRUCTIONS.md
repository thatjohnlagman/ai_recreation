# Quick Running Instructions

Quick guide to start and run the **IDS + Recall-Aware AFP** cybersecurity platform, the real-time SOC web dashboard, and the terminal attacker simulation.

---

## 1. Prerequisites

Ensure dependencies are installed in your virtual environment:

```powershell
# Windows PowerShell
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt fastapi uvicorn websockets requests
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
  - Real-time WebSocket connection streaming traffic metrics, recall gauge, and detection feeds.
  - Header status indicators (`IDS Active`, `AFP Defense Active`, `System Online`).

---

## 3. Run the Attacker Simulation

Open a **second terminal** and launch the attacker script:

### Interactive Menu Mode
```powershell
.\.venv\Scripts\python.exe attacker_sim.py
```
*(Provides an interactive prompt to select DDoS, Probing, Continuous Stream, or Benchmark)*

### Direct Command-Line Modes

| Scenario | Command | Description |
| :--- | :--- | :--- |
| **Continuous Stream** | `.\.venv\Scripts\python.exe attacker_sim.py --mode stream` | Streams mixed benign & malicious packets at 1 packet/sec to watch live dashboard updates. |
| **Adversarial Probe** | `.\.venv\Scripts\python.exe attacker_sim.py --mode probe` | Runs a 1D bisection search trying to find decision boundaries to evade detection. |
| **Volumetric DDoS** | `.\.venv\Scripts\python.exe attacker_sim.py --mode ddos --count 15` | Sends rapid bursts of high-volume malicious traffic to test throughput and alerts. |
| **Defense Benchmark** | `.\.venv\Scripts\python.exe attacker_sim.py --mode compare` | **Key Thesis Demo**: Shows evasion success (`200 OK`) when AFP is OFF vs. caught & blocked (`403 Forbidden`) when AFP is ON. |

---

## 4. Useful API Endpoints

The server exposes REST endpoints for automated testing or integration:

- `GET  /api/dashboard/stats` — Retrieve current SOC metrics, AFP intensity, and recent alerts.
- `POST /api/server/data` — Protected server endpoint; evaluated inline by IDS + AFP before processing.
- `POST /api/dashboard/toggle-afp` — Toggle Adaptive Feature Perturbation defense (`{"enabled": true|false}`).
- `POST /api/dashboard/reset` — Reset dashboard counters and threat logs.
- `WS   /ws` — Real-time telemetry WebSocket feed.
