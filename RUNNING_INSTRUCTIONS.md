# Running Instructions (Quick & Sequential Guide)

Follow these steps sequentially to run and test the ML-IDS Defense Platform.

---

## Step 1: Initial Setup (One-time)

Open PowerShell in the project directory:

```powershell
# 1. Create the virtual environment using Python 3.12
py -3.12 -m venv .venv

# 2. Activate the virtual environment
.\.venv\Scripts\Activate.ps1

# 3. Install required packages
pip install -r requirements.txt
```

---

## Step 2: Start the IDS Server (Terminal 1)

In your first terminal with `.venv` active, start the server:

```powershell
python server.py
```
* **What it does:**
  * Loads the 78-feature Random Forest model (`frozen_rf.joblib`).
  * Initializes the active defense (AFP) and Recall-Aware feedback controller.
  * Binds the server to `http://127.0.0.1:8000`.

*(Optional: Run `python server.py --launch` to start the server and automatically open the dashboard in your browser).*

---

## Step 3: Open the Dashboard (Browser)

Open your browser and navigate to:
```
http://127.0.0.1:8000/
```
* **Pages available:**
  * **Dashboard (`/`)**: Real-time SOC dashboard, live traffic telemetry, threat world map, and defense controls.
  * **Traffic Explorer (`/explorer`)**: Inspect and filter individual flows and their 78 network features.
  * **Sessions & Reports (`/sessions`)**: View historical simulation runs, confusion matrix, precision/recall, and export CSV reports.

---

## Step 4: Run Traffic & Attacks (Terminal 2)

Open a **second terminal**, activate `.venv`, and run the attacker simulation:

```powershell
.\.venv\Scripts\Activate.ps1
python attacker_sim.py
```

Choose from the interactive menu:
* **`[1]` Send Legitimate Benign Traffic**: Tests normal network traffic (shows `HTTP 200 Allowed`).
* **`[2]` Silent Probing**: Sends raw malicious flows without modifications to test baseline detection.
* **`[3]` Surrogate Transferability Attack**: Crafts evasion samples using a local Decision Tree surrogate.
* **`[4]` Decision-Boundary Attack**: Performs an interactive 1D bisection search to probe the model's decision boundary.
* **`[5]` Continuous Traffic Stream**: Streams mixed benign and attack flows in real time to the live dashboard.
* **`[0]` Exit**

### Direct CLI Commands (Headless / Non-Interactive):
```powershell
# Send 5 benign flows
python attacker_sim.py --mode benign --count 5

# Launch a continuous live traffic stream
python attacker_sim.py --mode stream

# Run specific attacks
python attacker_sim.py --mode silent
python attacker_sim.py --mode surrogate
python attacker_sim.py --mode boundary
```

---

## Step 5: (Optional) Run Operator Benchmarks (Terminal 2)

By default, benchmarks evaluate **200 batches (1,000 flows)** with statistical summaries:

```powershell
# Compare Base vs. Recall-Aware on AFP (default: 200 batches / 1000 flows)
python operator_benchmarks.py --mode compare

# Run comparison specifically on another defense (rs, fs, none)
python operator_benchmarks.py --mode compare --defense rs

# Compare all defense types (AFP, RS, FS, None) across both Base and Recall-Aware modes
python operator_benchmarks.py --mode compare-all

# Fast cross-defense comparison (e.g. AFP and Feature Squeezing only)
python operator_benchmarks.py --mode compare-all --defenses afp,fs

# Custom batch size (e.g. 50 batches = 250 flows)
python operator_benchmarks.py --mode compare --batches 50

# Reset metrics and controller state
python operator_benchmarks.py --mode reset
```

---

## Step 6: (Optional) Run Automated Tests

```powershell
# Run the pytest test suite
pytest -v

# Run the 12-point defense readiness verification
python tests/verify_defense_readiness.py

# Run master audit tests
python tests/run_audit_tests.py
```
