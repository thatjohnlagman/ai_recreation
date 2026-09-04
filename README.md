# Recall-Aware Adaptive Feature Perturbation (AFP-IDS)
## Research Evaluation Platform & Baseline Replication

This repository contains the standalone replication codebase and evaluation platform for our thesis research:
**Recall-Aware Adaptive Feature Perturbation for Machine Learning-Based Intrusion Detection Systems**.

---

## Quick Start (For Team Members / Collaborators)

Clone the repository and run the application in 3 steps:

```bash
# 1. Clone the repository
git clone https://github.com/thatjohnlagman/ai_recreation.git
cd ai_recreation

# 2. Set up virtual environment and install dependencies
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# 3. Launch the platform
streamlit run app.py
```

The application will open automatically in your browser at `http://localhost:8501`.

---

## Platform Contents

1. **Thesis Progress Platform (`app.py` / `thesis_progress_app.py`)**:
   - **Step 1: Data Preprocessing & Features**: CSE-CIC-IDS2018 cleaning, 77-feature extraction, statistical reference baselines ($\mu_{\text{ref}}, \sigma_{\text{ref}}$), and clamping bounds.
   - **Step 2: Live Baseline Classification**: Single-sample inference verifying that clean benign traffic is identified as Benign and raw attacks as Attack.
   - **Step 3: Measured Baseline Performance**: Control benchmark on 20,000 clean test flows (99.38% Accuracy, 96.64% Attack Recall, Confusion Matrix).
   - **Step 4: Attack & Defense Simulation (AFP ON / OFF)**: Interactive test demonstrating that adversarial probing evades the baseline IDS, but is blocked when AFP Defense is enabled.
   - **Step 5: Experiment Paper Steps**: Direct traceability mapping to Appendix 1 (Pre-experimentation, During experimentation RQ 1, and Post-experimentation RQ 2–4).

2. **Pre-Trained Model & Profiles Included**:
   - `models/rf_ids_cic.pkl.xz`: 200-tree Random Forest classifier checkpoint (compressed to 55 MB for direct Git tracking).
   - `models/X_ref_cic.json`: 77-feature benign reference distributions ($\mu_{\text{ref}}, \sigma_{\text{ref}}$).
   - `models/X_bounds_cic.json`: 77-feature physical clamping boundaries.
   - `datasets/demo/`: Balanced evaluation subsets for instant reproduction.

3. **Evaluation Notebooks**:
   - `final_evaluate_defenses.ipynb`: Full benchmark evaluation across Silent Probing, Decision Boundary, and Surrogate Transferability.
   - `final_train_eval_rf.ipynb`: Model training and standalone evaluation.
