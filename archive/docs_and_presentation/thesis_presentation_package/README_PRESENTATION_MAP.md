# Thesis Progress & Demonstration Package

This package contains **only the essential code, model checkpoints, data profiles, and notebooks** required to present and verify the thesis progress report for:

> **Recall-Aware Adaptive Feature Perturbation for Machine Learning-Based Intrusion Detection Systems**

---

## Folder Map & Step-by-Step Guide

```
thesis_presentation_package/
├── README_PRESENTATION_MAP.md           <-- You are here (Summary & Map)
├── 01_streamlit_dashboard/
│   └── thesis_progress_app.py           <-- Interactive Streamlit UI Platform
├── 02_data_preprocessing/
│   ├── prepare_demo_20k.ipynb           <-- Data sampling & reference statistics notebook
│   └── final_train_eval_rf.ipynb        <-- Data ingestion, splitting, & RF baseline training notebook
├── 03_baseline_model_and_data/
│   ├── models/
│   │   ├── rf_ids_cic.pkl.xz            <-- Trained Random Forest model checkpoint (200 trees)
│   │   ├── X_ref_cic.json               <-- 77-feature reference mean (μ) & std (σ) statistics
│   │   └── X_bounds_cic.json            <-- Physical feature min/max limits
│   └── datasets/
│       ├── X_test_demo.csv              <-- Preprocessed benchmark test flows matrix
│       └── y_test_demo.csv              <-- Benchmark test binary labels (Benign vs Attack)
├── 04_attack_and_defense_simulations/
│   ├── final_evaluate_defenses.ipynb    <-- Base AFP defense math & probing attack simulations notebook
│   └── simulate_unsw_pipeline.py        <-- Attack-defense pipeline script
├── 05_statistical_and_rq_analysis/
│   ├── run_paired_ttests.py             <-- Paired t-test statistical pipeline (α = 0.05) for RQ 3
│   ├── aggregate_thesis_metrics.py      <-- Confusion matrix & performance aggregator
│   └── afp_ablation_summary.csv         <-- Ablation & sensitivity data log for RQ 4
└── 06_presentation_guides/
    ├── PROGRESS_REPORT_RUNBOOK.md       <-- 3-minute presenter narrative guide
    └── presenter_script.md              <-- Spoken talking points per app step
```

---

## How Each UI Step Maps to Code Files

### 1. Data Preprocessing & Features (`Step 1` in UI)
* **Code Notebooks**: 
  * `02_data_preprocessing/prepare_demo_20k.ipynb`
  * `02_data_preprocessing/final_train_eval_rf.ipynb`
* **Data Profiles**:
  * `03_baseline_model_and_data/models/X_ref_cic.json`
  * `03_baseline_model_and_data/models/X_bounds_cic.json`

### 2. Live Baseline Classification (`Step 2` in UI)
* **Code App**: `01_streamlit_dashboard/thesis_progress_app.py`
* **Model Engine & Weights**:
  * `03_baseline_model_and_data/models/rf_ids_cic.pkl.xz`
* **Demo Flows**:
  * `03_baseline_model_and_data/datasets/X_test_demo.csv`

### 3. Measured Baseline Performance (`Step 3` in UI)
* **Code Notebook**: `02_data_preprocessing/final_train_eval_rf.ipynb` (Phase 4 Evaluation)
* **Metrics Aggregator**: `05_statistical_and_rq_analysis/aggregate_thesis_metrics.py`

### 4. Attack & Defense Simulation (`Step 4` in UI - RQ 1)
* **Code Notebook**: `04_attack_and_defense_simulations/final_evaluate_defenses.ipynb`
* **Attack Pipeline**: `04_attack_and_defense_simulations/simulate_unsw_pipeline.py`

### 5. Experiment Paper Protocol & Future Work (`Step 5` in UI - RQ 2, 3, 4)
* **Statistical Significance Script**: `05_statistical_and_rq_analysis/run_paired_ttests.py`
* **Ablation & Sensitivity Data**: `05_statistical_and_rq_analysis/afp_ablation_summary.csv`
* **Presenter Guides**: `06_presentation_guides/PROGRESS_REPORT_RUNBOOK.md`
