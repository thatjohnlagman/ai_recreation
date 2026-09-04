# Progress Report Runbook: Thesis Progress Presentation Guide
> **Document**: Presenter Instructions, Chronological Sequence, and Evidence Map  
> **Target Application**: `thesis_progress_app.py`  
> **Repository Root**: `/Users/trumpler-mac/dev/progress_report/ai_recreation`  
> **Thesis Project**: *Recall-Aware Adaptive Feature Perturbation for Machine Learning-Based Intrusion Detection Systems*  
> **Presentation Focus**: Chronological Progress Report Mapped to Appendix 1 (Experiment Paper)  

---

## 1. Quick Launch Commands

Execute the following commands from the repository root:

```bash
cd /Users/trumpler-mac/dev/progress_report/ai_recreation
source .venv/bin/activate
streamlit run thesis_progress_app.py --server.port 8501
```

- **Local URL**: `http://localhost:8501`

---

## 2. Chronological Progress Sequence (Matches Experiment Paper)

```
┌────────────────────────────────────────────────────────────────────────────────────────┐
│                        CHRONOLOGICAL PROGRESS SEQUENCE (5 STEPS)                       │
├─────────┬───────────────────────────────────────┬──────────────────────────────────────┤
│ Step 1  │ 1. Data Preprocessing & Features      │ Pre-experimentation data pipeline    │
│ Step 2  │ 2. Live Baseline Classification       │ Baseline model live flow test        │
│ Step 3  │ 3. Measured Baseline Performance      │ 20k clean benchmark (99.38% Acc)     │
│ Step 4  │ 4. Attack & Defense Simulation (AFP)  │ Probing attacks & AFP pass/fail test │
│ Step 5  │ 5. Experiment Paper Steps             │ Appendix 1 mapping: Done vs. Upcoming│
└─────────┴───────────────────────────────────────┴──────────────────────────────────────┘
```

---

## 3. Two-to-Three Minute Presentation Narrative (Chronological)

Follow this natural chronological sequence during your progress presentation:

### [0:00 – 0:30] Step 1: Data Preprocessing & Features (Pre-experimentation)
- **Click**: `1. Data Preprocessing & Features` in the sidebar.
- **Presenter Narrative**:
  > *"Good morning. Today I am presenting our progress report on **Recall-Aware Adaptive Feature Perturbation for ML-Based IDS**."*  
  > *"We begin with **Phase 1: Pre-experimentation**. We ingested the raw CSE-CIC-IDS2018 network captures, cleaned invalid and infinite values, converted labels into binary format, and stratified into a 70/30 split."*  
  > *"We extracted our standardized **77-feature reference baselines** (mean $\mu_{\text{ref}}$ and standard deviation $\sigma_{\text{ref}}$) along with physical clamping boundaries, which you can see in this searchable table."*

### [0:30 – 1:00] Step 2: Live Baseline Classification (Model Verification)
- **Click**: `2. Live Baseline Classification` in the sidebar.
- **Presenter Narrative**:
  > *"Using this preprocessed training data, we trained our baseline Random Forest classifier (200 trees, depth 20) and locked it for all evaluation runs."*  
  > *"Here is the live classifier test: when we submit a clean Benign flow, the model correctly predicts **Benign** with high confidence. When we submit a raw Attack flow, it correctly flags it as an **Attack**."*

### [1:00 – 1:30] Step 3: Measured Baseline Performance (Clean Control Benchmark)
- **Click**: `3. Measured Baseline Performance` in the sidebar.
- **Presenter Narrative**:
  > *"To establish our control benchmark, we evaluated the model across 20,000 clean test flows (10,000 benign, 10,000 attack)."*  
  > *"The baseline achieves **99.38% overall accuracy**, **96.64% attack detection recall**, and **97.90% F1-score**, with a very low false positive rate of 0.79%. Here is the complete $2 \times 2$ confusion matrix proving the model is solid under normal conditions."*

### [1:30 – 2:15] Step 4: Attack & Defense Simulation (During Experimentation - RQ 1)
- **Click**: `4. Attack & Defense Simulation (AFP ON / OFF)` in the sidebar.
- **Presenter Narrative**:
  > *"Now we move into **Phase 2: During Experimentation**. What happens when an adversary applies black-box probing to sneak past the IDS?"*  
  > *"Let's test an attack using **Silent Probing**: with AFP Defense **OFF**, the attacker modifies boundary features and the attack **PASSES** undetected—the model predicts Benign! The baseline is completely vulnerable."*  
  > *"Now we turn AFP Defense **ON**: the defense injects bounded adaptive perturbation, shifting the features away from the attacker's evasion point. The attack is **BLOCKED**! This verifies the efficacy of Base AFP, answering **Research Question 1**."*

### [2:15 – 2:45] Step 5: Experiment Paper Steps (Appendix 1 Protocol Mapping)
- **Click**: `5. Experiment Paper Steps` in the sidebar.
- **Presenter Narrative**:
  > *"Finally, here is where our work stands compared to our formal **Experiment Paper (Appendix 1)**:"*  
  > *"• **Steps Done:** Pre-experimentation is 100% complete, and Base Defense evaluations are completed for Silent Probing, Boundary Attack, and Surrogate Transfer (**RQ 1**)."*  
  > *"• **Steps To Be Done:** We are finalizing the dynamic Recall-Aware controller (**RQ 2**), after which we will run the paired t-test at $\alpha = 0.05$ (**RQ 3**) and the controller parameter sensitivity analysis (**RQ 4**)."*  
  > *"Thank you, and I welcome any questions or feedback on our progress."*

---

## 4. Key Metrics Reference

| Metric / Scenario | Value | Status in Progress Report |
| :--- | :---: | :--- |
| **Baseline Accuracy (20k clean)** | **99.38%** | Completed & Verified |
| **Baseline Attack Recall (20k clean)** | **96.64%** | Completed & Verified |
| **Silent Probing Undefended Recall** | **22.15%** | Vulnerability Confirmed |
| **Silent Probing AFP-Defended Recall** | **96.25%** | Recovery Verified (RQ 1) |
| **Decision Boundary Undefended Recall** | **12.35%** | Vulnerability Confirmed |
| **Decision Boundary AFP-Defended Recall** | **95.94%** | Recovery Verified (RQ 1) |
| **Surrogate Transfer Undefended Recall** | **86.42%** | Vulnerability Confirmed (78.5% FPR) |
| **Surrogate Transfer AFP-Defended Recall**| **96.11%** | Recovery Verified (RQ 1, 1.2% FPR) |
| **Controller-Augmented Evaluation (RQ 2)** | *In Progress* | Dynamic controller formulation |
| **Paired t-Test Comparison (RQ 3)** | *To Be Done* | Scheduled post-experimentation ($\alpha = 0.05$) |
| **Parameter Sensitivity Summary (RQ 4)** | *To Be Done* | Scheduled post-experimentation (C1..Cn) |
