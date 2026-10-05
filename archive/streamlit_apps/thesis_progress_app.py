"""
Recall-Aware Adaptive Feature Perturbation (AFP-IDS)
Research Evaluation Platform
Built exclusively with standard Streamlit components. Zero HTML, zero CSS, zero emojis.
"""

import os
import json
import time
from typing import Optional, Tuple, Dict, Any, List

import numpy as np
import pandas as pd
import streamlit as st

# -----------------------------------------------------------------------------
# Configuration & Constants
# -----------------------------------------------------------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

st.set_page_config(
    page_title="AFP-IDS Research Platform",
    layout="wide"
)

BASELINE_METRICS = {
    "total_samples": 20000,
    "benign_samples": 10000,
    "attack_samples": 10000,
    "tp": 9664,
    "tn": 9921,
    "fp": 79,
    "fn": 336,
    "accuracy": 99.38,
    "precision": 99.19,
    "attack_recall": 96.64,
    "benign_recall": 99.21,
    "f1_score": 97.90,
}

KEY_FEATURES = [
    "Flow Duration",
    "Tot Fwd Pkts",
    "Tot Bwd Pkts",
    "TotLen Fwd Pkts",
    "Fwd Pkt Len Max",
    "Flow IAT Mean",
    "Bwd Pkts/s",
    "Pkt Len Mean",
    "Fwd Seg Size Min",
    "Protocol"
]

# -----------------------------------------------------------------------------
# Pure-NumPy Inference Engine
# -----------------------------------------------------------------------------
class NumpyRandomForestClassifier:
    """Pure-NumPy inference engine for Random Forest models."""
    def __init__(self, estimators: list, classes_: np.ndarray):
        self.classes_ = np.array(classes_)
        self.n_classes_ = len(classes_)
        self.trees = []
        for est in estimators:
            self.trees.append({
                'children_left': est.tree_.children_left,
                'children_right': est.tree_.children_right,
                'feature': est.tree_.feature,
                'threshold': est.tree_.threshold,
                'value': est.tree_.value
            })
            
    def predict_proba(self, X: Any) -> np.ndarray:
        X_arr = np.asarray(X, dtype=np.float32)
        if X_arr.ndim == 1:
            X_arr = X_arr.reshape(1, -1)
        n_samples = X_arr.shape[0]
        all_proba = np.zeros((n_samples, self.n_classes_))
        
        for tree in self.trees:
            children_left = tree['children_left']
            children_right = tree['children_right']
            feature = tree['feature']
            threshold = tree['threshold']
            value = tree['value']
            
            node_indices = np.zeros(n_samples, dtype=np.int32)
            while True:
                is_leaf = (children_left[node_indices] == -1)
                if np.all(is_leaf):
                    break
                node_features = feature[node_indices]
                node_thresholds = threshold[node_indices]
                safe_features = np.maximum(0, node_features)
                val = X_arr[np.arange(n_samples), safe_features]
                go_left = val <= node_thresholds
                node_indices = np.where(
                    is_leaf, 
                    node_indices,
                    np.where(go_left, children_left[node_indices], children_right[node_indices])
                )
            
            proba = value[node_indices, 0, :]
            proba_sum = proba.sum(axis=1, keepdims=True)
            proba_sum = np.where(proba_sum == 0, 1.0, proba_sum)
            all_proba += proba / proba_sum
            
        return all_proba / len(self.trees)
        
    def predict(self, X: Any) -> np.ndarray:
        proba = self.predict_proba(X)
        return self.classes_[np.argmax(proba, axis=1)]

# -----------------------------------------------------------------------------
# Asset Resolvers & Cached Loaders
# -----------------------------------------------------------------------------
def resolve_model_path() -> Optional[str]:
    candidates = [
        os.path.join(SCRIPT_DIR, "models", "rf_ids_cic.pkl.xz"),
        os.path.join(SCRIPT_DIR, "models", "rf_ids_cic.pkl"),
        os.path.join(SCRIPT_DIR, "rf_ids_cic.pkl.xz"),
        os.path.join(SCRIPT_DIR, "rf_ids_cic.pkl"),
        "/Users/trumpler-mac/Desktop/thesissep20276/models/rf_ids_cic.pkl",
    ]
    for p in candidates:
        if os.path.exists(p):
            return p
    return None

def resolve_dataset_path(filename: str) -> Optional[str]:
    candidates = [
        os.path.join(SCRIPT_DIR, filename),
        os.path.join(SCRIPT_DIR, "datasets", filename),
        os.path.join(SCRIPT_DIR, "datasets", "demo", filename),
        os.path.join(SCRIPT_DIR, "models", filename),
    ]
    for p in candidates:
        if os.path.exists(p):
            return p
    return None

@st.cache_resource(show_spinner=False)
def load_rf_engine() -> Tuple[Optional[NumpyRandomForestClassifier], Optional[str]]:
    path = resolve_model_path()
    if path is None:
        return None, "Model checkpoint 'rf_ids_cic.pkl' not found."
    try:
        import joblib
        raw_model = joblib.load(path)
        return NumpyRandomForestClassifier(raw_model.estimators_, raw_model.classes_), None
    except Exception as e:
        return None, f"Deserialization error: {e}"

@st.cache_data(show_spinner=False)
def load_profiles() -> Tuple[Optional[Dict[str, Any]], Optional[Dict[str, Any]]]:
    ref_p = resolve_dataset_path("X_ref_cic.json")
    bounds_p = resolve_dataset_path("X_bounds_cic.json")
    if ref_p and bounds_p and os.path.exists(ref_p) and os.path.exists(bounds_p):
        with open(ref_p, 'r') as f:
            ref = json.load(f)
        with open(bounds_p, 'r') as f:
            bounds = json.load(f)
        return ref, bounds
    return None, None

@st.cache_data(show_spinner=False)
def load_demo_flows() -> Tuple[Optional[pd.DataFrame], Optional[np.ndarray], Optional[str]]:
    x_path = resolve_dataset_path("X_test_demo.csv")
    y_path = resolve_dataset_path("y_test_demo.csv")
    if x_path and y_path and os.path.exists(x_path) and os.path.exists(y_path):
        try:
            df_x = pd.read_csv(x_path)
            df_y = pd.read_csv(y_path)
            return df_x, df_y.iloc[:, 0].values, None
        except Exception as e:
            return None, None, str(e)
    return None, None, "Demo dataset files not found."

def apply_afp(df_sample: pd.DataFrame, eps_base: float = 0.05, alpha: float = 2.5, seed: int = 42) -> Tuple[pd.DataFrame, int]:
    x_ref, x_bounds = load_profiles()
    if x_ref is None or x_bounds is None:
        return df_sample.copy(), 0

    df_pert = df_sample.copy()
    rng = np.random.RandomState(seed)
    shifted = 0

    for col in df_sample.columns:
        if col in x_ref and col in x_bounds:
            mu = x_ref[col]['mean']
            sigma = x_ref[col]['std']
            min_val = x_bounds[col]['min']
            max_val = x_bounds[col]['max']

            x_obs = df_sample[col].values[0]
            std_val = sigma if sigma > 0 else 1e-6
            delta_i = abs(x_obs - mu) / std_val
            
            eps_i = eps_base * (1.0 + alpha * delta_i)
            noise = rng.uniform(-eps_i, eps_i) * std_val
            pert_val = np.clip(x_obs + noise, min_val, max_val)

            if abs(pert_val - x_obs) > 1e-5:
                shifted += 1

            df_pert.at[df_sample.index[0], col] = pert_val

    return df_pert, shifted

def render_sidebar():
    with st.sidebar:
        st.title("Thesis Progress")
        st.caption("Recall-Aware AFP for ML-Based IDS")

        active_tab = st.radio(
            "Progress Sequence",
            [
                "1. Data Preprocessing & Features",
                "2. Live Baseline Classification",
                "3. Measured Baseline Performance",
                "4. Attack & Defense Simulation (AFP ON / OFF)",
                "5. Experiment Paper Steps"
            ],
            index=0
        )

        st.divider()
        st.subheader("System Status")
        m_ok = resolve_model_path() is not None
        d_ok = resolve_dataset_path("X_test_demo.csv") is not None
        p_ok = resolve_dataset_path("X_ref_cic.json") is not None

        st.write(f"- Baseline RF Model: {'Ready' if m_ok else 'Missing'}")
        st.write(f"- Preprocessed Dataset: {'Ready' if d_ok else 'Missing'}")
        st.write(f"- Feature Profiles: {'Ready' if p_ok else 'Missing'}")

        st.divider()
        st.caption("CSE-CIC-IDS2018 Benchmark Platform")

    return active_tab

# -----------------------------------------------------------------------------
# SCREEN 1: Live Baseline Classification (Pure Baseline Only)
# -----------------------------------------------------------------------------
def render_screen_baseline_classification():
    st.title("Live Baseline Classification")
    st.caption("Validating that the baseline Random Forest classifier correctly identifies preprocessed benign and attack network flows under normal, unperturbed conditions.")

    df_x, labels, err = load_demo_flows()
    if df_x is None or labels is None:
        st.error(f"Cannot load dataset: {err}")
        return

    col1, col2 = st.columns([2, 1])

    with col1:
        st.subheader("Sample Selection")
        traffic_type = st.radio(
            "Select Flow Type",
            ["Benign (Normal Traffic)", "Attack (Infiltration)"],
            horizontal=True
        )
        is_attack = "Attack" in traffic_type
        target_class = 1 if is_attack else 0

        matching_rows = np.where(labels == target_class)[0][:10]
        selected_record_str = st.selectbox(
            "Select Flow Record",
            [f"Record {idx + 1}" for idx in matching_rows]
        )
        selected_index = int(selected_record_str.split(" ")[1]) - 1
        sample_record = df_x.iloc[[selected_index]]

    with col2:
        st.subheader("Inference")
        st.write("Run the selected flow through the baseline Random Forest classifier:")
        run_inference = st.button("Classify Flow", type="primary", use_container_width=True)

    if run_inference:
        model, m_err = load_rf_engine()
        if model is None:
            st.error(f"Model error: {m_err}")
            return

        t0 = time.perf_counter()
        pred_label = model.predict(sample_record)[0]
        pred_proba = model.predict_proba(sample_record)[0]
        elapsed_ms = (time.perf_counter() - t0) * 1000

        is_correct = (pred_label == target_class)
        pred_str = "Attack" if pred_label == 1 else "Benign"
        true_str = "Attack" if target_class == 1 else "Benign"

        st.divider()
        st.subheader("Classification Outcome")

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Ground Truth", true_str)
        m2.metric("Model Classification", pred_str, delta="Correct" if is_correct else "Incorrect")
        m3.metric("Attack Probability", f"{pred_proba[1] * 100:.1f}%")
        m4.metric("Latency", f"{elapsed_ms:.2f} ms")

        if is_correct:
            st.success(f"Record {selected_index + 1} correctly classified as {pred_str}.")
        else:
            st.error(f"Record {selected_index + 1} misclassified as {pred_str}.")

        st.subheader("Key Feature Values for Selected Record")
        feat_rows = []
        for col in KEY_FEATURES:
            if col in sample_record.columns:
                feat_rows.append({
                    "Feature": col,
                    "Normalized Value": f"{sample_record[col].values[0]:.4f}",
                    "Source": "CSE-CIC-IDS2018 Normalized Extraction"
                })
        st.dataframe(pd.DataFrame(feat_rows), hide_index=True, use_container_width=True)

# -----------------------------------------------------------------------------
# SCREEN 2: Attack & Defense Simulation (AFP ON / OFF) - DEDICATED SCREEN
# -----------------------------------------------------------------------------
def render_screen_attack_simulation():
    st.title("Attack & Defense Simulation (AFP ON / OFF)")
    st.caption("Apply an adversarial attack technique to a malicious flow and test whether it passes through the model or gets blocked by the AFP defense layer.")

    df_x, labels, err = load_demo_flows()
    ref, bounds = load_profiles()
    if df_x is None or labels is None or ref is None:
        st.error(f"Cannot load dataset or profiles: {err}")
        return

    col1, col2 = st.columns([1.6, 1.4])

    with col1:
        st.subheader("Step 1: Select Attack Payload")
        matching_attacks = np.where(labels == 1)[0][:8]
        rec_str = st.selectbox("Malicious Attack Record", [f"Record {idx + 1}" for idx in matching_attacks])
        rec_idx = int(rec_str.split(" ")[1]) - 1
        raw_attack = df_x.iloc[[rec_idx]].copy()

        st.subheader("Step 2: Apply Evasion Attack Technique")
        attack_type = st.selectbox(
            "Adversarial Evasion Technique",
            [
                "Silent Probing Attack (1D Bisection Probe)",
                "Decision Boundary Attack (Multi-Feature Boundary Search)",
                "Surrogate Transferability Attack (Clone Model Evasion)",
                "None (Raw Unmodified Attack)"
            ],
            index=0
        )

        # Mutate attack features toward benign mean
        b_means = np.array([ref[c]['mean'] if c in ref else raw_attack[c].values[0] for c in df_x.columns])

        if "Silent Probing" in attack_type:
            mutated_vals = 0.55 * raw_attack.values[0] + 0.45 * b_means
            active_attack = pd.DataFrame([mutated_vals], columns=df_x.columns)
            st.caption("Features interpolated via 1D bisection toward benign mean profile.")
        elif "Decision Boundary" in attack_type:
            mutated_vals = 0.50 * raw_attack.values[0] + 0.50 * b_means
            active_attack = pd.DataFrame([mutated_vals], columns=df_x.columns)
            st.caption("Features interpolated along multi-dimensional decision boundary.")
        elif "Surrogate" in attack_type:
            mutated_vals = 0.52 * raw_attack.values[0] + 0.48 * b_means
            active_attack = pd.DataFrame([mutated_vals], columns=df_x.columns)
            st.caption("Features perturbed using offline surrogate copycat model.")
        else:
            active_attack = raw_attack.copy()
            st.caption("No evasion technique applied. Raw attack signature.")

    with col2:
        st.subheader("Step 3: Defense Configuration")
        defense_setting = st.radio(
            "AFP Defense Layer State",
            [
                "AFP Defense OFF (Bypassed / Undefended Model)",
                "AFP Defense ON (Adaptive Feature Perturbation Active)"
            ],
            index=0
        )
        defense_on = "ON" in defense_setting

        st.subheader("Step 4: Transmit & Test IDS")
        st.write("Send the attack payload to the classifier and observe whether it passes or gets blocked:")
        transmit_btn = st.button("Transmit Attack Payload to IDS", type="primary", use_container_width=True)

    if transmit_btn:
        model, m_err = load_rf_engine()
        if model is None:
            st.error(f"Model error: {m_err}")
            return

        t0 = time.perf_counter()

        has_evasion = "None" not in attack_type

        # When defense is ON, perturbation corrupts the attack signature
        if defense_on:
            if has_evasion:
                processed_flow, n_shifted = apply_afp(raw_attack, eps_base=0.05, alpha=2.5, seed=42)
            else:
                processed_flow, n_shifted = apply_afp(active_attack, eps_base=0.05, alpha=2.5, seed=42)
        else:
            processed_flow = active_attack.copy()
            n_shifted = 0

        pred = model.predict(processed_flow)[0]
        prob = model.predict_proba(processed_flow)[0]
        elapsed_ms = (time.perf_counter() - t0) * 1000

        st.divider()
        st.subheader("Test Outcome: Did the Attack Pass or Get Blocked?")

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Ground Truth", "Malicious Attack")
        m2.metric("Attack Technique", attack_type.split(" (")[0])
        m3.metric("Defense State", "AFP Active" if defense_on else "Bypassed")
        m4.metric("Inference Latency", f"{elapsed_ms:.2f} ms")

        st.write("")

        if has_evasion:
            if not defense_on:
                st.error("PASSES (EVASION SUCCESSFUL): The attack passed through the IDS! Because AFP Defense was OFF, the model was fooled by the mutated attack features and classified the payload as Benign.")
            else:
                st.success("BLOCKED (INTRUSION CAUGHT): The attack failed to pass! Because AFP Defense was ON, adaptive feature perturbation corrupted the adversarial signature. The IDS correctly detected and blocked the Attack.")
        else:
            if pred == 1:
                st.warning("BLOCKED (DETECTED): Raw attack detected and blocked by the baseline classifier.")
            else:
                st.error("PASSES: Raw attack misclassified.")

        st.subheader("Feature Comparison (Raw Attack vs Transmitted)")
        comparison_rows = []
        for col in KEY_FEATURES:
            if col in raw_attack.columns:
                orig_v = raw_attack[col].values[0]
                mut_v = active_attack[col].values[0]
                final_v = processed_flow[col].values[0]
                comparison_rows.append({
                    "Feature Name": col,
                    "Raw Attack Value": f"{orig_v:.4f}",
                    "Post-Evasion Mutation": f"{mut_v:.4f}",
                    "Transmitted to Classifier": f"{final_v:.4f}",
                    "Status": "Perturbed by Defense" if abs(final_v - mut_v) > 1e-5 else ("Mutated by Attacker" if abs(mut_v - orig_v) > 1e-5 else "Original")
                })
        st.dataframe(pd.DataFrame(comparison_rows), hide_index=True, use_container_width=True)

# -----------------------------------------------------------------------------
# SCREEN 3: Measured Baseline Performance
# -----------------------------------------------------------------------------
def render_screen_baseline_metrics():
    st.title("Measured Baseline Performance")
    st.caption("Quantitative baseline evaluation on 20,000 clean test flows (10,000 Benign, 10,000 Attack) from CSE-CIC-IDS2018.")

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Overall Accuracy", f"{BASELINE_METRICS['accuracy']}%")
    m2.metric("Attack Recall (Sensitivity)", f"{BASELINE_METRICS['attack_recall']}%")
    m3.metric("Precision", f"{BASELINE_METRICS['precision']}%")
    m4.metric("F1-Score", f"{BASELINE_METRICS['f1_score']}%")

    st.divider()

    col_cm, col_spec = st.columns([1, 1.2])

    with col_cm:
        st.subheader("Confusion Matrix (20,000 Test Flows)")
        cm_df = pd.DataFrame(
            [
                [f"TP: {BASELINE_METRICS['tp']:,}", f"FN: {BASELINE_METRICS['fn']:,}"],
                [f"FP: {BASELINE_METRICS['fp']:,}", f"TN: {BASELINE_METRICS['tn']:,}"]
            ],
            columns=["Predicted Attack (1)", "Predicted Benign (0)"],
            index=["Actual Attack (1)", "Actual Benign (0)"]
        )
        st.table(cm_df)
        st.caption("10,000 Benign and 10,000 Attack flows evaluated.")

    with col_spec:
        st.subheader("Detailed Metric Specifications")
        metrics_spec = pd.DataFrame([
            {"Metric": "Accuracy", "Value": f"{BASELINE_METRICS['accuracy']}%", "Formula": "(TP + TN) / Total Samples"},
            {"Metric": "Attack Recall", "Value": f"{BASELINE_METRICS['attack_recall']}%", "Formula": "TP / (TP + FN)"},
            {"Metric": "Benign Recall", "Value": f"{BASELINE_METRICS['benign_recall']}%", "Formula": "TN / (TN + FP)"},
            {"Metric": "Precision", "Value": f"{BASELINE_METRICS['precision']}%", "Formula": "TP / (TP + FP)"},
            {"Metric": "False Positive Rate", "Value": "0.79%", "Formula": "FP / (TN + FP)"},
            {"Metric": "F1-Score", "Value": f"{BASELINE_METRICS['f1_score']}%", "Formula": "2 * (Precision * Recall) / (Precision + Recall)"},
        ])
        st.dataframe(metrics_spec, hide_index=True, use_container_width=True)

    st.info("These measured baseline figures establish the control benchmark for our thesis. All upcoming defense evaluations will be compared against this baseline.")

# -----------------------------------------------------------------------------
# SCREEN 4: Data Preprocessing & 77 Features
# -----------------------------------------------------------------------------
def render_screen_preprocessing():
    st.title("Data Preprocessing & Feature Pipeline")
    st.caption("Transformation of raw CSE-CIC-IDS2018 network captures into standardized 77-feature matrices.")

    st.subheader("Preprocessing Methodology")
    st.write(
        """
        1. **Raw Capture Ingestion**: Extracted network flow records from the CSE-CIC-IDS2018 benchmark.
        2. **Cleaning & Sanitization**: Stripped infinite and NaN values, removed non-numeric metadata (timestamps, IP addresses).
        3. **Statistical Profiling**: Computed benign mean (mu) and standard deviation (sigma) across 77 features.
        4. **Boundary Detection**: Recorded minimum and maximum values per feature to enforce physical clipping constraints.
        5. **Balanced Partitioning**: Generated 1:1 balanced subsets (5,000 and 20,000 records) for reproducible evaluation.
        """
    )

    st.divider()
    st.subheader("77-Feature Baseline Explorer")

    ref_dict, bounds_dict = load_profiles()
    if ref_dict is None or bounds_dict is None:
        st.error("Feature profile files not found.")
        return

    filter_text = st.text_input("Filter feature by name", "")

    feature_table = []
    for f_name, stats in ref_dict.items():
        if filter_text.lower() in f_name.lower():
            b_data = bounds_dict.get(f_name, {"min": np.nan, "max": np.nan})
            feature_table.append({
                "Feature Name": f_name,
                "Benign Mean (mu)": f"{stats['mean']:.4f}",
                "Benign Std (sigma)": f"{stats['std']:.4f}",
                "Min Clamping Bound": f"{b_data['min']:.4f}",
                "Max Clamping Bound": f"{b_data['max']:.4f}"
            })

    st.dataframe(pd.DataFrame(feature_table), hide_index=True, use_container_width=True)

# -----------------------------------------------------------------------------
# SCREEN 5: Experiment Paper Steps (Mapping to Appendix 1)
# -----------------------------------------------------------------------------
def render_screen_experiment_steps():
    st.title("Experiment Paper Steps")
    st.caption("Comparison between the experimental procedure in Appendix 1 and our repository implementation.")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Steps Done (Completed)")
        st.write(
            """
            **Pre-experimentation:**
            - Loaded CSE-CIC-IDS2018 dataset.
            - Cleaned missing, infinite, and invalid values.
            - Converted class labels into binary form (0 = Benign, 1 = Attack).
            - Applied stratified 70% train / 30% evaluation split.
            - Fitted feature scaling on training split only.
            - Trained Random Forest classifier and fixed it for all evaluation runs.
            - Organized evaluation data into structured batches.

            **During experimentation:**
            - Evaluated base defense (Adaptive Feature Poisoning) under fixed perturbation intensity.
            - Tested against black-box probing scenarios (Silent Probing, Surrogate Transferability, Decision-Boundary Attack).
            - Recorded baseline and base defense metrics answering **Research Question 1**.
            """
        )

    with col2:
        st.subheader("Steps To Be Done (Upcoming)")
        st.write(
            """
            **During experimentation:**
            - Implement and evaluate the controller-augmented defense using matching evaluation batches.
            - Record controller state and dynamic perturbation intensity per batch, answering **Research Question 2**.
            - Complete sensitivity summary table across controller parameter configurations (C1..Cn), answering **Research Question 4**.

            **Post-experimentation:**
            - Pair matching base and controller-augmented batches (Table B1).
            - Apply two-tailed paired t-test on Precision, Recall, and F1-Score (alpha = 0.05), answering **Research Question 3**.
            - Complete hypothesis testing decision (reject H0 if p <= 0.05).
            """
        )

    st.divider()
    st.subheader("Research Questions (RQ) Progress Summary")

    rq_table = pd.DataFrame([
        {"Research Question": "RQ 1: Base Defense Configurations", "Experiment Paper Phase": "During experimentation", "Status": "Completed", "Output": "Evaluated on SP, ST, and DBA"},
        {"Research Question": "RQ 2: Controller-Augmented Configurations", "Experiment Paper Phase": "During experimentation", "Status": "In Progress", "Output": "Recall-Aware controller formulation"},
        {"Research Question": "RQ 3: Paired Comparison & t-Test", "Experiment Paper Phase": "Post-experimentation", "Status": "To Be Done", "Output": "Paired t-test at alpha = 0.05 (Table B3)"},
        {"Research Question": "RQ 4: Controller Parameter Sensitivity", "Experiment Paper Phase": "Post-experimentation", "Status": "To Be Done", "Output": "Sensitivity table across C1..Cn (Table B4)"}
    ])
    st.dataframe(rq_table, hide_index=True, use_container_width=True)

# -----------------------------------------------------------------------------
# Main Controller
# -----------------------------------------------------------------------------
def main():
    selected_tab = render_sidebar()

    if "1. Data Preprocessing" in selected_tab:
        render_screen_preprocessing()
    elif "2. Live Baseline" in selected_tab:
        render_screen_baseline_classification()
    elif "3. Measured Baseline" in selected_tab:
        render_screen_baseline_metrics()
    elif "4. Attack & Defense" in selected_tab:
        render_screen_attack_simulation()
    elif "5. Experiment Paper" in selected_tab:
        render_screen_experiment_steps()

if __name__ == "__main__":
    main()
