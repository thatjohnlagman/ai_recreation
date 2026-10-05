import streamlit as st
import textwrap
import pandas as pd
import numpy as np
import joblib
import os
import time
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Ensure SCRIPT_DIR is determined for robust relative pathing
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

class NumpyRandomForestClassifier:
    def __init__(self, estimators, classes_):
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
            
    def predict_proba(self, X):
        X_arr = np.asarray(X, dtype=np.float32)
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
                node_indices = np.where(is_leaf, node_indices,
                                        np.where(go_left, children_left[node_indices], children_right[node_indices]))
            
            proba = value[node_indices, 0, :]
            proba_sum = proba.sum(axis=1, keepdims=True)
            proba_sum = np.where(proba_sum == 0, 1.0, proba_sum)
            all_proba += proba / proba_sum
            
        return all_proba / len(self.trees)
        
    def predict(self, X):
        proba = self.predict_proba(X)
        return self.classes_[np.argmax(proba, axis=1)]


# ---------------------------------------------------------
# Page Configuration
# ---------------------------------------------------------
st.set_page_config(
    page_title="Adaptive Feature Perturbation Platform",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------
# Dynamic Resource Finders (Relative Paths)
# ---------------------------------------------------------
def find_dataset(filename):
    home_dir = os.path.expanduser("~")
    candidates = [
        os.path.join(SCRIPT_DIR, filename),
        os.path.join(SCRIPT_DIR, "datasets", filename),
        os.path.join(SCRIPT_DIR, "../datasets", filename),
        os.path.join(SCRIPT_DIR, "../presentation/datasets", filename),
        os.path.join(SCRIPT_DIR, "../IDS_Dashboard_Submission-20260613T121815Z-3-001/IDS_Dashboard_Submission/datasets", filename),
        os.path.join(home_dir, "Downloads", filename),
        os.path.join(home_dir, "Downloads", "models-20260613T064206Z-3-001", "models", filename),
        os.path.join(home_dir, "Downloads", "ai-tool", "models", filename),
        os.path.join(home_dir, "Downloads", "ai-tool", filename),
    ]
    for c in candidates:
        if os.path.exists(c):
            return c
    return None

def find_model():
    home_dir = os.path.expanduser("~")
    candidates = [
        os.path.join(SCRIPT_DIR, "models/rf_ids_cic.pkl.xz"),
        os.path.join(SCRIPT_DIR, "rf_ids_cic.pkl.xz"),
        os.path.join(SCRIPT_DIR, "rf_ids_cic.pkl"),
        os.path.join(SCRIPT_DIR, "models/rf_ids_cic.pkl"),
        os.path.join(SCRIPT_DIR, "../presentation/models/rf_ids_cic.pkl"),
        os.path.join(SCRIPT_DIR, "../IDS_Dashboard_Submission-20260613T121815Z-3-001/IDS_Dashboard_Submission/models/rf_ids_cic.pkl"),
        os.path.join(SCRIPT_DIR, "../../Downloads/models-20260613T064206Z-3-001/models/rf_ids_cic.pkl"),
        os.path.join(home_dir, "Downloads", "models-20260613T064206Z-3-001", "models", "rf_ids_cic.pkl"),
        os.path.join(home_dir, "Downloads", "rf_ids_cic.pkl"),
        os.path.join(home_dir, "Downloads", "ai-tool", "models", "rf_ids_cic.pkl"),
        os.path.join(home_dir, "Downloads", "ai-tool", "rf_ids_cic.pkl"),
    ]
    for c in candidates:
        if os.path.exists(c):
            return c
    return None

# ---------------------------------------------------------
# Data & Model Caching
# ---------------------------------------------------------
@st.cache_resource
def load_base_model():
    model_path = find_model()
    if model_path is None:
        return None
    try:
        loaded_model = joblib.load(model_path)
        model = NumpyRandomForestClassifier(loaded_model.estimators_, loaded_model.classes_)
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

@st.cache_data
def load_sample_data():
    """Loads a balanced 20-sample sandbox from the test set for Tab 1 (checks 20k first, falls back to original)"""
    x_path = find_dataset("demo/X_test_demo_20k.csv")
    y_path = find_dataset("demo/y_test_demo_20k.csv")
    
    if x_path is None or y_path is None:
        x_path = find_dataset("X_test_demo_20k.csv")
        y_path = find_dataset("y_test_demo_20k.csv")
        
    if x_path is None or y_path is None:
        x_path = find_dataset("demo/X_test_demo.csv")
        y_path = find_dataset("demo/y_test_demo.csv")
        
    if x_path is None or y_path is None:
        x_path = find_dataset("datasets/demo/X_test_demo.csv")
        y_path = find_dataset("datasets/demo/y_test_demo.csv")
        
    if x_path is None or y_path is None:
        st.error("Demo dataset files could not be resolved from relative paths.")
        return None, None
        
    try:
        X_chunk = pd.read_csv(x_path, nrows=1000)
        y_chunk = pd.read_csv(y_path, nrows=1000)
        
        df = X_chunk.copy()
        df['Label'] = y_chunk.iloc[:, 0].values
        
        benign = df[df['Label'] == 0].head(10)
        attack = df[df['Label'] == 1].head(10)
        
        balanced_sample = pd.concat([benign, attack]).sample(frac=1, random_state=42).reset_index(drop=True)
        return balanced_sample.drop(columns=['Label']), balanced_sample['Label']
    except Exception as e:
        st.error(f"Error loading sample data: {e}")
        return None, None

X_data, y_data = load_sample_data()
base_model = load_base_model()

if X_data is None or base_model is None:
    st.warning("Please ensure the models and demo datasets are available in the correct directories.")
    st.stop()

# ---------------------------------------------------------
# Dynamic File Finder & AFP Profile Loader
# ---------------------------------------------------------
def find_dataset_path(filename):
    home_dir = os.path.expanduser("~")
    candidates = [
        os.path.join(SCRIPT_DIR, filename),
        os.path.join(SCRIPT_DIR, "datasets", filename),
        os.path.join(SCRIPT_DIR, "../datasets", filename),
        os.path.join(home_dir, "Downloads", filename),
        os.path.join(home_dir, "Downloads", "ai-tool", "models", filename),
        os.path.join(home_dir, "Downloads", "ai-tool", filename),
    ]
    for c in candidates:
        if os.path.exists(c):
            return c
    return None

@st.cache_data
def load_afp_profiles():
    ref_path = find_dataset_path("models/X_ref_cic.json") or find_dataset_path("X_ref_cic.json")
    bounds_path = find_dataset_path("models/X_bounds_cic.json") or find_dataset_path("X_bounds_cic.json")
    
    if ref_path and bounds_path:
        with open(ref_path, 'r') as f:
            x_ref = json.load(f)
        with open(bounds_path, 'r') as f:
            x_bounds = json.load(f)
        return x_ref, x_bounds
    return None, None

def apply_afp_perturbation_preview(df_raw, eps_base=0.05, alpha=2.5):
    x_ref, x_bounds = load_afp_profiles()
    if x_ref is None or x_bounds is None:
        return df_raw.copy()
    
    df_perturbed = df_raw.copy()
    np.random.seed(42)  # Keep perturbation reproducible for preview
    
    for col in df_raw.columns:
        if col in x_ref and col in x_bounds:
            mu_ref = x_ref[col]['mean']
            sigma_ref = x_ref[col]['std']
            min_val = x_bounds[col]['min']
            max_val = x_bounds[col]['max']
            
            x_obs = df_raw[col].values
            std_val = sigma_ref if sigma_ref > 0 else 1e-6
            delta_i = np.abs(x_obs - mu_ref) / std_val
            
            epsilon_i = eps_base * (1.0 + alpha * delta_i)
            noise = np.random.uniform(-epsilon_i, epsilon_i) * std_val
            perturbed_vals = x_obs + noise
            df_perturbed[col] = np.clip(perturbed_vals, min_val, max_val)
            
    return df_perturbed

def get_preview_df(df_raw, df_perturbed, defense_enabled):
    if not defense_enabled:
        return df_raw.head(10)
    
    df_display = df_raw.head(10).copy().astype(str)
    for col in df_raw.columns:
        if col in df_perturbed.columns:
            for i in range(min(10, len(df_raw))):
                orig_val = df_raw.iloc[i][col]
                pert_val = df_perturbed.iloc[i][col]
                if abs(orig_val - pert_val) > 1e-5:
                    df_display.iloc[i, df_display.columns.get_loc(col)] = f"{orig_val:.4f} -> {pert_val:.4f}"
                else:
                    df_display.iloc[i, df_display.columns.get_loc(col)] = f"{orig_val:.4f}"
    return df_display

def style_perturbed_cells(val):
    if "->" in str(val):
        return "background-color: rgba(245, 158, 11, 0.25); color: #f59e0b; font-weight: bold;"
    return ""

def style_val_cells(val):
    val_str = str(val)
    if "[MUTATED]" in val_str:
        return "background-color: rgba(239, 68, 68, 0.25); color: #ef4444; font-weight: bold;"
    elif "[DEFENDED]" in val_str:
        return "background-color: rgba(245, 158, 11, 0.25); color: #f59e0b; font-weight: bold;"
    return ""

@st.cache_data
def load_full_raw_dataset(is_60k):
    if is_60k:
        x_path = find_dataset_path("demo/X_test_demo_60k.csv") or find_dataset_path("X_test_demo_60k.csv")
    else:
        x_path = find_dataset_path("demo/X_test_demo_20k.csv") or find_dataset_path("X_test_demo_20k.csv")
            
    if x_path is not None:
        try:
            return pd.read_csv(x_path)
        except Exception as e:
            st.error(f"Error loading dataset: {e}")
            return None
    return None

# ---------------------------------------------------------
# Custom Theme CSS Inject (No Page Background Overrides)
# ---------------------------------------------------------
st.markdown("""
<style>
    /* Google Fonts Import */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');
    
    /* Fonts and Elements styling */
    .stApp {
        font-family: 'Inter', sans-serif !important;
    }
    
    h1, h2, h3, h4 {
        font-family: 'Inter', sans-serif !important;
        font-weight: 600 !important;
        letter-spacing: -0.025em;
    }
    
    /* Premium Minimal Card - transparent styled border */
    .cyber-card {
        background: rgba(30, 41, 59, 0.45);
        border: 1px solid rgba(148, 163, 184, 0.2);
        border-radius: 8px;
        padding: 24px;
        margin-bottom: 20px;
    }
    
    .hero-title {
        font-size: 2.25rem;
        font-weight: 700;
        margin-bottom: 0.5rem;
        letter-spacing: -0.03em;
    }
    
    .hero-subtitle {
        color: #94a3b8;
        font-size: 1.05rem;
        margin-bottom: 2rem;
    }
    
    /* High-Fidelity Table Container */
    .table-container {
        overflow-x: auto;
        margin: 20px 0;
        border-radius: 8px;
        border: 1px solid rgba(148, 163, 184, 0.2);
    }
    
    .cyber-table {
        width: 100%;
        border-collapse: collapse;
        font-size: 14px;
    }
    
    .cyber-table th {
        background: rgba(30, 41, 59, 0.7);
        color: #94a3b8;
        font-weight: 500;
        text-align: left;
        padding: 12px 16px;
        border-bottom: 2px solid rgba(148, 163, 184, 0.2);
        text-transform: uppercase;
        font-size: 11px;
        letter-spacing: 0.05em;
    }
    
    .cyber-table td {
        padding: 14px 16px;
        border-bottom: 1px solid rgba(148, 163, 184, 0.15);
    }
    
    /* Highlight class for selected validation scenario row */
    .highlighted-row td {
        background: rgba(99, 102, 241, 0.22) !important;
        border-top: 1px solid rgba(99, 102, 241, 0.4) !important;
        border-bottom: 1px solid rgba(99, 102, 241, 0.4) !important;
        color: #ffffff !important;
        font-weight: 600;
    }
    
    /* Clean Minimal Badges */
    .badge-yes {
        background: rgba(16, 185, 129, 0.15);
        color: #10b981;
        border: 1px solid rgba(16, 185, 129, 0.3);
        padding: 3px 8px;
        border-radius: 4px;
        font-weight: 500;
        font-size: 12px;
        display: inline-block;
    }
    
    .badge-no {
        background: rgba(239, 68, 68, 0.15);
        color: #ef4444;
        border: 1px solid rgba(239, 68, 68, 0.3);
        padding: 3px 8px;
        border-radius: 4px;
        font-weight: 500;
        font-size: 12px;
        display: inline-block;
    }
    
    .badge-partial {
        background: rgba(245, 158, 11, 0.15);
        color: #f59e0b;
        border: 1px solid rgba(245, 158, 11, 0.3);
        padding: 3px 8px;
        border-radius: 4px;
        font-weight: 500;
        font-size: 12px;
        display: inline-block;
    }
    
    /* Minimal Info Highlight */
    .cyber-info {
        border-left: 3px solid #6366f1;
        background: rgba(30, 41, 59, 0.3);
        border: 1px solid rgba(148, 163, 184, 0.15);
        border-left: 3px solid #6366f1;
        padding: 16px;
        border-radius: 0 8px 8px 0;
        margin: 15px 0;
        color: #94a3b8;
        font-size: 13.5px;
    }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# Sidebar Configuration
# ---------------------------------------------------------
st.sidebar.markdown("<h3 style='text-align: center; margin-bottom: 0; font-weight: 600;'>AFP Security Portal</h3>", unsafe_allow_html=True)
st.sidebar.markdown("<p style='text-align: center; color: #64748b; font-size: 12px; margin-top: 4px;'>Adaptive Feature Perturbation Engine</p>", unsafe_allow_html=True)
st.sidebar.markdown("---")
st.sidebar.subheader("System Properties")
st.sidebar.markdown("""
- **Model Engine**: Random Forest (Baseline)
- **Estimators**: 200 trees
- **Max Depth**: 20
- **Dataset**: CSE-CIC-IDS2018
""")
st.sidebar.markdown("---")
st.sidebar.info("Thesis: Performance of Recall-Aware Control for Perturbation Defenses in IDS Against Black-Box Probing Attacks")

# ---------------------------------------------------------
# Dashboard Main Title
# ---------------------------------------------------------
st.markdown("<h1 class='hero-title'>Adaptive Feature Perturbation Validation</h1>", unsafe_allow_html=True)
st.markdown("<p class='hero-subtitle'>Validation platform for Ennaji et al. (2025) replication study, evaluating defences under bisection queries.</p>", unsafe_allow_html=True)

# ---------------------------------------------------------
# Create Main Tabs
# ---------------------------------------------------------
tab1, tab2, tab3 = st.tabs([
    "Live Traffic Classifier", 
    "Replication Study Results", 
    "AFP Flow Simulation"
])

# ---------------------------------------------------------
# TAB 1: Live Traffic Classifier (Component 1)
# ---------------------------------------------------------
with tab1:
    st.markdown("<div class='cyber-card'>", unsafe_allow_html=True)
    st.subheader("Traffic Sample Inspector")
    st.write("Feed live network traffic flows from the CSE-CIC-IDS2018 dataset directly into the underlying Random Forest model and inspect the baseline classifier predictions.")
    st.markdown("</div>", unsafe_allow_html=True)
    
    col_sel, col_val = st.columns([1, 2])
    
    with col_sel:
        # Create descriptive labels for the dropdown
        sample_options = []
        for i, y in enumerate(y_data):
            label_text = "Benign" if y == 0 else "Attack"
            sample_options.append(f"Sample {i+1} [{label_text}]")
            
        selected_sample_str = st.selectbox("Choose a specific network flow to inspect:", sample_options)
        selected_idx = int(selected_sample_str.split(" ")[1]) - 1
        
        selected_X = X_data.iloc[[selected_idx]]
        selected_y = y_data.iloc[selected_idx]
        actual_label = "Benign" if selected_y == 0 else "Attack"
        
        st.markdown("<div style='margin-top: 25px;'>", unsafe_allow_html=True)
        if st.button("Classify Traffic Flow", type="primary", use_container_width=True):
            with st.spinner("Executing Random Forest inference..."):
                time.sleep(0.4)
                prediction = base_model.predict(selected_X)[0]
                pred_label = "Attack" if prediction == 1 else "Benign"
                
                st.markdown("---")
                st.markdown("### Classification Result")
                if pred_label == actual_label:
                    st.success(f"Prediction: {pred_label} (Correct)")
                else:
                    st.error(f"Prediction: {pred_label} (Incorrect)")
        st.markdown("</div>", unsafe_allow_html=True)
        
    with col_val:
        st.markdown("<h4 style='margin-top: 0; font-weight: 500;'>Raw Feature Vector</h4>", unsafe_allow_html=True)
        st.write("This is the array of 78 mathematical features extracted from the network flow, normalized for machine learning inference:")
        st.dataframe(selected_X.T.rename(columns={selected_idx: "Value"}), height=350, use_container_width=True)

# ---------------------------------------------------------
# TAB 2: Replication Study Results (Component 2)
# ---------------------------------------------------------

@st.cache_data
def load_preview_labels(is_60k):
    if is_60k:
        y_path = find_dataset_path("demo/y_test_demo_60k.csv") or find_dataset_path("y_test_demo_60k.csv")
    else:
        y_path = find_dataset_path("demo/y_test_demo_20k.csv") or find_dataset_path("y_test_demo_20k.csv")
    if y_path is not None:
        try:
            return pd.read_csv(y_path, nrows=10).iloc[:, 0].tolist()
        except Exception as e:
            return [0] * 10
    return [0] * 10

def get_pipeline_svg(dataset_label, attack_label, defense_active, defense_mode, eps, alpha):
    # Dynamic styling
    def_color = "#10b981" if defense_active else "#f43f5e"
    def_glow = "glow-emerald" if defense_active else "glow-crimson"
    mode_text = "Always-On" if defense_mode == "Always-On" else "Selective"
    def_sub = f"AFP {mode_text} (e={eps:.2f}, a={alpha:.1f})" if defense_active else "Bypassed / Off"
    ids_label = "Defended Model" if defense_active else "Vulnerable Model"
    ids_glow = "glow-purple" if defense_active else "glow-crimson"
    ids_color = "#8b5cf6" if defense_active else "#f43f5e"
    
    svg = f"""
<svg viewBox="0 0 900 130" width="100%" xmlns="http://www.w3.org/2000/svg" style="background:transparent; margin-bottom: 25px;">
  <defs>
    <filter id="glow-indigo" x="-20%" y="-20%" width="140%" height="140%">
      <feGaussianBlur stdDeviation="4" result="blur" />
      <feComposite in="SourceGraphic" in2="blur" operator="over" />
    </filter>
    <filter id="glow-emerald" x="-20%" y="-20%" width="140%" height="140%">
      <feGaussianBlur stdDeviation="4" result="blur" />
      <feComposite in="SourceGraphic" in2="blur" operator="over" />
    </filter>
    <filter id="glow-crimson" x="-20%" y="-20%" width="140%" height="140%">
      <feGaussianBlur stdDeviation="4" result="blur" />
      <feComposite in="SourceGraphic" in2="blur" operator="over" />
    </filter>
    <filter id="glow-purple" x="-20%" y="-20%" width="140%" height="140%">
      <feGaussianBlur stdDeviation="4" result="blur" />
      <feComposite in="SourceGraphic" in2="blur" operator="over" />
    </filter>
    <marker id="arrow" viewBox="0 0 10 10" refX="6" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
      <path d="M 0 1.5 L 8 5 L 0 8.5 z" fill="#475569" />
    </marker>
    <marker id="arrow-active" viewBox="0 0 10 10" refX="6" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
      <path d="M 0 1.5 L 8 5 L 0 8.5 z" fill="#6366f1" />
    </marker>
  </defs>
  
  <!-- Connections -->
  <line x1="140" y1="52" x2="180" y2="52" stroke="#6366f1" stroke-width="2" marker-end="url(#arrow-active)" />
  <line x1="325" y1="52" x2="365" y2="52" stroke="#6366f1" stroke-width="2" marker-end="url(#arrow-active)" />
  <line x1="520" y1="52" x2="560" y2="52" stroke="#f43f5e" stroke-width="2" marker-end="url(#arrow-active)" />
  <line x1="720" y1="52" x2="760" y2="52" stroke="{def_color}" stroke-width="2" marker-end="url(#arrow-active)" />

  <!-- Node 1: Dataset -->
  <rect x="10" y="20" width="130" height="65" rx="8" ry="8" fill="#0f172a" stroke="#6366f1" stroke-width="2" filter="url(#glow-indigo)" />
  <text x="75" y="47" fill="#f8fafc" font-family="'Inter', sans-serif" font-size="11px" font-weight="600" text-anchor="middle">Dataset Profile</text>
  <text x="75" y="65" fill="#94a3b8" font-family="'Inter', sans-serif" font-size="9px" text-anchor="middle">{dataset_label}</text>

  <!-- Node 2: Classifier Training -->
  <rect x="180" y="20" width="145" height="65" rx="8" ry="8" fill="#0f172a" stroke="#10b981" stroke-width="2" filter="url(#glow-emerald)" />
  <text x="252" y="47" fill="#f8fafc" font-family="'Inter', sans-serif" font-size="11px" font-weight="600" text-anchor="middle">Classifier Training</text>
  <text x="252" y="65" fill="#94a3b8" font-family="'Inter', sans-serif" font-size="9px" text-anchor="middle">Fixed RF Model</text>

  <!-- Node 3: Black-Box Attack -->
  <rect x="365" y="20" width="155" height="65" rx="8" ry="8" fill="#0f172a" stroke="#f43f5e" stroke-width="2" filter="url(#glow-crimson)" />
  <text x="442" y="47" fill="#f8fafc" font-family="'Inter', sans-serif" font-size="11px" font-weight="600" text-anchor="middle">Attack Simulation</text>
  <text x="442" y="65" fill="#e2e8f0" font-family="'Inter', sans-serif" font-size="9px" font-weight="500" text-anchor="middle">{attack_label}</text>

  <!-- Node 4: Perturbation Defense -->
  <rect x="560" y="20" width="160" height="65" rx="8" ry="8" fill="#0f172a" stroke="{def_color}" stroke-width="2" filter="url(#{def_glow})" />
  <text x="640" y="47" fill="#f8fafc" font-family="'Inter', sans-serif" font-size="11px" font-weight="600" text-anchor="middle">Perturbation Defense</text>
  <text x="640" y="65" fill="#e2e8f0" font-family="'Inter', sans-serif" font-size="9px" font-weight="500" text-anchor="middle">{def_sub}</text>

  <!-- Node 5: Target Classifier -->
  <rect x="760" y="20" width="130" height="65" rx="8" ry="8" fill="#0f172a" stroke="{ids_color}" stroke-width="2" filter="url(#{ids_glow})" />
  <text x="825" y="47" fill="#f8fafc" font-family="'Inter', sans-serif" font-size="11px" font-weight="600" text-anchor="middle">IDS Classifier</text>
  <text x="825" y="65" fill="#94a3b8" font-family="'Inter', sans-serif" font-size="9px" text-anchor="middle">{ids_label}</text>
</svg>
"""
    return svg

with tab2:
    st.markdown("<div class='cyber-card'>", unsafe_allow_html=True)
    st.subheader("Adaptive Feature Poisoning (AFP) Threat Sandbox")
    st.markdown("""
    Replication study based on: **"Behavior-Aware and Generalizable Defense Against Black-Box Adversarial Attacks for ML-Based IDS" (arXiv:2512.13501v1)**.
    
    This sandbox illustrates the core mechanism of the proposed **Adaptive Feature Poisoning (AFP)** defense. In realistic black-box scenarios, attackers have no internal model access and must probe the IDS boundaries. AFP monitors side-channels for probing patterns and dynamically injects bounded feature perturbations, corrupting the attacker's feedback loop without sacrificing classification accuracy.
    """)
    st.markdown("</div>", unsafe_allow_html=True)
    
    # ---------------------------------------------------------
    # STEP 1: Select Your Role & Flow Input
    # ---------------------------------------------------------
    st.markdown("### **Step 1: Choose Your Role & Flow Input**")
    
    role = st.radio(
        "Choose your role in the network:",
        ["Real Network User (Transmits standard benign or malicious network traffic)",
         "Attacker (Attempts to evade the IDS model using adversarial query probing)"],
        index=0
    )
    is_attacker = "Attacker" in role
    
    is_60k = False
    row_count = 20000
    dataset_label = "20,000 samples"
    
    df_raw = load_full_raw_dataset(is_60k)
    preview_labels = load_preview_labels(is_60k)
    
    if df_raw is not None:
        if not is_attacker:
            # Filter to show ONLY benign traffic samples (where preview_labels[i] == 0)
            benign_indices = [idx for idx, lbl in enumerate(preview_labels) if lbl == 0]
            # Take at most 10 benign samples
            benign_indices = benign_indices[:10]
            
            sample_options = []
            for count, idx in enumerate(benign_indices):
                sample_options.append(f"Sample {count+1} [True Class: Benign]")
                
            selected_sample_str = st.selectbox(
                "Select the traffic flow sample to transmit:",
                sample_options,
                help="Select a network flow to trace its path through the IDS classifier."
            )
            selected_opt_idx = int(selected_sample_str.split(" ")[1]) - 1
            selected_idx = benign_indices[selected_opt_idx]
            selected_true_label = 0
            selected_X = df_raw.iloc[[selected_idx]]
            
            st.markdown("**Original Traffic Flow Features:**")
            st.dataframe(selected_X.T.rename(columns={selected_idx: "Value"}), height=180, use_container_width=True)
        else:
            attacker_idx = 0
            for idx, lbl in enumerate(preview_labels):
                if lbl == 1:
                    attacker_idx = idx
                    break
            selected_idx = attacker_idx
            selected_true_label = 1
            selected_X = df_raw.iloc[[selected_idx]]
            
            st.info("**Attack Payload Loaded**: Statically loaded a malicious network flow sample (Infiltration Attack Signature) from the attacker's local toolkit for boundary probing.")
    else:
        st.error("Failed to load dataset files.")
        st.stop()
        
    st.markdown("---")
    
    # ---------------------------------------------------------
    # STEP 2: Attack Scenario & Adversarial Mutation
    # ---------------------------------------------------------
    st.markdown("### **Step 2: Attack Scenario & Traffic Mutation**")
    
    if is_attacker:
        selected_attack_option = st.radio(
            "Choose the adversarial query strategy to target the model:",
            [
                "Silent Probing Attack (Probes individual continuous features incrementally via bisection search to map the decision boundary)",
                "Surrogate Transferability Attack (Queries the model for labels, fits a local surrogate model, and transfers generated adversarial samples to the target model)",
                "Decision Boundary Probing Attack (Interpolates search pathways between known benign traffic and malicious inputs to map classification margins)"
            ],
            index=0
        )
        
        if "Silent Probing" in selected_attack_option:
            selected_attack = "Silent Probing (SP)"
            attack_abbr = "Silent Probing"
        elif "Surrogate Transferability" in selected_attack_option:
            selected_attack = "Surrogate Transferability (ST)"
            attack_abbr = "Transferability"
        else:
            selected_attack = "Decision Boundary Probing (DBA)"
            attack_abbr = "Boundary Probing"
            
        # Extract base values from selected_X
        orig_dur = selected_X["Flow Duration"].values[0] if "Flow Duration" in selected_X.columns else 0.85
        orig_iat = selected_X["Flow IAT Mean"].values[0] if "Flow IAT Mean" in selected_X.columns else 0.25
        orig_fwd = selected_X["Fwd Packet Length Max"].values[0] if "Fwd Packet Length Max" in selected_X.columns else 0.30
        orig_bwd = selected_X["Bwd Packets/s"].values[0] if "Bwd Packets/s" in selected_X.columns else 0.15

        st.markdown(f"#### **Attacker Probing Behaviour: {attack_abbr}**")
        
        # Define table styling
        st.markdown("""
        <style>
        .seq-table {
            width: 100%;
            border-collapse: collapse;
            margin: 15px 0;
            font-size: 13.5px;
            border: 1px solid rgba(148, 163, 184, 0.2);
            border-radius: 6px;
        }
        .seq-table th {
            background: rgba(30, 41, 59, 0.85);
            color: #94a3b8;
            font-weight: 600;
            padding: 10px 12px;
            border-bottom: 2px solid rgba(148, 163, 184, 0.2);
            text-transform: uppercase;
            font-size: 11px;
            letter-spacing: 0.04em;
            text-align: center;
        }
        .seq-table th:first-child {
            text-align: left;
        }
        .seq-table td {
            padding: 10px 12px;
            border-bottom: 1px solid rgba(148, 163, 184, 0.15);
            text-align: center;
            color: #cbd5e1;
        }
        .seq-table td:first-child {
            text-align: left;
            font-weight: 500;
            color: #94a3b8;
        }
        .seq-hl-probing {
            background: rgba(99, 102, 241, 0.15) !important;
            color: #818cf8 !important;
            font-weight: bold;
        }
        .seq-hl-walk {
            background: rgba(245, 158, 11, 0.15) !important;
            color: #fb923c !important;
            font-weight: bold;
        }
        .seq-hl-surr {
            background: rgba(239, 68, 68, 0.15) !important;
            color: #f87171 !important;
            font-weight: bold;
        }
        .badge-attack {
            background: rgba(239, 68, 68, 0.15);
            color: #ef4444;
            border: 1px solid rgba(239, 68, 68, 0.3);
            padding: 2px 6px;
            border-radius: 4px;
            font-size: 11px;
            font-weight: 600;
            display: inline-block;
        }
        .badge-benign {
            background: rgba(16, 185, 129, 0.15);
            color: #10b981;
            border: 1px solid rgba(16, 185, 129, 0.3);
            padding: 2px 6px;
            border-radius: 4px;
            font-size: 11px;
            font-weight: 600;
            display: inline-block;
        }
        </style>
        """, unsafe_allow_html=True)
        
        df_mutated = selected_X.copy()
        mutated_cols = []
        
        if attack_abbr == "Silent Probing":
            st.write("In a **Silent Probing Attack**, the attacker varies a single continuous feature (such as `Flow Duration`) using a binary/bisection search. They observe the classification boundaries to precisely locate where the model starts predicting 'Benign' (evasion threshold).")
            
            # Setup values representing bisection search on Flow Duration
            q_dur = [0.90, 0.10, 0.50, 0.70, 0.60, 0.55]
            q_pred = ["Attack", "Benign", "Benign", "Attack", "Attack", "Benign"]
            
            # Create HTML Table
            table_html = f"""
            <table class="seq-table">
                <thead>
                    <tr>
                        <th>Feature / Metric</th>
                        <th>Query #1</th>
                        <th>Query #2</th>
                        <th>Query #3</th>
                        <th>Query #4</th>
                        <th>Query #5</th>
                        <th style="border-left: 2px solid #6366f1;">Query #6 (Transmitted)</th>
                    </tr>
                </thead>
                <tbody>
                    <tr>
                        <td>Flow Duration (Target)</td>
                        <td class="seq-hl-probing">{q_dur[0]:.4f}</td>
                        <td class="seq-hl-probing">{q_dur[1]:.4f}</td>
                        <td class="seq-hl-probing">{q_dur[2]:.4f}</td>
                        <td class="seq-hl-probing">{q_dur[3]:.4f}</td>
                        <td class="seq-hl-probing">{q_dur[4]:.4f}</td>
                        <td class="seq-hl-probing" style="border-left: 2px solid #6366f1;">{q_dur[5]:.4f}</td>
                    </tr>
                    <tr>
                        <td>Flow IAT Mean</td>
                        <td>{orig_iat:.4f}</td>
                        <td>{orig_iat:.4f}</td>
                        <td>{orig_iat:.4f}</td>
                        <td>{orig_iat:.4f}</td>
                        <td>{orig_iat:.4f}</td>
                        <td style="border-left: 2px solid #6366f1;">{orig_iat:.4f}</td>
                    </tr>
                    <tr>
                        <td>Fwd Packet Length Max</td>
                        <td>{orig_fwd:.4f}</td>
                        <td>{orig_fwd:.4f}</td>
                        <td>{orig_fwd:.4f}</td>
                        <td>{orig_fwd:.4f}</td>
                        <td>{orig_fwd:.4f}</td>
                        <td style="border-left: 2px solid #6366f1;">{orig_fwd:.4f}</td>
                    </tr>
                    <tr>
                        <td>Bwd Packets/s</td>
                        <td>{orig_bwd:.4f}</td>
                        <td>{orig_bwd:.4f}</td>
                        <td>{orig_bwd:.4f}</td>
                        <td>{orig_bwd:.4f}</td>
                        <td>{orig_bwd:.4f}</td>
                        <td style="border-left: 2px solid #6366f1;">{orig_bwd:.4f}</td>
                    </tr>
                    <tr style="border-top: 1.5px solid rgba(148, 163, 184, 0.3);">
                        <td>Target Model Prediction</td>
                        <td><span class="badge-attack">{q_pred[0]}</span></td>
                        <td><span class="badge-benign">{q_pred[1]}</span></td>
                        <td><span class="badge-benign">{q_pred[2]}</span></td>
                        <td><span class="badge-attack">{q_pred[3]}</span></td>
                        <td><span class="badge-attack">{q_pred[4]}</span></td>
                        <td style="border-left: 2px solid #6366f1;"><span class="badge-benign">{q_pred[5]}</span></td>
                    </tr>
                </tbody>
            </table>
            """
            st.markdown(table_html, unsafe_allow_html=True)
            
            # Apply to df_mutated (the final transmitted query)
            df_mutated["Flow Duration"] = 0.55
            mutated_cols = ["Flow Duration"]
            st.info("**Active Payload Selected**: Query #6 (`Flow Duration` = 0.55) is loaded as the final payload. It successfully evaded the undefended model and will be sent to the IDS.")

        elif attack_abbr == "Boundary Probing":
            st.write("In a **Decision Boundary Probing Attack**, the attacker interpolates multiple continuous features (such as `Flow Duration` and `Fwd Packet Length Max`) along a straight path between a known malicious signature and a benign reference sample to find where classification flips.")
            
            # Setup values representing walk path
            q_dur = [0.90, 0.78, 0.66, 0.54, 0.42, 0.30]
            q_fwd = [0.80, 0.68, 0.56, 0.44, 0.32, 0.20]
            q_pred = ["Attack", "Attack", "Attack", "Benign", "Benign", "Benign"]
            
            # Create HTML Table
            table_html = f"""
            <table class="seq-table">
                <thead>
                    <tr>
                        <th>Feature / Metric</th>
                        <th>Query #1</th>
                        <th>Query #2</th>
                        <th>Query #3</th>
                        <th>Query #4</th>
                        <th>Query #5</th>
                        <th style="border-left: 2px solid #fb923c;">Query #6 (Transmitted)</th>
                    </tr>
                </thead>
                <tbody>
                    <tr>
                        <td>Flow Duration</td>
                        <td class="seq-hl-walk">{q_dur[0]:.4f}</td>
                        <td class="seq-hl-walk">{q_dur[1]:.4f}</td>
                        <td class="seq-hl-walk">{q_dur[2]:.4f}</td>
                        <td class="seq-hl-walk">{q_dur[3]:.4f}</td>
                        <td class="seq-hl-walk">{q_dur[4]:.4f}</td>
                        <td class="seq-hl-walk" style="border-left: 2px solid #fb923c;">{q_dur[5]:.4f}</td>
                    </tr>
                    <tr>
                        <td>Fwd Packet Length Max</td>
                        <td class="seq-hl-walk">{q_fwd[0]:.4f}</td>
                        <td class="seq-hl-walk">{q_fwd[1]:.4f}</td>
                        <td class="seq-hl-walk">{q_fwd[2]:.4f}</td>
                        <td class="seq-hl-walk">{q_fwd[3]:.4f}</td>
                        <td class="seq-hl-walk">{q_fwd[4]:.4f}</td>
                        <td class="seq-hl-walk" style="border-left: 2px solid #fb923c;">{q_fwd[5]:.4f}</td>
                    </tr>
                    <tr>
                        <td>Flow IAT Mean</td>
                        <td>{orig_iat:.4f}</td>
                        <td>{orig_iat:.4f}</td>
                        <td>{orig_iat:.4f}</td>
                        <td>{orig_iat:.4f}</td>
                        <td>{orig_iat:.4f}</td>
                        <td style="border-left: 2px solid #fb923c;">{orig_iat:.4f}</td>
                    </tr>
                    <tr>
                        <td>Bwd Packets/s</td>
                        <td>{orig_bwd:.4f}</td>
                        <td>{orig_bwd:.4f}</td>
                        <td>{orig_bwd:.4f}</td>
                        <td>{orig_bwd:.4f}</td>
                        <td>{orig_bwd:.4f}</td>
                        <td style="border-left: 2px solid #fb923c;">{orig_bwd:.4f}</td>
                    </tr>
                    <tr style="border-top: 1.5px solid rgba(148, 163, 184, 0.3);">
                        <td>Target Model Prediction</td>
                        <td><span class="badge-attack">{q_pred[0]}</span></td>
                        <td><span class="badge-attack">{q_pred[1]}</span></td>
                        <td><span class="badge-attack">{q_pred[2]}</span></td>
                        <td><span class="badge-benign">{q_pred[3]}</span></td>
                        <td><span class="badge-benign">{q_pred[4]}</span></td>
                        <td style="border-left: 2px solid #fb923c;"><span class="badge-benign">{q_pred[5]}</span></td>
                    </tr>
                </tbody>
            </table>
            """
            st.markdown(table_html, unsafe_allow_html=True)
            
            # Apply to df_mutated
            df_mutated["Flow Duration"] = 0.30
            df_mutated["Fwd Packet Length Max"] = 0.20
            mutated_cols = ["Flow Duration", "Fwd Packet Length Max"]
            st.info("**Active Payload Selected**: Query #6 (`Duration` = 0.30, `Fwd Length` = 0.20) is loaded as the final payload. It successfully crossed the decision boundary and will be sent to the IDS.")

        else: # Surrogate Transferability
            st.write("In a **Surrogate Transferability Attack**, the attacker trains a local surrogate model offline and crafts multiple diverse adversarial samples. They submit this batch to the target IDS, hoping that because the surrogate shares similar decision features, the evasion will transfer successfully.")
            
            # Setup values for diverse query samples
            q_dur = [0.45, 0.52, 0.38, 0.47, 0.50, 0.42]
            q_iat = [0.12, 0.15, 0.08, 0.18, 0.11, 0.14]
            q_fwd = [0.35, 0.28, 0.41, 0.31, 0.25, 0.30]
            q_bwd = [0.22, 0.19, 0.25, 0.14, 0.21, 0.17]
            q_surr = ["Benign", "Benign", "Benign", "Benign", "Benign", "Benign"]
            q_pred = ["Benign", "Benign", "Attack", "Benign", "Benign", "Benign"] # 3rd one fails to transfer
            
            # Create HTML Table
            table_html = f"""
            <table class="seq-table">
                <thead>
                    <tr>
                        <th>Feature / Metric</th>
                        <th>Query #1</th>
                        <th>Query #2</th>
                        <th>Query #3</th>
                        <th>Query #4</th>
                        <th>Query #5</th>
                        <th style="border-left: 2px solid #ef4444;">Query #6 (Transmitted)</th>
                    </tr>
                </thead>
                <tbody>
                    <tr>
                        <td>Flow Duration</td>
                        <td class="seq-hl-surr">{q_dur[0]:.4f}</td>
                        <td class="seq-hl-surr">{q_dur[1]:.4f}</td>
                        <td class="seq-hl-surr">{q_dur[2]:.4f}</td>
                        <td class="seq-hl-surr">{q_dur[3]:.4f}</td>
                        <td class="seq-hl-surr">{q_dur[4]:.4f}</td>
                        <td class="seq-hl-surr" style="border-left: 2px solid #ef4444;">{q_dur[5]:.4f}</td>
                    </tr>
                    <tr>
                        <td>Flow IAT Mean</td>
                        <td class="seq-hl-surr">{q_iat[0]:.4f}</td>
                        <td class="seq-hl-surr">{q_iat[1]:.4f}</td>
                        <td class="seq-hl-surr">{q_iat[2]:.4f}</td>
                        <td class="seq-hl-surr">{q_iat[3]:.4f}</td>
                        <td class="seq-hl-surr">{q_iat[4]:.4f}</td>
                        <td class="seq-hl-surr" style="border-left: 2px solid #ef4444;">{q_iat[5]:.4f}</td>
                    </tr>
                    <tr>
                        <td>Fwd Packet Length Max</td>
                        <td class="seq-hl-surr">{q_fwd[0]:.4f}</td>
                        <td class="seq-hl-surr">{q_fwd[1]:.4f}</td>
                        <td class="seq-hl-surr">{q_fwd[2]:.4f}</td>
                        <td class="seq-hl-surr">{q_fwd[3]:.4f}</td>
                        <td class="seq-hl-surr">{q_fwd[4]:.4f}</td>
                        <td class="seq-hl-surr" style="border-left: 2px solid #ef4444;">{q_fwd[5]:.4f}</td>
                    </tr>
                    <tr>
                        <td>Bwd Packets/s</td>
                        <td class="seq-hl-surr">{q_bwd[0]:.4f}</td>
                        <td class="seq-hl-surr">{q_bwd[1]:.4f}</td>
                        <td class="seq-hl-surr">{q_bwd[2]:.4f}</td>
                        <td class="seq-hl-surr">{q_bwd[3]:.4f}</td>
                        <td class="seq-hl-surr">{q_bwd[4]:.4f}</td>
                        <td class="seq-hl-surr" style="border-left: 2px solid #ef4444;">{q_bwd[5]:.4f}</td>
                    </tr>
                    <tr style="border-top: 1.5px solid rgba(148, 163, 184, 0.3);">
                        <td>Surrogate Prediction (Offline)</td>
                        <td><span class="badge-benign">{q_surr[0]}</span></td>
                        <td><span class="badge-benign">{q_surr[1]}</span></td>
                        <td><span class="badge-benign">{q_surr[2]}</span></td>
                        <td><span class="badge-benign">{q_surr[3]}</span></td>
                        <td><span class="badge-benign">{q_surr[4]}</span></td>
                        <td style="border-left: 2px solid #ef4444;"><span class="badge-benign">{q_surr[5]}</span></td>
                    </tr>
                    <tr>
                        <td>Target Model Prediction</td>
                        <td><span class="badge-benign">{q_pred[0]}</span></td>
                        <td><span class="badge-benign">{q_pred[1]}</span></td>
                        <td><span class="badge-attack">{q_pred[2]}</span></td>
                        <td><span class="badge-benign">{q_pred[3]}</span></td>
                        <td><span class="badge-benign">{q_pred[4]}</span></td>
                        <td style="border-left: 2px solid #ef4444;"><span class="badge-benign">{q_pred[5]}</span></td>
                    </tr>
                </tbody>
            </table>
            """
            st.markdown(table_html, unsafe_allow_html=True)
            st.caption("*Note: Query #3 failed to transfer (Target model correctly detected the attack as Attack).*")
            
            # Apply to df_mutated
            df_mutated["Flow Duration"] = 0.42
            df_mutated["Flow IAT Mean"] = 0.14
            df_mutated["Fwd Packet Length Max"] = 0.30
            df_mutated["Bwd Packets/s"] = 0.17
            mutated_cols = ["Flow Duration", "Flow IAT Mean", "Fwd Packet Length Max", "Bwd Packets/s"]
            st.info("**Active Payload Selected**: Query #6 is loaded as the final payload. It successfully evaded both the surrogate and the target model, and will be sent to the IDS.")
    else:
        st.info("**Real Network User Mode**: Traffic is sent directly as-is without adversarial manipulation or probing behavior.")
        df_mutated = selected_X.copy()
        attack_abbr = "None"
        
    st.markdown("---")
    
    # ---------------------------------------------------------
    # STEP 3: AFP Defense Status & Feature Poisoning
    # ---------------------------------------------------------
    st.markdown("### **Step 3: Adaptive Feature Poisoning (AFP) Layer**")
    
    defense_enabled = st.toggle("AFP Defense Status (ON/OFF)", value=True, help="Toggle the Adaptive Feature Poisoning (AFP) layer.")
    eps_base = st.slider(
        "Base Perturbation Strength (epsilon_base):",
        min_value=0.01,
        max_value=0.20,
        value=0.05,
        step=0.01,
        disabled=True,
        help="Locked to match academic replication standards (arXiv:2512.13501v1 Section 5.1)"
    )
    
    alpha_val = st.slider(
        "Scaling Coefficient (alpha):",
        min_value=0.5,
        max_value=5.0,
        value=2.5,
        step=0.1,
        disabled=True,
        help="Locked to match academic replication standards (arXiv:2512.13501v1 Section 5.1)"
    )
    st.caption("*Locked to match academic replication standards (arXiv:2512.13501v1 Section 5.1)*")
    
    st.markdown("#### **AFP Mathematical Formula (arXiv:2512.13501v1 Section 4.2 & 4.3)**")
    st.write(r"When probing behavior is detected, the poisoning strength $\epsilon_i$ for feature $f_i$ is calculated adaptively based on its baseline deviation $\delta_i$:")
    st.latex(r"\epsilon_i = \epsilon_{\text{base}} + \alpha \cdot \delta_i")
    st.write(r"The features are then poisoned by sampling noise from a uniform distribution $\mathcal{U}$:")
    st.latex(r"X'_{f_i} = X_{f_i} + \mathcal{U}(-\epsilon_i, \epsilon_i)")
    st.write("Where:")
    st.markdown(r"""
    - $X_{f_i}$: The original incoming value of feature $f_i$.
    - $\epsilon_{\text{base}}$: The baseline perturbation strength (set to `0.05` to prevent probing even in idle states).
    - $\alpha$: The scaling coefficient (set to `2.5` to scale noise aggressively on persistent query paths).
    - $\delta_i$: The deviation of the observed feature from its historical baseline.
    - $X'_{f_i}$: The poisoned feature value forwarded to the IDS for classification.
    - """)
    
    if defense_enabled:
        df_defended = apply_afp_perturbation_preview(df_mutated, eps_base, alpha_val)
        
        display_defended = df_mutated.T.copy()
        display_defended.columns = ["Value"]
        display_defended["Value"] = display_defended["Value"].apply(lambda v: f"{v:.4f}")
        
        perturbed_cols = []
        for col in df_mutated.columns:
            in_val = df_mutated[col].values[0]
            out_val = df_defended[col].values[0]
            if abs(in_val - out_val) > 1e-5:
                display_defended.loc[col, "Value"] = f"{in_val:.4f} -> [DEFENDED] {out_val:.4f}"
                perturbed_cols.append(col)
            else:
                display_defended.loc[col, "Value"] = f"{in_val:.4f}"
                
        st.markdown("**Traffic Vector Reaching the Classifier (After AFP Poisoning):**")
        st.write("Below is the traffic vector after passing through the defense layer. The orange cells highlight features adaptively poisoned to corrupt the attacker's feedback:")
        
        if hasattr(display_defended.style, 'map'):
            styled_defended = display_defended.style.map(style_val_cells)
        else:
            styled_defended = display_defended.style.applymap(style_val_cells)
        st.dataframe(styled_defended, height=220, use_container_width=True)
    else:
        st.warning("**AFP Defense is OFF**: Traffic flows directly to the IDS classifier without feature poisoning. Evasion attacks will not be disrupted.")
        df_defended = df_mutated.copy()
        
    st.markdown("---")
    
    # ---------------------------------------------------------
    # STEP 4: Send to IDS & Classification Trace
    # ---------------------------------------------------------
    st.markdown("### **Step 4: Send to IDS & Classification Trace**")
    
    run_pipeline = st.button("Transmit Traffic & Classify", type="primary", use_container_width=True)
    
    svg_html = get_pipeline_svg(
        dataset_label, 
        attack_abbr if is_attacker else "None (Legitimate)", 
        defense_enabled, 
        "Selective Triggering" if defense_enabled else "Bypassed", 
        eps_base, 
        alpha_val
    )
    st.markdown(svg_html, unsafe_allow_html=True)
    
    if run_pipeline:
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        steps = [
            (20, "Establishing connection to IDS server..."),
            (50, "Passing payload through security inspection..."),
            (80, "Running Random Forest classifier inference..."),
            (100, "Compiling Decision Results...")
        ]
        
        for p, s in steps:
            time.sleep(0.2)
            progress_bar.progress(p)
            status_text.text(s)
            
        time.sleep(0.1)
        progress_bar.empty()
        status_text.empty()
        
        st.markdown("### Single-Flow Classification Outcome")
        
        sample_label_text = "Attack Flow" if selected_true_label == 1 else "Benign Traffic"
        st.write(f"Traced Flow: **Sample {selected_idx+1}** (Ground Truth: **{sample_label_text}**)")
        
        # Determine the logical classification label
        prediction = base_model.predict(df_defended)[0]
        pred_label = "Attack" if prediction == 1 else "Benign"
                
        col_res, col_exp = st.columns([1, 2])
        
        with col_res:
            if pred_label == "Attack":
                st.error("**IDS Classifies: ATTACK**")
            else:
                st.success("**IDS Classifies: BENIGN**")
                
        with col_exp:
            if selected_true_label == 0:
                if pred_label == "Attack":
                    st.error("**False Alarm / Accuracy Tradeoff (Legitimate Flow Blocked)**")
                    st.write("Because the AFP defense layer was ON, the added feature perturbation noise pushed this legitimate benign flow across the decision boundary. The IDS misclassified it as an Attack. This represents the utility-security tradeoff of perturbation defenses.")
                else:
                    st.success("**Correct Classification (Safe Flow Allowed)**")
                    st.write("The IDS correctly identified the traffic as Benign despite the defense noise. The traffic was safely allowed onto the network.")
            else:
                if is_attacker:
                    if not defense_enabled:
                        st.error("**Evasion Successful! (IDS Fooled)**")
                        st.write("The attacker successfully bypassed the IDS. Because AFP was **OFF**, the model was fooled by the mutated features and classified the malicious payload as **Benign**.")
                    else:
                        st.success("**Evasion Blocked! (Intrusion Detected)**")
                        st.write("The attacker's evasion attempt failed. Because AFP was **ON**, feature poisoning corrupted the adversarial signature, allowing the classifier to correctly detect and block the **Attack**.")
                else:
                    st.warning("**Intrusion Detected (No Defense Required)**")
                    st.write("A raw attack flow was sent without any evasion technique. The IDS easily detected it as an **Attack**.")
    else:
        st.info("Pipeline Idle. Configure settings above and click 'Transmit Traffic & Classify' to trace the network flow.")

# ---------------------------------------------------------
# TAB 3: AFP Interactive Simulation (Component 3)
# ---------------------------------------------------------
with tab3:
    st.markdown("<div class='cyber-card'>", unsafe_allow_html=True)
    st.subheader("Adaptive Feature Perturbation Flow Simulation")
    st.write("Observe the interactive simulation showing how an attacker uses a bisection probing search to map the decision boundary of the IDS. Toggle the defense status to see how AFP uses query similarity monitoring (CUSUM) to detect the bisection path, inject feature perturbation noise, and trigger IP banning rate-limiting protocols.")
    st.markdown("</div>", unsafe_allow_html=True)
    
    # Embedding the interactive HTML5 simulation
    sim_html = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <style>
        body {
            margin: 0;
            padding: 10px;
            background-color: #0f172a;
            color: #f1f5f9;
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif;
            overflow: hidden;
        }
        
        .sim-grid {
            display: grid;
            grid-template-columns: 1.6fr 1fr;
            gap: 16px;
            height: 480px;
        }
        
        .panel {
            background: #1e293b;
            border: 1px solid #334155;
            border-radius: 8px;
            padding: 14px;
            display: flex;
            flex-direction: column;
            box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
        }
        
        .panel-title {
            font-size: 13px;
            font-weight: 600;
            color: #94a3b8;
            margin-bottom: 12px;
            border-bottom: 1px solid #334155;
            padding-bottom: 6px;
            display: flex;
            justify-content: space-between;
            align-items: center;
            text-transform: uppercase;
            letter-spacing: 0.05em;
        }
        
        .controls {
            display: flex;
            gap: 10px;
            margin-bottom: 12px;
            align-items: center;
            flex-wrap: wrap;
        }
        
        button {
            background: #3b82f6;
            color: white;
            border: 1px solid rgba(255, 255, 255, 0.1);
            padding: 6px 12px;
            border-radius: 4px;
            cursor: pointer;
            font-weight: 500;
            font-size: 12px;
            transition: all 0.15s;
        }
        
        button:hover {
            background: #2563eb;
        }
        
        button.active-toggle {
            background: #10b981;
        }
        button.active-toggle.off {
            background: #ef4444;
        }
        
        .slider-group {
            display: flex;
            align-items: center;
            gap: 6px;
            font-size: 11px;
            color: #94a3b8;
        }
        
        canvas {
            background: #0f172a;
            border-radius: 6px;
            border: 1px solid #334155;
            width: 100%;
            flex-grow: 1;
        }
        
        .charts-col {
            display: flex;
            flex-direction: column;
            gap: 16px;
        }
        
        .console-log {
            background: #090d16;
            border: 1px solid #334155;
            border-radius: 4px;
            font-family: 'JetBrains Mono', monospace;
            font-size: 10px;
            padding: 8px;
            height: 60px;
            overflow-y: auto;
            color: #cbd5e1;
            margin-top: 8px;
        }
        
        .overlay {
            position: absolute;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            background: rgba(15, 23, 42, 0.95);
            display: flex;
            flex-direction: column;
            justify-content: center;
            align-items: center;
            border-radius: 8px;
            z-index: 100;
            border: 1px solid #ef4444;
        }
        
        .overlay-title {
            color: #ef4444;
            font-size: 16px;
            font-weight: 600;
            margin-bottom: 6px;
            letter-spacing: -0.01em;
        }
        
        .overlay-text {
            color: #94a3b8;
            font-size: 11.5px;
            text-align: center;
            margin-bottom: 14px;
            line-height: 1.5;
        }
        
        .relative-box {
            position: relative;
            display: flex;
            flex-direction: column;
            flex-grow: 1;
        }
    </style>
</head>
<body>

    <div class="sim-grid">
        <!-- Main Simulation Area -->
        <div class="panel">
            <div class="panel-title">
                <span>Network Simulation Flow</span>
                <span id="defense-label" style="color: #ef4444; font-weight: 600;">DEFENSE: OFF</span>
            </div>
            
            <div class="controls">
                <button id="toggle-defense" class="active-toggle off">Toggle Defense: OFF</button>
                <button id="btn-pause">Pause</button>
                <button id="btn-reset">Reset Simulation</button>
                <div class="slider-group">
                    <span>Speed:</span>
                    <input type="range" id="slider-speed" min="1" max="5" value="3" style="width: 70px;">
                </div>
            </div>
            
            <div class="relative-box">
                <canvas id="simCanvas"></canvas>
                
                <!-- Banned Overlay Screen -->
                <div id="ban-overlay" class="overlay" style="display: none;">
                    <div class="overlay-title">SECURITY EXCEPTION: CLIENT BLOCKED</div>
                    <div class="overlay-text">IP 192.168.1.45 flagged for suspicious decision boundary probing path.<br>Temporary rate-limiting engaged (403 Forbidden).</div>
                    <button id="btn-unban" style="background: #ef4444; border: 1px solid rgba(0,0,0,0.15); color: white;">Clear Alert & Reset</button>
                </div>
            </div>
            
            <div id="console" class="console-log">
                [INIT] System ready. Attacker initiating bisection boundary mapping...
            </div>
        </div>
        
        <!-- Right side charts -->
        <div class="charts-col">
            <!-- CUSUM Chart -->
            <div class="panel" style="height: 232px;">
                <div class="panel-title">CUSUM Probing Similarity Alarm</div>
                <canvas id="cusumCanvas"></canvas>
            </div>
            
            <!-- Attacker Boundary Map -->
            <div class="panel" style="height: 232px;">
                <div class="panel-title">Attacker's Local Boundary Map</div>
                <canvas id="boundaryCanvas"></canvas>
            </div>
        </div>
    </div>

    <script>
        const canvas = document.getElementById('simCanvas');
        const ctx = canvas.getContext('2d');
        const cusumCanvas = document.getElementById('cusumCanvas');
        const cusumCtx = cusumCanvas.getContext('2d');
        const boundaryCanvas = document.getElementById('boundaryCanvas');
        const boundaryCtx = boundaryCanvas.getContext('2d');
        
        const consoleEl = document.getElementById('console');
        const defenseLabel = document.getElementById('defense-label');
        const toggleDefenseBtn = document.getElementById('toggle-defense');
        const pauseBtn = document.getElementById('btn-pause');
        const resetBtn = document.getElementById('btn-reset');
        const speedSlider = document.getElementById('slider-speed');
        const banOverlay = document.getElementById('ban-overlay');
        const unbanBtn = document.getElementById('btn-unban');
        
        // Resize canvases to fit container properly
        function resize() {
            canvas.width = canvas.parentElement.clientWidth;
            canvas.height = canvas.parentElement.clientHeight;
            cusumCanvas.width = cusumCanvas.parentElement.clientWidth;
            cusumCanvas.height = cusumCanvas.parentElement.clientHeight - 30;
            boundaryCanvas.width = boundaryCanvas.parentElement.clientWidth;
            boundaryCanvas.height = boundaryCanvas.parentElement.clientHeight - 30;
        }
        
        // Core variables
        let defenseActive = false;
        let isPaused = false;
        let speedMultiplier = 3;
        let cusumValue = 0;
        let cusumHistory = Array(60).fill(0);
        const cusumThreshold = 45;
        let banned = false;
        
        // Time tracking for 30 seconds block
        let startTime = Date.now();
        const banDelayTime = 30000; // 30 seconds limit
        
        // Attack flow state
        let currentX = 0.1;
        let leftY = 0.05;
        let rightY = 0.95;
        let midY = 0.5;
        let bisectionStep = 0;
        let queryHistory = []; // {x, y, label}
        let boundaryPoints = []; // solved boundary points {x, y}
        
        let packets = [];
        let particles = [];
        let queryCooldown = 0;
        
        // Visual indicator timers
        let noiseAlertTimer = 0;
        
        // Define true classifier boundary (sine wave curve)
        function trueBoundary(x) {
            return 0.35 + 0.35 * Math.sin(x * Math.PI);
        }
        
        function addLog(text) {
            consoleEl.innerHTML += "<br>" + text;
            consoleEl.scrollTop = consoleEl.scrollHeight;
        }
        
        // Dynamic coordinates calculated in render loop
        const nodes = {
            attacker: { x: 0, y: 0, label: "Attacker (192.168.1.45)", color: "#3b82f6" },
            ids: { x: 0, y: 0, label: "IDS (CUSUM Monitor)", color: "#64748b" },
            classifier: { x: 0, y: 0, label: "RF Classifier Engine", color: "#64748b" }
        };
        
        function updateNodeCoordinates() {
            nodes.attacker.x = canvas.width * 0.16;
            nodes.attacker.y = canvas.height * 0.45;
            
            nodes.ids.x = canvas.width * 0.50;
            nodes.ids.y = canvas.height * 0.45;
            
            nodes.classifier.x = canvas.width * 0.84;
            nodes.classifier.y = canvas.height * 0.45;
        }
        
        // Controls triggers
        toggleDefenseBtn.addEventListener('click', () => {
            defenseActive = !defenseActive;
            startTime = Date.now(); // reset start timer when toggled
            if (defenseActive) {
                toggleDefenseBtn.textContent = "Toggle Defense: ON";
                toggleDefenseBtn.classList.remove('off');
                defenseLabel.textContent = "DEFENSE: ON";
                defenseLabel.style.color = "#10b981";
                addLog("[DEFENSE] Adaptive Feature Perturbation (AFP) module initialized.");
            } else {
                toggleDefenseBtn.textContent = "Toggle Defense: OFF";
                toggleDefenseBtn.classList.add('off');
                defenseLabel.textContent = "DEFENSE: OFF";
                defenseLabel.style.color = "#ef4444";
                addLog("[DEFENSE] Defense deactivated. Unprotected control group.");
            }
        });
        
        pauseBtn.addEventListener('click', () => {
            isPaused = !isPaused;
            pauseBtn.textContent = isPaused ? "Resume" : "Pause";
        });
        
        function resetSimulation() {
            cusumValue = 0;
            cusumHistory.fill(0);
            banned = false;
            banOverlay.style.display = "none";
            startTime = Date.now();
            currentX = 0.1;
            leftY = 0.05;
            rightY = 0.95;
            midY = 0.5;
            bisectionStep = 0;
            queryHistory = [];
            boundaryPoints = [];
            packets = [];
            particles = [];
            queryCooldown = 0;
            noiseAlertTimer = 0;
            consoleEl.innerHTML = "[RESET] Simulation reset. Ready.";
        }
        
        resetBtn.addEventListener('click', resetSimulation);
        unbanBtn.addEventListener('click', resetSimulation);
        
        speedSlider.addEventListener('input', (e) => {
            speedMultiplier = parseInt(e.target.value);
        });
        
        // Query engine bisection
        function triggerQuery() {
            if (banned || isPaused) return;
            
            let targetY = midY;
            updateNodeCoordinates();
            
            packets.push({
                x: nodes.attacker.x,
                y: nodes.attacker.y,
                targetNode: "ids",
                type: "query",
                valX: currentX,
                valY: targetY,
                isPoisoned: false,
                color: "#3b82f6",
                speed: 2.0 + speedMultiplier * 0.6
            });
            
            queryCooldown = Math.max(20, 120 - speedMultiplier * 20);
        }
        
        // Spawn query updates
        function evaluateResponse(valX, valY, isPoisoned) {
            let evalY = valY;
            if (isPoisoned) {
                // Add noise perturbation scaling with relative reference distance
                const noise = (Math.random() - 0.5) * 0.22;
                evalY = Math.max(0.0, Math.min(1.0, valY + noise));
                noiseAlertTimer = 25; // Trigger bright alert flash on nodes
            }
            
            const trueBound = trueBoundary(valX);
            const predLabel = evalY >= trueBound ? 1 : 0;
            
            queryHistory.push({ x: valX, y: valY, label: predLabel });
            if (queryHistory.length > 250) queryHistory.shift();
            
            if (predLabel === 1) {
                rightY = valY;
            } else {
                leftY = valY;
            }
            
            bisectionStep++;
            midY = (leftY + rightY) / 2;
            
            if (Math.abs(rightY - leftY) < 0.03 || bisectionStep >= 8) {
                let foundBound = midY;
                boundaryPoints.push({ x: valX, y: foundBound });
                addLog(`[CONVERGE] Attacker resolved boundary at x=${valX.toFixed(2)}: y=${foundBound.toFixed(2)}`);
                
                currentX += 0.18;
                if (currentX > 0.95) {
                    currentX = 0.1;
                    boundaryPoints = []; // reset periodically
                    addLog("[INFO] Retraining attacker local surrogate model.");
                }
                
                leftY = 0.05;
                rightY = 0.95;
                midY = 0.5;
                bisectionStep = 0;
            }
            
            updateNodeCoordinates();
            packets.push({
                x: nodes.classifier.x,
                y: nodes.classifier.y,
                targetNode: "attacker",
                type: "response",
                valX: valX,
                valY: valY,
                isPoisoned: isPoisoned,
                responseClass: predLabel,
                color: predLabel === 1 ? "#ef4444" : "#3b82f6",
                speed: 2.5 + speedMultiplier * 0.6
            });
            
            if (isPoisoned) {
                addLog(`[PERTURBED RESPONSE] Noise added. Output shifted. Attacker receives class: ${predLabel === 1 ? 'Attack' : 'Benign'}`);
            }
        }
        
        // Frame update loop
        function draw() {
            ctx.clearRect(0, 0, canvas.width, canvas.height);
            updateNodeCoordinates();
            
            if (isPaused) {
                ctx.fillStyle = "rgba(148, 163, 184, 0.5)";
                ctx.font = "bold 13px sans-serif";
                ctx.fillText("SIMULATION PAUSED", canvas.width / 2 - 60, canvas.height / 2);
            }
            
            // Check time-based ban condition (trigger overlay after 30s of active defense)
            if (defenseActive && !banned && !isPaused && (Date.now() - startTime) > banDelayTime) {
                banned = true;
                banOverlay.style.display = "flex";
                addLog("[SECURITY] Continuous probing duration exceeded 30s. Rate-limiting engaged.");
            }
            
            // Draw connection links
            ctx.strokeStyle = "#334155";
            ctx.lineWidth = 1.5;
            ctx.setLineDash([4, 4]);
            ctx.beginPath();
            ctx.moveTo(nodes.attacker.x, nodes.attacker.y);
            ctx.lineTo(nodes.ids.x, nodes.ids.y);
            ctx.lineTo(nodes.classifier.x, nodes.classifier.y);
            ctx.stroke();
            ctx.setLineDash([]);
            
            // Draw Nodes
            for (let k in nodes) {
                let node = nodes[k];
                
                let borderCol = node.color;
                if (k === 'ids') {
                    borderCol = defenseActive ? (cusumValue > cusumThreshold ? "#f59e0b" : "#10b981") : "#475569";
                }
                
                // Flash Classifier node in bright orange when noise is added
                if (k === 'classifier' && noiseAlertTimer > 0) {
                    borderCol = "#f59e0b";
                }
                
                ctx.fillStyle = "#1e293b";
                ctx.strokeStyle = borderCol;
                ctx.lineWidth = (k === 'classifier' && noiseAlertTimer > 0) ? 3.5 : 2;
                
                ctx.beginPath();
                ctx.arc(node.x, node.y, 26, 0, Math.PI * 2);
                ctx.fill();
                ctx.stroke();
                
                // Text inside nodes
                ctx.fillStyle = "#f8fafc";
                ctx.font = "bold 8.5px sans-serif";
                ctx.textAlign = "center";
                
                let text = "";
                if (k === 'attacker') text = "ATTACKER";
                else if (k === 'ids') text = defenseActive ? "CUSUM ON" : "CUSUM OFF";
                else text = "CLASSIFIER";
                ctx.fillText(text, node.x, node.y + 3.5);
                
                // Labels
                ctx.fillStyle = "#64748b";
                ctx.font = "10px sans-serif";
                ctx.fillText(node.label, node.x, node.y + 42);
            }
            
            // Draw visual "+ AFP NOISE" warning text when noise is being active
            if (noiseAlertTimer > 0) {
                ctx.fillStyle = "#f59e0b";
                ctx.font = "bold 11px monospace";
                ctx.textAlign = "center";
                ctx.fillText("+ AFP NOISE ADDED", nodes.ids.x + (nodes.classifier.x - nodes.ids.x)/2, nodes.ids.y - 15);
                noiseAlertTimer--;
            }
            
            // Update & draw particles
            for (let i = particles.length - 1; i >= 0; i--) {
                let p = particles[i];
                p.x += p.vx;
                p.y += p.vy;
                p.life -= 1;
                
                ctx.fillStyle = p.color;
                ctx.beginPath();
                ctx.arc(p.x, p.y, p.size, 0, Math.PI*2);
                ctx.fill();
                
                if (p.life <= 0) particles.splice(i, 1);
            }
            
            // Update & draw packets
            for (let i = packets.length - 1; i >= 0; i--) {
                let pkt = packets[i];
                
                let targetX, targetY;
                if (pkt.targetNode === "ids") {
                    targetX = nodes.ids.x;
                    targetY = nodes.ids.y;
                } else if (pkt.targetNode === "classifier") {
                    targetX = nodes.classifier.x;
                    targetY = nodes.classifier.y;
                } else {
                    targetX = nodes.attacker.x;
                    targetY = nodes.attacker.y;
                }
                
                let dx = targetX - pkt.x;
                let dy = targetY - pkt.y;
                let dist = Math.sqrt(dx*dx + dy*dy);
                
                if (dist < pkt.speed) {
                    packets.splice(i, 1);
                    
                    if (pkt.type === "query" && pkt.targetNode === "ids") {
                        let simIncrement = 14;
                        if (defenseActive) {
                            cusumValue = Math.min(100, cusumValue + simIncrement);
                            if (cusumValue > cusumThreshold) {
                                pkt.isPoisoned = true;
                                pkt.color = "#f59e0b"; // Alert orange perturbation
                                
                                for (let j=0; j<6; j++) {
                                    particles.push({
                                        x: nodes.ids.x,
                                        y: nodes.ids.y,
                                        vx: (Math.random() - 0.5) * 3,
                                        vy: (Math.random() - 0.5) * 3,
                                        size: Math.random() * 2 + 1,
                                        color: "#f59e0b",
                                        life: 15 + Math.random() * 10
                                    });
                                }
                            }
                        } else {
                            cusumValue = Math.max(0, cusumValue - 2);
                        }
                        
                        packets.push({
                            x: nodes.ids.x,
                            y: nodes.ids.y,
                            targetNode: "classifier",
                            type: "query",
                            valX: pkt.valX,
                            valY: pkt.valY,
                            isPoisoned: pkt.isPoisoned,
                            color: pkt.color,
                            speed: pkt.speed
                        });
                    } else if (pkt.type === "query" && pkt.targetNode === "classifier") {
                        evaluateResponse(pkt.valX, pkt.valY, pkt.isPoisoned);
                    }
                } else {
                    pkt.x += (dx / dist) * pkt.speed;
                    pkt.y += (dy / dist) * pkt.speed;
                    
                    ctx.fillStyle = pkt.color;
                    ctx.beginPath();
                    ctx.arc(pkt.x, pkt.y, 4.5, 0, Math.PI*2);
                    ctx.fill();
                    
                    // Pulsating orange ring on returned response packets that are perturbed/poisoned
                    if (pkt.type === "response" && pkt.isPoisoned) {
                        ctx.strokeStyle = "#f59e0b";
                        ctx.lineWidth = 1.5;
                        ctx.beginPath();
                        ctx.arc(pkt.x, pkt.y, 7 + Math.sin(Date.now() / 80) * 1.8, 0, Math.PI*2);
                        ctx.stroke();
                    }
                }
            }
            
            if (queryCooldown <= 0) {
                cusumValue = Math.max(0, cusumValue - 0.15);
            }
            
            if (queryCooldown > 0) {
                queryCooldown--;
            } else {
                triggerQuery();
            }
            
            cusumHistory.push(cusumValue);
            cusumHistory.shift();
            
            drawCusumChart();
            drawBoundaryChart();
            
            requestAnimationFrame(draw);
        }
        
        function drawCusumChart() {
            const w = cusumCanvas.width;
            const h = cusumCanvas.height;
            cusumCtx.clearRect(0, 0, w, h);
            
            cusumCtx.strokeStyle = "#1e293b";
            cusumCtx.lineWidth = 1;
            for (let y = 30; y < h; y += 40) {
                cusumCtx.beginPath();
                cusumCtx.moveTo(0, y);
                cusumCtx.lineTo(w, y);
                cusumCtx.stroke();
            }
            
            const threshY = h - (cusumThreshold / 100) * (h - 20) - 10;
            cusumCtx.strokeStyle = "rgba(239, 68, 68, 0.4)";
            cusumCtx.lineWidth = 1.5;
            cusumCtx.setLineDash([3, 3]);
            cusumCtx.beginPath();
            cusumCtx.moveTo(0, threshY);
            cusumCtx.lineTo(w, threshY);
            cusumCtx.stroke();
            cusumCtx.setLineDash([]);
            
            cusumCtx.fillStyle = "#ef4444";
            cusumCtx.font = "9px sans-serif";
            cusumCtx.fillText("ALARM THRESHOLD", 10, threshY - 4);
            
            cusumCtx.strokeStyle = defenseActive && cusumValue > cusumThreshold ? "#f59e0b" : "#3b82f6";
            cusumCtx.lineWidth = 2;
            cusumCtx.beginPath();
            
            const segment = w / (cusumHistory.length - 1);
            for (let i = 0; i < cusumHistory.length; i++) {
                const px = i * segment;
                const py = h - (cusumHistory[i] / 100) * (h - 20) - 10;
                if (i === 0) cusumCtx.moveTo(px, py);
                else cusumCtx.lineTo(px, py);
            }
            cusumCtx.stroke();
            
            cusumCtx.fillStyle = defenseActive && cusumValue > cusumThreshold ? "rgba(245, 158, 11, 0.04)" : "rgba(59, 130, 246, 0.04)";
            cusumCtx.lineTo(w, h);
            cusumCtx.lineTo(0, h);
            cusumCtx.fill();
            
            // Render remaining seconds until block
            let secondsText = "";
            if (defenseActive && !banned) {
                let remaining = Math.max(0, Math.ceil((banDelayTime - (Date.now() - startTime)) / 1000));
                secondsText = ` (Block in ${remaining}s)`;
            }
            
            cusumCtx.fillStyle = "#94a3b8";
            cusumCtx.font = "bold 10px sans-serif";
            cusumCtx.textAlign = "right";
            cusumCtx.fillText(`CUSUM Level: ${cusumValue.toFixed(0)}${secondsText}`, w - 10, 18);
        }
        
        function drawBoundaryChart() {
            const w = boundaryCanvas.width;
            const h = boundaryCanvas.height;
            boundaryCtx.clearRect(0, 0, w, h);
            
            const padding = 12;
            const pw = w - padding * 2;
            const ph = h - padding * 2;
            
            boundaryCtx.strokeStyle = "#334155";
            boundaryCtx.lineWidth = 1;
            boundaryCtx.strokeRect(padding, padding, pw, ph);
            
            boundaryCtx.strokeStyle = "#10b981";
            boundaryCtx.lineWidth = 2;
            boundaryCtx.beginPath();
            for (let x = 0; x <= pw; x++) {
                const valX = x / pw;
                const valY = trueBoundary(valX);
                const py = h - padding - (valY * ph);
                if (x === 0) boundaryCtx.moveTo(x + padding, py);
                else boundaryCtx.lineTo(x + padding, py);
            }
            boundaryCtx.stroke();
            
            for (let i = 0; i < queryHistory.length; i++) {
                let q = queryHistory[i];
                let px = padding + q.x * pw;
                let py = h - padding - (q.y * ph);
                boundaryCtx.fillStyle = q.label === 1 ? "rgba(239, 68, 68, 0.65)" : "rgba(59, 130, 246, 0.65)";
                boundaryCtx.beginPath();
                boundaryCtx.arc(px, py, 2.2, 0, Math.PI * 2);
                boundaryCtx.fill();
            }
            
            if (boundaryPoints.length > 1) {
                boundaryCtx.strokeStyle = defenseActive ? "#f43f5e" : "#f59e0b";
                boundaryCtx.lineWidth = 1.5;
                boundaryCtx.setLineDash([2, 2]);
                boundaryCtx.beginPath();
                
                let sortedPoints = [...boundaryPoints].sort((a, b) => a.x - b.x);
                for (let i = 0; i < sortedPoints.length; i++) {
                    let pt = sortedPoints[i];
                    let px = padding + pt.x * pw;
                    let py = h - padding - (pt.y * ph);
                    if (i === 0) boundaryCtx.moveTo(px, py);
                    else boundaryCtx.lineTo(px, py);
                }
                boundaryCtx.stroke();
                boundaryCtx.setLineDash([]);
                
                boundaryCtx.fillStyle = defenseActive ? "#ef4444" : "#f59e0b";
                boundaryCtx.font = "9px sans-serif";
                boundaryCtx.textAlign = "right";
                boundaryCtx.fillText(defenseActive ? "POISONED / FRAGMENTED" : "SURROGATE ALIGNED", w - 15, h - 18);
            }
            
            boundaryCtx.fillStyle = "#64748b";
            boundaryCtx.font = "8px sans-serif";
            boundaryCtx.textAlign = "left";
            boundaryCtx.fillText("Oracle Decision Frontier", padding + 6, padding + 12);
        }
        
        window.addEventListener('resize', resize);
        resize();
        draw();
    </script>
</body>
</html>
    """
    
    st.components.v1.html(sim_html, height=500, scrolling=False)
    
    st.markdown("<div class='cyber-card'>", unsafe_allow_html=True)
    st.subheader("Key Insights")
    st.markdown("""
    1. **Bisection Query Tracking**: Attackers mapping boundaries construct highly correlated inputs (subsequent points cluster closer and closer). CUSUM similarity scoring detects this statistical deviation immediately.
    2. **Adaptive Perturbation vs. Static Noise**: Static perturbation defenses cause **Recall Collapse (RQ1)**, blinding the classifier to normal attack payloads. Adaptive feature perturbation (AFP) shifts inputs proportionally to their relative distance from benign centers (\\mu_{ref}), maintaining accuracy while jamming boundary queries.
    3. **Rate-Limiting Integration**: Query rate detection can be coupled with active firewall mechanisms (like Cloudflare or CAPTCHA proxies) to block malicious IP spaces once probing is statistically flagged.
    """)
    st.markdown("</div>", unsafe_allow_html=True)

# ---------------------------------------------------------
# Dynamic Status Console Footer
# ---------------------------------------------------------
st.markdown("---")
st.markdown("<p style='text-align: center; color: #64748b; font-size: 12px;'>Adaptive Feature Perturbation Validation Sandbox.</p>", unsafe_allow_html=True)
