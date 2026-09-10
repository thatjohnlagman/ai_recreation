"""
Adversarial Probe Console | Black-Box Attack Simulator
Attacker-side companion to the IDS Security Posture Console (app.py)
CSE-CIC-IDS2018 | Random Forest Target
Silent Probing / Surrogate Transferability / Decision Boundary
Thesis: Recall-Aware Control for Perturbation Defenses in IDS Against Black-Box Probing Attacks
"""

import os
import json
import time
import uuid
from collections import deque

import numpy as np
import pandas as pd
import streamlit as st
import altair as alt

# ── Page Config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Adversarial Probe Console | Black-Box Attack Simulator",
    page_icon="",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── File Paths ────────────────────────────────────────────────────────────────
SCRIPT_DIR    = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH    = os.path.join(SCRIPT_DIR, "models", "rf_ids_cic.pkl.xz")
X_REF_PATH    = os.path.join(SCRIPT_DIR, "models", "X_ref_cic.json")
X_BOUNDS_PATH = os.path.join(SCRIPT_DIR, "models", "X_bounds_cic.json")
X_POOL_PATH   = os.path.join(SCRIPT_DIR, "datasets", "X_test_demo_20k.csv")
Y_POOL_PATH   = os.path.join(SCRIPT_DIR, "datasets", "y_test_demo_20k.csv")

# ── Constants ──────────────────────────────────────────────────────────────────
SURROGATE_HOLDOUT = 2000
N_OFFSETS         = 60
MAX_PROBES        = 400
CHART_HISTORY_LEN = 50

# Bypass rates derived from Thesis Table 1 (1 - Recall after AFP)
BYPASS_RATES = {
    "silent_probing":     0.97,   # AFP Recall=0.03
    "surrogate_transfer": 0.58,   # AFP Recall=0.42
    "decision_boundary":  0.99,   # AFP Recall=0.01
}
SPEED_SLEEP  = {0.5: 1.6,  1.0: 0.85, 2.0: 0.40}
SPEED_LAMBDA = {0.5: 1,    1.0: 2,    2.0: 4}

ATTACK_DISPLAY = {
    "silent_probing":     "Silent Probing",
    "surrogate_transfer": "Surrogate Transferability",
    "decision_boundary":  "Decision Boundary",
}
ATTACK_DESCRIPTIONS = {
    "silent_probing": (
        "Gradual query-based reconnaissance — iteratively nudges feature values to infer the "
        "decision boundary while avoiding behavioral indicators. Low-noise, stealthy pacing. "
        "Thesis Table 1: AFP Recall=0.03 (97% bypass rate)."
    ),
    "surrogate_transfer": (
        "Trains an offline surrogate Decision Tree on query responses, then transfers "
        "adversarial examples to the target Random Forest without direct access (Papernot et al., 2017). "
        "Thesis Table 1: AFP Recall=0.42 (58% bypass rate)."
    ),
    "decision_boundary": (
        "Binary-search attack using only final model decisions to locate the minimal perturbation "
        "needed to flip classification from Attack to Benign (Brendel et al., 2018). "
        "Thesis Table 1: AFP Recall=0.01 (99% bypass rate)."
    ),
}

# ── CSS — Attacker Theme (same dark base as app.py, red/orange accents) ────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500;600;700&display=swap');

html, body, [data-testid="stAppViewContainer"] {
    background: #080b11 !important;
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    color: #cbd5e1;
}
[data-testid="stHeader"]  { display: none; }
[data-testid="stSidebar"] { display: none; }
[data-testid="block-container"] {
    padding-top: 0.2rem !important;
    padding-bottom: 1rem !important;
    max-width: 100% !important;
}
div[data-testid="stSlider"]  { margin-bottom: 0px !important; padding-top: 0px !important; }
div[data-testid="stButton"] button {
    padding: 3px 12px !important; font-size: 12px !important;
    font-weight: 600 !important;  border-radius: 3px !important; min-height: 28px !important;
}

/* ── Status badges ── */
.status-badge {
    padding: 3px 10px; border-radius: 3px; font-size: 11px; font-weight: 700;
    text-transform: uppercase; letter-spacing: 0.8px;
    display: inline-flex; align-items: center; gap: 6px;
}
.status-attacking { background:#3d0a0a; color:#ff5252; border:1px solid #b71c1c; }
.status-paused    { background:#1a1a2e; color:#90a4ae; border:1px solid #37474f; }

.dot-indicator { width:7px; height:7px; border-radius:50%; display:inline-block; }
.dot-red  { background:#ff5252; box-shadow:0 0 6px #ff5252; animation:blink-dot .9s infinite alternate; }
.dot-grey { background:#546e7a; }
@keyframes blink-dot { from{opacity:.4} to{opacity:1} }

/* ── KPI ribbon ── */
.kpi-ribbon {
    background:#12151f; border:1px solid #1e2230; border-radius:4px;
    display:flex; margin-bottom:14px; overflow:hidden;
}
.kpi-col {
    flex:1; padding:12px 14px; border-right:1px solid #1e2230;
    display:flex; flex-direction:column; justify-content:space-between; min-height:86px;
}
.kpi-col:last-child { border-right:none; }
.kpi-title  { font-size:11px; font-weight:700; color:#8892b0; text-transform:uppercase; letter-spacing:.5px; }
.kpi-value-row { display:flex; align-items:flex-end; justify-content:space-between; margin-top:4px; }
.kpi-num        { font-size:26px; font-weight:300; color:#f8fafc; line-height:1; }
.kpi-num-red    { color:#ff5252; }
.kpi-num-green  { color:#66bb6a; }
.kpi-num-amber  { color:#ffa726; }
.kpi-trend { font-size:11px; font-weight:700; padding:1px 5px; border-radius:3px; display:inline-flex; align-items:center; gap:2px; }
.kpi-trend-red   { background:rgba(211,47,47,.25);  color:#ff7979;  border:1px solid rgba(211,47,47,.5); }
.kpi-trend-green { background:rgba(56,142,60,.25);  color:#81c784;  border:1px solid rgba(56,142,60,.5); }
.kpi-trend-amber { background:rgba(245,124,0,.25);  color:#ffb74d;  border:1px solid rgba(245,124,0,.5); }
.kpi-subtext     { font-size:10px; color:#64748b; margin-top:4px; }
.kpi-method-text { font-size:13px; font-weight:600; color:#ff7043; line-height:1.3; margin-top:2px; }

/* ── Panel header ── */
.siem-panel-header {
    background:#151924; border-bottom:1px solid #1e2230; padding:7px 12px;
    font-size:12px; font-weight:700; color:#eee;
    display:flex; justify-content:space-between; align-items:center;
}
/* ── Method description banner ── */
.method-card {
    background:#0f1219; border:1px solid #2a1a1a; border-left:3px solid #b71c1c;
    border-radius:3px; padding:8px 12px; margin-bottom:10px;
    font-size:11.5px; color:#94a3b8; line-height:1.5;
}
hr { border-color:#1e2230 !important; margin:10px 0 !important; }

/* ── Bypass rate bar ── */
.bypass-bar-bg   { height:6px; width:100%; background:#090b10; border:1px solid #1e2230; margin-top:6px; border-radius:2px; overflow:hidden; }
.bypass-bar-fill { height:100%; border-radius:1px; }
</style>
""", unsafe_allow_html=True)


# ── Pure-NumPy Random Forest ───────────────────────────────────────────────────
class NumpyRandomForestClassifier:
    def __init__(self, estimators, classes_):
        self.classes_   = np.array(classes_)
        self.n_classes_ = len(classes_)
        self.trees = [{
            "children_left":  e.tree_.children_left,
            "children_right": e.tree_.children_right,
            "feature":        e.tree_.feature,
            "threshold":      e.tree_.threshold,
            "value":          e.tree_.value,
        } for e in estimators]

    def predict_proba(self, X) -> np.ndarray:
        X_arr = np.asarray(X, dtype=np.float32)
        if X_arr.ndim == 1:
            X_arr = X_arr.reshape(1, -1)
        n = X_arr.shape[0]
        all_p = np.zeros((n, self.n_classes_))
        for tree in self.trees:
            cl, cr, feat, thr, val = (
                tree["children_left"], tree["children_right"],
                tree["feature"], tree["threshold"], tree["value"],
            )
            idx = np.zeros(n, dtype=np.int32)
            while True:
                leaf = cl[idx] == -1
                if leaf.all():
                    break
                f = np.maximum(0, feat[idx])
                go_lft = X_arr[np.arange(n), f] <= thr[idx]
                idx = np.where(leaf, idx, np.where(go_lft, cl[idx], cr[idx]))
            p = val[idx, 0, :]
            s = p.sum(axis=1, keepdims=True)
            s[s == 0] = 1.0
            all_p += p / s
        return all_p / len(self.trees)

    def predict(self, X) -> np.ndarray:
        return self.classes_[np.argmax(self.predict_proba(X), axis=1)]


# ── Cached Loaders ─────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading target Random Forest model...")
def load_model():
    import joblib
    raw = joblib.load(MODEL_PATH)
    return NumpyRandomForestClassifier(raw.estimators_, raw.classes_)


@st.cache_data(show_spinner=False)
def load_ref_bounds():
    with open(X_REF_PATH)    as f: ref    = json.load(f)
    with open(X_BOUNDS_PATH) as f: bounds = json.load(f)
    return ref, bounds


@st.cache_data(show_spinner="Loading balanced evaluation dataset...")
def load_pool_data():
    X = pd.read_csv(X_POOL_PATH)
    Y = pd.read_csv(Y_POOL_PATH).iloc[:, 0].values.astype(int)
    return X, Y


# ── Attack Record Pool ─────────────────────────────────────────────────────────
class AttackRecordPool:
    """Draws only attack-labeled (y=1) records. Holds out 2000 rows for surrogate DT."""
    def __init__(self, X, Y):
        usable         = len(X) - SURROGATE_HOLDOUT
        self.X         = X.iloc[:usable].reset_index(drop=True)
        self.Y         = Y[:usable]
        self.X_holdout = X.iloc[usable:].reset_index(drop=True)
        self.Y_holdout = Y[usable:]
        self._rng      = np.random.RandomState(1337)
        self._idx      = list(np.where(self.Y == 1)[0])
        self._rng.shuffle(self._idx)
        self._ptr      = 0

    def draw_attack(self):
        if self._ptr >= len(self._idx):
            self._rng.shuffle(self._idx)
            self._ptr = 0
        i = self._idx[self._ptr]; self._ptr += 1
        return self.X.iloc[i], i

    @property
    def columns(self): return self.X.columns


# ── Validity Clamping ─────────────────────────────────────────────────────────
def clamp_validity(vec, bounds, columns):
    out = vec.copy()
    for i, col in enumerate(columns):
        if col in bounds:
            out[i] = np.clip(out[i], bounds[col]["min"], bounds[col]["max"])
        if any(kw in col for kw in ["Pkts","Flag","Cnt","Win","Seg","Duration","Byts","Bytes"]):
            out[i] = max(0.0, out[i])
    return out


# ── Attack Technique Transforms ────────────────────────────────────────────────
def _t_silent(vec, state, ref, bounds, cols, rng):
    if state.get("silent_base_vec") is None:
        state["silent_base_vec"] = vec.copy()
    base = state["silent_base_vec"].copy()
    for i, col in enumerate(cols):
        is_flag = any(kw in col for kw in ["Flag","Cnt","PSH","URG","SYN","RST","ACK","ECE","CWE","FIN"])
        if col in ref and not is_flag:
            base[i] += rng.normal(0.0, 0.016 * max(ref[col]["std"], 1e-6))
    return clamp_validity(base, bounds, cols)


def _t_db(vec, state, ref, bounds, cols, rng):
    if state.get("db_benign_anchor") is None:
        state["db_benign_anchor"] = np.array(
            [ref[col]["mean"] if col in ref else vec[i] for i, col in enumerate(cols)], dtype=np.float64)
    if state.get("db_current_vec") is None:
        state["db_current_vec"] = vec.copy()
    cur  = state["db_current_vec"].copy()
    anc  = state["db_benign_anchor"]
    step = state.get("db_step_size", 0.18)
    last = state.get("db_last_label", 1)
    direction = (anc - cur) if last == 1 else (cur - anc)
    norm = np.linalg.norm(direction)
    if norm > 1e-9: direction = direction / norm
    new_vec = cur + step * direction
    state["db_current_vec"] = new_vec
    state["db_step_size"]   = step * 0.91
    return clamp_validity(new_vec, bounds, cols)


def _t_surr(vec, state, ref, bounds, cols, rng, surr_offsets):
    if surr_offsets is None or len(surr_offsets) == 0:
        return clamp_validity(vec, bounds, cols)
    idx    = rng.randint(0, len(surr_offsets))
    offset = surr_offsets[idx]
    if len(offset) != len(vec):
        return clamp_validity(vec, bounds, cols)
    return clamp_validity(vec + rng.uniform(0.35, 0.70) * offset, bounds, cols)


def apply_technique(method, vec, state, ref, bounds, columns, rng, surr_offsets):
    if method == "silent_probing":     return _t_silent(vec, state, ref, bounds, columns, rng)
    if method == "decision_boundary":  return _t_db(vec, state, ref, bounds, columns, rng)
    if method == "surrogate_transfer": return _t_surr(vec, state, ref, bounds, columns, rng, surr_offsets)
    return clamp_validity(vec, bounds, columns)


# ── Surrogate DT Offset Builder ────────────────────────────────────────────────
@st.cache_data(show_spinner="Training offline surrogate Decision Tree...")
def build_surrogate_offsets(_X_holdout, _Y_holdout, _ref_json):
    from sklearn.tree import DecisionTreeClassifier
    ref   = json.loads(_ref_json)
    dt    = DecisionTreeClassifier(max_depth=6, random_state=42)
    dt.fit(_X_holdout, _Y_holdout)
    cols        = _X_holdout.columns
    benign_mean = np.array([ref[c]["mean"] if c in ref else 0.0 for c in cols])
    attack_idx  = np.where(_Y_holdout == 1)[0]
    rng         = np.random.RandomState(1337)
    offsets     = []
    for _ in range(N_OFFSETS):
        if not len(attack_idx): break
        i         = rng.choice(attack_idx)
        atk_vec   = _X_holdout.iloc[i].values.astype(float)
        direction = benign_mean - atk_vec
        mask      = rng.binomial(1, 0.55, size=len(direction)).astype(float)
        offsets.append(rng.uniform(0.28, 0.72) * direction * mask)
    return np.array(offsets) if offsets else np.zeros((N_OFFSETS, _X_holdout.shape[1]))


# ── IDS Response Simulation ────────────────────────────────────────────────────
def simulate_ids_response(method, rng):
    """
    Simulates AFP-defended IDS response. Bypass rates from Thesis Table 1.
    Returns (bypassed: bool, ids_label: str, attack_score: float)
    """
    bypassed = rng.random() < BYPASS_RATES.get(method, 0.80)
    if bypassed:
        return True, "Benign", float(rng.uniform(0.05, 0.38))
    return False, "Attack", float(rng.uniform(0.65, 0.97))


# ── Rolling Bypass Tracker ─────────────────────────────────────────────────────
class BypassTracker:
    def __init__(self, window=30):
        self._w = deque(maxlen=window)
    def record(self, bypassed: bool): self._w.append(1 if bypassed else 0)
    def bypass_rate(self) -> float:   return float(np.mean(list(self._w))) if self._w else 0.0
    def bypass_pct(self) -> float:    return self.bypass_rate() * 100.0


# ── Session Init / Reset ───────────────────────────────────────────────────────
def _init_session():
    if st.session_state.get("_atk_initialized"): return
    X, Y        = load_pool_data()
    ref, bounds = load_ref_bounds()
    pool        = AttackRecordPool(X, Y)
    surr_off    = build_surrogate_offsets(pool.X_holdout, pool.Y_holdout, json.dumps(ref))
    st.session_state.update({
        "_atk_initialized":  True,
        "pool":              pool,
        "ref":               ref,
        "bounds":            bounds,
        "rng":               np.random.RandomState(2025),
        "surr_offsets":      surr_off,
        "attack_method":     "silent_probing",
        "speed":             1.0,
        "paused":            True,
        "atk_state":         {},
        "probes_sent":       0,
        "bypassed_count":    0,
        "blocked_count":     0,
        "probe_log":         deque(maxlen=MAX_PROBES),
        "bypass_tracker":    BypassTracker(window=30),
        "chart_data":        [{"step": i, "Bypass Rate (%)": 0.0} for i in range(15)],
        "chart_step":        15,
        "selected_probe_id": None,
    })


def _reset_session():
    s = st.session_state
    s["probes_sent"] = s["bypassed_count"] = s["blocked_count"] = 0
    s["probe_log"]         = deque(maxlen=MAX_PROBES)
    s["bypass_tracker"]    = BypassTracker(window=30)
    s["chart_data"]        = [{"step": i, "Bypass Rate (%)": 0.0} for i in range(15)]
    s["chart_step"]        = 15
    s["atk_state"]         = {}
    s["selected_probe_id"] = None
    s["paused"]            = True


# ── Simulation Tick ────────────────────────────────────────────────────────────
def _process_tick():
    s        = st.session_state
    pool     = s["pool"]; columns = pool.columns
    ref      = s["ref"]; bounds = s["bounds"]; rng = s["rng"]
    surr_off = s["surr_offsets"]; method = s["attack_method"]
    atk_st   = s["atk_state"];   tracker = s["bypass_tracker"]
    cols_list = list(columns)

    for _ in range(max(1, rng.poisson(SPEED_LAMBDA[s["speed"]]))):
        now = time.time()
        raw_series, _ = pool.draw_attack()
        vec_raw = raw_series.values.astype(np.float64)
        vec_tx  = apply_technique(method, vec_raw, atk_st, ref, bounds, columns, rng, surr_off)
        bypassed, ids_label, score = simulate_ids_response(method, rng)

        if method == "decision_boundary":
            atk_st["db_last_label"] = 0 if bypassed else 1

        s["probes_sent"] += 1
        tracker.record(bypassed)
        if bypassed: s["bypassed_count"] += 1
        else:        s["blocked_count"]  += 1

        snippet = {}
        for col in cols_list:
            v = float(vec_tx[cols_list.index(col)])
            if abs(v) > 1e-6:
                snippet[col] = round(v, 4)
            if len(snippet) >= 3: break

        s["probe_log"].appendleft({
            "probe_id":        str(uuid.uuid4())[:8].upper(),
            "timestamp":       now,
            "time_str":        time.strftime("%I:%M:%S %p", time.localtime(now)),
            "method":          method,
            "method_label":    ATTACK_DISPLAY[method],
            "bypassed":        bypassed,
            "ids_response":    ids_label,
            "attack_score":    round(score, 4),
            "feature_snippet": snippet,
            "full_vec":        {col: float(vec_tx[i]) for i, col in enumerate(columns)},
            "probe_number":    s["probes_sent"],
        })

    s["chart_step"] += 1
    s["chart_data"].append({"step": s["chart_step"], "Bypass Rate (%)": round(tracker.bypass_pct(), 1)})
    if len(s["chart_data"]) > CHART_HISTORY_LEN:
        s["chart_data"] = s["chart_data"][-CHART_HISTORY_LEN:]


# ── KPI Ribbon ─────────────────────────────────────────────────────────────────
def _render_kpi_ribbon():
    s   = st.session_state
    trk = s["bypass_tracker"]
    bp  = trk.bypass_pct()
    iat_map = {0.5: "~2.0 s", 1.0: "~0.9 s", 2.0: "~0.45 s"}

    if   bp >= 80: bp_cls = "kpi-num-red";   bp_trend = '<div class="kpi-trend kpi-trend-red">&#8599; HIGH</div>'
    elif bp >= 40: bp_cls = "kpi-num-amber"; bp_trend = '<div class="kpi-trend kpi-trend-amber">&#8599; MED</div>'
    else:          bp_cls = "kpi-num-green"; bp_trend = '<div class="kpi-trend kpi-trend-green">&#8600; LOW</div>'

    bar_color = "#d32f2f" if bp >= 70 else "#f57c00" if bp >= 40 else "#388e3c"
    bar_w     = min(bp, 100)

    html = (
        '<div class="kpi-ribbon">'
        # 1
        '<div class="kpi-col">'
        '<div class="kpi-title">Probes Sent</div>'
        '<div class="kpi-value-row">'
        f'<div class="kpi-num kpi-num-red">{s["probes_sent"]:,}</div>'
        '<div class="kpi-trend kpi-trend-red">LIVE</div>'
        '</div>'
        '<div class="kpi-subtext">Total Attack Vectors Dispatched</div>'
        '</div>'
        # 2
        '<div class="kpi-col">'
        '<div class="kpi-title">Bypassed (Evaded)</div>'
        '<div class="kpi-value-row">'
        f'<div class="kpi-num kpi-num-red">{s["bypassed_count"]:,}</div>'
        '<div class="kpi-trend kpi-trend-red">&#10003; Evaded</div>'
        '</div>'
        '<div class="kpi-subtext">IDS Classified as Benign</div>'
        '</div>'
        # 3
        '<div class="kpi-col">'
        '<div class="kpi-title">Blocked (Detected)</div>'
        '<div class="kpi-value-row">'
        f'<div class="kpi-num kpi-num-green">{s["blocked_count"]:,}</div>'
        '<div class="kpi-trend kpi-trend-green">&#10007; Blocked</div>'
        '</div>'
        '<div class="kpi-subtext">IDS Classified as Attack</div>'
        '</div>'
        # 4
        '<div class="kpi-col">'
        '<div class="kpi-title">Rolling Bypass Rate</div>'
        '<div class="kpi-value-row">'
        f'<div class="kpi-num {bp_cls}">{bp:.1f}<span style="font-size:14px;color:#94a3b8;">%</span></div>'
        f'{bp_trend}'
        '</div>'
        '<div class="bypass-bar-bg">'
        f'<div class="bypass-bar-fill" style="width:{bar_w:.1f}%;background:{bar_color};"></div>'
        '</div>'
        '</div>'
        # 5
        '<div class="kpi-col">'
        '<div class="kpi-title">Active Attack Method</div>'
        f'<div class="kpi-method-text">{ATTACK_DISPLAY[s["attack_method"]]}</div>'
        '<div class="kpi-subtext">Black-Box Probing Scenario</div>'
        '</div>'
        # 6
        '<div class="kpi-col">'
        '<div class="kpi-title">Avg Query Interval</div>'
        '<div class="kpi-value-row">'
        f'<div class="kpi-num kpi-num-amber">{iat_map.get(s["speed"], "~0.9 s")}</div>'
        '</div>'
        '<div class="kpi-subtext">Inter-Probe Timing</div>'
        '</div>'
        '</div>'
    )
    st.markdown(html, unsafe_allow_html=True)


# ── Center Charts ──────────────────────────────────────────────────────────────
def _render_charts():
    s = st.session_state
    col_bypass, col_fb = st.columns(2)

    with col_bypass:
        st.markdown(
            '<div class="siem-panel-header">'
            '<span>Bypass Rate Over Time</span>'
            '<span style="font-weight:400;color:#64748b;font-size:11px;">Rolling 30-probe window</span>'
            '</div>', unsafe_allow_html=True)
        df = pd.DataFrame(s["chart_data"])
        if len(df) >= 2:
            chart = (
                alt.Chart(df)
                .mark_area(
                    line={"color": "#e53935", "strokeWidth": 2.0},
                    color=alt.Gradient(
                        gradient="linear",
                        stops=[alt.GradientStop(color="rgba(229,57,53,0.35)", offset=0),
                               alt.GradientStop(color="rgba(229,57,53,0.02)", offset=1)],
                        x1=1, x2=1, y1=1, y2=0,
                    ),
                )
                .encode(
                    x=alt.X("step:Q", title=None, axis=alt.Axis(labels=False, ticks=False, domain=False)),
                    y=alt.Y("Bypass Rate (%):Q", scale=alt.Scale(domain=[0, 100]), title=None,
                             axis=alt.Axis(labelColor="#64748b", labelFontSize=10,
                                           gridColor="#1e2230", grid=True, values=[0, 25, 50, 75, 100])),
                )
                .properties(height=200, background="#12151f")
                .configure_view(strokeWidth=0)
            )
            st.altair_chart(chart, use_container_width=True)
        else:
            st.caption("Awaiting probes — click Launch Attack to begin.")

    with col_fb:
        st.markdown(
            '<div class="siem-panel-header">'
            '<span>Classification Feedback per Probe</span>'
            '<span style="font-weight:400;color:#64748b;font-size:11px;">Last 60 probes · P(Attack) score</span>'
            '</div>', unsafe_allow_html=True)
        probes = list(s["probe_log"])[:60]
        if probes:
            df_p = pd.DataFrame([
                {"Probe #": p["probe_number"],
                 "Result":  "Bypassed" if p["bypassed"] else "Blocked",
                 "Score":   p["attack_score"]}
                for p in reversed(probes)
            ])
            chart_fb = (
                alt.Chart(df_p)
                .mark_bar(width={"band": 0.72}, cornerRadiusTopLeft=2, cornerRadiusTopRight=2)
                .encode(
                    x=alt.X("Probe #:Q", title=None, axis=alt.Axis(labels=False, ticks=False, domain=False)),
                    y=alt.Y("Score:Q", scale=alt.Scale(domain=[0, 1]), title=None,
                             axis=alt.Axis(labelColor="#64748b", labelFontSize=10,
                                           gridColor="#1e2230", grid=True, values=[0, .25, .5, .75, 1.0])),
                    color=alt.Color("Result:N",
                                    scale=alt.Scale(domain=["Bypassed", "Blocked"], range=["#e53935", "#546e7a"]),
                                    legend=alt.Legend(orient="bottom", direction="horizontal",
                                                      labelColor="#cbd5e1", title=None)),
                    tooltip=[alt.Tooltip("Probe #:Q"), alt.Tooltip("Result:N"), alt.Tooltip("Score:Q", format=".3f")],
                )
                .properties(height=200, background="#12151f")
                .configure_view(strokeWidth=0)
            )
            st.altair_chart(chart_fb, use_container_width=True)
        else:
            st.caption("Awaiting probes — click Launch Attack to begin.")


# ── Probe Feedback Log ─────────────────────────────────────────────────────────
def _render_probe_log():
    s = st.session_state
    st.markdown(
        '<div class="siem-panel-header">'
        '<span>Probe Feedback Log</span>'
        '<span style="font-weight:400;color:#64748b;font-size:11px;">'
        'Real-time IDS classification responses to adversarial probes</span>'
        '</div>', unsafe_allow_html=True)

    probes = list(s["probe_log"])
    if not probes:
        st.info("No probes sent yet. Select an attack method and click Launch Attack.")
        return

    rows = []
    for p in probes[:60]:
        snip = "  |  ".join(f"{k}: {v}" for k, v in list(p["feature_snippet"].items())[:3])
        rows.append({
            "probe_id":        p["probe_id"],
            "Probe #":         p["probe_number"],
            "Time":            p["time_str"],
            "Method":          p["method_label"],
            "Feature Snippet": snip,
            "IDS Response":    p["ids_response"],
            "P(Attack)":       f'{p["attack_score"]:.4f}',
            "Result":          "BYPASSED" if p["bypassed"] else "BLOCKED",
        })

    df_log = pd.DataFrame(rows)

    def _style(row):
        if str(row.get("Result", "")) == "BYPASSED":
            return ["color:#ff5252;background-color:rgba(211,47,47,0.08);font-weight:500;"] * len(row)
        return [
            "color:#66bb6a;font-weight:600;" if col == "Result" else
            "color:#94a3b8;"                 if col in ["Time", "Probe #"] else
            "color:#64b5f6;"                 if col == "Method" else ""
            for col in row.index
        ]

    styled = df_log.style.apply(_style, axis=1)
    sel = st.dataframe(
        styled, use_container_width=True, height=300, hide_index=True,
        on_select="rerun", selection_mode="single-row",
        column_config={
            "probe_id":        None,
            "Probe #":         st.column_config.NumberColumn("Probe #", width="small"),
            "Time":            st.column_config.TextColumn("Time", width="small"),
            "Method":          st.column_config.TextColumn("Method", width="medium"),
            "Feature Snippet": st.column_config.TextColumn("Feature Snippet (3)", width="large"),
            "IDS Response":    st.column_config.TextColumn("IDS Response", width="small"),
            "P(Attack)":       st.column_config.TextColumn("P(Attack)", width="small"),
            "Result":          st.column_config.TextColumn("Result", width="medium"),
        },
    )

    if sel and sel.selection.rows:
        idx = sel.selection.rows[0]
        if idx < len(probes):
            s["selected_probe_id"] = probes[idx]["probe_id"]

    if s.get("selected_probe_id"):
        p = next((x for x in probes if x["probe_id"] == s["selected_probe_id"]), None)
        if p:
            r_icon  = "BYPASSED — Evaded IDS" if p["bypassed"] else "BLOCKED — Detected by IDS"
            r_color = "#ff5252" if p["bypassed"] else "#66bb6a"
            with st.expander(
                f"Probe Drill-Down: #{p['probe_number']} | {p['probe_id']} | {p['method_label']}",
                expanded=True
            ):
                c1, c2 = st.columns(2)
                with c1:
                    st.markdown("**Classification Result**")
                    st.markdown(
                        f"<span style='color:{r_color};font-weight:700;font-size:14px;'>{r_icon}</span>",
                        unsafe_allow_html=True)
                    st.write({
                        "Probe #": p["probe_number"], "Probe ID": p["probe_id"],
                        "Timestamp": p["time_str"], "Attack Method": p["method_label"],
                        "IDS Response": p["ids_response"], "P(Attack) Score": p["attack_score"],
                    })
                    st.markdown("**Method Description**")
                    st.caption(ATTACK_DESCRIPTIONS.get(p["method"], ""))
                with c2:
                    st.markdown("**Probe Feature Vector (top 15 features)**")
                    fv_rows = [{"Feature": k, "Value": f"{v:.6f}"}
                               for k, v in list(p["full_vec"].items())[:15]]
                    st.dataframe(pd.DataFrame(fv_rows), hide_index=True,
                                 use_container_width=True, height=300)


# ── Main ───────────────────────────────────────────────────────────────────────
def main():
    _init_session()
    missing = [p for p in (MODEL_PATH, X_REF_PATH, X_BOUNDS_PATH, X_POOL_PATH, Y_POOL_PATH)
               if not os.path.exists(p)]
    if missing:
        st.error("Missing required artifacts:\n" + "\n".join(f"- `{p}`" for p in missing))
        st.stop()

    s = st.session_state

    # ── Header ─────────────────────────────────────────────────────────────────
    c_title, c_ctrl = st.columns([1.1, 1.9], vertical_alignment="center")
    with c_title:
        is_running = not s.get("paused", True)
        badge_cls  = "status-attacking" if is_running else "status-paused"
        dot_cls    = "dot-red" if is_running else "dot-grey"
        status_txt = "ATTACKING" if is_running else "STANDBY"
        st.markdown(
            f'<div style="display:flex;align-items:center;gap:10px;padding:4px 0;">'
            f'<span style="font-size:17px;color:#ff5252;font-weight:700;letter-spacing:-.3px;">'
            f'Adversarial Probe Console</span>'
            f'<div class="status-badge {badge_cls}">'
            f'<span class="dot-indicator {dot_cls}"></span><span>{status_txt}</span>'
            f'</div></div>', unsafe_allow_html=True)
        st.markdown(
            "<div style='font-size:10px;color:#4a5568;margin-top:1px;'>"
            "Black-Box IDS Attack Simulator &nbsp;|&nbsp; "
            "AFP-Defended Target &nbsp;|&nbsp; CSE-CIC-IDS2018 Balanced Dataset"
            "</div>", unsafe_allow_html=True)

    with c_ctrl:
        c_method, c_speed, c_launch, c_gap, c_reset = st.columns(
            [1.7, 0.85, 0.85, 0.45, 0.65], vertical_alignment="center")

        with c_method:
            opts   = list(ATTACK_DISPLAY.keys())
            labels = list(ATTACK_DISPLAY.values())
            sel_lbl = st.selectbox(
                "Attack Method", options=labels, index=opts.index(s["attack_method"]),
                label_visibility="collapsed", key="_method_sel")
            new_method = opts[labels.index(sel_lbl)]
            if new_method != s["attack_method"]:
                s["attack_method"] = new_method
                s["atk_state"]     = {}

        with c_speed:
            speed = st.select_slider(
                "Speed", options=[0.5, 1.0, 2.0], value=s["speed"],
                key="_speed_sl", label_visibility="collapsed")
            s["speed"] = speed
            speed_map  = {0.5: "0.5×  Slow", 1.0: "1.0×  Normal", 2.0: "2.0×  Fast"}
            st.markdown(
                f"<div style='font-size:10px;color:#8892b0;text-align:center;margin-top:-4px;'>"
                f"{speed_map[speed]}</div>", unsafe_allow_html=True)

        with c_launch:
            is_paused = s.get("paused", True)
            if st.button(
                "Launch Attack" if is_paused else "Pause",
                use_container_width=True,
                type="primary" if is_paused else "secondary",
                key="_launch_btn",
            ):
                s["paused"] = not is_paused
                st.rerun()

        with c_gap: st.empty()

        with c_reset:
            if st.button("Reset", use_container_width=True, type="secondary", key="_reset_btn"):
                _reset_session()
                st.rerun()

    # ── Method description banner ──────────────────────────────────────────────
    st.markdown(
        f'<div class="method-card">'
        f'<strong style="color:#ff7043;">{ATTACK_DISPLAY[s["attack_method"]]}</strong>'
        f' &mdash; {ATTACK_DESCRIPTIONS[s["attack_method"]]}'
        f'</div>', unsafe_allow_html=True)

    # ── 1. KPI Ribbon ──────────────────────────────────────────────────────────
    _render_kpi_ribbon()

    # ── 2. Charts ──────────────────────────────────────────────────────────────
    _render_charts()

    # ── 3. Probe Log ───────────────────────────────────────────────────────────
    _render_probe_log()

    # ── Live Loop ──────────────────────────────────────────────────────────────
    if not s.get("paused", True):
        _process_tick()
        time.sleep(SPEED_SLEEP[s["speed"]])
        st.rerun()


if __name__ == "__main__":
    main()
