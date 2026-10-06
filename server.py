"""
IDS + Recall-Aware Research Runtime Platform & Simulated Protected Server
Connects:
- Pretrained 78-feature Random Forest Classifier (frozen_rf.joblib, CSE-CIC-IDS2018)
- Real Defenses: Adaptive Feature Poisoning (AFP), Randomized Smoothing (RS), Feature Squeezing (FS)
- Recall-Aware Controller (C1 configuration: rolling recall window, atomic batch updates)
- Inline IDS Guarded Protected Server Endpoints (/api/server/data)
- High-Performance Cyber SOC Web Dashboard (REST + WebSockets)
"""

import os
import sys
import json
import time
import math
import hashlib
import asyncio
import warnings
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, Any, List
from collections import deque

import numpy as np
import pandas as pd
import joblib
import yaml
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Request, Response, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Filter scikit-learn version mismatch warnings on deserialization
warnings.filterwarnings("ignore", category=UserWarning)

# -----------------------------------------------------------------------------
# Path Resolution
# -----------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
RUNTIME_DIR = BASE_DIR / "runtime_package"
FRONTEND_DIR = BASE_DIR / "frontend"

if str(RUNTIME_DIR) not in sys.path:
    sys.path.insert(0, str(RUNTIME_DIR))

# Import Research Runtime Modules
from controller.recall_controller import RecallAwareController
from defenses.afp import AdaptiveFeaturePoisoning
from defenses.randomized_smoothing import RandomizedSmoothing
from defenses.feature_squeezing import FeatureSqueezing
from runtime_package.data_loader import get_dataset, LoadedDataset

# -----------------------------------------------------------------------------
# Operator Authorization Configuration & Verification
# -----------------------------------------------------------------------------
DEFAULT_OPERATOR_TOKEN = "ids-operator-secret-2026"
TOKEN_FILE = BASE_DIR / "operator_token.txt"

def get_configured_operator_token() -> str:
    """Resolves authorized operator token from env var, local token file, or default demo secret."""
    env_token = os.environ.get("IDS_OPERATOR_TOKEN")
    if env_token and env_token.strip():
        return env_token.strip()
    if TOKEN_FILE.exists():
        try:
            content = TOKEN_FILE.read_text(encoding="utf-8").strip()
            if content:
                return content
        except Exception:
            pass
    return DEFAULT_OPERATOR_TOKEN

def verify_operator_authorization(request: Request) -> bool:
    """
    Verifies that incoming management request includes a valid operator token.
    Accepts token via 'X-Operator-Token' header or 'Authorization: Bearer <token>'.
    """
    expected = get_configured_operator_token()
    token = request.headers.get("X-Operator-Token")
    if not token:
        auth_header = request.headers.get("Authorization", "")
        if auth_header.startswith("Bearer "):
            token = auth_header[7:].strip()
    return bool(token and token == expected)

# -----------------------------------------------------------------------------
# Canonical Attack Procedure Validation
# Active demonstration permits only the 3 canonical study procedures or None.
# -----------------------------------------------------------------------------
CANONICAL_ATTACK_PROCEDURES = {
    "silent probing": "Silent Probing",
    "silent": "Silent Probing",
    "silent_probing": "Silent Probing",
    "surrogate transferability": "Surrogate Transferability",
    "surrogate": "Surrogate Transferability",
    "surrogate_transfer": "Surrogate Transferability",
    "surrogate_transferability": "Surrogate Transferability",
    "decision-boundary attack": "Decision-Boundary Attack",
    "decision boundary attack": "Decision-Boundary Attack",
    "decision_boundary_attack": "Decision-Boundary Attack",
    "decision_boundary": "Decision-Boundary Attack",
    "boundary": "Decision-Boundary Attack",
    "boundary_attack": "Decision-Boundary Attack",
}

def validate_attack_procedure(scenario: Optional[str]) -> Optional[str]:
    """
    Validates attack procedure name.
    Permits only the three canonical study procedures, their deliberate aliases, or absent/None.
    Rejects unsupported procedure names with ValueError.
    """
    if scenario is None:
        return None
    s = str(scenario).strip()
    if s.lower() in ["", "none"]:
        return None
    canonical = CANONICAL_ATTACK_PROCEDURES.get(s.lower())
    if canonical is None:
        raise ValueError(
            f"Unsupported attack procedure '{s}'. Active demonstration supports only the three canonical "
            f"study procedures: 'Silent Probing', 'Surrogate Transferability', 'Decision-Boundary Attack' (or None)."
        )
    return canonical

# -----------------------------------------------------------------------------
# Security & Defense Engine
# -----------------------------------------------------------------------------
class SecurityEngine:
    def __init__(self):
        self.model = None
        self.feature_names: List[str] = []
        self.modifiable_mask: np.ndarray = np.array([])
        self.bounds_df = None
        self.afp_profile_df = None
        self.X_demo: Optional[pd.DataFrame] = None
        self.meta_demo: Optional[pd.DataFrame] = None

        # Defenses and Configs
        self.defenses: Dict[str, Any] = {}
        self.def_configs: Dict[str, Any] = {}
        self.ctrl_configs: Dict[str, Any] = {}

        # Active Defense State
        self.active_defense_name: str = "afp"  # "afp", "rs", "fs", "none"
        self.controller_mode: str = "recall-aware"  # "recall-aware" or "base"
        self.controller: Optional[RecallAwareController] = None
        self.controller_state: str = "Green"
        self.current_intensity: float = 0.0003
        self.intensity_min: float = 0.0
        self.intensity_max: float = 0.0003
        self.afp_alpha: float = 0.5
        self.batch_id: int = 0
        self.batch_tp: int = 0
        self.batch_fn: int = 0
        self.batch_size: int = 5

        # Simulation Session State
        self.active_attack_scenario: str = "None"
        self.active_simulation_session: Optional[str] = None

        self.data_profile: str = os.environ.get("IDS_DATA_PROFILE", "expanded").strip().lower()
        self.dataset: Optional[LoadedDataset] = None
        self.query_count: int = 0

        # Cumulative Metrics (Honest cold start: initialized to zero / session baseline)
        self.total_traffic: int = 0
        self.detected_attacks: int = 0
        self.tp: int = 0
        self.fp: int = 0
        self.fn: int = 0
        self.tn: int = 0
        self.recall: Optional[float] = None
        self.fpr: Optional[float] = None

        # Sliding window for live timeline tracking (empty on cold start)
        self.history_labels = deque([], maxlen=20)
        self.history_recall = deque([], maxlen=20)
        self.history_intensity = deque([], maxlen=20)

        # Threat locations on map (Empty on cold start - no fabricated coordinates)
        self.threat_locations: List[Dict[str, Any]] = []

        self.top_threat_ips: List[Dict[str, Any]] = []

        self.recent_feed: deque = deque([], maxlen=50)
        self.recent_attacks: deque = deque([], maxlen=50)

    def load_resources(self):
        """Loads models, feature schemas, training bounds, and defenses from runtime_package."""
        m_dir = RUNTIME_DIR / "model"
        c_dir = RUNTIME_DIR / "configs"

        print(f"[Engine] Loading frozen Random Forest model from {m_dir / 'frozen_rf.joblib'}...")
        self.model = joblib.load(m_dir / "frozen_rf.joblib")

        with open(m_dir / "feature_names.json", "r") as f:
            self.feature_names = json.load(f)

        with open(m_dir / "feature_mask.json", "r") as f:
            mask_dict = json.load(f)
            self.modifiable_mask = np.array(list(mask_dict.values())[0], dtype=bool)

        self.bounds_df = pd.read_parquet(m_dir / "training_bounds.parquet")
        self.afp_profile_df = pd.read_parquet(m_dir / "afp_benign_profile.parquet")

        with open(c_dir / "defenses.yaml", "r") as f:
            self.def_configs = yaml.safe_load(f)
        with open(c_dir / "controllers.yaml", "r") as f:
            self.ctrl_configs = yaml.safe_load(f)

        # Initialize Defenses
        self.defenses["afp"] = AdaptiveFeaturePoisoning(
            self.feature_names, self.modifiable_mask, self.bounds_df, self.afp_profile_df
        )
        self.defenses["rs"] = RandomizedSmoothing(
            self.feature_names, self.modifiable_mask, self.bounds_df, ensemble_size=11
        )
        self.defenses["fs"] = FeatureSqueezing(
            self.feature_names, self.modifiable_mask, self.bounds_df
        )

        # Load Dataset via Profile Loader
        self.dataset = get_dataset(self.data_profile)
        self.X_demo = self.dataset.X
        self.meta_demo = self.dataset.metadata
        print(f"[Engine] Loaded data profile '{self.data_profile}': {len(self.feature_names)} features, 3 defenses, and {self.dataset.total_rows:,} samples ({self.dataset.measurement_rows:,} measurement, {self.dataset.crafting_rows:,} crafting).")

        # Initialize default controller
        self.set_defense(self.active_defense_name)
        self.set_mode(self.controller_mode)

    def set_defense(self, defense_name: str):
        """Sets the active defense mechanism ('afp', 'rs', 'fs', 'none')."""
        if defense_name not in ["afp", "rs", "fs", "none"]:
            raise ValueError(f"Unknown defense: {defense_name}")

        self.active_defense_name = defense_name
        if defense_name == "none":
            self.controller = None
            self.current_intensity = 0.0
            self.controller_state = "Bypassed"
            return

        def_key = "afp"
        if defense_name == "rs":
            def_key = "randomized_smoothing"
        elif defense_name == "fs":
            def_key = "feature_squeezing"

        d_cfg = self.def_configs[def_key]
        c1_cfg = self.ctrl_configs.get("controller_configurations", {}).get("C1", {})
        if not c1_cfg:
            c1_cfg = {"id": "C1", "window_size": 5, "Rcritical": 0.85, "Rmin": 0.95, "fast_decay": 0.40, "slow_decay": 0.90, "growth_factor": 1.05}

        self.intensity_min = float(d_cfg.get("intensity_min", 0.0))
        self.intensity_max = float(d_cfg.get("intensity_max", 0.0003))

        self.controller = RecallAwareController(c1_cfg, d_cfg, def_key)
        self.batch_id = 0
        self.batch_tp = 0
        self.batch_fn = 0
        self.controller_state = "Green" if self.controller_mode == "recall-aware" else "Base"

        if self.controller_mode == "recall-aware":
            decision = self.controller.get_intensity(self.batch_id)
            self.current_intensity = decision.intensity
        else:
            self.current_intensity = self.controller.base_intensity

    def set_mode(self, mode: str):
        """Toggles between 'recall-aware' (dynamic feedback) and 'base' (static intensity)."""
        if mode not in ["recall-aware", "base"]:
            raise ValueError(f"Unknown mode: {mode}")

        self.controller_mode = mode
        if self.active_defense_name == "none":
            self.controller_state = "Bypassed"
            return

        if self.controller is not None:
            self.controller.reset()
            self.batch_id = 0
            self.batch_tp = 0
            self.batch_fn = 0

            if mode == "recall-aware":
                decision = self.controller.get_intensity(self.batch_id)
                self.current_intensity = decision.intensity
                self.controller_state = "Green"
            else:
                self.current_intensity = self.controller.base_intensity
                self.controller_state = "Base"

    def update_metrics_and_controller(self, ground_truth: Optional[int], predicted: int, used_intensity: float):
        """Updates confusion matrix counters (if ground truth known) and triggers batch-level controller updates."""
        if ground_truth is not None:
            if ground_truth == 1:
                if predicted == 1:
                    self.tp += 1
                    self.batch_tp += 1
                else:
                    self.fn += 1
                    self.batch_fn += 1
            else:
                if predicted == 1:
                    self.fp += 1
                else:
                    self.tn += 1

            # Real detection recall and false positive rate
            total_positives = self.tp + self.fn
            if total_positives > 0:
                self.recall = round(float(self.tp) / float(total_positives), 3)
            else:
                self.recall = None

            total_negatives = self.tn + self.fp
            if total_negatives > 0:
                self.fpr = round(float(self.fp) / float(total_negatives), 3)
            else:
                self.fpr = None

            # Batch-level Controller Update (Atomically advances when batch completes)
            if self.active_defense_name != "none" and self.controller is not None:
                if self.controller_mode == "recall-aware":
                    # Evaluate when batch accumulates batch_size attack decisions
                    if (self.batch_tp + self.batch_fn) >= self.batch_size:
                        try:
                            update = self.controller.submit_observations(self.batch_id, self.batch_tp, self.batch_fn)
                            self.controller_state = update.state
                            self.current_intensity = update.clipped_next_intensity
                            self.batch_id += 1
                            self.batch_tp = 0
                            self.batch_fn = 0
                            # Pre-request intensity for next batch
                            self.controller.get_intensity(self.batch_id)
                        except Exception as e:
                            print(f"[Engine] Controller update error: {e}")

        # Update timeline history
        now_str = datetime.now().strftime("%H:%M:%S")
        self.history_labels.append(now_str)
        self.history_recall.append(self.recall if self.recall is not None else 0.0)
        self.history_intensity.append(used_intensity)

    def record_attack_ip(self, ip: str, location: Optional[str] = None):
        """Updates top threat IP counts."""
        loc_str = location if location else resolve_ip_location(ip)

        # Update Top Threat IPs
        found = False
        for entry in self.top_threat_ips:
            if entry["ip"] == ip:
                entry["attacks"] += 1
                found = True
                break
        if not found:
            self.top_threat_ips.append({"ip": ip, "location": loc_str, "attacks": 1})
        self.top_threat_ips.sort(key=lambda x: x["attacks"], reverse=True)
        self.top_threat_ips = self.top_threat_ips[:5]

    def start_simulation(self, scenario: Optional[str], session_id: Optional[str] = None):
        """Starts an active black-box attack simulation session."""
        canonical = validate_attack_procedure(scenario) if scenario else None
        self.active_attack_scenario = canonical if canonical else "None"
        self.active_simulation_session = session_id or f"sim-{int(time.time()*1000)%1000000:06d}"

    def stop_simulation(self):
        """Ends the active attack simulation session."""
        self.active_attack_scenario = "None"
        self.active_simulation_session = None

    def get_dashboard_payload(self) -> Dict[str, Any]:
        """Assembles complete telemetry payload for WebSocket / REST clients."""
        fingerprint = self.dataset.fingerprint if self.dataset else ""
        total_rows = self.dataset.total_rows if self.dataset else len(self.X_demo)
        meas_rows = self.dataset.measurement_rows if self.dataset else len(self.X_demo)
        craft_rows = self.dataset.crafting_rows if self.dataset else 0

        return {
            "data_profile": self.data_profile,
            "data_fingerprint": fingerprint,
            "data_stats": {
                "profile": self.data_profile,
                "fingerprint": fingerprint,
                "total_rows": total_rows,
                "measurement_rows": meas_rows,
                "crafting_rows": craft_rows,
                "available_target_count": meas_rows,
                "available_reference_count": craft_rows,
                "measurement_attack_rows": len(self.dataset.measurement_attack_indices) if self.dataset else 0,
                "measurement_benign_rows": len(self.dataset.measurement_benign_indices) if self.dataset else 0,
                "crafting_attack_rows": len(self.dataset.crafting_attack_indices) if self.dataset else 0,
                "crafting_benign_rows": len(self.dataset.crafting_benign_indices) if self.dataset else 0,
                "query_count": self.query_count
            },
            "simulation": {
                "active_scenario": self.active_attack_scenario,
                "session_id": self.active_simulation_session,
                "is_active": (self.active_attack_scenario != "None")
            },
            "stats": {
                "total_traffic": f"{self.total_traffic:,}",
                "detected_attacks": str(self.detected_attacks),
                "detection_recall": f"{self.recall:.3f}" if self.recall is not None else "—",
                "false_positive_rate": f"{self.fpr:.3f}" if self.fpr is not None else "—",
                "recall_numeric": self.recall,
                "fpr_numeric": self.fpr,
                "tp": self.tp,
                "fp": self.fp,
                "fn": self.fn,
                "tn": self.tn,
                "query_count": self.query_count,
                "metric_scope": "target_flows_only" if self.data_profile == "expanded" else "all_evaluated_flows",
                "traffic_delta": "Session Baseline",
                "attacks_delta": "Session Baseline",
                "recall_delta": "Session Baseline",
                "fpr_delta": "Session Baseline",
            },
            "afp": {
                "enabled": (self.active_defense_name != "none"),
                "status": "Active" if self.active_defense_name != "none" else "Bypassed",
                "defense_name": self.active_defense_name,
                "mode": self.controller_mode,
                "intensity": float(self.current_intensity),
                "intensity_min": float(self.intensity_min),
                "intensity_max": float(self.intensity_max),
                "threshold_warning": 0.85,
                "threshold_critical": 0.95,
                "health_status": "Awaiting Data" if self.recall is None else ("Healthy" if self.recall >= 0.85 else "Alert"),
                "controller_state": self.controller_state,
                "batch_id": self.batch_id,
            },
            "threat_locations": self.threat_locations,
            "top_threat_ips": self.top_threat_ips,
            "recent_feed": list(self.recent_feed),
            "recent_attacks": list(self.recent_attacks),
            "history": {
                "labels": list(self.history_labels),
                "recall": list(self.history_recall),
                "afp_intensity": list(self.history_intensity),
            }
        }


# Global Security Engine Singleton
engine = SecurityEngine()
engine.load_resources()

# -----------------------------------------------------------------------------
# WebSocket Connection Manager
# -----------------------------------------------------------------------------
class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket):
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)

    async def broadcast(self, message: dict):
        for connection in list(self.active_connections):
            try:
                await connection.send_json(message)
            except Exception:
                self.disconnect(connection)

ws_manager = ConnectionManager()

# -----------------------------------------------------------------------------
# FastAPI Application
# -----------------------------------------------------------------------------
app = FastAPI(
    title="IDS + Recall-Aware Research Runtime Platform",
    description="File-backed research demonstration runtime for Adaptive Feature Perturbation, Randomized Smoothing, Feature Squeezing, and Recall-Aware Feedback Control.",
    version="3.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.middleware("http")
async def add_no_cache_headers(request: Request, call_next):
    response = await call_next(request)
    if request.url.path.startswith("/static") or request.url.path == "/":
        response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
        response.headers["Pragma"] = "no-cache"
        response.headers["Expires"] = "0"
    return response

def resolve_ip_location(ip: str, explicit_location: Optional[str] = None) -> str:
    """Accurately identifies IP location, strictly distinguishing private subnets from unknown public IPs."""
    if not ip:
        return "Unknown"
    # RFC 1918 & Loopback Private Address Check
    parts = ip.split(".")
    if len(parts) == 4 and all(p.isdigit() for p in parts):
        first, second = int(parts[0]), int(parts[1])
        if first == 10:
            return "Private Network"
        if first == 172 and 16 <= second <= 31:
            return "Private Network"
        if first == 192 and second == 168:
            return "Private Network"
        if first == 127:
            return "Private Network"

    # Public IP: reject untrusted caller-claimed country strings; no trusted GeoIP DB is bundled
    return "Unknown"

# -----------------------------------------------------------------------------
# Request Schemas
# -----------------------------------------------------------------------------
class ServerRequestModel(BaseModel):
    source_ip: str
    destination_ip: Optional[str] = "192.168.1.10"
    traffic_family: Optional[str] = None
    traffic_family_source: Optional[str] = None
    flow_type: Optional[str] = None
    country: Optional[str] = None
    is_attack: Optional[bool] = None
    attack_scenario: Optional[str] = None
    session_id: Optional[str] = None
    feature_vector: Optional[List[float]] = None
    sample_id: Optional[int] = None
    data_profile: Optional[str] = None
    is_query: Optional[bool] = False
    query_stage: Optional[str] = None

class SimulationSessionModel(BaseModel):
    scenario: Optional[str] = None  # Canonical study procedures: "Silent Probing", "Surrogate Transferability", "Decision-Boundary Attack" (or None)
    session_id: Optional[str] = None

class SetDefenseModel(BaseModel):
    defense: str  # "afp", "rs", "fs", "none"

class SetModeModel(BaseModel):
    mode: str  # "recall-aware", "base"

class ToggleAFPModel(BaseModel):
    enabled: bool

# -----------------------------------------------------------------------------
# Protected Server Endpoints (Guarded by Inline IDS Pipeline)
# -----------------------------------------------------------------------------
@app.post("/api/server/request")
@app.post("/api/server/data")
async def protected_server_handler(req: ServerRequestModel):
    """
    Simulated Protected Server endpoint guarded by inline IDS + Defense + Recall-Aware Controller.
    Pipeline:
      1. Resolve 78-feature vector (from request or real demo parquet dataset).
      2. Apply Active Defense (AFP / RS / FS / None) using current intensity.
      3. Classify with frozen Random Forest model.
      4. Update Confusion Matrix & Recall-Aware Controller state atomically.
      5. Broadcast live telemetry to dashboard WebSocket.
      6. Return 200 OK (Allowed) or 403 Forbidden (Blocked).
    """
    t_now = datetime.now().strftime("%H:%M:%S")

    # 0. Data Profile Validation
    if req.data_profile is not None:
        req_prof = str(req.data_profile).strip().lower()
        if req_prof not in ["expanded", "fixture20"]:
            return JSONResponse(
                status_code=400,
                content={"status": "error", "message": f"Unknown dataset profile '{req_prof}'. Supported profiles are 'expanded' and 'fixture20'."}
            )
        if req_prof != engine.data_profile:
            return JSONResponse(
                status_code=400,
                content={"status": "error", "message": f"Dataset profile mismatch: server is running profile '{engine.data_profile}' but request specified '{req_prof}'."}
            )

    # 0b. Attack Scenario Procedure Validation
    if req.attack_scenario is not None:
        try:
            validate_attack_procedure(req.attack_scenario)
        except ValueError as e:
            return JSONResponse(
                status_code=400,
                content={"status": "error", "message": str(e)}
            )

    # 1. Sample ID Range & Type Validation (When sample_id is provided)
    if req.sample_id is not None:
        max_id = (len(engine.X_demo) - 1) if engine.X_demo is not None else 0
        if not isinstance(req.sample_id, int) or engine.X_demo is None or not (0 <= req.sample_id < len(engine.X_demo)):
            return JSONResponse(
                status_code=400,
                content={"status": "error", "message": f"Invalid sample_id: must be an integer between 0 and {max_id}."}
            )

    # 2. Session Binding Resolution
    is_session_bound = (
        engine.active_simulation_session is not None and
        req.session_id is not None and
        req.session_id == engine.active_simulation_session
    )

    # 3. Determine Query vs Target Role
    is_crafting_row = (
        engine.data_profile == "expanded" and
        req.sample_id is not None and
        engine.dataset is not None and
        engine.dataset.is_crafting(req.sample_id)
    )
    # Define query intent consistently; empty stage does not independently suppress target accounting
    has_query_intent = bool(req.is_query) or bool(req.query_stage and req.query_stage.strip())

    is_query_flow = False
    if engine.data_profile == "expanded":
        if is_crafting_row:
            # Crafting-role rows never contribute to the measurement confusion matrix
            is_query_flow = True
        elif has_query_intent:
            # Query intent on measurement-origin or synthetic candidate requests requires an active matching session
            if not is_session_bound:
                if engine.active_simulation_session is None:
                    err_msg = "Query scope rejected: no active simulation session on server. Start a session before submitting queries."
                elif req.session_id is None:
                    err_msg = "Query scope rejected: missing session_id. Queries on measurement or synthetic flows require a bound simulation session."
                else:
                    err_msg = f"Query scope rejected: invalid or mismatched session token '{req.session_id}'. Server active session is '{engine.active_simulation_session}'."
                return JSONResponse(status_code=400, content={"status": "error", "message": err_msg})
            is_query_flow = True

    # 4. Feature Vector Validation and Resolution (WHAT EXACT FLOW SHOULD THE IDS CLASSIFY?)
    # Strict validation: accept finite 78-element vector or valid server sample_id. Reject malformed inputs with 400.
    features: Optional[np.ndarray] = None

    if req.feature_vector is not None:
        if not isinstance(req.feature_vector, (list, tuple)) or len(req.feature_vector) != 78:
            return JSONResponse(
                status_code=400,
                content={"status": "error", "message": "Invalid feature_vector: must contain exactly 78 numerical features."}
            )
        try:
            fv = [float(v) for v in req.feature_vector]
            if not all(math.isfinite(v) for v in fv):
                return JSONResponse(
                    status_code=400,
                    content={"status": "error", "message": "Invalid feature_vector: all 78 values must be finite numbers."}
                )
            features = np.array(fv, dtype=np.float32)
            if not np.isfinite(features).all():
                return JSONResponse(
                    status_code=400,
                    content={"status": "error", "message": "Invalid feature_vector: values overflow 32-bit float or are non-finite."}
                )
        except (ValueError, TypeError):
            return JSONResponse(
                status_code=400,
                content={"status": "error", "message": "Invalid feature_vector: elements must be valid real numbers."}
            )
    else:
        # feature_vector is None; sample_id must be provided and valid
        if req.sample_id is None:
            return JSONResponse(
                status_code=400,
                content={"status": "error", "message": "Missing input: Request must include either a finite 78-element feature_vector or a valid demo sample_id."}
            )
        features = np.array(engine.X_demo.iloc[req.sample_id].values, dtype=np.float32)

    # 5. Metadata Provenance Resolution & Origin Verification
    is_exact_sample = False

    if req.feature_vector is not None:
        if req.sample_id is not None and engine.X_demo is not None and engine.meta_demo is not None and 0 <= req.sample_id < len(engine.meta_demo):
            orig_row = engine.X_demo.iloc[req.sample_id].values.astype(np.float32)
            if np.array_equal(features, orig_row):
                is_exact_sample = True
                resolved_family = str(engine.meta_demo.iloc[req.sample_id].get("attack_family", "Normal"))
                resolved_family_source = "Dataset-derived (Exact sample)"
            else:
                orig_fam = str(engine.meta_demo.iloc[req.sample_id].get("attack_family", "Unknown"))
                if is_session_bound:
                    resolved_family = f"Transformed (Original: {orig_fam})"
                    resolved_family_source = "Original sample family (Transformed via verified simulation session)"
                else:
                    resolved_family = f"Claimed original family: {orig_fam}"
                    resolved_family_source = "Unverified claim (Mismatched sample ID)"
        else:
            resolved_family = "Unknown"
            resolved_family_source = "Synthetic / Non-dataset"
    else:
        is_exact_sample = True
        resolved_family = str(engine.meta_demo.iloc[req.sample_id].get("attack_family", "Normal"))
        resolved_family_source = "Dataset-derived (Exact sample)"

    # 3. Ground Truth Provenance Resolution (Post-decision feedback, NOT classifier input)
    # Production IDS has no autonomous ground truth.
    # Ground truth is ONLY resolved if:
    #   (a) Explicit caller feedback (is_attack is not None) within an active, server-validated simulation session
    #   (b) Exact server-selected demo sample -> derived from authoritative metadata_demo.parquet
    # All other flows are strictly Unlabeled and will NOT update confusion matrix or controller.
    ground_truth: Optional[int] = None
    ground_truth_status = "Unlabeled"

    if is_exact_sample:
        ground_truth = int(engine.meta_demo.iloc[req.sample_id]["y_binary"])
        ground_truth_status = f"Dataset: {ground_truth}"
        if req.is_attack is not None and (1 if req.is_attack else 0) != ground_truth:
            # Conflicting simulator feedback is rejected/ignored; dataset label takes authoritative precedence
            print(f"[Provenance] Ignored conflicting simulator feedback (is_attack={req.is_attack}) for exact dataset sample {req.sample_id}; authoritative dataset label is {ground_truth}")
    elif req.is_attack is not None and is_session_bound:
        ground_truth = 1 if req.is_attack else 0
        ground_truth_status = f"Simulator: {ground_truth}"
    else:
        ground_truth = None
        ground_truth_status = "Unlabeled"

    # 4. Inline Defense & Inference
    active_def = engine.active_defense_name
    used_intensity = float(engine.current_intensity)
    pred_label: int = 0
    attack_prob: float = 0.05

    # Trusted simulation context for defense RNG seed (immune to caller scenario injection)
    # Uses fixed trusted context to make perturbation scenario-independent without modifying frozen defense modules.
    trusted_seed_scenario = "live_simulation"
    flow_seed = int(hashlib.md5(features.tobytes()).hexdigest(), 16) % (2**31 - 1)

    if active_def == "afp" and "afp" in engine.defenses:
        X_proj, _, _ = engine.defenses["afp"].defend(
            features.reshape(1, -1),
            epsilon_base=used_intensity,
            alpha=engine.afp_alpha,
            seed=flow_seed,
            attack_scenario=trusted_seed_scenario,
            batch_id=engine.batch_id
        )
        proba = engine.model.predict_proba(X_proj)[0]
        pred_label = int(np.argmax(proba))
        attack_prob = float(proba[1])

    elif active_def == "rs" and "rs" in engine.defenses:
        preds, _, vote_frac = engine.defenses["rs"].predict_ensemble(
            features.reshape(1, -1),
            sigma=used_intensity,
            seed=flow_seed,
            attack_scenario=trusted_seed_scenario,
            batch_id=engine.batch_id,
            predict_func=engine.model.predict,
            return_scores=True
        )
        pred_label = int(preds[0])
        attack_prob = float(vote_frac[0])

    elif active_def == "fs" and "fs" in engine.defenses:
        X_proj, _, _ = engine.defenses["fs"].defend(
            features.reshape(1, -1),
            intensity=used_intensity,
            seed=42,
            attack_scenario=trusted_seed_scenario,
            batch_id=engine.batch_id
        )
        proba = engine.model.predict_proba(X_proj)[0]
        pred_label = int(np.argmax(proba))
        attack_prob = float(proba[1])

    else:
        proba = engine.model.predict_proba(features.reshape(1, -1))[0]
        pred_label = int(np.argmax(proba))
        attack_prob = float(proba[1])

    # 5. Outcome Assessment (Authoritative binary classification: Benign vs Attack)
    is_malicious = (pred_label == 1)
    status_str = "Attack" if is_malicious else "Benign"

    # Resolve active attack scenario display (truthfully limited to canonical study procedures or None)
    if engine.active_attack_scenario != "None":
        scenario_display = engine.active_attack_scenario
    elif req.attack_scenario and req.attack_scenario.strip() not in ["None", "", "none"]:
        scenario_display = validate_attack_procedure(req.attack_scenario) or "None"
    else:
        scenario_display = "None"

    loc_display = resolve_ip_location(req.source_ip, req.country)

    if is_query_flow:
        engine.query_count += 1
    else:
        engine.total_traffic += 1
        if is_malicious:
            engine.detected_attacks += 1
            if loc_display != "Private Network":
                engine.record_attack_ip(req.source_ip, loc_display)

        # 6. Update Engine Metrics & Controller Feedback (Using used_intensity)
        engine.update_metrics_and_controller(ground_truth=ground_truth, predicted=pred_label, used_intensity=used_intensity)

    # 7. Append to Telemetry Feeds
    feed_entry = {
        "timestamp": t_now,
        "source_ip": req.source_ip,
        "destination_ip": req.destination_ip or "192.168.1.10",
        "attack_scenario": f"[Query: {req.query_stage or 'search'}] {scenario_display}" if is_query_flow else scenario_display,
        "traffic_family": f"[Query] {resolved_family}" if is_query_flow else resolved_family,
        "traffic_family_source": f"Query Telemetry ({resolved_family_source})" if is_query_flow else resolved_family_source,
        "type": resolved_family,
        "confidence": round(attack_prob if is_malicious else (1.0 - attack_prob), 2),
        "status": status_str,
        "location": loc_display,
        "defense": engine.active_defense_name.upper(),
        "mode": "Recall-Aware" if engine.controller_mode == "recall-aware" else "Base",
        "intensity": used_intensity,
        "ground_truth_status": ground_truth_status,
        "sample_id": req.sample_id,
        "data_profile": engine.data_profile,
        "is_query": is_query_flow,
        "action": "BLOCKED (403)" if is_malicious else "ALLOWED (200)"
    }
    engine.recent_feed.appendleft(feed_entry)

    if is_malicious and not is_query_flow:
        attack_entry = {
            "time": t_now,
            "source_ip": req.source_ip,
            "destination_ip": req.destination_ip or "192.168.1.10",
            "location": loc_display,
            "attack_scenario": scenario_display,
            "traffic_family": resolved_family,
            "traffic_family_source": resolved_family_source,
            "type": resolved_family,
            "confidence": round(attack_prob, 2),
            "status": "Attack",
            "defense": engine.active_defense_name.upper(),
            "mode": "Recall-Aware" if engine.controller_mode == "recall-aware" else "Base",
            "intensity": used_intensity,
            "ground_truth_status": ground_truth_status,
            "sample_id": req.sample_id,
            "data_profile": engine.data_profile,
            "action": "BLOCKED (403)"
        }
        engine.recent_attacks.appendleft(attack_entry)

    # 8. Broadcast Real-time Event to WebSocket Clients
    await ws_manager.broadcast({
        "event_type": "traffic_event",
        "entry": feed_entry,
        "payload": engine.get_dashboard_payload()
    })

    # 9. Response (Includes used_intensity, next_intensity, data_profile, is_query)
    if is_malicious:
        return JSONResponse(
            status_code=403,
            content={
                "status": "blocked",
                "code": "IDS_INTRUSION_BLOCKED",
                "message": "Connection dropped by IDS: Attack detected.",
                "details": {
                    "source_ip": req.source_ip,
                    "ids_classification": "Attack",
                    "traffic_family": resolved_family,
                    "attack_scenario": scenario_display,
                    "confidence": round(attack_prob, 4),
                    "defense": engine.active_defense_name,
                    "mode": engine.controller_mode,
                    "intensity": used_intensity,
                    "used_intensity": used_intensity,
                    "next_intensity": float(engine.current_intensity),
                    "controller_state": engine.controller_state,
                    "ground_truth_status": ground_truth_status,
                    "sample_id": req.sample_id,
                    "data_profile": engine.data_profile,
                    "is_query": is_query_flow,
                    "verdict": "DROPPED"
                }
            }
        )
    else:
        return JSONResponse(
            status_code=200,
            content={
                "status": "success",
                "message": "Request evaluated and passed by protected server inline IDS.",
                "details": {
                    "source_ip": req.source_ip,
                    "ids_classification": "Benign",
                    "traffic_family": resolved_family,
                    "attack_scenario": scenario_display,
                    "confidence": round(1.0 - attack_prob, 4),
                    "defense": engine.active_defense_name,
                    "mode": engine.controller_mode,
                    "intensity": used_intensity,
                    "used_intensity": used_intensity,
                    "next_intensity": float(engine.current_intensity),
                    "controller_state": engine.controller_state,
                    "ground_truth_status": ground_truth_status,
                    "sample_id": req.sample_id,
                    "data_profile": engine.data_profile,
                    "is_query": is_query_flow,
                    "verdict": "FORWARDED"
                }
            }
        )

# -----------------------------------------------------------------------------
# Simulation Session Management API Endpoints
# -----------------------------------------------------------------------------
@app.post("/api/simulation/start")
async def start_simulation_session(req: SimulationSessionModel):
    try:
        canonical_sc = validate_attack_procedure(req.scenario)
    except ValueError as e:
        return JSONResponse(status_code=400, content={"status": "error", "message": str(e)})

    engine.start_simulation(canonical_sc, req.session_id)
    await ws_manager.broadcast({
        "event_type": "simulation_session_change",
        "payload": engine.get_dashboard_payload()
    })
    return {
        "status": "started",
        "scenario": engine.active_attack_scenario,
        "session_id": engine.active_simulation_session
    }

@app.post("/api/simulation/stop")
async def stop_simulation_session():
    engine.stop_simulation()
    await ws_manager.broadcast({
        "event_type": "simulation_session_change",
        "payload": engine.get_dashboard_payload()
    })
    return {"status": "stopped", "scenario": "None"}

# -----------------------------------------------------------------------------
# Defense & Controller Management API Endpoints (Authorized Operators Only)
# -----------------------------------------------------------------------------
@app.post("/api/dashboard/set-defense")
async def api_set_defense(req: SetDefenseModel, request: Request):
    """Switches active defense (afp, rs, fs, none) with operator authorization."""
    if not verify_operator_authorization(request):
        return JSONResponse(
            status_code=401,
            content={"status": "error", "message": "Unauthorized: Valid operator token required in 'X-Operator-Token' header."}
        )
    engine.set_defense(req.defense)
    payload = engine.get_dashboard_payload()
    await ws_manager.broadcast({"event_type": "defense_changed", "payload": payload})
    return JSONResponse(content={"defense": engine.active_defense_name, "status": "Active" if engine.active_defense_name != "none" else "Bypassed"})

@app.post("/api/dashboard/set-mode")
async def api_set_mode(req: SetModeModel, request: Request):
    """Switches mode between 'recall-aware' and 'base' with operator authorization."""
    if not verify_operator_authorization(request):
        return JSONResponse(
            status_code=401,
            content={"status": "error", "message": "Unauthorized: Valid operator token required in 'X-Operator-Token' header."}
        )
    engine.set_mode(req.mode)
    payload = engine.get_dashboard_payload()
    await ws_manager.broadcast({"event_type": "mode_changed", "payload": payload})
    return JSONResponse(content={"mode": engine.controller_mode, "state": engine.controller_state})

@app.post("/api/dashboard/toggle-afp")
async def toggle_afp(req: ToggleAFPModel, request: Request):
    """Backward compatibility toggle button with operator authorization."""
    if not verify_operator_authorization(request):
        return JSONResponse(
            status_code=401,
            content={"status": "error", "message": "Unauthorized: Valid operator token required in 'X-Operator-Token' header."}
        )
    if req.enabled:
        engine.set_defense("afp")
    else:
        engine.set_defense("none")
    payload = engine.get_dashboard_payload()
    await ws_manager.broadcast({"event_type": "afp_toggled", "payload": payload})
    return JSONResponse(content={"defense": engine.active_defense_name, "status": "Active" if engine.active_defense_name != "none" else "Bypassed"})

@app.post("/api/dashboard/reset")
async def reset_metrics(request: Request):
    """Resets counters and controller state to cold start baseline with operator authorization."""
    if not verify_operator_authorization(request):
        return JSONResponse(
            status_code=401,
            content={"status": "error", "message": "Unauthorized: Valid operator token required in 'X-Operator-Token' header."}
        )
    engine.total_traffic = 0
    engine.detected_attacks = 0
    engine.tp = 0
    engine.fp = 0
    engine.fn = 0
    engine.tn = 0
    engine.query_count = 0
    engine.recall = None
    engine.fpr = None
    engine.batch_id = 0
    engine.batch_tp = 0
    engine.batch_fn = 0
    engine.history_labels.clear()
    engine.history_recall.clear()
    engine.history_intensity.clear()
    engine.threat_locations.clear()
    engine.top_threat_ips.clear()
    engine.recent_feed.clear()
    engine.recent_attacks.clear()
    engine.active_attack_scenario = "None"
    engine.active_simulation_session = None
    engine.set_defense(engine.active_defense_name)
    payload = engine.get_dashboard_payload()
    await ws_manager.broadcast({"event_type": "reset", "payload": payload})
    return JSONResponse(content=payload)

@app.get("/api/dashboard/stats")
async def get_dashboard_stats():
    """Returns complete state payload for frontend initialization."""
    return JSONResponse(content=engine.get_dashboard_payload())

# -----------------------------------------------------------------------------
# WebSocket Live Telemetry Feed
# -----------------------------------------------------------------------------
@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await ws_manager.connect(websocket)
    await websocket.send_json({
        "event_type": "initial_state",
        "payload": engine.get_dashboard_payload()
    })
    try:
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        ws_manager.disconnect(websocket)

# -----------------------------------------------------------------------------
# Static Files & Frontend Routing
# -----------------------------------------------------------------------------
if os.path.exists(FRONTEND_DIR):
    app.mount("/static", StaticFiles(directory=str(FRONTEND_DIR)), name="static")

@app.get("/")
async def serve_index():
    index_path = FRONTEND_DIR / "index.html"
    if index_path.exists():
        return FileResponse(str(index_path))
    return HTMLResponse("<h1>SOC dashboard not found. Please verify /frontend directory.</h1>")

# -----------------------------------------------------------------------------
# Main Execution Entrypoint
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    import uvicorn
    print("\n========================================================")
    print("  IDS + Recall-Aware Research Runtime Platform")
    print("  SOC Web Dashboard: http://localhost:8000")
    print("  Protected Server:  http://localhost:8000/api/server/data")
    print("  WebSocket Feed:    ws://localhost:8000/ws")
    print("========================================================\n")
    uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=False)
