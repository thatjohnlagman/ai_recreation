"""
IDS + Recall-Aware Research Runtime Platform & Simulated Protected Server
Connects:
- Pretrained 78-feature Random Forest Classifier (frozen_rf.joblib, CSE-CIC-IDS2018)
- Real Defenses: Adaptive Feature Poisoning (AFP), Randomized Smoothing (RS), Feature Squeezing (FS)
- Recall-Aware Controller (C1 configuration: rolling recall window, atomic batch updates)
- Inline IDS Guarded Protected Server Endpoints (/api/server/data)
- High-Performance Cyber SOC Web Dashboard (REST + WebSockets)
"""

import copy
import re
import logging
import os
import sys
import json
import time
import math
import hashlib
import asyncio
import secrets
import webbrowser
import warnings
import uuid
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, Any, List
from collections import deque

import traffic_history

import numpy as np
import pandas as pd
import joblib
import yaml
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Request, Response, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse, FileResponse, RedirectResponse
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
from runtime_package.geoip.resolver import resolve_ip_geo, DB_ATTRIBUTION_TEXT, DB_ATTRIBUTION_HTML


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
        self.reporting_session_id: str = self._generate_session_id()

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
        
    def _generate_session_id(self):
        import uuid
        return f"rev-{datetime.utcnow().strftime('%Y%m%d-%H%M%S%f')}-{uuid.uuid4().hex[:4]}"
        
    def rotate_session(self):
        """Starts a new reporting session and resets live dashboard counters."""
        self.reporting_session_id = self._generate_session_id()
        self.total_traffic = 0
        self.detected_attacks = 0
        self.tp = 0
        self.fp = 0
        self.fn = 0
        self.tn = 0
        self.query_count = 0
        self.recall = None
        self.fpr = None
        
        if self.controller is not None:
            self.controller.reset()
            self.batch_id = 0
            self.batch_tp = 0
            self.batch_fn = 0
            if self.controller_mode == "recall-aware":
                decision = self.controller.get_intensity(self.batch_id)
                self.current_intensity = decision.intensity
            else:
                self.current_intensity = self.controller.base_intensity
                
        self.history_labels.clear()
        self.history_recall.clear()
        self.history_intensity.clear()
        self.threat_locations.clear()
        self.top_threat_ips.clear()
        self.recent_feed.clear()
        self.recent_attacks.clear()

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
            
        self.rotate_session()

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
                
        self.rotate_session()

    def update_metrics_and_controller(self, ground_truth: Optional[int], predicted: int, used_intensity: float):
        """Updates confusion matrix counters (if ground truth known) and triggers batch-level controller updates."""
        triggered = False
        old_state_dict = {
            "batch_tp": self.batch_tp,
            "batch_fn": self.batch_fn,
            "batch_size": getattr(self, "batch_size", 50),
            "recall": self.recall,
            "controller_state": str(self.controller_state) if self.controller_state else "None"
        }
        
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
                        triggered = True
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
        
        new_state_dict = {
            "batch_tp": self.batch_tp,
            "batch_fn": self.batch_fn,
            "batch_size": getattr(self, "batch_size", 50),
            "recall": self.recall,
            "controller_state": str(self.controller_state) if self.controller_state else "None"
        }
        return triggered, old_state_dict, new_state_dict

    def record_attack_ip(self, ip: str, location: Optional[str] = None):
        """Updates threat locations on map and top threat IPs table using real MMDB geolocation."""
        geo = resolve_ip_geo(ip)
        loc_str = location if location else geo.get("location_str", "Unknown")
        t_now = datetime.now().strftime("%H:%M:%S")

        # Update threat_locations (if valid latitude/longitude coordinates exist)
        if geo.get("lat") is not None and geo.get("lng") is not None:
            found_loc = False
            for entry in self.threat_locations:
                if entry.get("ip") == ip:
                    entry["attacks"] = entry.get("attacks", 1) + 1
                    entry["last_seen"] = t_now
                    found_loc = True
                    break
            if not found_loc:
                self.threat_locations.append({
                    "ip": ip,
                    "country": geo.get("country", "—"),
                    "city": geo.get("city", "—"),
                    "lat": geo.get("lat"),
                    "lng": geo.get("lng"),
                    "attacks": 1,
                    "last_seen": t_now
                })

        # Update Top Threat IPs
        found = False
        for entry in self.top_threat_ips:
            if entry.get("ip") == ip:
                entry["attacks"] = entry.get("attacks", 1) + 1
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
                "threshold_critical": 0.85,
                "threshold_target": 0.95,
                "health_status": "Awaiting Data" if self.recall is None else ("Healthy" if self.recall >= 0.85 else "Alert"),
                "controller_state": self.controller_state,
                "batch_id": self.batch_id,
                "rolling_recall": float(self.controller.current_recall) if self.controller and getattr(self.controller, 'current_recall', None) is not None else None,
                "window_batch_count": int(self.controller.window_batch_count) if self.controller and hasattr(self.controller, 'window_batch_count') else 0
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

from uvicorn.config import LOGGING_CONFIG
from uvicorn.logging import AccessFormatter

class RedactedAccessFormatter(AccessFormatter):
    """Access log formatter that redacts ticket query parameters."""
    def formatMessage(self, record: logging.LogRecord) -> str:
        if record.args and len(record.args) >= 3:
            full_path = str(record.args[2])
            if "ticket=" in full_path:
                redacted_path = re.sub(r"ticket=[^&\s]+", "ticket=[REDACTED]", full_path)
                args_list = list(record.args)
                args_list[2] = redacted_path
                record.args = tuple(args_list)
        return super().formatMessage(record)

class RedactTicketFilter(logging.Filter):
    """Logging filter that redacts ticket query strings from records."""
    def filter(self, record: logging.LogRecord) -> bool:
        if record.args and len(record.args) >= 3:
            full_path = str(record.args[2])
            if "ticket=" in full_path:
                redacted_path = re.sub(r"ticket=[^&\s]+", "ticket=[REDACTED]", full_path)
                args_list = list(record.args)
                args_list[2] = redacted_path
                record.args = tuple(args_list)
        if isinstance(record.msg, str) and "ticket=" in record.msg:
            record.msg = re.sub(r"ticket=[^&\s]+", "ticket=[REDACTED]", record.msg)
        return True

def get_redacted_log_config() -> dict:
    cfg = copy.deepcopy(LOGGING_CONFIG)
    cfg["formatters"]["access"]["()"] = "server.RedactedAccessFormatter"
    return cfg

class RedactTicketASGIMiddleware:
    """ASGI middleware that redacts launch ticket values from scope before access logging."""
    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] == "http" and scope.get("path") == "/launch":
            raw_qs = scope.get("query_string", b"").decode("latin1", errors="ignore")
            async def wrapped_send(message):
                if message["type"] == "http.response.start":
                    if "ticket=" in raw_qs:
                        redacted = re.sub(r"ticket=[^&\s]+", "ticket=[REDACTED]", raw_qs)
                        scope["query_string"] = redacted.encode("latin1")
                await send(message)
            await self.app(scope, receive, wrapped_send)
        else:
            await self.app(scope, receive, send)

# -----------------------------------------------------------------------------
# FastAPI Application
# -----------------------------------------------------------------------------
app = FastAPI(
    title="IDS + Recall-Aware Research Runtime Platform",
    description="File-backed research demonstration runtime for Adaptive Feature Perturbation, Randomized Smoothing, Feature Squeezing, and Recall-Aware Feedback Control.",
    version="3.0.0"
)

app.add_middleware(RedactTicketASGIMiddleware)
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
    """Accurately identifies IP location using local MMDB database and RFC address classification."""
    geo = resolve_ip_geo(ip)
    return geo.get("location_str", "Unknown")

# -----------------------------------------------------------------------------
# Request Schemas
# -----------------------------------------------------------------------------
class ServerRequestModel(BaseModel):
    source_ip: str
    destination_ip: Optional[str] = None
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
    
    event_id = str(uuid.uuid4())
    received_vector = features.copy()
    classifier_input_vector = None
    rs_member_vectors = None
    rs_member_preds = None

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
        classifier_input_vector = X_proj.copy()
        proba = engine.model.predict_proba(X_proj)[0]
        pred_label = int(np.argmax(proba))
        attack_prob = float(proba[1])

    elif active_def == "rs" and "rs" in engine.defenses:
        rs_member_inputs = []
        rs_member_outputs = []
        def wrapped_predict_func(X):
            nonlocal rs_member_inputs, rs_member_outputs
            rs_member_inputs.append(X.copy())
            preds = engine.model.predict(X)
            rs_member_outputs.append(preds.copy())
            return preds
            
        preds, _, vote_frac = engine.defenses["rs"].predict_ensemble(
            features.reshape(1, -1),
            sigma=used_intensity,
            seed=flow_seed,
            attack_scenario=trusted_seed_scenario,
            batch_id=engine.batch_id,
            predict_func=wrapped_predict_func,
            return_scores=True
        )
        if len(rs_member_inputs) > 0:
            rs_member_vectors = np.vstack(rs_member_inputs)
            rs_member_preds = np.concatenate(rs_member_outputs)
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
        classifier_input_vector = X_proj.copy()
        proba = engine.model.predict_proba(X_proj)[0]
        pred_label = int(np.argmax(proba))
        attack_prob = float(proba[1])

    else:
        classifier_input_vector = features.reshape(1, -1).copy()
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

    triggered, old_state, new_state = False, {}, {}
    if is_query_flow:
        engine.query_count += 1
    else:
        engine.total_traffic += 1
        if is_malicious:
            engine.detected_attacks += 1
            if loc_display != "Private Network":
                engine.record_attack_ip(req.source_ip, loc_display)

        # 6. Update Engine Metrics & Controller Feedback (Using used_intensity)
        triggered, old_state, new_state = engine.update_metrics_and_controller(ground_truth=ground_truth, predicted=pred_label, used_intensity=used_intensity)

    try:
        if active_def == "rs":
            conf_str = f"RS Vote Fraction: {attack_prob:.3f}"
            conf_type = "RS vote fraction"
        else:
            conf_str = f"RF Class Probability: {attack_prob:.3f}"
            conf_type = "RF probability"

        traffic_history.save_flow_event({
            "event_id": event_id,
            "timestamp_utc": datetime.utcnow().isoformat() + "Z",
            "source_ip": req.source_ip,
            "destination_ip": req.destination_ip or "",
            "data_profile": engine.data_profile,
            "role": "Query" if is_query_flow else "Measured Target",
            "sample_id": req.sample_id,
            "binary_prediction": pred_label,
            "action": "Request rejected (HTTP 403)" if is_malicious else "Request allowed (HTTP 200)",
            "confidence_meaning": conf_str,
            "defense": active_def.upper(),
            "used_intensity": used_intensity,
            "ground_truth_status": ground_truth_status,
            "controller_triggered": 1 if triggered else 0,
            "controller_state": {"old": old_state, "new": new_state},
            "controller_mode": engine.controller_mode,
            "session_id": engine.reporting_session_id,
            "received_vector": received_vector,
            "classifier_input_vector": classifier_input_vector,
            "rs_member_vectors": rs_member_vectors,
            "rs_member_preds": rs_member_preds,
            "attack_score": float(attack_prob),
            "predicted_class_confidence": float(attack_prob if is_malicious else (1.0 - attack_prob)),
            "confidence_type": conf_type,
            "next_intensity": float(engine.current_intensity),
            "controller_rolling_recall": float(engine.controller.current_recall) if engine.controller and getattr(engine.controller, 'current_recall', None) is not None else None,
            "controller_state_before": str(old_state.get('color', 'Base')) if old_state else "Base",
            "controller_state_after": str(new_state.get('color', 'Base')) if new_state else "Base",
            "controller_batch_id": engine.batch_id
        })
    except Exception as e:
        print(f"[History] Failed to save history: {e}")

    # 7. Append to Telemetry Feeds
    feed_entry = {
        "event_id": event_id,
        "timestamp": t_now,
        "source_ip": req.source_ip,
        "destination_ip": req.destination_ip if req.destination_ip else None,
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
        "action": "Request rejected (HTTP 403)" if is_malicious else "Request allowed (HTTP 200)"
    }
    engine.recent_feed.appendleft(feed_entry)

    if is_malicious and not is_query_flow:
        attack_entry = {
            "time": t_now,
            "source_ip": req.source_ip,
            "destination_ip": req.destination_ip if req.destination_ip else None,
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
            "action": "Request rejected (HTTP 403)"
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
                    "action": "Request rejected (HTTP 403)",
                    "verdict": "DROPPED",
                    "event_id": event_id
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
                    "action": "Request allowed (HTTP 200)",
                    "verdict": "FORWARDED",
                    "event_id": event_id
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
async def api_set_defense(req: SetDefenseModel):
    """Switches active defense (afp, rs, fs, none)."""
    engine.set_defense(req.defense)
    payload = engine.get_dashboard_payload()
    await ws_manager.broadcast({"event_type": "defense_changed", "payload": payload})
    return JSONResponse(content={"defense": engine.active_defense_name, "status": "Active" if engine.active_defense_name != "none" else "Bypassed"})

@app.post("/api/dashboard/set-mode")
async def api_set_mode(req: SetModeModel):
    """Switches mode between 'recall-aware' and 'base'."""
    engine.set_mode(req.mode)
    payload = engine.get_dashboard_payload()
    await ws_manager.broadcast({"event_type": "mode_changed", "payload": payload})
    return JSONResponse(content={"mode": engine.controller_mode, "state": engine.controller_state})

@app.post("/api/dashboard/toggle-afp")
async def toggle_afp(req: ToggleAFPModel):
    """Backward compatibility toggle button."""
    if req.enabled:
        engine.set_defense("afp")
    else:
        engine.set_defense("none")
    payload = engine.get_dashboard_payload()
    await ws_manager.broadcast({"event_type": "afp_toggled", "payload": payload})
    return JSONResponse(content={"defense": engine.active_defense_name, "status": "Active" if engine.active_defense_name != "none" else "Bypassed"})

@app.post("/api/dashboard/reset")
async def reset_metrics():
    """Resets counters and controller state to cold start baseline."""
    engine.rotate_session()
    engine.active_attack_scenario = "None"
    engine.active_simulation_session = None
    # Ensure controller reset explicitly if needed, but rotate_session covers it
    engine.set_defense(engine.active_defense_name)
    payload = engine.get_dashboard_payload()
    await ws_manager.broadcast({"event_type": "reset", "payload": payload})
    return JSONResponse(content=payload)

def _protect_csv(val):
    if isinstance(val, str) and val.startswith(('=', '+', '-', '@')):
        return "'" + val
    return val

@app.get("/api/history")
async def api_get_history(
    limit: int = 50,
    offset: int = 0,
    session_id: Optional[str] = None,
    role: Optional[str] = None,
    action: Optional[str] = None,
    defense: Optional[str] = None,
    is_attack: Optional[bool] = None,
    query_only: bool = False,
    include_queries: bool = False,
    order: str = "desc"
):
    import traffic_history
    results, total = traffic_history.get_history(
        limit=limit,
        offset=offset,
        session_id=session_id,
        role=role,
        action=action,
        defense=defense,
        is_attack=is_attack,
        query_only=query_only,
        include_queries=include_queries,
        order=order
    )
    return {"data": results, "total": total}

@app.get("/api/history/schema")
async def api_get_schema():
    # engine.modifiable_mask is a numpy boolean array, need tolist()
    mask = engine.modifiable_mask
    if isinstance(mask, np.ndarray):
        mask = mask.tolist()

    return {
        "feature_names": engine.feature_names,
        "modifiable_mask": mask
    }

@app.get("/api/history/event/{event_id}")
async def api_get_event(event_id: str):
    import traffic_history
    event = traffic_history.get_event(event_id)
    if not event:
        raise HTTPException(status_code=404, detail="Event not found")
    
    # We must convert numpy arrays to lists for JSON serialization
    def make_serializable(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj

    event["received_vector"] = make_serializable(event.get("received_vector"))
    event["classifier_input_vector"] = make_serializable(event.get("classifier_input_vector"))
    event["rs_member_vectors"] = make_serializable(event.get("rs_member_vectors"))
    event["rs_member_preds"] = make_serializable(event.get("rs_member_preds"))
    
    return event

class ReviewUpdateModel(BaseModel):
    review_status: str
    analyst_notes: str

@app.post("/api/history/event/{event_id}/review")
async def api_update_review(event_id: str, req: ReviewUpdateModel):
    import traffic_history
    traffic_history.update_review(event_id, req.review_status, req.analyst_notes)
    return {"status": "success"}

@app.get("/api/history/sessions")
async def api_get_sessions():
    import traffic_history
    return traffic_history.get_sessions()

import csv
import io
from fastapi.responses import StreamingResponse

def _export_generator(fieldnames, results, get_row_dict):
    output = io.StringIO()
    writer = csv.DictWriter(output, fieldnames=fieldnames)
    writer.writeheader()
    yield output.getvalue()
    output.seek(0)
    output.truncate(0)
    
    for row in results:
        row_dict = get_row_dict(row)
        if row_dict:
            safe_dict = {k: _protect_csv(v) for k, v in row_dict.items()}
            writer.writerow(safe_dict)
            yield output.getvalue()
            output.seek(0)
            output.truncate(0)

@app.get("/api/history/export")
async def api_export_metadata(
    session_id: Optional[str] = None,
    role: Optional[str] = None,
    action: Optional[str] = None,
    defense: Optional[str] = None,
    is_attack: Optional[bool] = None,
    query_only: bool = False,
    include_queries: bool = False,
    preview: bool = False
):
    import traffic_history
    import re
    fetch_limit = 5 if preview else 1000000
    results, _ = traffic_history.get_history(limit=fetch_limit, offset=0, session_id=session_id, role=role, action=action, defense=defense, is_attack=is_attack, query_only=query_only, include_queries=include_queries)
    
    if not results:
        return StreamingResponse(iter(["No data"]), media_type="text/csv")
        
    ts = datetime.utcnow().strftime('%Y%m%d_%H%M%S')
    filename = "traffic_metadata.csv"
    if session_id:
        clean_id = re.sub(r'[^a-zA-Z0-9_-]', '', session_id)
        filename = f"traffic_metadata_session_{clean_id}.csv"
    elif any([role, action, defense, is_attack, query_only, include_queries]):
        filename = f"traffic_metadata_filtered_{ts}.csv"
    else:
        filename = f"traffic_metadata_all_{ts}.csv"
        
    return StreamingResponse(_export_generator(list(results[0].keys()), results, lambda r: r), media_type="text/csv", headers={"Content-Disposition": f"attachment; filename={filename}"})

@app.get("/api/history/export/sessions")
async def api_export_sessions(session_id: Optional[str] = None, preview: bool = False):
    import traffic_history
    import re
    sessions = traffic_history.get_sessions()
    
    if session_id:
        sessions = [s for s in sessions if s.get("session_id") == session_id]
        
    if preview:
        sessions = sessions[:5]
    
    if not sessions:
        return StreamingResponse(iter(["No data"]), media_type="text/csv")
        
    ts = datetime.utcnow().strftime('%Y%m%d_%H%M%S')
    filename = "session_summary.csv"
    if session_id:
        clean_id = re.sub(r'[^a-zA-Z0-9_-]', '', session_id)
        filename = f"session_summary_{clean_id}.csv"
    else:
        filename = f"session_summary_all_{ts}.csv"
        
    return StreamingResponse(_export_generator(list(sessions[0].keys()), sessions, lambda r: r), media_type="text/csv", headers={"Content-Disposition": f"attachment; filename={filename}"})


@app.get("/api/history/export/features")
async def api_export_features(
    session_id: Optional[str] = None,
    role: Optional[str] = None,
    action: Optional[str] = None,
    defense: Optional[str] = None,
    is_attack: Optional[bool] = None,
    query_only: bool = False,
    include_queries: bool = False,
    preview: bool = False
):
    import traffic_history
    import re
    fetch_limit = 5 if preview else 1000000
    results, _ = traffic_history.get_history(limit=fetch_limit, offset=0, session_id=session_id, role=role, action=action, defense=defense, is_attack=is_attack, query_only=query_only, include_queries=include_queries)
    
    if not results:
        return StreamingResponse(iter(["No data"]), media_type="text/csv")
    
    fieldnames = ["event_id", "stage", "rs_member_index"] + engine.feature_names
    
    def get_feature_rows():
        output = io.StringIO()
        writer = csv.DictWriter(output, fieldnames=fieldnames)
        writer.writeheader()
        yield output.getvalue()
        output.seek(0)
        output.truncate(0)
        
        for row in results:
            event = traffic_history.get_event(row["event_id"])
            
            # Received
            vec_rec = event.get("received_vector")
            if vec_rec is not None:
                row_dict = {"event_id": row["event_id"], "stage": "received", "rs_member_index": ""}
                for i, val in enumerate(vec_rec):
                    row_dict[engine.feature_names[i]] = _protect_csv(val)
                writer.writerow(row_dict)
                yield output.getvalue()
                output.seek(0)
                output.truncate(0)
                
            # Final Classifier Input
            vec_clf = event.get("classifier_input_vector")
            if vec_clf is not None:
                row_dict = {"event_id": row["event_id"], "stage": "final_classifier_input", "rs_member_index": ""}
                for i, val in enumerate(vec_clf):
                    row_dict[engine.feature_names[i]] = _protect_csv(val)
                writer.writerow(row_dict)
                yield output.getvalue()
                output.seek(0)
                output.truncate(0)
                
            # RS Members
            vec_rs = event.get("rs_member_vectors")
            if vec_rs is not None:
                for idx, member in enumerate(vec_rs):
                    row_dict = {"event_id": row["event_id"], "stage": "rs_member", "rs_member_index": str(idx)}
                    for i, val in enumerate(member):
                        row_dict[engine.feature_names[i]] = _protect_csv(val)
                    writer.writerow(row_dict)
                    yield output.getvalue()
                    output.seek(0)
                    output.truncate(0)
                    
    ts = datetime.utcnow().strftime('%Y%m%d_%H%M%S')
    filename = "traffic_features.csv"
    if session_id:
        clean_id = re.sub(r'[^a-zA-Z0-9_-]', '', session_id)
        filename = f"traffic_features_session_{clean_id}.csv"
    elif any([role, action, defense, is_attack, query_only, include_queries]):
        filename = f"traffic_features_filtered_{ts}.csv"
    else:
        filename = f"traffic_features_all_{ts}.csv"
        
    return StreamingResponse(get_feature_rows(), media_type="text/csv", headers={"Content-Disposition": f"attachment; filename={filename}"})

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

# Removed operator dashboard launch endpoints to simplify presentation

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

@app.get("/explorer")
async def serve_explorer():
    return FileResponse(FRONTEND_DIR / "explorer.html")

@app.get("/sessions")
async def serve_sessions():
    return FileResponse(FRONTEND_DIR / "sessions.html")

# -----------------------------------------------------------------------------
# Main Execution Entrypoint
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    import uvicorn
    import argparse
    import threading
    import urllib.request
    import time
    import webbrowser

    parser = argparse.ArgumentParser(description="Standalone IDS + Recall-Aware Defense Server")
    parser.add_argument("--host", default="127.0.0.1", help="Bind host (default: 127.0.0.1 loopback)")
    parser.add_argument("--port", type=int, default=8000, help="Bind port (default: 8000)")
    parser.add_argument("--launch", "--open", action="store_true", help="Automatically launch dashboard in browser")
    args = parser.parse_args()

    launch_url = f"http://{args.host}:{args.port}/"
    print("\n========================================================")
    print("  IDS + Recall-Aware Research Runtime Platform")
    print(f"  SOC Web Dashboard: http://{args.host}:{args.port}")
    print(f"  Protected Server:  http://{args.host}:{args.port}/api/server/data")
    print(f"  WebSocket Feed:    ws://{args.host}:{args.port}/ws")
    print("========================================================\n")

    if args.launch:
        def open_browser():
            ready_url = f"http://{args.host}:{args.port}/api/dashboard/stats"
            deadline = time.time() + 15.0
            server_ready = False
            while time.time() < deadline:
                try:
                    req = urllib.request.Request(ready_url)
                    with urllib.request.urlopen(req, timeout=0.5) as resp:
                        if resp.status == 200:
                            server_ready = True
                            break
                except Exception:
                    time.sleep(0.1)
            if server_ready:
                webbrowser.open(launch_url)
        threading.Thread(target=open_browser, daemon=True).start()

    log_config = get_redacted_log_config()
    uvicorn.run(app, host=args.host, port=args.port, reload=False, log_config=log_config)
