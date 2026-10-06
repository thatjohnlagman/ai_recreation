#!/usr/bin/env python3
"""
Attacker Simulation Console for IDS + Recall-Aware Research Platform
Simulates adversarial network attacks, research black-box evasion algorithms,
and volumetric floods against the protected target server.

Research Attacks Supported:
  1. Silent Probing Attack (Offline perturbation without querying oracle)
  2. Surrogate Transferability Attack (Locally trained surrogate decision tree)
  3. Decision Boundary Attack (1D Bisection search vs benign reference pool)

Supported Data Profiles:
  - 'expanded' (Default): 90,000 evaluation rows (72,000 measurement targets, 18,000 crafting references)
  - 'fixture20': 20-flow regression/fallback fixture
"""

import sys
import os
import time
import json
import random
import argparse
import urllib.request
import urllib.error
from pathlib import Path
from collections import deque
from typing import Optional, Dict, Any, List

import numpy as np
import pandas as pd

# Ensure UTF-8 output on Windows consoles if supported
if hasattr(sys.stdout, 'reconfigure'):
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass

# Add runtime_package to path for research attack imports
BASE_DIR = Path(__file__).resolve().parent
RUNTIME_DIR = BASE_DIR / "runtime_package"
if str(RUNTIME_DIR) not in sys.path:
    sys.path.insert(0, str(RUNTIME_DIR))

from attacks.silent_probing import SilentProbingAttack
from attacks.surrogate_transfer import SurrogateTransferAttack
from attacks.boundary_attack import DecisionBoundaryAttack
from data_loader import get_dataset, LoadedDataset

# ANSI Color Codes
RESET = "\033[0m"
BOLD = "\033[1m"
DIM = "\033[2m"
RED = "\033[91m"
GREEN = "\033[92m"
YELLOW = "\033[93m"
BLUE = "\033[94m"
MAGENTA = "\033[95m"
CYAN = "\033[96m"
WHITE = "\033[97m"
BG_RED = "\033[41m"
BG_GREEN = "\033[42m"
BG_BLUE = "\033[44m"

DEFAULT_SERVER_URL = "http://localhost:8000/api/server/data"
DEFAULT_DEFENSE_URL = "http://localhost:8000/api/dashboard/set-defense"
DEFAULT_MODE_URL = "http://localhost:8000/api/dashboard/set-mode"
DEFAULT_TOGGLE_URL = "http://localhost:8000/api/dashboard/toggle-afp"
DEFAULT_SIM_START_URL = "http://localhost:8000/api/simulation/start"
DEFAULT_SIM_STOP_URL = "http://localhost:8000/api/simulation/stop"

def get_server_url() -> str:
    return DEFAULT_SERVER_URL

def get_base_url() -> str:
    if "/api/" in DEFAULT_SERVER_URL:
        return DEFAULT_SERVER_URL.split("/api/")[0]
    return DEFAULT_SERVER_URL.rstrip("/")

def get_defense_url() -> str:
    return f"{get_base_url()}/api/dashboard/set-defense"

def get_mode_url() -> str:
    return f"{get_base_url()}/api/dashboard/set-mode"

def get_sim_start_url() -> str:
    return f"{get_base_url()}/api/simulation/start"

def get_sim_stop_url() -> str:
    return f"{get_base_url()}/api/simulation/stop"

def get_reset_url() -> str:
    return f"{get_base_url()}/api/dashboard/reset"

def get_stats_url() -> str:
    return f"{get_base_url()}/api/dashboard/stats"

def set_target_url(target: str):
    global DEFAULT_SERVER_URL, DEFAULT_DEFENSE_URL, DEFAULT_MODE_URL, DEFAULT_TOGGLE_URL, DEFAULT_SIM_START_URL, DEFAULT_SIM_STOP_URL
    if "/api/" in target:
        base_url = target.split("/api/")[0]
        DEFAULT_SERVER_URL = target
    else:
        base_url = target.rstrip("/")
        DEFAULT_SERVER_URL = f"{base_url}/api/server/data"
    DEFAULT_DEFENSE_URL = f"{base_url}/api/dashboard/set-defense"
    DEFAULT_MODE_URL = f"{base_url}/api/dashboard/set-mode"
    DEFAULT_TOGGLE_URL = f"{base_url}/api/dashboard/toggle-afp"
    DEFAULT_SIM_START_URL = f"{base_url}/api/simulation/start"
    DEFAULT_SIM_STOP_URL = f"{base_url}/api/simulation/stop"

CURRENT_SIMULATION_SESSION_ID: Optional[str] = None

# Origin Pools for Simulation (Private IPs strictly labeled Private Network, public IPs Unknown)
ORIGIN_POOL = [
    {"ip": "203.0.113.45", "country": "Unknown"},
    {"ip": "185.199.110.23", "country": "Unknown"},
    {"ip": "103.21.54.12", "country": "Unknown"},
    {"ip": "45.76.32.18", "country": "Unknown"},
    {"ip": "89.248.163.77", "country": "Unknown"},
    {"ip": "114.119.130.88", "country": "Unknown"},
    {"ip": "177.54.144.20", "country": "Unknown"},
    {"ip": "197.232.12.9", "country": "Unknown"},
    {"ip": "10.0.2.45", "country": "Private Network"},
    {"ip": "172.16.0.12", "country": "Private Network"},
    {"ip": "10.0.3.77", "country": "Private Network"},
    {"ip": "192.168.1.25", "country": "Private Network"}
]

BENIGN_POOL = [
    {"ip": "172.16.0.5", "country": "Private Network"},
    {"ip": "192.168.1.100", "country": "Private Network"},
    {"ip": "10.0.1.22", "country": "Private Network"},
    {"ip": "192.168.1.45", "country": "Private Network"}
]


def print_banner():
    print(f"\n{CYAN}{BOLD}======================================================================{RESET}")
    print(f"{CYAN}{BOLD}  CYBER ATTACKER SIMULATION CONSOLE — RESEARCH RUNTIME{RESET}")
    print(f"{WHITE}  Target Protected Server: {BLUE}{get_server_url()}{RESET}")
    print(f"{WHITE}  Active Data Profile:     {YELLOW}{ctx.profile.upper()}{RESET} ({ctx.dataset.measurement_rows:,} measurement targets, {ctx.dataset.crafting_rows:,} crafting references)")
    print(f"{WHITE}  Defense Layer:          {GREEN}Inline IDS + Defenses (AFP/RS/FS) + Recall-Aware Controller{RESET}")
    print(f"{CYAN}{BOLD}======================================================================{RESET}\n")


def send_packet(target_url: str, payload: dict) -> dict:
    """Sends HTTP POST to protected server and captures IDS verdict."""
    data_bytes = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        target_url,
        data=data_bytes,
        headers={"Content-Type": "application/json"}
    )
    t0 = time.perf_counter()
    try:
        with urllib.request.urlopen(req, timeout=5) as response:
            latency_ms = (time.perf_counter() - t0) * 1000
            res_body = json.loads(response.read().decode("utf-8"))
            return {
                "status_code": response.status,
                "latency_ms": latency_ms,
                "body": res_body,
                "blocked": (response.status == 403),
                "allowed": (response.status == 200),
                "error": None
            }
    except urllib.error.HTTPError as e:
        latency_ms = (time.perf_counter() - t0) * 1000
        try:
            body = json.loads(e.read().decode("utf-8"))
        except Exception:
            body = {"message": str(e)}
        return {
            "status_code": e.code,
            "latency_ms": latency_ms,
            "body": body,
            "blocked": (e.code == 403),
            "allowed": (e.code == 200),
            "error": None if e.code in (200, 403) else f"HTTP {e.code}"
        }
    except Exception as e:
        return {
            "status_code": 0,
            "latency_ms": 0,
            "body": {"error": str(e)},
            "blocked": False,
            "allowed": False,
            "error": str(e)
        }


def set_server_defense(defense_name: str) -> bool:
    """Configures server defense ('afp', 'rs', 'fs', 'none')."""
    payload = json.dumps({"defense": defense_name}).encode("utf-8")
    req = urllib.request.Request(get_defense_url(), data=payload, headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=3) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            return True
    except Exception as e:
        print(f"{RED}[Error] Failed to set defense: {e}{RESET}")
        return False


def set_server_mode(mode: str) -> bool:
    """Configures controller mode ('recall-aware', 'base')."""
    payload = json.dumps({"mode": mode}).encode("utf-8")
    req = urllib.request.Request(get_mode_url(), data=payload, headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=3) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            return True
    except Exception as e:
        print(f"{RED}[Error] Failed to set mode: {e}{RESET}")
        return False


def start_simulation_session(scenario_name: str, session_id: Optional[str] = None) -> str:
    """
    Notifies the protected server that an attack simulation session has begun.
    Session creation must succeed before feedback-dependent candidate queries.
    """
    global CURRENT_SIMULATION_SESSION_ID
    sess_id = session_id or f"sim-{int(time.time()*1000)%1000000:06d}"
    try:
        payload = json.dumps({"scenario": scenario_name, "session_id": sess_id}).encode("utf-8")
        req = urllib.request.Request(get_sim_start_url(), data=payload, headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=3) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            confirmed_id = data.get("session_id")
            if not confirmed_id:
                raise RuntimeError("Server returned empty session_id")
            CURRENT_SIMULATION_SESSION_ID = confirmed_id
    except Exception as e:
        CURRENT_SIMULATION_SESSION_ID = None
        raise RuntimeError(
            f"Failed to start simulation session on server at {get_sim_start_url()}: {e}. "
            f"Session creation must succeed before feedback-dependent candidate queries."
        )
    return CURRENT_SIMULATION_SESSION_ID


def stop_simulation_session():
    """Notifies the protected server that the attack simulation session has ended."""
    global CURRENT_SIMULATION_SESSION_ID
    try:
        req = urllib.request.Request(get_sim_stop_url(), data=b"{}", headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=3) as resp:
            pass
    except Exception:
        pass
    CURRENT_SIMULATION_SESSION_ID = None


def check_backend_compatibility(require_profile: Optional[str] = None) -> bool:
    """
    Verifies that the target backend is running with a matching data profile and data fingerprint.
    Aborts with a clear message on mismatch or error.
    """
    try:
        req = urllib.request.Request(get_stats_url())
        with urllib.request.urlopen(req, timeout=3) as resp:
            data = json.loads(resp.read().decode("utf-8"))
    except Exception as e:
        print(f"{YELLOW}[Warning] Could not reach target server at {get_stats_url()}: {e}{RESET}")
        return False

    server_profile = str(data.get("data_profile", "")).strip().lower()
    server_fp = str(data.get("data_fingerprint", ""))
    expected_profile = (require_profile or ctx.profile).strip().lower()
    expected_fp = ctx.dataset.fingerprint

    if server_profile != expected_profile:
        raise RuntimeError(
            f"Profile mismatch: Attacker is running profile '{expected_profile}' (fingerprint {expected_fp[:12]}...) "
            f"but server at {get_server_url()} is running profile '{server_profile}' (fingerprint {server_fp[:12]}...). "
            f"Launch server with matching IDS_DATA_PROFILE or set attacker --dataset {server_profile}."
        )

    if server_fp and expected_fp and server_fp != expected_fp:
        raise RuntimeError(
            f"Dataset fingerprint mismatch: Attacker fingerprint is '{expected_fp}' but server fingerprint is '{server_fp}'. "
            f"Ensure identical dataset files are loaded in both processes."
        )

    data_stats = data.get("data_stats", {})
    meas_rows = data_stats.get("measurement_rows", data_stats.get("total_rows", "N/A"))
    craft_rows = data_stats.get("crafting_rows", 0)
    print(f"{GREEN}[OK] Verified backend connection & data contract:{RESET} Profile='{server_profile}', Measurement Targets={meas_rows:,}, Crafting References={craft_rows:,}, Fingerprint={server_fp[:16]}...")
    return True


class RemoteServerOracle:
    """
    Oracle interface querying the target server via HTTP POST to /api/server/data.
    In expanded mode, queries are marked with is_query=True and do not pollute measured target metrics.
    """
    def __init__(self, target_url: Optional[str] = None, max_queries_per_sample: int = 50, scenario_name: str = "Adversarial Query", session_id: Optional[str] = None, data_profile: Optional[str] = None):
        self.target_url = target_url or get_server_url()
        self.max_queries_per_sample = max_queries_per_sample
        self.scenario_name = scenario_name
        self.session_id = session_id
        self.data_profile = data_profile or ctx.profile
        self.queries = {}

    def predict(self, X, sample_ids=None, stage=None):
        if not isinstance(X, np.ndarray):
            X = np.array(X, dtype=np.float32)
        if X.ndim == 1:
            X = X.reshape(1, -1)

        preds = []
        sess_id = self.session_id or CURRENT_SIMULATION_SESSION_ID
        for i, row in enumerate(X):
            sid = sample_ids[i] if (sample_ids is not None and i < len(sample_ids)) else None

            origin = random.choice(ORIGIN_POOL)
            family = "Unknown"
            source_type = "Synthetic / Non-dataset"
            is_attack_val = None

            if sid is not None and ctx.dataset.metadata is not None and 0 <= sid < len(ctx.dataset.metadata):
                family = str(ctx.dataset.metadata.iloc[sid].get("attack_family", "Unknown"))
                true_y = int(ctx.dataset.metadata.iloc[sid]["y_binary"])
                is_attack_val = bool(true_y == 1)

                # Check if this row is the exact dataset row or a transformed candidate
                if ctx.dataset.X is not None and 0 <= sid < len(ctx.dataset.X):
                    orig_row = ctx.dataset.X.iloc[sid].values.astype(np.float32)
                    if np.array_equal(row.astype(np.float32), orig_row):
                        source_type = "Dataset-derived"
                    else:
                        source_type = "Original sample family (Transformed via verified simulation session)"
                        family = f"Transformed (Original: {family})"
                else:
                    source_type = "Dataset-derived"

            payload = {
                "source_ip": origin["ip"],
                "destination_ip": "192.168.1.10",
                "attack_scenario": self.scenario_name,
                "traffic_family": family,
                "traffic_family_source": source_type,
                "country": origin["country"],
                "is_attack": is_attack_val,
                "session_id": sess_id,
                "sample_id": int(sid) if sid is not None else None,
                "feature_vector": row.tolist(),
                "data_profile": self.data_profile,
                "is_query": True,
                "query_stage": stage or self.scenario_name
            }
            res = send_packet(self.target_url, payload)
            query_key = sid if sid is not None else "unbound"
            # Accept only 403 as Attack (1) and 200 as Benign (0)
            if res["status_code"] == 403:
                self.queries[query_key] = self.queries.get(query_key, 0) + 1
                preds.append(1)
            elif res["status_code"] == 200:
                self.queries[query_key] = self.queries.get(query_key, 0) + 1
                preds.append(0)
            else:
                # 400/422/500 or transport error is an oracle error, not an evasion!
                err_body = res.get("body", {})
                err_msg = err_body.get("message") or err_body.get("error") or f"HTTP {res['status_code']}"
                raise RuntimeError(
                    f"Target oracle failure: HTTP {res['status_code']} from {self.target_url}: {err_msg}. "
                    f"This error cannot be interpreted as Benign or evasion."
                )

        return np.array(preds, dtype=int)

    def get_query_count(self, sample_id):
        return self.queries.get(sample_id, 0)


TargetOracle = RemoteServerOracle


class AttackerContext:
    """
    Manages schemas, models, datasets, and seeded shuffled queues without replacement.
    Supports both 'expanded' (72k measurement targets) and 'fixture20' profiles.
    """
    def __init__(self, profile: Optional[str] = None, seed: int = 42):
        self.seed = seed
        self.rng = random.Random(seed)
        self.profile = profile or os.environ.get("IDS_DATA_PROFILE", "expanded").strip().lower()

        m_dir = RUNTIME_DIR / "model"
        with open(m_dir / "feature_names.json", "r") as f:
            self.feature_names = json.load(f)
        with open(m_dir / "feature_mask.json", "r") as f:
            mask_dict = json.load(f)
            self.modifiable_mask = np.array(list(mask_dict.values())[0], dtype=bool)
        self.bounds_df = pd.read_parquet(m_dir / "training_bounds.parquet")

        self.dataset: Optional[LoadedDataset] = None
        self.X_demo: Optional[pd.DataFrame] = None
        self.meta_demo: Optional[pd.DataFrame] = None
        self.attack_indices: np.ndarray = np.array([], dtype=int)
        self.benign_indices: np.ndarray = np.array([], dtype=int)

        self.benign_queue = deque()
        self.attack_queue = deque()
        self.ddos_queue = deque()
        self.cycle_counts = {"benign": 0, "attack": 0, "ddos": 0}

        self.load_dataset(self.profile, seed=self.seed)

    def load_dataset(self, profile: str, seed: Optional[int] = None):
        """Loads dataset and initializes seeded queues without replacement."""
        self.profile = profile
        if seed is not None:
            self.seed = seed
            self.rng = random.Random(seed)

        self.dataset = get_dataset(self.profile)
        self.X_demo = self.dataset.X
        self.meta_demo = self.dataset.metadata

        if self.profile == "expanded":
            self.attack_indices = self.dataset.measurement_attack_indices
            self.benign_indices = self.dataset.measurement_benign_indices
        else:
            y = self.meta_demo["y_binary"].values
            self.attack_indices = np.where(y == 1)[0]
            self.benign_indices = np.where(y == 0)[0]

        self.reset_queues(self.seed)

    def reset_queues(self, seed: Optional[int] = None):
        """Resets and reshuffles measurement target queues using specified or current seed."""
        if seed is not None:
            self.seed = seed
            self.rng = random.Random(seed)

        # Shuffle measurement targets into queues
        b_list = list(self.benign_indices)
        a_list = list(self.attack_indices)
        d_list = list(self.dataset.measurement_ddos_indices)

        self.rng.shuffle(b_list)
        self.rng.shuffle(a_list)
        self.rng.shuffle(d_list)

        self.benign_queue = deque(b_list)
        self.attack_queue = deque(a_list)
        self.ddos_queue = deque(d_list)
        self.cycle_counts = {"benign": 0, "attack": 0, "ddos": 0}

    def set_profile(self, profile: str, seed: Optional[int] = None):
        """Switches active dataset profile and reinitializes queues."""
        if profile != self.profile or seed != self.seed:
            self.load_dataset(profile, seed=seed)

    def draw_measurement_target(self, is_attack: bool = False, family_subset: Optional[str] = None) -> int:
        """
        Draws a measurement target ID without replacement.
        When a queue is exhausted, logs the transition and reshuffles for a new cycle.
        """
        if self.profile == "fixture20":
            # Fixture mode legacy behavior
            indices = self.attack_indices if is_attack else self.benign_indices
            return int(self.rng.choice(indices))

        if is_attack:
            if family_subset == "DDoS":
                if len(self.ddos_queue) == 0:
                    self.cycle_counts["ddos"] += 1
                    print(f"{YELLOW}[Sampling] DDoS measurement pool exhausted ({len(self.dataset.measurement_ddos_indices):,} rows). Reshuffling pool for cycle {self.cycle_counts['ddos']} (seed={self.seed})...{RESET}")
                    d_list = list(self.dataset.measurement_ddos_indices)
                    self.rng.shuffle(d_list)
                    self.ddos_queue = deque(d_list)
                return self.ddos_queue.popleft()
            else:
                if len(self.attack_queue) == 0:
                    self.cycle_counts["attack"] += 1
                    print(f"{YELLOW}[Sampling] Attack measurement pool exhausted ({len(self.dataset.measurement_attack_indices):,} rows). Reshuffling pool for cycle {self.cycle_counts['attack']} (seed={self.seed})...{RESET}")
                    a_list = list(self.attack_indices)
                    self.rng.shuffle(a_list)
                    self.attack_queue = deque(a_list)
                return self.attack_queue.popleft()
        else:
            if len(self.benign_queue) == 0:
                self.cycle_counts["benign"] += 1
                print(f"{YELLOW}[Sampling] Benign measurement pool exhausted ({len(self.dataset.measurement_benign_indices):,} rows). Reshuffling pool for cycle {self.cycle_counts['benign']} (seed={self.seed})...{RESET}")
                b_list = list(self.benign_indices)
                self.rng.shuffle(b_list)
                self.benign_queue = deque(b_list)
            return self.benign_queue.popleft()


ctx = AttackerContext()


def run_single_flow(
    flow_type: Optional[str] = None,
    is_attack: bool = False,
    vector: Optional[np.ndarray] = None,
    sample_id: Optional[int] = None,
    scenario: str = "Background Traffic",
    session_id: Optional[str] = None,
    is_query: bool = False,
    query_stage: Optional[str] = None
):
    """
    Sends a single network packet to the server with real dataset metadata and profile tracking.
    Draws targets without replacement from the measurement pool if sample_id is not specified.
    """
    origin = random.choice(ORIGIN_POOL if is_attack else BENIGN_POOL)
    t_str = time.strftime("%H:%M:%S")

    feat_list = None
    chosen_idx = sample_id
    if vector is not None:
        feat_list = vector.tolist()
        resolved_family = flow_type or "Unknown"
        resolved_family_source = "Dataset-derived" if chosen_idx is not None else "Synthetic / Non-dataset"
    else:
        if chosen_idx is None:
            chosen_idx = ctx.draw_measurement_target(is_attack=is_attack, family_subset=flow_type)
        feat_list = ctx.dataset.X.iloc[chosen_idx].values.tolist()
        resolved_family = str(ctx.dataset.metadata.iloc[chosen_idx].get("attack_family", "Normal" if not is_attack else "Unknown"))
        resolved_family_source = "Dataset-derived"

    active_sess = session_id if session_id is not None else CURRENT_SIMULATION_SESSION_ID

    payload = {
        "source_ip": origin["ip"],
        "destination_ip": "192.168.1.10",
        "attack_scenario": scenario,
        "traffic_family": resolved_family,
        "traffic_family_source": resolved_family_source,
        "country": origin["country"],
        "is_attack": is_attack,
        "session_id": active_sess,
        "sample_id": int(chosen_idx) if chosen_idx is not None else None,
        "feature_vector": feat_list,
        "data_profile": ctx.profile,
        "is_query": is_query,
        "query_stage": query_stage
    }

    res = send_packet(get_server_url(), payload)

    if res["status_code"] == 0:
        print(f"{RED}[FAIL] Could not connect to target server at {get_server_url()}. Is server.py running?{RESET}")
        return res

    if res["status_code"] not in [200, 403]:
        print(f"[{t_str}] {BG_RED}{WHITE}{BOLD} ERROR (HTTP {res['status_code']}) {RESET} Server returned error: {res['body']}")
        return res

    sample_tag = f"(ID: {chosen_idx})" if chosen_idx is not None else ""
    if res["blocked"]:
        badge = f"{BG_RED}{WHITE}{BOLD} BLOCKED (403) {RESET}"
        conf = res["body"].get("details", {}).get("confidence", "0.95")
        defense = res["body"].get("details", {}).get("defense", "active").upper()
        mode = res["body"].get("details", {}).get("mode", "").upper()
        used_int = res["body"].get("details", {}).get("used_intensity", "")
        print(f"[{t_str}] {badge} {resolved_family:<22} {sample_tag:<12} from {origin['ip']:<15} | Conf: {conf} | Defense: {defense} [{mode}] (used: {used_int})")
    else:
        badge = f"{BG_GREEN}{WHITE}{BOLD} ALLOWED (200) {RESET}"
        print(f"[{t_str}] {badge} {resolved_family:<22} {sample_tag:<12} from {origin['ip']:<15} | Server Status: Resource Granted (Benign)")

    return res


def run_volumetric_burst(attack_type: str = "DDoS", count: int = 6, delay: float = 0.25):
    """Sends a rapid burst of malicious packets drawing real DDoS families from the measurement pool."""
    print(f"\n{RED}{BOLD}>>> Launching Volumetric Burst: {attack_type} ({count} packets from measurement pool)...{RESET}\n")
    for i in range(count):
        run_single_flow(flow_type=attack_type, is_attack=True, scenario="Volumetric Burst")
        time.sleep(delay)
    print(f"\n{GREEN}[OK] Volumetric burst completed.{RESET}\n")


def run_silent_probing_attack():
    """
    Executes Research Attack 1: Silent Probing
    Sequential submission of unchanged baseline flows from measurement pool without querying oracle (0 queries).
    """
    print(f"\n{MAGENTA}{BOLD}======================================================================{RESET}")
    print(f"{MAGENTA}{BOLD}  RESEARCH ATTACK: SILENT PROBING ATTACK{RESET}")
    print(f"{WHITE}  Model: Sequential Submission of Unchanged Measurement Flows (0 Queries){RESET}")
    print(f"{MAGENTA}{BOLD}======================================================================{RESET}\n")

    sess_id = start_simulation_session("Silent Probing")
    try:
        attack = SilentProbingAttack(ctx.feature_names, ctx.modifiable_mask, ctx.bounds_df)

        # Draw attack target strictly from measurement pool
        chosen_idx = ctx.draw_measurement_target(is_attack=True)
        x_orig = ctx.dataset.X.iloc[chosen_idx].values
        family = str(ctx.dataset.metadata.iloc[chosen_idx].get("attack_family", "Unknown"))

        print(f"{WHITE}Selected Measurement Target: {CYAN}{family}{RESET} (Evaluation ID: {chosen_idx}, Role: Measurement)")
        print(f"{DIM}Preparing unchanged baseline flow (Silent Probing makes 0 oracle queries)...{RESET}")

        res = attack.generate(x_orig, oracle=None, sample_id=chosen_idx, true_label=1)
        print(f"{GREEN}[OK] Flow prepared (0 queries). Transmitting final measured candidate to server...{RESET}")

        resp = run_single_flow(
            flow_type=family,
            is_attack=True,
            vector=res.X_adv,
            sample_id=chosen_idx,
            scenario="Silent Probing",
            session_id=sess_id,
            is_query=False
        )

        if resp is None or resp["status_code"] not in [200, 403]:
            return

        t_str = time.strftime("%H:%M:%S")
        if resp["blocked"]:
            details = resp["body"].get("details", {})
            print(f"[{t_str}] {BG_RED}{WHITE}{BOLD} DETECTED & BLOCKED (403) {RESET}")
            print(f"IDS Classification: ATTACK | Family: {family} | Defense: {details.get('defense', '').upper()} [{details.get('mode', '').upper()}]")
            print(f"{CYAN}Analysis: Target IDS classified silent probing flow as Attack.{RESET}\n")
        else:
            print(f"[{t_str}] {BG_GREEN}{WHITE}{BOLD} EVADED & ALLOWED (200) {RESET}")
            print(f"IDS Classification: BENIGN | Family: {family}")
            print(f"{YELLOW}Warning: Attack penetrated protected server under current defense configuration.{RESET}\n")
    finally:
        stop_simulation_session()


def run_surrogate_transfer_attack():
    """
    Executes Research Attack 2: Surrogate Transferability
    Trains a local Decision Tree surrogate from crafting query pool, then crafts candidate
    against a measurement target and submits the final candidate once as a measured flow.
    """
    print(f"\n{MAGENTA}{BOLD}======================================================================{RESET}")
    print(f"{MAGENTA}{BOLD}  RESEARCH ATTACK: SURROGATE TRANSFERABILITY ATTACK{RESET}")
    print(f"{WHITE}  Model: Local Decision Tree Surrogate with Boundary Transfer{RESET}")
    print(f"{WHITE}  Crafting Query References: Disclosed Crafting Pool | Measured Target: Measurement Pool{RESET}")
    print(f"{MAGENTA}{BOLD}======================================================================{RESET}\n")

    sess_id = start_simulation_session("Surrogate Transferability")
    try:
        attack = SurrogateTransferAttack(ctx.feature_names, ctx.modifiable_mask, ctx.bounds_df)
        oracle = RemoteServerOracle(target_url=get_server_url(), scenario_name="Surrogate Transferability", session_id=sess_id)

        # Select reference query pool strictly from CRAFTING pool
        if ctx.profile == "expanded":
            # Select bounded set of 20 crafting samples (10 benign, 10 attack)
            ref_b = ctx.dataset.crafting_benign_indices[:10]
            ref_a = ctx.dataset.crafting_attack_indices[:10]
            ref_ids = np.concatenate([ref_b, ref_a]).tolist()
            x_pool = ctx.dataset.X.iloc[ref_ids].values
            print(f"{DIM}Selecting 20 bounded reference flows from 18,000 crafting pool (10 benign, 10 attack)...{RESET}")
            print(f"{WHITE}Crafting Query Sample IDs: {CYAN}{ref_ids[:5]}... + {len(ref_ids)-5} more{RESET}")
        else:
            ref_ids = list(range(len(ctx.dataset.X)))
            x_pool = ctx.dataset.X.values
            print(f"{DIM}Using {len(ref_ids)} fixture samples for surrogate training...{RESET}")

        print(f"{DIM}Fitting surrogate model via query telemetry (is_query=True)...{RESET}")
        y_oracle = oracle.predict(x_pool, sample_ids=ref_ids, stage="surrogate_fitting")
        attack.fit_surrogate(x_pool, y_oracle)
        print(f"{GREEN}[OK] Surrogate decision tree fitted successfully with benign class resolved.{RESET}")

        # Draw target flow strictly from MEASUREMENT pool
        target_idx = ctx.draw_measurement_target(is_attack=True)
        x_orig = ctx.dataset.X.iloc[target_idx].values
        family = str(ctx.dataset.metadata.iloc[target_idx].get("attack_family", "Unknown"))

        print(f"{WHITE}Targeting Measurement Attack Sample: {CYAN}{family}{RESET} (Evaluation ID: {target_idx}, Role: Measurement)")
        x_cand, mags = attack.generate_candidate(x_orig)
        if x_cand is None:
            print(f"{YELLOW}[!] Surrogate yielded no feasible candidate for target {target_idx}.{RESET}\n")
            return

        print(f"{DIM}Submitting finalized candidate once as measured target flow...{RESET}")
        resp = run_single_flow(
            flow_type=family,
            is_attack=True,
            vector=x_cand,
            sample_id=target_idx,
            scenario="Surrogate Transferability",
            session_id=sess_id,
            is_query=False
        )

        t_str = time.strftime("%H:%M:%S")
        if resp is not None and resp.get("status_code") == 200:
            print(f"[{t_str}] {BG_GREEN}{WHITE}{BOLD} TRANSFER SUCCESS (200 OK) {RESET}")
            print(f"IDS Classification: BENIGN (EVASION SUCCESS) | Family: {family}")
            print(f"{YELLOW}Result: Adversarial candidate transferred and evaded target classifier!{RESET}\n")
        else:
            print(f"[{t_str}] {BG_RED}{WHITE}{BOLD} TRANSFER BLOCKED (403 FORBIDDEN) {RESET}")
            print(f"IDS Classification: ATTACK (DEFENSE HELD) | Family: {family}")
            print(f"{CYAN}Result: Target IDS defense prevented surrogate boundary transfer.{RESET}\n")
    finally:
        stop_simulation_session()


def run_decision_boundary_attack(max_queries: int = 50, steps: int = 10):
    """
    Executes Research Attack 3: Decision Boundary Bisection Search
    Uses crafting benign references for binary search, targeting a measurement attack flow.
    Submits the final candidate once as a measured flow.
    """
    print(f"\n{MAGENTA}{BOLD}======================================================================{RESET}")
    print(f"{MAGENTA}{BOLD}  RESEARCH ATTACK: DECISION BOUNDARY ATTACK (1D BISECTION){RESET}")
    print(f"{WHITE}  Model: Query-Guided Binary Search Boundary Finding{RESET}")
    print(f"{WHITE}  References: Crafting Benign Pool | Target: Measurement Attack Pool{RESET}")
    print(f"{MAGENTA}{BOLD}======================================================================{RESET}\n")

    sess_id = start_simulation_session("Decision-Boundary Attack")
    try:
        attack = DecisionBoundaryAttack(
            ctx.feature_names, ctx.modifiable_mask, ctx.bounds_df,
            max_queries=max_queries, binary_search_steps=steps
        )
        oracle = RemoteServerOracle(target_url=get_server_url(), max_queries_per_sample=max_queries, scenario_name="Decision-Boundary Attack", session_id=sess_id)

        # Benign reference pool drawn strictly from CRAFTING pool
        if ctx.profile == "expanded":
            ref_ids = ctx.dataset.crafting_benign_indices[:50].tolist()
            reference_pool = ctx.dataset.X.iloc[ref_ids].values
            print(f"{DIM}Selecting 50 crafting benign references from 14,936 crafting benign pool...{RESET}")
            print(f"{WHITE}Crafting Benign Reference IDs: {CYAN}{ref_ids[:5]}... + {len(ref_ids)-5} more{RESET}")
        else:
            ref_ids = ctx.benign_indices.tolist()
            reference_pool = ctx.dataset.X.iloc[ref_ids].values
            print(f"{DIM}Using {len(ref_ids)} fixture benign references...{RESET}")

        # Draw target flow strictly from MEASUREMENT pool
        target_idx = ctx.draw_measurement_target(is_attack=True)
        x_orig = ctx.dataset.X.iloc[target_idx].values
        family = str(ctx.dataset.metadata.iloc[target_idx].get("attack_family", "Unknown"))

        print(f"{WHITE}Target Measurement Attack Sample: {CYAN}{family}{RESET} (Evaluation ID: {target_idx}, Role: Measurement)")
        print(f"{DIM}Running bisection search via query telemetry (max query budget: {max_queries})...{RESET}")

        res = attack.generate(x_orig, oracle, sample_id=target_idx, true_label=1, reference_pool=reference_pool)
        queries_used = oracle.get_query_count(target_idx)

        print(f"\n{BOLD}Boundary Search Finished:{RESET} queries_used={queries_used}/{max_queries}, status={res.status_code}, message={res.message}")

        # Submit finalized candidate once as measured target flow
        final_cand = res.X_adv if (res.success and res.X_adv is not None) else x_orig
        print(f"{DIM}Submitting finalized candidate once as measured target flow...{RESET}")
        resp = run_single_flow(
            flow_type=family,
            is_attack=True,
            vector=final_cand,
            sample_id=target_idx,
            scenario="Decision-Boundary Attack",
            session_id=sess_id,
            is_query=False
        )

        t_str = time.strftime("%H:%M:%S")
        if resp is not None and resp.get("status_code") == 200:
            print(f"\n[{t_str}] {BG_GREEN}{WHITE}{BOLD} EVADED BOUNDARY (200 OK) {RESET}")
            print(f"Verdict: Adversarial candidate evaded target classifier!\n")
        else:
            print(f"\n[{t_str}] {BG_RED}{WHITE}{BOLD} DEFENSE HELD (403 BLOCKED) {RESET}")
            print(f"Verdict: Target IDS defense detected and blocked flow.\n")
    finally:
        stop_simulation_session()


def run_comparative_benchmark(defense: Optional[str] = None):
    """
    Local Control-Path Demonstration:
    Compares Base (fixed intensity) versus Recall-Aware (adaptive controller)
    for the selected defense across an identical 5-flow measurement sequence.
    """
    target_def = (defense or "afp").lower()
    print(f"\n{YELLOW}{BOLD}======================================================================{RESET}")
    print(f"{YELLOW}{BOLD}  LOCAL CONTROL-PATH DEMONSTRATION: BASE vs. RECALL-AWARE ({target_def.upper()}){RESET}")
    print(f"{WHITE}  Evaluating identical 5-flow measurement sequence under fixed vs adaptive controller{RESET}")
    print(f"{YELLOW}{BOLD}======================================================================{RESET}\n")

    # Pick 5 fixed measurement attack samples for exact repeatability
    if ctx.profile == "fixture20":
        fixed_indices = ctx.attack_indices[:5].tolist()
    else:
        fixed_indices = ctx.dataset.measurement_attack_indices[:5].tolist()

    print(f"{WHITE}Saved Measurement Target IDs: {CYAN}{fixed_indices}{RESET} (Profile: {ctx.profile})\n")

    # Step 1: Base Mode
    print(f"{WHITE}Step 1: Setting {target_def.upper()} Defense to {YELLOW}BASE MODE (Static Calibrated Intensity){RESET}...")
    if not set_server_defense(target_def) or not set_server_mode("base"):
        print(f"{RED}[FAIL] Setup failed for Base mode.{RESET}")
        return

    # Reset dashboard metrics to clean baseline
    try:
        req = urllib.request.Request(get_reset_url(), data=b"{}", headers={"Content-Type": "application/json"})
        urllib.request.urlopen(req, timeout=3)
    except Exception as e:
        print(f"{RED}[FAIL] Reset failed: {e}{RESET}")
        return
    time.sleep(0.5)

    print(f"{DIM}Sending 5 fixed attack flows under Base {target_def.upper()}...{RESET}")
    results_base = []
    for idx in fixed_indices:
        x_row = ctx.dataset.X.iloc[idx].values
        fam = str(ctx.dataset.metadata.iloc[idx].get("attack_family", "Unknown"))
        r = run_single_flow(flow_type=fam, is_attack=True, vector=x_row, sample_id=idx, scenario="Base Demonstration")
        if r is None or r.get("status_code") not in (200, 403):
            print(f"{RED}[FAIL] Error during Base mode flow transmission.{RESET}")
            return
        results_base.append(r)
        time.sleep(0.2)

    time.sleep(0.8)

    # Step 2: Recall-Aware Mode
    print(f"\n{WHITE}Step 2: Activating {GREEN}RECALL-AWARE CONTROLLER (Dynamic Feedback Window){RESET}...")
    if not set_server_mode("recall-aware"):
        print(f"{RED}[FAIL] Setup failed for Recall-Aware mode.{RESET}")
        return

    # Reset dashboard metrics to clean baseline
    try:
        req = urllib.request.Request(get_reset_url(), data=b"{}", headers={"Content-Type": "application/json"})
        urllib.request.urlopen(req, timeout=3)
    except Exception as e:
        print(f"{RED}[FAIL] Reset failed: {e}{RESET}")
        return
    time.sleep(0.5)

    sess_id = start_simulation_session("Recall-Aware Demonstration")
    results_recall = []
    try:
        print(f"{DIM}Sending the SAME 5 attack flows under Recall-Aware {target_def.upper()}...{RESET}")
        for idx in fixed_indices:
            x_row = ctx.dataset.X.iloc[idx].values
            fam = str(ctx.dataset.metadata.iloc[idx].get("attack_family", "Unknown"))
            r = run_single_flow(flow_type=fam, is_attack=True, vector=x_row, sample_id=idx, scenario="Recall-Aware Demonstration", session_id=sess_id)
            if r is None or r.get("status_code") not in (200, 403):
                print(f"{RED}[FAIL] Error during Recall-Aware mode flow transmission.{RESET}")
                return
            results_recall.append(r)
            time.sleep(0.2)
    finally:
        stop_simulation_session()

    # Query final telemetry from server to verify actual observed state
    dashboard_data = None
    try:
        req = urllib.request.Request(get_stats_url())
        with urllib.request.urlopen(req, timeout=3) as resp:
            dashboard_data = json.loads(resp.read().decode())
    except Exception as e:
        print(f"{YELLOW}[!] Could not fetch final dashboard telemetry: {e}{RESET}")

    stats = dashboard_data.get("stats", {}) if isinstance(dashboard_data, dict) else {}
    afp_info = dashboard_data.get("afp", {}) if isinstance(dashboard_data, dict) else {}

    raw_intensity = afp_info.get("intensity") if isinstance(afp_info, dict) else None
    obs_state = afp_info.get("controller_state") if isinstance(afp_info, dict) else None

    obs_intensity_str = f"{raw_intensity:.5f}" if isinstance(raw_intensity, (int, float)) else None
    has_controller_telemetry = (obs_intensity_str is not None and obs_state is not None)

    detected_count = sum(1 for r in results_recall if r.get("blocked"))
    total_count = len(results_recall)

    tp_count = stats.get("tp", detected_count)
    fn_count = stats.get("fn", total_count - detected_count)

    print(f"\n{CYAN}{BOLD}Demonstration completed. Check the SOC dashboard!{RESET}")
    print(f"{WHITE}Observed Run Results: {detected_count}/{total_count} attack flows detected (TP={tp_count}, FN={fn_count}).{RESET}")

    if not has_controller_telemetry:
        print(f"{YELLOW}Controller telemetry is unavailable; cannot confirm final intensity or state.{RESET}\n")
    elif detected_count == total_count:
        print(f"{WHITE}With all {total_count} attack flows detected (Recall = 1.000 >= Rmin), the controller maintained intensity={obs_intensity_str} in {obs_state} state.{RESET}")
        print(f"{DIM}Note: If lower recall occurs (e.g., under adversarial evasion), the controller adapts downward (0.00030 -> 0.00012). See the automated test suite for the verified adaptation sequence.{RESET}\n")
    else:
        print(f"{WHITE}Observed evasion resulted in controller adaptation: intensity={obs_intensity_str}, state={obs_state}.{RESET}\n")


def run_cross_defense_comparison(count: int = 5):
    """
    Replays one saved measurement sequence across AFP, RS, FS, and None in Base mode.
    Sequence contains both benign and attack rows to evaluate TP, FN, FP, TN, Recall, and FPR.
    """
    print(f"\n{CYAN}{BOLD}======================================================================{RESET}")
    print(f"{CYAN}{BOLD}  CROSS-DEFENSE BENCHMARK: REPLAY IDENTICAL SEQUENCE (BASE MODE){RESET}")
    print(f"{WHITE}  Evaluating AFP, RS, FS, and None across identical saved measurement flows{RESET}")
    print(f"{CYAN}{BOLD}======================================================================{RESET}\n")

    # Select saved sequence containing both classes (e.g. 2 benign, 3 attack)
    if ctx.profile == "fixture20":
        sample_sequence = [0, 1, 10, 11, 12]
    else:
        sample_sequence = list(ctx.dataset.measurement_benign_indices[:2]) + list(ctx.dataset.measurement_attack_indices[:3])

    print(f"{WHITE}Replay Sequence IDs: {CYAN}{sample_sequence}{RESET} (Profile: {ctx.profile}, Seed: {ctx.seed})")
    print(f"{DIM}Classes in sequence: {sum(1 for sid in sample_sequence if ctx.dataset.metadata.iloc[sid]['y_binary'] == 0)} Benign, {sum(1 for sid in sample_sequence if ctx.dataset.metadata.iloc[sid]['y_binary'] == 1)} Attack{RESET}\n")

    defenses = ["afp", "rs", "fs", "none"]
    arm_results = {}

    for def_name in defenses:
        print(f"{WHITE}Evaluating Defense: {YELLOW}{def_name.upper()}{RESET} (Base Mode)...")
        if not set_server_defense(def_name) or not set_server_mode("base"):
            print(f"{RED}[FAIL] Setup failed for {def_name}.{RESET}")
            continue

        # Reset counters & controller before arm
        try:
            req = urllib.request.Request(get_reset_url(), data=b"{}", headers={"Content-Type": "application/json"})
            urllib.request.urlopen(req, timeout=3)
        except Exception as e:
            print(f"{RED}[FAIL] Reset failed for {def_name}: {e}{RESET}")
            continue
        time.sleep(0.3)

        flow_verdicts = []
        for sid in sample_sequence:
            x_row = ctx.dataset.X.iloc[sid].values
            true_y = int(ctx.dataset.metadata.iloc[sid]["y_binary"])
            fam = str(ctx.dataset.metadata.iloc[sid].get("attack_family", "Normal" if true_y == 0 else "Unknown"))
            r = run_single_flow(flow_type=fam, is_attack=(true_y == 1), vector=x_row, sample_id=sid, scenario=f"Benchmark-{def_name.upper()}")
            if r and r.get("status_code") in (200, 403):
                flow_verdicts.append(r)
            time.sleep(0.15)

        # Retrieve server stats
        try:
            req = urllib.request.Request(get_stats_url())
            with urllib.request.urlopen(req, timeout=3) as resp:
                stats_payload = json.loads(resp.read().decode())
        except Exception:
            stats_payload = {}

        st = stats_payload.get("stats", {})
        afp_info = stats_payload.get("afp", {})

        tp = st.get("tp", 0)
        fn = st.get("fn", 0)
        fp = st.get("fp", 0)
        tn = st.get("tn", 0)

        positives = tp + fn
        negatives = fp + tn
        recall_val = f"{(tp / positives):.3f}" if positives > 0 else "N/A"
        fpr_val = f"{(fp / negatives):.3f}" if negatives > 0 else "N/A"

        conf_type = "Vote fraction (ensemble)" if def_name == "rs" else "RF probability"
        intensity_val = afp_info.get("intensity", "N/A")

        arm_results[def_name] = {
            "defense": def_name.upper(),
            "tp": tp, "fn": fn, "fp": fp, "tn": tn,
            "recall": recall_val, "fpr": fpr_val,
            "intensity": intensity_val,
            "conf_semantics": conf_type
        }
        time.sleep(0.5)

    # Print comparative results table
    print(f"\n{CYAN}{BOLD}========================================================================================{RESET}")
    print(f"{CYAN}{BOLD}  CROSS-DEFENSE EVALUATION SUMMARY (IDENTICAL 5-FLOW MEASUREMENT SEQUENCE){RESET}")
    print(f"{CYAN}{BOLD}========================================================================================{RESET}")
    print(f"{BOLD}{'Defense':<10} {'Mode':<8} {'TP':<5} {'FN':<5} {'FP':<5} {'TN':<5} {'Recall':<10} {'FPR':<10} {'Intensity':<12} {'Confidence Semantics':<24}{RESET}")
    print(f"{'-'*96}")
    for def_name in defenses:
        if def_name in arm_results:
            r = arm_results[def_name]
            int_str = f"{r['intensity']:.5f}" if isinstance(r['intensity'], (int, float)) else str(r['intensity'])
            print(f"{r['defense']:<10} {'Base':<8} {r['tp']:<5} {r['fn']:<5} {r['fp']:<5} {r['tn']:<5} {r['recall']:<10} {r['fpr']:<10} {int_str:<12} {r['conf_semantics']:<24}")
    print(f"{'-'*96}")
    print(f"{DIM}Note: RS confidence reflects ensemble vote fractions (11 sub-models), whereas AFP, FS, and None report direct Random Forest probability estimates. Equal verdicts do not imply identical internal representations.{RESET}\n")


def run_continuous_stream(delay: float = 1.0):
    """Streams live network traffic to demonstrate real-time dashboard updates."""
    print(f"\n{CYAN}{BOLD}>>> Starting Continuous Traffic & Threat Stream (Press Ctrl+C to stop)...{RESET}\n")
    attack_types = ["DDoS", "Port Scan", "Brute Force", "Malware", "Bot", "SlowHTTPTest"]

    try:
        while True:
            is_attack = (random.random() < 0.35)
            if is_attack:
                a_type = random.choice(attack_types)
                run_single_flow(flow_type=a_type, is_attack=True)
            else:
                run_single_flow(flow_type="Normal", is_attack=False)
            time.sleep(delay)
    except KeyboardInterrupt:
        print(f"\n{YELLOW}[!] Stream paused by user.{RESET}\n")


def interactive_menu():
    print_banner()
    while True:
        print(f"{WHITE}{BOLD}======================================================================{RESET}")
        print(f"{CYAN}{BOLD}  ATTACKER ACTIONS (Black-Box Simulation POV):{RESET}")
        print(f"  {CYAN}[1]{RESET} Send Legitimate Benign Traffic Flow (Sampled from Measurement Pool)")
        print(f"  {RED}[2]{RESET} Launch DDoS Volumetric Flood Burst (Sampled from DDoS Measurement Pool)")
        print(f"  {MAGENTA}[3]{RESET} Launch Silent Probing (Sequential Unchanged Baseline Flows, 0 Queries)")
        print(f"  {MAGENTA}[4]{RESET} Launch Surrogate Transferability Attack (Decision Tree Surrogate, Crafting References)")
        print(f"  {MAGENTA}[5]{RESET} Launch Decision Boundary Attack (1D Bisection Search, Crafting Benign References)")
        print(f"  {BLUE}[6]{RESET} Continuous Real-time Traffic Stream (1 flow / sec)")
        print(f"")
        print(f"{YELLOW}{BOLD}  OPERATOR SETUP CONTROLS (Local Experiment Configuration):{RESET}")
        print(f"  {YELLOW}[7]{RESET} Local Control-Path Demonstration: Base vs. Recall-Aware (Selected Defense, 5 Flows)")
        print(f"  {YELLOW}[8]{RESET} Cross-Defense Comparison: Replay Saved Sequence Across [AFP, RS, FS, None] (Base Mode)")
        print(f"  {WHITE}[9]{RESET} Switch Active Defense: [AFP -> RS -> FS -> None]")
        print(f"  {WHITE}[10]{RESET} Toggle Controller Mode: [Recall-Aware <-> Base]")
        print(f"  {DIM}[0]{RESET} Exit")
        print(f"{WHITE}{BOLD}======================================================================{RESET}\n")

        try:
            choice = input(f"{BOLD}Enter option (0-10): {RESET}").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nExiting.")
            sys.exit(0)

        if choice == "1":
            run_single_flow(flow_type="Normal", is_attack=False)
        elif choice == "2":
            run_volumetric_burst("DDoS", count=6, delay=0.25)
        elif choice == "3":
            run_silent_probing_attack()
        elif choice == "4":
            run_surrogate_transfer_attack()
        elif choice == "5":
            run_decision_boundary_attack()
        elif choice == "6":
            run_continuous_stream(delay=1.0)
        elif choice == "7":
            run_comparative_benchmark()
        elif choice == "8":
            run_cross_defense_comparison(count=5)
        elif choice == "9":
            cycle = {"afp": "rs", "rs": "fs", "fs": "none", "none": "afp"}
            try:
                with urllib.request.urlopen(get_stats_url(), timeout=3) as resp:
                    data = json.loads(resp.read().decode("utf-8"))
                    curr = data.get("afp", {}).get("defense_name", "afp")
                    nxt = cycle.get(curr, "afp")
                    set_server_defense(nxt)
                    print(f"Defense switched to: {GREEN}{nxt.upper()}{RESET}\n")
            except Exception as e:
                print(f"{RED}Could not cycle defense: {e}{RESET}\n")
        elif choice == "10":
            try:
                with urllib.request.urlopen(get_stats_url(), timeout=3) as resp:
                    data = json.loads(resp.read().decode("utf-8"))
                    curr_mode = data.get("afp", {}).get("mode", "recall-aware")
                    nxt_mode = "base" if curr_mode == "recall-aware" else "recall-aware"
                    set_server_mode(nxt_mode)
                    print(f"Controller Mode switched to: {CYAN}{nxt_mode.upper()}{RESET}\n")
            except Exception as e:
                print(f"{RED}Could not toggle mode: {e}{RESET}\n")
        elif choice == "0":
            print("Exiting attacker simulation console.")
            break
        else:
            print(f"{RED}Invalid selection. Please choose 0-10.{RESET}\n")


def main():
    parser = argparse.ArgumentParser(description="Attacker Simulation Console for IDS + Recall-Aware Platform")
    parser.add_argument("--mode", choices=["menu", "silent", "surrogate", "boundary", "ddos", "benign", "stream", "compare", "compare-all"], default="menu")
    parser.add_argument("--dataset", choices=["expanded", "fixture20"], default=None, help="Dataset profile (expanded or fixture20)")
    parser.add_argument("--seed", type=int, default=42, help="RNG seed for target sampling without replacement")
    parser.add_argument("--defense", choices=["afp", "rs", "fs", "none"], default=None)
    parser.add_argument("--controller", choices=["recall-aware", "base"], default=None)
    parser.add_argument("--target", default=DEFAULT_SERVER_URL, help="Target URL for protected server endpoint")
    parser.add_argument("--count", type=int, default=5)
    parser.add_argument("--delay", type=float, default=0.3)

    args = parser.parse_args()

    # Make --target effective for all server endpoints dynamically
    if args.target:
        set_target_url(args.target)

    # Initialize / update profile and seed based on CLI flags
    active_profile = args.dataset or os.environ.get("IDS_DATA_PROFILE", "expanded").strip().lower()
    ctx.set_profile(active_profile, seed=args.seed)

    # Check backend compatibility before sending flows
    check_backend_compatibility()

    if args.defense:
        set_server_defense(args.defense)
    if args.controller:
        set_server_mode(args.controller)

    if args.mode == "menu":
        interactive_menu()
    elif args.mode == "silent":
        run_silent_probing_attack()
    elif args.mode == "surrogate":
        run_surrogate_transfer_attack()
    elif args.mode == "boundary":
        run_decision_boundary_attack(max_queries=50)
    elif args.mode == "compare":
        run_comparative_benchmark(defense=args.defense)
    elif args.mode == "compare-all":
        run_cross_defense_comparison(count=args.count)
    elif args.mode == "ddos":
        run_volumetric_burst("DDoS", count=args.count, delay=args.delay)
    elif args.mode == "benign":
        run_single_flow("Normal", is_attack=False)
    elif args.mode == "stream":
        run_continuous_stream(delay=args.delay)


if __name__ == "__main__":
    main()
