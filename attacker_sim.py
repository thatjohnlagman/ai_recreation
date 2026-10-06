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


class AttackerContext:
    """
    Manages schemas, models, datasets, and seeded shuffled queues without replacement.
    Supports both 'expanded' (72k measurement targets) and 'fixture20' profiles.
    Coordinates target consumption across general attack and DDoS queues via shared consumed ledger.
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
        self._ddos_set: set = set()

        self.benign_queue = deque()
        self.attack_queue = deque()
        self.ddos_queue = deque()
        self.consumed_attack_ids: set = set()
        self.consumed_ddos_ids: set = set()
        self.consumed_benign_ids: set = set()
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
            self._ddos_set = set(self.dataset.measurement_ddos_indices)
        else:
            y = self.meta_demo["y_binary"].values
            self.attack_indices = np.where(y == 1)[0]
            self.benign_indices = np.where(y == 0)[0]
            self._ddos_set = set(self.dataset.measurement_ddos_indices)

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

        self.consumed_attack_ids.clear()
        self.consumed_ddos_ids.clear()
        self.consumed_benign_ids.clear()
        self.cycle_counts = {"benign": 0, "attack": 0, "ddos": 0}

    def set_profile(self, profile: str, seed: Optional[int] = None):
        """Switches active dataset profile and reinitializes queues."""
        if profile != self.profile or seed != self.seed:
            self.load_dataset(profile, seed=seed)

    def draw_measurement_target(self, is_attack: bool = False, family_subset: Optional[str] = None) -> int:
        """
        Draws a measurement target ID without replacement.
        Coordinates consumption across overlapping queues using a shared consumed ledger.
        When a queue is exhausted, logs the transition accurately and reshuffles for a new cycle.
        """
        if self.profile == "fixture20":
            indices = self.attack_indices if is_attack else self.benign_indices
            return int(self.rng.choice(indices))

        if is_attack:
            if family_subset == "DDoS":
                total_ddos = len(self.dataset.measurement_ddos_indices)
                unconsumed_ddos = [x for x in self.ddos_queue if x not in self.consumed_ddos_ids]
                if not unconsumed_ddos or len(self.consumed_ddos_ids) >= total_ddos:
                    self.cycle_counts["ddos"] += 1
                    self.consumed_ddos_ids.clear()
                    rem_general = len(self.attack_indices) - len(self.consumed_attack_ids)
                    print(f"{YELLOW}[Sampling] DDoS family subset exhausted ({total_ddos:,} rows). Reshuffling DDoS pool for cycle {self.cycle_counts['ddos']} (seed={self.seed}). General attack pool has {rem_general:,} unused targets remaining.{RESET}")
                    d_list = list(self.dataset.measurement_ddos_indices)
                    self.rng.shuffle(d_list)
                    self.ddos_queue = deque(d_list)

                while self.ddos_queue:
                    tid = self.ddos_queue.popleft()
                    if tid not in self.consumed_ddos_ids:
                        self.consumed_ddos_ids.add(tid)
                        self.consumed_attack_ids.add(tid)
                        return tid

                self.cycle_counts["ddos"] += 1
                self.consumed_ddos_ids.clear()
                rem_general = len(self.attack_indices) - len(self.consumed_attack_ids)
                print(f"{YELLOW}[Sampling] DDoS family subset exhausted ({total_ddos:,} rows). Reshuffling DDoS pool for cycle {self.cycle_counts['ddos']} (seed={self.seed}). General attack pool has {rem_general:,} unused targets remaining.{RESET}")
                d_list = list(self.dataset.measurement_ddos_indices)
                self.rng.shuffle(d_list)
                self.ddos_queue = deque(d_list)
                tid = self.ddos_queue.popleft()
                self.consumed_ddos_ids.add(tid)
                self.consumed_attack_ids.add(tid)
                return tid

            else:
                total_attack = len(self.attack_indices)
                unconsumed_attack = [x for x in self.attack_queue if x not in self.consumed_attack_ids]
                if not unconsumed_attack or len(self.consumed_attack_ids) >= total_attack:
                    self.cycle_counts["attack"] += 1
                    self.consumed_attack_ids.clear()
                    self.consumed_ddos_ids.clear()
                    print(f"{YELLOW}[Sampling] General attack measurement pool exhausted ({total_attack:,} rows). Reshuffling pool for cycle {self.cycle_counts['attack']} (seed={self.seed})...{RESET}")
                    a_list = list(self.attack_indices)
                    self.rng.shuffle(a_list)
                    self.attack_queue = deque(a_list)

                while self.attack_queue:
                    tid = self.attack_queue.popleft()
                    if tid not in self.consumed_attack_ids:
                        self.consumed_attack_ids.add(tid)
                        if tid in self._ddos_set:
                            self.consumed_ddos_ids.add(tid)
                        return tid

                self.cycle_counts["attack"] += 1
                self.consumed_attack_ids.clear()
                self.consumed_ddos_ids.clear()
                a_list = list(self.attack_indices)
                self.rng.shuffle(a_list)
                self.attack_queue = deque(a_list)
                tid = self.attack_queue.popleft()
                self.consumed_attack_ids.add(tid)
                if tid in self._ddos_set:
                    self.consumed_ddos_ids.add(tid)
                return tid

        else:
            total_benign = len(self.benign_indices)
            unconsumed_benign = [x for x in self.benign_queue if x not in self.consumed_benign_ids]
            if not unconsumed_benign or len(self.consumed_benign_ids) >= total_benign:
                self.cycle_counts["benign"] += 1
                self.consumed_benign_ids.clear()
                print(f"{YELLOW}[Sampling] Benign measurement pool exhausted ({total_benign:,} rows). Reshuffling pool for cycle {self.cycle_counts['benign']} (seed={self.seed})...{RESET}")
                b_list = list(self.benign_indices)
                self.rng.shuffle(b_list)
                self.benign_queue = deque(b_list)

            while self.benign_queue:
                tid = self.benign_queue.popleft()
                if tid not in self.consumed_benign_ids:
                    self.consumed_benign_ids.add(tid)
                    return tid

            self.cycle_counts["benign"] += 1
            self.consumed_benign_ids.clear()
            b_list = list(self.benign_indices)
            self.rng.shuffle(b_list)
            self.benign_queue = deque(b_list)
            tid = self.benign_queue.popleft()
            self.consumed_benign_ids.add(tid)
            return tid

    def select_crafting_references_surrogate(self, n_benign: int = 10, n_attack: int = 10) -> List[int]:
        """
        Selects bounded, seeded references strictly from the crafting pool for surrogate fitting.
        """
        if self.profile == "expanded":
            b_candidates = list(self.dataset.crafting_benign_indices)
            a_candidates = list(self.dataset.crafting_attack_indices)
            chosen_b = self.rng.sample(b_candidates, min(n_benign, len(b_candidates)))
            chosen_a = self.rng.sample(a_candidates, min(n_attack, len(a_candidates)))
            return chosen_b + chosen_a
        else:
            b_idx = self.benign_indices[:n_benign].tolist()
            a_idx = self.attack_indices[:n_attack].tolist()
            return b_idx + a_idx

    def select_crafting_references_boundary(self, n_benign: int = 50) -> List[int]:
        """
        Selects bounded, seeded benign references strictly from the crafting pool for boundary search.
        """
        if self.profile == "expanded":
            b_candidates = list(self.dataset.crafting_benign_indices)
            return self.rng.sample(b_candidates, min(n_benign, len(b_candidates)))
        else:
            return self.benign_indices[:n_benign].tolist()


ctx: Optional[AttackerContext] = None

def get_ctx(profile: Optional[str] = None, seed: Optional[int] = None) -> AttackerContext:
    global ctx
    if ctx is None:
        p = profile or os.environ.get("IDS_DATA_PROFILE", "expanded").strip().lower()
        s = seed if seed is not None else 42
        ctx = AttackerContext(profile=p, seed=s)
    else:
        if profile is not None and profile != ctx.profile:
            ctx.set_profile(profile, seed=seed)
        elif seed is not None and seed != ctx.seed:
            ctx.reset_queues(seed=seed)
    return ctx


def print_banner():
    c = get_ctx()
    print(f"\n{CYAN}{BOLD}======================================================================{RESET}")
    print(f"{CYAN}{BOLD}  CYBER ATTACKER SIMULATION CONSOLE — RESEARCH RUNTIME{RESET}")
    print(f"{WHITE}  Target Protected Server: {BLUE}{get_server_url()}{RESET}")
    print(f"{WHITE}  Active Data Profile:     {YELLOW}{c.profile.upper()}{RESET} ({c.dataset.measurement_rows:,} measurement targets, {c.dataset.crafting_rows:,} crafting references)")
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
    c = get_ctx(require_profile)
    try:
        req = urllib.request.Request(get_stats_url())
        with urllib.request.urlopen(req, timeout=3) as resp:
            data = json.loads(resp.read().decode("utf-8"))
    except Exception as e:
        print(f"{YELLOW}[Warning] Could not reach target server at {get_stats_url()}: {e}{RESET}")
        return False

    server_profile = str(data.get("data_profile", "")).strip().lower()
    server_fp = str(data.get("data_fingerprint", ""))
    expected_profile = (require_profile or c.profile).strip().lower()
    expected_fp = c.dataset.fingerprint

    if not server_profile:
        print(f"{RED}[Error] Target server at {get_stats_url()} did not report an active data profile.{RESET}")
        return False

    if server_profile != expected_profile:
        print(
            f"{RED}[Profile Mismatch]{RESET} Attacker is running profile '{expected_profile}' (fingerprint {expected_fp[:12]}...) "
            f"but server at {get_server_url()} is running profile '{server_profile}' (fingerprint {server_fp[:12]}...).\n"
            f"Launch server with matching IDS_DATA_PROFILE or set attacker --dataset {server_profile}."
        )
        return False

    if not server_fp:
        print(f"{RED}[Fingerprint Error]{RESET} Server telemetry at {get_stats_url()} did not provide a data_fingerprint.")
        return False

    if not expected_fp:
        print(f"{RED}[Fingerprint Error]{RESET} Attacker dataset fingerprint could not be computed.")
        return False

    if server_fp != expected_fp:
        print(
            f"{RED}[Fingerprint Mismatch]{RESET} Attacker fingerprint is '{expected_fp}' "
            f"but server fingerprint is '{server_fp}'.\n"
            f"Ensure identical dataset files are loaded in both processes."
        )
        return False

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
        if data_profile is not None:
            self.data_profile = data_profile
        else:
            self.data_profile = get_ctx().profile
        self.queries = {}

    def predict(self, X, sample_ids=None, stage=None):
        if not isinstance(X, np.ndarray):
            X = np.array(X, dtype=np.float32)
        if X.ndim == 1:
            X = X.reshape(1, -1)

        c = get_ctx()
        preds = []
        sess_id = self.session_id or CURRENT_SIMULATION_SESSION_ID
        for i, row in enumerate(X):
            sid = sample_ids[i] if (sample_ids is not None and i < len(sample_ids)) else None

            origin = random.choice(ORIGIN_POOL)
            family = "Unknown"
            source_type = "Synthetic / Non-dataset"
            is_attack_val = None

            if sid is not None and c.dataset.metadata is not None and 0 <= sid < len(c.dataset.metadata):
                family = str(c.dataset.metadata.iloc[sid].get("attack_family", "Unknown"))
                true_y = int(c.dataset.metadata.iloc[sid]["y_binary"])
                is_attack_val = bool(true_y == 1)

                # Check if this row is the exact dataset row or a transformed candidate
                if c.dataset.X is not None and 0 <= sid < len(c.dataset.X):
                    orig_row = c.dataset.X.iloc[sid].values.astype(np.float32)
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


def run_single_flow(
    flow_type: Optional[str] = None,
    is_attack: bool = False,
    vector: Optional[np.ndarray] = None,
    sample_id: Optional[int] = None,
    scenario: Optional[str] = None,
    session_id: Optional[str] = None,
    is_query: bool = False,
    query_stage: Optional[str] = None
):
    """
    Sends a single network packet to the server with real dataset metadata and profile tracking.
    Draws targets without replacement from the measurement pool if sample_id is not specified.
    """
    c = get_ctx()
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
            chosen_idx = c.draw_measurement_target(is_attack=is_attack, family_subset=flow_type)
        feat_list = c.dataset.X.iloc[chosen_idx].values.tolist()
        resolved_family = str(c.dataset.metadata.iloc[chosen_idx].get("attack_family", "Normal" if not is_attack else "Unknown"))
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
        "data_profile": c.profile,
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



def run_silent_probing_attack() -> bool:
    """
    Executes Research Attack 1: Silent Probing
    Sequential submission of unchanged baseline flows from measurement pool without querying oracle (0 queries).
    """
    c = get_ctx()
    print(f"\n{MAGENTA}{BOLD}======================================================================{RESET}")
    print(f"{MAGENTA}{BOLD}  RESEARCH ATTACK: SILENT PROBING ATTACK{RESET}")
    print(f"{WHITE}  Model: Sequential Submission of Unchanged Measurement Flows (0 Queries){RESET}")
    print(f"{MAGENTA}{BOLD}======================================================================{RESET}\n")

    sess_id = start_simulation_session("Silent Probing")
    try:
        attack = SilentProbingAttack(c.feature_names, c.modifiable_mask, c.bounds_df)

        # Draw attack target strictly from measurement pool
        chosen_idx = c.draw_measurement_target(is_attack=True)
        x_orig = c.dataset.X.iloc[chosen_idx].values
        family = str(c.dataset.metadata.iloc[chosen_idx].get("attack_family", "Unknown"))

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

        t_str = time.strftime("%H:%M:%S")
        if resp is None or resp.get("status_code") not in [200, 403]:
            status = resp.get("status_code") if resp else "No response"
            err = resp.get("error", "Unknown error") if resp else "Failed"
            print(f"[{t_str}] {RED}{BOLD}[EXECUTION ERROR] Submission failed (HTTP {status}: {err}){RESET}")
            print(f"{YELLOW}No IDS classification or defense verdict could be obtained due to server/transport error.{RESET}\n")
            return False

        if resp["blocked"]:
            details = resp["body"].get("details", {})
            print(f"[{t_str}] {BG_RED}{WHITE}{BOLD} DETECTED & BLOCKED (403) {RESET}")
            print(f"IDS Classification: ATTACK | Family: {family} | Defense: {details.get('defense', '').upper()} [{details.get('mode', '').upper()}]")
            print(f"{CYAN}Analysis: Target IDS classified silent probing flow as Attack.{RESET}\n")
        else:
            print(f"[{t_str}] {BG_GREEN}{WHITE}{BOLD} EVADED & ALLOWED (200) {RESET}")
            print(f"IDS Classification: BENIGN | Family: {family}")
            print(f"{YELLOW}Warning: Attack penetrated protected server under current defense configuration.{RESET}\n")
        return True
    finally:
        stop_simulation_session()


def run_surrogate_transfer_attack() -> bool:
    """
    Executes Research Attack 2: Surrogate Transferability
    Trains a local Decision Tree surrogate from crafting query pool, then crafts candidate
    against a measurement target and submits the final candidate once as a measured flow.
    """
    c = get_ctx()
    print(f"\n{MAGENTA}{BOLD}======================================================================{RESET}")
    print(f"{MAGENTA}{BOLD}  RESEARCH ATTACK: SURROGATE TRANSFERABILITY ATTACK{RESET}")
    print(f"{WHITE}  Model: Local Decision Tree Surrogate with Boundary Transfer{RESET}")
    print(f"{WHITE}  Crafting Query References: Disclosed Crafting Pool | Measured Target: Measurement Pool{RESET}")
    print(f"{MAGENTA}{BOLD}======================================================================{RESET}\n")

    sess_id = start_simulation_session("Surrogate Transferability")
    try:
        attack = SurrogateTransferAttack(c.feature_names, c.modifiable_mask, c.bounds_df)
        oracle = RemoteServerOracle(target_url=get_server_url(), scenario_name="Surrogate Transferability", session_id=sess_id)

        # Select reference query pool strictly from CRAFTING pool using seeded selection
        ref_ids = c.select_crafting_references_surrogate(n_benign=10, n_attack=10)
        x_pool = c.dataset.X.iloc[ref_ids].values
        print(f"{DIM}Selecting {len(ref_ids)} bounded reference flows from crafting pool (10 benign, 10 attack) with seed={c.seed}...{RESET}")
        print(f"{WHITE}Crafting Query Sample IDs (Role: Crafting): {CYAN}{ref_ids}{RESET}")

        print(f"{DIM}Fitting surrogate model via query telemetry (is_query=True)...{RESET}")
        y_oracle = oracle.predict(x_pool, sample_ids=ref_ids, stage="surrogate_fitting")
        attack.fit_surrogate(x_pool, y_oracle)
        print(f"{GREEN}[OK] Surrogate decision tree fitted successfully with classes {attack.surrogate.classes_}.{RESET}")

        # Draw target flow strictly from MEASUREMENT pool
        target_idx = c.draw_measurement_target(is_attack=True)
        x_orig = c.dataset.X.iloc[target_idx].values
        family = str(c.dataset.metadata.iloc[target_idx].get("attack_family", "Unknown"))

        print(f"{WHITE}Targeting Measurement Attack Sample: {CYAN}{family}{RESET} (Evaluation ID: {target_idx}, Role: Measurement)")
        x_cand, mags = attack.generate_candidate(x_orig)
        if x_cand is None:
            print(f"{YELLOW}[!] Surrogate yielded no feasible candidate for target {target_idx}.{RESET}\n")
            return False

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
        if resp is None or resp.get("status_code") not in (200, 403):
            status = resp.get("status_code") if resp else "No response"
            err = resp.get("error", "Unknown error") if resp else "Failed"
            print(f"[{t_str}] {RED}{BOLD}[EXECUTION ERROR] Submission failed (HTTP {status}: {err}){RESET}")
            print(f"{YELLOW}No IDS classification or defense verdict could be obtained due to server/transport error.{RESET}\n")
            return False
        elif resp.get("status_code") == 200:
            print(f"[{t_str}] {BG_GREEN}{WHITE}{BOLD} TRANSFER SUCCESS (200 OK) {RESET}")
            print(f"IDS Classification: BENIGN (EVASION SUCCESS) | Family: {family}")
            print(f"{YELLOW}Result: Adversarial candidate transferred and evaded target classifier!{RESET}\n")
            return True
        elif resp.get("status_code") == 403:
            print(f"[{t_str}] {BG_RED}{WHITE}{BOLD} TRANSFER BLOCKED (403 FORBIDDEN) {RESET}")
            print(f"IDS Classification: ATTACK (DEFENSE HELD) | Family: {family}")
            print(f"{CYAN}Result: Target IDS defense prevented surrogate boundary transfer.{RESET}\n")
            return True
    finally:
        stop_simulation_session()


def run_decision_boundary_attack(max_queries: int = 50, steps: int = 10) -> bool:
    """
    Executes Research Attack 3: Decision Boundary Bisection Search
    Uses crafting benign references for binary search, targeting a measurement attack flow.
    Submits the final candidate once as a measured flow.
    """
    c = get_ctx()
    print(f"\n{MAGENTA}{BOLD}======================================================================{RESET}")
    print(f"{MAGENTA}{BOLD}  RESEARCH ATTACK: DECISION BOUNDARY ATTACK (1D BISECTION){RESET}")
    print(f"{WHITE}  Model: Query-Guided Binary Search Boundary Finding{RESET}")
    print(f"{WHITE}  References: Crafting Benign Pool | Target: Measurement Attack Pool{RESET}")
    print(f"{MAGENTA}{BOLD}======================================================================{RESET}\n")

    sess_id = start_simulation_session("Decision-Boundary Attack")
    try:
        attack = DecisionBoundaryAttack(
            c.feature_names, c.modifiable_mask, c.bounds_df,
            max_queries=max_queries, binary_search_steps=steps
        )
        oracle = RemoteServerOracle(target_url=get_server_url(), max_queries_per_sample=max_queries, scenario_name="Decision-Boundary Attack", session_id=sess_id)

        # Benign reference pool drawn strictly from CRAFTING pool using seeded selection
        ref_ids = c.select_crafting_references_boundary(n_benign=50)
        reference_pool = c.dataset.X.iloc[ref_ids].values
        print(f"{DIM}Selecting {len(ref_ids)} crafting benign references from crafting benign pool with seed={c.seed}...{RESET}")
        print(f"{WHITE}Crafting Benign Reference IDs (Role: Crafting): {CYAN}{ref_ids[:10]}... ({len(ref_ids)} total){RESET}")

        # Draw target flow strictly from MEASUREMENT pool
        target_idx = c.draw_measurement_target(is_attack=True)
        x_orig = c.dataset.X.iloc[target_idx].values
        family = str(c.dataset.metadata.iloc[target_idx].get("attack_family", "Unknown"))

        print(f"{WHITE}Target Measurement Attack Sample: {CYAN}{family}{RESET} (Evaluation ID: {target_idx}, Role: Measurement)")
        print(f"{DIM}Running bisection search via query telemetry (budget: {max_queries}; candidate queries retain measurement target origin ID {target_idx})...{RESET}")

        res = attack.generate(x_orig, oracle, sample_id=target_idx, true_label=1, reference_pool=reference_pool)
        queries_used = oracle.get_query_count(target_idx)

        print(f"\n{BOLD}Boundary Search Finished:{RESET} queries_used={queries_used}/{max_queries}, status={res.status_code}, message={res.message}")

        is_fallback = (not res.success or res.X_adv is None)
        final_cand = res.X_adv if not is_fallback else x_orig
        if is_fallback:
            print(f"{YELLOW}[!] Bisection search was unsuccessful or ineligible ({res.message}). Submitting unchanged original flow as baseline fallback...{RESET}")
        else:
            print(f"{DIM}Submitting generated boundary evasion candidate once as measured target flow...{RESET}")

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
        if resp is None or resp.get("status_code") not in (200, 403):
            status = resp.get("status_code") if resp else "No response"
            err = resp.get("error", "Unknown error") if resp else "Failed"
            print(f"\n[{t_str}] {RED}{BOLD}[EXECUTION ERROR] Submission failed (HTTP {status}: {err}){RESET}")
            print(f"{YELLOW}No IDS classification or defense verdict could be obtained due to server/transport error.{RESET}\n")
            return False
        elif resp.get("status_code") == 200:
            if is_fallback:
                print(f"\n[{t_str}] {YELLOW}{BOLD} ORIGINAL FLOW ALLOWED (200 OK - BASELINE FALSE NEGATIVE) {RESET}")
                print(f"Verdict: Unperturbed baseline attack was not detected by classifier (fallback flow, not an evasion candidate).\n")
            else:
                print(f"\n[{t_str}] {BG_GREEN}{WHITE}{BOLD} EVADED BOUNDARY (200 OK) {RESET}")
                print(f"Verdict: Generated adversarial candidate crossed decision boundary and evaded target classifier!\n")
            return True
        elif resp.get("status_code") == 403:
            if is_fallback:
                print(f"\n[{t_str}] {BG_RED}{WHITE}{BOLD} BASELINE FLOW BLOCKED (403 BLOCKED) {RESET}")
                print(f"Verdict: Unperturbed baseline attack was correctly detected and blocked.\n")
            else:
                print(f"\n[{t_str}] {BG_RED}{WHITE}{BOLD} DEFENSE HELD (403 BLOCKED) {RESET}")
                print(f"Verdict: Target IDS defense detected and blocked boundary candidate.\n")
            return True
    finally:
        stop_simulation_session()


def run_continuous_stream(delay: float = 1.0):
    """Streams recorded traffic to demonstrate real-time dashboard updates."""
    print(f"\n{CYAN}{BOLD}>>> Starting Continuous Recorded-Flow Stream (Press Ctrl+C to stop)...{RESET}\n")
    attack_types = ["DDoS", "Port Scan", "Brute Force", "Malware", "Bot", "SlowHTTPTest"]

    try:
        while True:
            is_attack = (random.random() < 0.35)
            if is_attack:
                a_type = random.choice(attack_types)
                run_single_flow(flow_type=a_type, is_attack=True, scenario=None)
            else:
                run_single_flow(flow_type="Normal", is_attack=False, scenario=None)
            time.sleep(delay)
    except KeyboardInterrupt:
        print(f"\n{YELLOW}[!] Stream paused by user.{RESET}\n")


def interactive_menu():
    print_banner()
    while True:
        print(f"{WHITE}{BOLD}======================================================================{RESET}")
        print(f"{CYAN}{BOLD}  ATTACKER ACTIONS (Black-Box Simulation POV):{RESET}")
        print(f"  {CYAN}[1]{RESET} Send recorded benign flow")
        print(f"  {MAGENTA}[2]{RESET} Silent Probing: submit unchanged malicious flow, zero preliminary queries")
        print(f"  {MAGENTA}[3]{RESET} Surrogate Transferability: craft using a local surrogate and crafting references")
        print(f"  {MAGENTA}[4]{RESET} Decision-Boundary Attack: search using server responses and crafting references")
        print(f"  {BLUE}[5]{RESET} Continuous recorded-flow stream")
        print(f"  {DIM}[0]{RESET} Exit")
        print(f"{WHITE}{BOLD}======================================================================{RESET}\n")

        try:
            choice = input(f"{BOLD}Enter option (0-5): {RESET}").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nExiting.")
            sys.exit(0)

        if choice == "1":
            run_single_flow(flow_type="Normal", is_attack=False, scenario=None)
        elif choice == "2":
            run_silent_probing_attack()
        elif choice == "3":
            run_surrogate_transfer_attack()
        elif choice == "4":
            run_decision_boundary_attack()
        elif choice == "5":
            run_continuous_stream(delay=1.0)
        elif choice == "0":
            print("Exiting attacker simulation console.")
            break
        else:
            print(f"{RED}Invalid selection. Please choose 0-5.{RESET}\n")


def main():
    parser = argparse.ArgumentParser(description="Attacker Simulation Console for IDS + Recall-Aware Platform")
    parser.add_argument("--mode", choices=["menu", "benign", "silent", "surrogate", "boundary", "stream"], default="menu")
    parser.add_argument("--dataset", choices=["expanded", "fixture20"], default=None, help="Dataset profile (expanded or fixture20)")
    parser.add_argument("--seed", type=int, default=42, help="RNG seed for target sampling without replacement (default: 42 for deterministic session replay)")
    parser.add_argument("--target", default=DEFAULT_SERVER_URL, help="Target URL for protected server endpoint")
    parser.add_argument("--count", type=int, default=5, help="Number of flows for stream utility")
    parser.add_argument("--delay", type=float, default=0.3)

    args = parser.parse_args()

    # Make --target effective for all server endpoints dynamically
    if args.target:
        set_target_url(args.target)

    # Initialize profile and seed based on CLI flags
    active_profile = args.dataset or os.environ.get("IDS_DATA_PROFILE", "expanded").strip().lower()
    try:
        current_ctx = get_ctx(active_profile, seed=args.seed)
    except Exception as e:
        print(f"{RED}[Fatal] Failed to initialize dataset profile '{active_profile}': {e}{RESET}")
        sys.exit(1)

    # Check backend compatibility before sending flows; halt with nonzero exit if unsuccessful
    if not check_backend_compatibility(require_profile=active_profile):
        print(f"{RED}[Fatal] Backend compatibility check failed. Halting.{RESET}")
        sys.exit(1)

    success = True
    if args.mode == "menu":
        interactive_menu()
    elif args.mode == "silent":
        success = run_silent_probing_attack()
    elif args.mode == "surrogate":
        success = run_surrogate_transfer_attack()
    elif args.mode == "boundary":
        success = run_decision_boundary_attack(max_queries=50)
    elif args.mode == "benign":
        r = run_single_flow("Normal", is_attack=False, scenario=None)
        success = (r is not None and r.get("status_code") == 200)
    elif args.mode == "stream":
        run_continuous_stream(delay=args.delay)

    if not success:
        sys.exit(1)


if __name__ == "__main__":
    main()
