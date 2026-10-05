#!/usr/bin/env python3
"""
Attacker Simulation Console for IDS + Recall-Aware Research Platform
Simulates adversarial network attacks, research black-box evasion algorithms,
and volumetric floods against the protected target server.

Research Attacks Supported:
  1. Silent Probing Attack (Offline perturbation without querying oracle)
  2. Surrogate Transferability Attack (Locally trained surrogate decision tree)
  3. Decision Boundary Attack (1D Bisection search vs benign reference pool)

Usage:
  python attacker_sim.py                     (Interactive Menu Console)
  python attacker_sim.py --mode silent       (Silent Probing Attack)
  python attacker_sim.py --mode surrogate    (Surrogate Transferability Attack)
  python attacker_sim.py --mode boundary     (Decision Boundary Bisection Attack)
  python attacker_sim.py --mode ddos         (DDoS Volumetric Burst)
  python attacker_sim.py --mode stream       (Continuous Real-time Traffic Stream)
  python attacker_sim.py --mode compare      (Base Defense vs Recall-Aware Benchmark)
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

# Origin Pools for Simulation
ORIGIN_POOL = [
    {"ip": "203.0.113.45", "country": "Singapore"},
    {"ip": "185.199.110.23", "country": "Ukraine"},
    {"ip": "103.21.54.12", "country": "United States"},
    {"ip": "45.76.32.18", "country": "Germany"},
    {"ip": "89.248.163.77", "country": "Russia"},
    {"ip": "114.119.130.88", "country": "China"},
    {"ip": "177.54.144.20", "country": "Brazil"},
    {"ip": "197.232.12.9", "country": "Kenya"},
    {"ip": "10.0.2.45", "country": "Internal Network"},
    {"ip": "172.16.0.12", "country": "DMZ Zone"},
    {"ip": "10.0.3.77", "country": "Internal Network"},
    {"ip": "192.168.1.25", "country": "Local Subnet"}
]

BENIGN_POOL = [
    {"ip": "172.16.0.5", "country": "Internal Corp"},
    {"ip": "192.168.1.100", "country": "Local Workstation"},
    {"ip": "10.0.1.22", "country": "Database Client"},
    {"ip": "192.168.1.45", "country": "Authorized API Gateway"}
]


def print_banner():
    print(f"\n{CYAN}{BOLD}======================================================================{RESET}")
    print(f"{CYAN}{BOLD}  CYBER ATTACKER SIMULATION CONSOLE — RESEARCH RUNTIME{RESET}")
    print(f"{WHITE}  Target Protected Server: {BLUE}{DEFAULT_SERVER_URL}{RESET}")
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
                "blocked": False
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
            "blocked": (e.code == 403)
        }
    except Exception as e:
        return {
            "status_code": 0,
            "latency_ms": 0,
            "body": {"error": str(e)},
            "blocked": False
        }


def set_server_defense(defense_name: str) -> bool:
    """Configures server defense ('afp', 'rs', 'fs', 'none')."""
    payload = json.dumps({"defense": defense_name}).encode("utf-8")
    req = urllib.request.Request(DEFAULT_DEFENSE_URL, data=payload, headers={"Content-Type": "application/json"})
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
    req = urllib.request.Request(DEFAULT_MODE_URL, data=payload, headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=3) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            return True
    except Exception as e:
        print(f"{RED}[Error] Failed to set mode: {e}{RESET}")
        return False


class RemoteServerOracle:
    """Oracle interface querying the target server via HTTP POST to /api/server/data."""
    def __init__(self, target_url: str = DEFAULT_SERVER_URL, max_queries_per_sample: int = 50):
        self.target_url = target_url
        self.max_queries_per_sample = max_queries_per_sample
        self.queries = {}

    def predict(self, X, sample_ids=None, stage=None):
        if not isinstance(X, np.ndarray):
            X = np.array(X, dtype=np.float32)
        if X.ndim == 1:
            X = X.reshape(1, -1)

        preds = []
        for i, row in enumerate(X):
            sid = sample_ids[i] if (sample_ids is not None and i < len(sample_ids)) else 0
            self.queries[sid] = self.queries.get(sid, 0) + 1

            origin = random.choice(ORIGIN_POOL)
            payload = {
                "source_ip": origin["ip"],
                "destination_ip": "192.168.1.10",
                "flow_type": f"Probe ({stage})" if stage else "Adversarial Query",
                "country": origin["country"],
                "is_attack": True,
                "attack_scenario": "adversarial_oracle",
                "feature_vector": row.tolist()
            }
            res = send_packet(self.target_url, payload)
            # Blocked (403) -> Model classified as Malicious (1)
            # Allowed (200) -> Model classified as Benign (0)
            preds.append(1 if res["blocked"] else 0)

        return np.array(preds, dtype=int)

    def get_query_count(self, sample_id):
        return self.queries.get(sample_id, 0)


class AttackerContext:
    """Loads required schemas and demo data for research attack execution."""
    def __init__(self):
        m_dir = RUNTIME_DIR / "model"
        d_dir = RUNTIME_DIR / "demo_data"

        with open(m_dir / "feature_names.json", "r") as f:
            self.feature_names = json.load(f)
        with open(m_dir / "feature_mask.json", "r") as f:
            mask_dict = json.load(f)
            self.modifiable_mask = np.array(list(mask_dict.values())[0], dtype=bool)

        self.bounds_df = pd.read_parquet(m_dir / "training_bounds.parquet")
        self.X_demo = pd.read_parquet(d_dir / "X_demo.parquet")
        self.meta_demo = pd.read_parquet(d_dir / "metadata_demo.parquet")

        self.attack_indices = np.where(self.meta_demo["y_binary"].values == 1)[0]
        self.benign_indices = np.where(self.meta_demo["y_binary"].values == 0)[0]


ctx = AttackerContext()


def run_single_flow(flow_type: str = "Normal", is_attack: bool = False, vector: Optional[np.ndarray] = None):
    """Sends a single legitimate or volumetric packet to the server."""
    origin = random.choice(ORIGIN_POOL if is_attack else BENIGN_POOL)
    t_str = time.strftime("%H:%M:%S")

    feat_list = None
    if vector is not None:
        feat_list = vector.tolist()
    else:
        indices = ctx.attack_indices if is_attack else ctx.benign_indices
        chosen_idx = np.random.choice(indices)
        feat_list = ctx.X_demo.iloc[chosen_idx].values.tolist()

    payload = {
        "source_ip": origin["ip"],
        "destination_ip": "192.168.1.10",
        "flow_type": flow_type,
        "country": origin["country"],
        "is_attack": is_attack,
        "attack_scenario": "traffic_flow",
        "feature_vector": feat_list
    }

    res = send_packet(DEFAULT_SERVER_URL, payload)

    if res["status_code"] == 0:
        print(f"{RED}[FAIL] Could not connect to target server at {DEFAULT_SERVER_URL}. Is server.py running?{RESET}")
        return

    if res["blocked"]:
        badge = f"{BG_RED}{WHITE}{BOLD} BLOCKED (403) {RESET}"
        conf = res["body"].get("details", {}).get("confidence", "0.95")
        defense = res["body"].get("details", {}).get("defense", "active").upper()
        mode = res["body"].get("details", {}).get("mode", "").upper()
        print(f"[{t_str}] {badge} {flow_type:<16} from {origin['ip']:<15} ({origin['country']:<14}) | Conf: {conf} | Defense: {defense} [{mode}]")
    else:
        badge = f"{BG_GREEN}{WHITE}{BOLD} ALLOWED (200) {RESET}"
        print(f"[{t_str}] {badge} {flow_type:<16} from {origin['ip']:<15} ({origin['country']:<14}) | Server Status: Resource Granted")


def run_volumetric_burst(attack_type: str = "DDoS", count: int = 6, delay: float = 0.25):
    """Sends a rapid burst of malicious packets to test throughput and alerts."""
    print(f"\n{RED}{BOLD}>>> Launching Volumetric Burst: {attack_type} ({count} packets)...{RESET}\n")
    for i in range(count):
        run_single_flow(flow_type=attack_type, is_attack=True)
        time.sleep(delay)
    print(f"\n{GREEN}[OK] Volumetric burst completed.{RESET}\n")


def run_silent_probing_attack():
    """
    Executes Research Attack 1: Silent Probing
    Iteratively perturbs features without querying the target oracle.
    """
    print(f"\n{MAGENTA}{BOLD}======================================================================{RESET}")
    print(f"{MAGENTA}{BOLD}  RESEARCH ATTACK: SILENT PROBING ATTACK{RESET}")
    print(f"{WHITE}  Model: Black-Box Zero-Query Feature Space Exploration{RESET}")
    print(f"{MAGENTA}{BOLD}======================================================================{RESET}\n")

    attack = SilentProbingAttack(ctx.feature_names, ctx.modifiable_mask, ctx.bounds_df)
    oracle = RemoteServerOracle()

    # Pick an attack sample
    chosen_idx = np.random.choice(ctx.attack_indices)
    x_orig = ctx.X_demo.iloc[chosen_idx].values
    family = ctx.meta_demo.iloc[chosen_idx]["attack_family"]

    print(f"{WHITE}Selected Base Attack Vector: {CYAN}{family}{RESET}")
    print(f"{DIM}Generating silent perturbation candidate (0 target queries)...{RESET}")

    res = attack.generate(x_orig, oracle, sample_id=chosen_idx, true_label=1)
    print(f"{GREEN}[OK] Candidate generated. Transmitting to protected server endpoint...{RESET}")

    origin = random.choice(ORIGIN_POOL)
    payload = {
        "source_ip": origin["ip"],
        "destination_ip": "192.168.1.10",
        "flow_type": "Silent Probe",
        "country": origin["country"],
        "is_attack": True,
        "attack_scenario": "silent_probing",
        "feature_vector": res.X_adv.tolist()
    }

    t0 = time.time()
    resp = send_packet(DEFAULT_SERVER_URL, payload)
    t_str = time.strftime("%H:%M:%S")

    if resp["blocked"]:
        details = resp["body"].get("details", {})
        print(f"[{t_str}] {BG_RED}{WHITE}{BOLD} DETECTED & BLOCKED (403) {RESET}")
        print(f"Defense: {details.get('defense', '').upper()} | Mode: {details.get('mode', '').upper()} | Confidence: {details.get('confidence', 0.95)}")
        print(f"{CYAN}Analysis: Defense perturbation successfully neutralized silent probing vector.{RESET}\n")
    else:
        print(f"[{t_str}] {BG_GREEN}{WHITE}{BOLD} EVADED & ALLOWED (200) {RESET}")
        print(f"{YELLOW}Warning: Attack penetrated protected server under current defense configuration.{RESET}\n")


def run_surrogate_transfer_attack():
    """
    Executes Research Attack 2: Surrogate Transferability
    Trains a local Decision Tree surrogate from oracle queries and crafts transfer attacks.
    """
    print(f"\n{MAGENTA}{BOLD}======================================================================{RESET}")
    print(f"{MAGENTA}{BOLD}  RESEARCH ATTACK: SURROGATE TRANSFERABILITY ATTACK{RESET}")
    print(f"{WHITE}  Model: Local Decision Tree Surrogate with Boundary Transfer{RESET}")
    print(f"{MAGENTA}{BOLD}======================================================================{RESET}\n")

    attack = SurrogateTransferAttack(ctx.feature_names, ctx.modifiable_mask, ctx.bounds_df)
    oracle = RemoteServerOracle()

    print(f"{DIM}Fitting surrogate model on local sample distribution...{RESET}")
    # Fit surrogate on known samples with oracle labels
    x_pool = ctx.X_demo.values
    y_oracle = oracle.predict(x_pool, sample_ids=list(range(len(x_pool))), stage="surrogate_fitting")
    attack.fit_surrogate(x_pool, y_oracle)
    print(f"{GREEN}[OK] Surrogate decision tree fitted with benign class resolved.{RESET}")

    # Pick an attack sample
    chosen_idx = np.random.choice(ctx.attack_indices)
    x_orig = ctx.X_demo.iloc[chosen_idx].values
    family = ctx.meta_demo.iloc[chosen_idx]["attack_family"]

    print(f"{WHITE}Targeting Attack Sample: {CYAN}{family}{RESET}")
    x_cand, _ = attack.generate_candidate(x_orig)

    print(f"{DIM}Evaluating transfer vector against protected target server...{RESET}")
    transfer_res = attack.evaluate_transfer(x_cand, x_orig, oracle, sample_id=chosen_idx, true_label=1)

    t_str = time.strftime("%H:%M:%S")
    if transfer_res.success:
        print(f"[{t_str}] {BG_GREEN}{WHITE}{BOLD} TRANSFER SUCCESS (200 OK) {RESET}")
        print(f"{YELLOW}Result: Adversarial candidate transferred and evaded target classifier!{RESET}\n")
    else:
        print(f"[{t_str}] {BG_RED}{WHITE}{BOLD} TRANSFER BLOCKED (403 FORBIDDEN) {RESET}")
        print(f"{CYAN}Result: Target IDS defense prevented surrogate boundary transfer.{RESET}\n")


def run_decision_boundary_attack(max_queries: int = 50, steps: int = 10):
    """
    Executes Research Attack 3: Decision Boundary Bisection Search
    Uses benign references and binary search to identify adversarial boundary vectors.
    """
    print(f"\n{MAGENTA}{BOLD}======================================================================{RESET}")
    print(f"{MAGENTA}{BOLD}  RESEARCH ATTACK: DECISION BOUNDARY ATTACK (1D BISECTION){RESET}")
    print(f"{WHITE}  Model: Query-Guided Binary Search Boundary Finding{RESET}")
    print(f"{MAGENTA}{BOLD}======================================================================{RESET}\n")

    attack = DecisionBoundaryAttack(
        ctx.feature_names, ctx.modifiable_mask, ctx.bounds_df,
        max_queries=max_queries, binary_search_steps=steps
    )
    oracle = RemoteServerOracle(max_queries_per_sample=max_queries)
    reference_pool = ctx.X_demo.iloc[ctx.benign_indices].values

    # Pick attack sample
    chosen_idx = np.random.choice(ctx.attack_indices)
    x_orig = ctx.X_demo.iloc[chosen_idx].values
    family = ctx.meta_demo.iloc[chosen_idx]["attack_family"]

    print(f"{WHITE}Target Attack Sample: {CYAN}{family}{RESET} (ID: {chosen_idx})")
    print(f"{DIM}Running bisection search via live server queries (max budget: {max_queries})...{RESET}")

    res = attack.generate(x_orig, oracle, sample_id=chosen_idx, true_label=1, reference_pool=reference_pool)

    t_str = time.strftime("%H:%M:%S")
    queries_used = oracle.get_query_count(chosen_idx)
    print(f"\n[{t_str}] {BOLD}Boundary Search Outcome:{RESET}")
    print(f"  Status Code:   {CYAN}{res.status_code}{RESET}")
    print(f"  Total Queries: {YELLOW}{queries_used}{RESET} / {max_queries}")
    print(f"  Message:       {res.message}")

    if res.success:
        print(f"  Verdict:       {BG_GREEN}{WHITE}{BOLD} EVADED BOUNDARY {RESET}\n")
    else:
        print(f"  Verdict:       {BG_RED}{WHITE}{BOLD} DEFENSE HELD / ATTACK STOPPED {RESET}\n")


def run_comparative_benchmark():
    """
    Demonstrates the Core Research Hypothesis:
    Base Defense (fixed intensity) vs. Recall-Aware Defense (adaptive feedback control).
    """
    print(f"\n{YELLOW}{BOLD}======================================================================{RESET}")
    print(f"{YELLOW}{BOLD}  COMPARATIVE RESEARCH BENCHMARK: BASE DEFENSE vs. RECALL-AWARE{RESET}")
    print(f"{YELLOW}{BOLD}======================================================================{RESET}\n")

    # Step 1: Base Mode
    print(f"{WHITE}Step 1: Setting AFP Defense to {YELLOW}BASE MODE (Static Calibrated Intensity){RESET}...")
    set_server_defense("afp")
    set_server_mode("base")
    time.sleep(0.5)

    print(f"{DIM}Sending 5 adversarial burst packets under Base AFP...{RESET}")
    for i in range(5):
        run_single_flow("Base AFP Flow", is_attack=True)
        time.sleep(0.2)

    time.sleep(0.8)

    # Step 2: Recall-Aware Mode
    print(f"\n{WHITE}Step 2: Activating {GREEN}RECALL-AWARE CONTROLLER (Dynamic Feedback Window){RESET}...")
    set_server_mode("recall-aware")
    time.sleep(0.5)

    print(f"{DIM}Sending 5 adversarial burst packets under Recall-Aware AFP...{RESET}")
    for i in range(5):
        run_single_flow("Recall-Aware Flow", is_attack=True)
        time.sleep(0.2)

    print(f"\n{CYAN}{BOLD}Benchmark completed. Check the SOC dashboard to observe dynamic intensity adjustments!{RESET}\n")


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
        print(f"{WHITE}{BOLD}Select Action:{RESET}")
        print(f"  {CYAN}[1]{RESET} Send Legitimate Benign Traffic Flow")
        print(f"  {RED}[2]{RESET} Launch DDoS Volumetric Flood Burst")
        print(f"  {MAGENTA}[3]{RESET} Launch Silent Probing Attack (Zero Queries, Offline Evasion)")
        print(f"  {MAGENTA}[4]{RESET} Launch Surrogate Transferability Attack (Decision Tree Surrogate)")
        print(f"  {MAGENTA}[5]{RESET} Launch Decision Boundary Attack (1D Bisection Search)")
        print(f"  {YELLOW}[6]{RESET} Comparative Benchmark: Base Defense vs. Recall-Aware Defense")
        print(f"  {BLUE}[7]{RESET} Continuous Real-time Traffic Stream (1 flow / sec)")
        print(f"  {WHITE}[8]{RESET} Switch Active Defense: [AFP -> RS -> FS -> None]")
        print(f"  {WHITE}[9]{RESET} Toggle Controller Mode: [Recall-Aware <-> Base]")
        print(f"  {DIM}[0]{RESET} Exit\n")

        try:
            choice = input(f"{BOLD}Enter option (0-9): {RESET}").strip()
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
            run_comparative_benchmark()
        elif choice == "7":
            run_continuous_stream(delay=1.0)
        elif choice == "8":
            cycle = {"afp": "rs", "rs": "fs", "fs": "none", "none": "afp"}
            # Fetch current
            try:
                with urllib.request.urlopen("http://localhost:8000/api/dashboard/stats", timeout=3) as resp:
                    data = json.loads(resp.read().decode("utf-8"))
                    curr = data.get("afp", {}).get("defense_name", "afp")
                    nxt = cycle.get(curr, "afp")
                    set_server_defense(nxt)
                    print(f"Defense switched to: {GREEN}{nxt.upper()}{RESET}\n")
            except Exception as e:
                print(f"{RED}Could not cycle defense: {e}{RESET}\n")
        elif choice == "9":
            try:
                with urllib.request.urlopen("http://localhost:8000/api/dashboard/stats", timeout=3) as resp:
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
            print(f"{RED}Invalid selection. Please choose 0-9.{RESET}\n")


def main():
    parser = argparse.ArgumentParser(description="Attacker Simulation Console for IDS + Recall-Aware Platform")
    parser.add_argument("--mode", choices=["menu", "silent", "surrogate", "boundary", "ddos", "benign", "stream", "compare"], default="menu")
    parser.add_argument("--defense", choices=["afp", "rs", "fs", "none"], default=None)
    parser.add_argument("--controller", choices=["recall-aware", "base"], default=None)
    parser.add_argument("--target", default=DEFAULT_SERVER_URL)
    parser.add_argument("--count", type=int, default=5)
    parser.add_argument("--delay", type=float, default=0.3)

    args = parser.parse_args()

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
        run_comparative_benchmark()
    elif args.mode == "ddos":
        run_volumetric_burst("DDoS", count=args.count, delay=args.delay)
    elif args.mode == "benign":
        run_single_flow("Normal", is_attack=False)
    elif args.mode == "stream":
        run_continuous_stream(delay=args.delay)


if __name__ == "__main__":
    main()
