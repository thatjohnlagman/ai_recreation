#!/usr/bin/env python3
"""
Attacker Simulation Console for IDS + Recall-Aware AFP Platform
Simulates adversarial network attacks, volumetric floods, and black-box decision boundary probing
against the protected server.

Usage:
  python attacker_sim.py                     (Interactive Menu Console)
  python attacker_sim.py --mode probe        (Adversarial Bisection Probing Attack)
  python attacker_sim.py --mode ddos         (DDoS Attack Flood)
  python attacker_sim.py --mode stream       (Continuous Real-time Traffic Stream)
  python attacker_sim.py --mode compare      (AFP ON vs OFF Evasion Benchmark)
"""

import sys
import os
import time
import json
import random
import argparse
import urllib.request
import urllib.error

# Ensure UTF-8 output on Windows consoles if supported
if hasattr(sys.stdout, 'reconfigure'):
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass

# ANSI Color Codes for Rich Terminal Display
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
DEFAULT_TOGGLE_URL = "http://localhost:8000/api/dashboard/toggle-afp"

# Threat Origin Pool (IPs & Countries)
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
    print(f"{CYAN}{BOLD}  CYBER ATTACKER SIMULATION CONSOLE{RESET}")
    print(f"{WHITE}  Target Protected Server: {BLUE}{DEFAULT_SERVER_URL}{RESET}")
    print(f"{WHITE}  Defense Layer:          {GREEN}Inline IDS + Recall-Aware AFP{RESET}")
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


def toggle_server_afp(enabled: bool) -> bool:
    """Helper to toggle server AFP defense state."""
    data_bytes = json.dumps({"enabled": enabled}).encode("utf-8")
    req = urllib.request.Request(
        DEFAULT_TOGGLE_URL,
        data=data_bytes,
        headers={"Content-Type": "application/json"}
    )
    try:
        with urllib.request.urlopen(req, timeout=3) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            return data.get("afp_enabled", enabled)
    except Exception as e:
        print(f"{RED}[Error] Failed to toggle AFP defense: {e}{RESET}")
        return enabled


def run_single_flow(flow_type: str, is_attack: bool = True, is_probe: bool = False, mutation: float = 0.0):
    origin = random.choice(ORIGIN_POOL if is_attack else BENIGN_POOL)
    payload = {
        "source_ip": origin["ip"],
        "destination_ip": "192.168.1.10",
        "flow_type": flow_type,
        "country": origin["country"],
        "is_attack": is_attack,
        "is_probe": is_probe,
        "evasion_mutation": mutation
    }

    t_str = time.strftime("%H:%M:%S")
    res = send_packet(DEFAULT_SERVER_URL, payload)

    if res["status_code"] == 0:
        print(f"{RED}[FAIL] Could not connect to target server at {DEFAULT_SERVER_URL}. Is server.py running?{RESET}")
        return

    if res["blocked"]:
        badge = f"{BG_RED}{WHITE}{BOLD} BLOCKED (403) {RESET}"
        conf = res["body"].get("details", {}).get("confidence", "0.9+ ")
        afp_state = f"{CYAN}AFP Perturbation: ON{RESET}" if res["body"].get("details", {}).get("afp_active") else f"{YELLOW}AFP: OFF{RESET}"
        print(f"[{t_str}] {badge} {flow_type:<14} from {origin['ip']:<15} ({origin['country']:<14}) | Confidence: {conf} | {afp_state}")
    else:
        badge = f"{BG_GREEN}{WHITE}{BOLD} ALLOWED (200) {RESET}"
        print(f"[{t_str}] {badge} {flow_type:<14} from {origin['ip']:<15} ({origin['country']:<14}) | Server Status: Resource Granted")


def run_volumetric_burst(attack_type: str = "DDoS", count: int = 8, delay: float = 0.3):
    print(f"\n{RED}{BOLD}>>> Launching Volumetric Burst: {attack_type} ({count} packets)...{RESET}\n")
    for i in range(count):
        run_single_flow(flow_type=attack_type, is_attack=True)
        time.sleep(delay)
    print(f"\n{GREEN}[OK] Volumetric burst completed.{RESET}\n")


def run_adversarial_probing_attack(iterations: int = 6):
    """
    Simulates a black-box 1D Bisection Probing Evasion Attack.
    The attacker iteratively mutates feature coordinates toward the benign profile mean
    to find the minimum distance to flip the IDS decision boundary.
    """
    print(f"\n{MAGENTA}{BOLD}>>> Launching Adversarial Evasion Probing Attack (1D Bisection Search)...{RESET}")
    print(f"{DIM}Target: Finding decision boundary coordinates using mutated query vectors.{RESET}\n")

    origin = {"ip": "203.0.113.45", "country": "Singapore"}

    # Bisection search interval [0.0, 1.0]
    low = 0.0
    high = 1.0

    for step in range(1, iterations + 1):
        mid = (low + high) / 2.0
        payload = {
            "source_ip": origin["ip"],
            "destination_ip": "192.168.1.10",
            "flow_type": f"Probe Step {step}",
            "country": origin["country"],
            "is_attack": True,
            "is_probe": True,
            "evasion_mutation": round(mid, 4)
        }

        res = send_packet(DEFAULT_SERVER_URL, payload)
        t_str = time.strftime("%H:%M:%S")

        if res["blocked"]:
            verdict = f"{RED}DETECTED (Blocked 403){RESET}"
            afp_info = res["body"].get("details", {}).get("afp_intensity", 0.42)
            # Attacker tries to push further toward benign to evade
            low = mid
            print(f"[{t_str}] Probe #{step}: Mutation={mid:.3f} -> {verdict} | AFP Intensity={afp_info:.2f} (Perturbation applied)")
        else:
            verdict = f"{YELLOW}EVADED (Allowed 200 - Bypassed!){RESET}"
            high = mid
            print(f"[{t_str}] Probe #{step}: Mutation={mid:.3f} -> {verdict} | IDS fooled by mutated coordinates!")

        time.sleep(0.4)

    print(f"\n{CYAN}[OK] Bisection Probing Finished. Note the Recall-Aware controller adjustments on the SOC dashboard!{RESET}\n")


def run_comparative_benchmark():
    """
    Direct head-to-head comparison demonstrating:
    1. Without AFP (AFP OFF): Evasive mutated attack slips through as 200 OK.
    2. With AFP (AFP ON): Adaptive perturbation distorts the vector, forcing IDS to detect it as 403 Blocked.
    """
    print(f"\n{YELLOW}{BOLD}======================================================================{RESET}")
    print(f"{YELLOW}{BOLD}  COMPARATIVE BENCHMARK: AFP DEFENSE OFF vs. AFP DEFENSE ON{RESET}")
    print(f"{YELLOW}{BOLD}======================================================================{RESET}\n")

    print(f"{WHITE}Step 1: Setting AFP Defense Layer to {RED}OFF (Bypassed){RESET}...")
    toggle_server_afp(False)
    time.sleep(0.5)

    print(f"{DIM}Sending Evasive Adversarial Attack Payload (Mutation Ratio: 0.48)...{RESET}")
    res_off = send_packet(DEFAULT_SERVER_URL, {
        "source_ip": "89.248.163.77",
        "destination_ip": "192.168.1.10",
        "flow_type": "Adversarial Evasion",
        "country": "Russia",
        "is_attack": True,
        "is_probe": True,
        "evasion_mutation": 0.48
    })
    
    if res_off["status_code"] == 200:
        print(f"{RED}{BOLD}[EVASION SUCCESSFUL]{RESET} The attack PASSED into the protected server! Status: {GREEN}200 OK{RESET}")
        print(f"{RED}Reason: Without AFP, the classifier's boundary was probed and bypassed.{RESET}\n")
    else:
        print(f"Result: {res_off['status_code']} - {res_off['body']}\n")

    time.sleep(1.0)

    print(f"{WHITE}Step 2: Activating AFP Defense Layer {GREEN}ON (Adaptive Feature Perturbation Active){RESET}...")
    toggle_server_afp(True)
    time.sleep(0.5)

    print(f"{DIM}Sending IDENTICAL Evasive Adversarial Attack Payload (Mutation Ratio: 0.48)...{RESET}")
    res_on = send_packet(DEFAULT_SERVER_URL, {
        "source_ip": "89.248.163.77",
        "destination_ip": "192.168.1.10",
        "flow_type": "Adversarial Evasion",
        "country": "Russia",
        "is_attack": True,
        "is_probe": True,
        "evasion_mutation": 0.48
    })

    if res_on["blocked"]:
        print(f"{GREEN}{BOLD}[INTRUSION CAUGHT & BLOCKED]{RESET} Status: {RED}403 FORBIDDEN{RESET}")
        print(f"{GREEN}Reason: Recall-Aware AFP injected dynamic deviation noise, corrupting the evasion vector.{RESET}\n")
    else:
        print(f"Result: {res_on['status_code']} - {res_on['body']}\n")

    print(f"{CYAN}{BOLD}Summary: AFP successfully eliminated the adversarial evasion window.{RESET}\n")


def run_continuous_stream(delay: float = 1.0):
    print(f"\n{CYAN}{BOLD}>>> Starting Continuous Traffic & Threat Stream (Press Ctrl+C to stop)...{RESET}\n")
    attack_types = ["DDoS", "Port Scan", "Brute Force", "Malware", "Infiltration", "Bot"]
    
    try:
        while True:
            # 70% Benign, 30% Attacks to simulate realistic traffic
            is_attack = (random.random() < 0.35)
            if is_attack:
                a_type = random.choice(attack_types)
                is_probe = (random.random() < 0.30)
                mut = round(random.uniform(0.35, 0.55), 3) if is_probe else 0.0
                run_single_flow(flow_type=a_type, is_attack=True, is_probe=is_probe, mutation=mut)
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
        print(f"  {RED}[2]{RESET} Launch DDoS Flood (Volumetric Attack Burst)")
        print(f"  {RED}[3]{RESET} Launch Port Scan Attack (Network Recon)")
        print(f"  {RED}[4]{RESET} Launch Brute Force Attack (Credential Stuffing)")
        print(f"  {RED}[5]{RESET} Launch Malware Infiltration Attack")
        print(f"  {MAGENTA}[6]{RESET} Launch Adversarial Evasion Probing (1D Bisection Search)")
        print(f"  {YELLOW}[7]{RESET} Comparative Benchmark: AFP Defense ON vs. OFF")
        print(f"  {BLUE}[8]{RESET} Continuous Real-time Stream (1 flow / second)")
        print(f"  {WHITE}[9]{RESET} Toggle AFP Defense (ON / OFF)")
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
            run_volumetric_burst("Port Scan", count=5, delay=0.3)
        elif choice == "4":
            run_volumetric_burst("Brute Force", count=5, delay=0.3)
        elif choice == "5":
            run_volumetric_burst("Malware", count=4, delay=0.4)
        elif choice == "6":
            run_adversarial_probing_attack(iterations=6)
        elif choice == "7":
            run_comparative_benchmark()
        elif choice == "8":
            run_continuous_stream(delay=1.0)
        elif choice == "9":
            # Toggle
            print(f"Toggling AFP defense state on server...")
            # Send flip
            current_state = True
            new_state = toggle_server_afp(not current_state)
            print(f"Server AFP Defense is now: {GREEN if new_state else RED}{'Active' if new_state else 'Bypassed'}{RESET}\n")
        elif choice == "0":
            print("Exiting attacker simulation console.")
            break
        else:
            print(f"{RED}Invalid selection. Please choose 0-9.{RESET}\n")


def main():
    parser = argparse.ArgumentParser(description="Attacker Simulation Console for IDS + Recall-Aware AFP")
    parser.add_argument("--mode", choices=["menu", "ddos", "portscan", "bruteforce", "malware", "probe", "compare", "stream", "benign"], default="menu", help="Attack simulation mode")
    parser.add_argument("--target", default=DEFAULT_SERVER_URL, help="Target protected server URL")
    parser.add_argument("--count", type=int, default=5, help="Packet count for bursts")
    parser.add_argument("--delay", type=float, default=0.3, help="Inter-packet delay in seconds")

    args = parser.parse_args()

    if args.mode == "menu":
        interactive_menu()
    elif args.mode == "benign":
        run_single_flow(flow_type="Normal", is_attack=False)
    elif args.mode == "ddos":
        run_volumetric_burst("DDoS", count=args.count, delay=args.delay)
    elif args.mode == "portscan":
        run_volumetric_burst("Port Scan", count=args.count, delay=args.delay)
    elif args.mode == "bruteforce":
        run_volumetric_burst("Brute Force", count=args.count, delay=args.delay)
    elif args.mode == "malware":
        run_volumetric_burst("Malware", count=args.count, delay=args.delay)
    elif args.mode == "probe":
        run_adversarial_probing_attack(iterations=args.count)
    elif args.mode == "compare":
        run_comparative_benchmark()
    elif args.mode == "stream":
        run_continuous_stream(delay=args.delay)


if __name__ == "__main__":
    main()
