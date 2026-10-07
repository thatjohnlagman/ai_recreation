#!/usr/bin/env python3
"""
Operator Benchmark & Automated Comparison Suite
Standalone IDS Tool + Recall-Aware Research Platform

Authorized Operator-Only Orchestration:
- Base vs. Recall-Aware Control-Path Demonstration (5-flow measurement sequence)
- Cross-Defense Benchmark Replay across [AFP, RS, FS, None] in Base mode
- Server Configuration & Management with Operator Token Authorization

NOTE:
- Attacker simulation scripts have no configuration or reset authority.
- All management calls require valid X-Operator-Token authentication.
- Benchmark flows omit procedure names (scenario=None); exact measurement samples
  use server-side reference labels.
"""

import sys
import os
import time
import json
import argparse
import urllib.request
import urllib.error
from pathlib import Path
from typing import Optional, Dict, Any, List

# Ensure UTF-8 output on Windows consoles if supported
if hasattr(sys.stdout, 'reconfigure'):
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass

BASE_DIR = Path(__file__).resolve().parent
RUNTIME_DIR = BASE_DIR / "runtime_package"
if str(RUNTIME_DIR) not in sys.path:
    sys.path.insert(0, str(RUNTIME_DIR))

import attacker_sim
from attacker_sim import (
    get_ctx,
    run_single_flow,
    get_server_url,
    get_stats_url,
    get_base_url,
    set_target_url,
    check_backend_compatibility,
    RESET, BOLD, DIM, RED, GREEN, YELLOW, BLUE, MAGENTA, CYAN, WHITE
)

PRIVATE_TOKEN_FILE = BASE_DIR / ".operator_token"
TOKEN_FILE = BASE_DIR / "operator_token.txt"


def get_operator_token(cli_token: Optional[str] = None) -> str:
    """
    Resolves the operator token with the following priority:
    1. CLI argument (--token)
    2. Environment variable (IDS_OPERATOR_TOKEN)
    3. Server-generated private token (.operator_token)
    4. Local configuration token (operator_token.txt)
    """
    if cli_token and cli_token.strip():
        return cli_token.strip()
    env_token = os.environ.get("IDS_OPERATOR_TOKEN")
    if env_token and env_token.strip():
        return env_token.strip()
    if PRIVATE_TOKEN_FILE.exists():
        try:
            content = PRIVATE_TOKEN_FILE.read_text(encoding="utf-8").strip()
            if content:
                return content
        except Exception:
            pass
    if TOKEN_FILE.exists():
        try:
            content = TOKEN_FILE.read_text(encoding="utf-8").strip()
            if content:
                return content
        except Exception:
            pass
    raise RuntimeError(
        "Operator authorization token not found. Please start the server (which creates "
        ".operator_token) or configure $env:IDS_OPERATOR_TOKEN or pass --token."
    )


def get_management_url(path: str) -> str:
    base = get_base_url()
    return f"{base}/api/dashboard/{path.lstrip('/')}"


def set_server_defense(defense_name: str, token: Optional[str] = None) -> bool:
    """Configures server defense ('afp', 'rs', 'fs', 'none') with operator authorization."""
    auth_token = get_operator_token(token)
    url = get_management_url("set-defense")
    payload = json.dumps({"defense": defense_name}).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=payload,
        headers={
            "Content-Type": "application/json",
            "X-Operator-Token": auth_token
        }
    )
    try:
        with urllib.request.urlopen(req, timeout=5) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            return True
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8", errors="replace") if hasattr(e, "read") else ""
        print(f"{RED}[Operator Error] Failed to set defense ({e.code}): {body}{RESET}")
        return False
    except Exception as e:
        print(f"{RED}[Operator Error] Failed to set defense: {e}{RESET}")
        return False


def set_server_mode(mode: str, token: Optional[str] = None) -> bool:
    """Configures controller mode ('recall-aware', 'base') with operator authorization."""
    auth_token = get_operator_token(token)
    url = get_management_url("set-mode")
    payload = json.dumps({"mode": mode}).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=payload,
        headers={
            "Content-Type": "application/json",
            "X-Operator-Token": auth_token
        }
    )
    try:
        with urllib.request.urlopen(req, timeout=5) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            return True
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8", errors="replace") if hasattr(e, "read") else ""
        print(f"{RED}[Operator Error] Failed to set controller mode ({e.code}): {body}{RESET}")
        return False
    except Exception as e:
        print(f"{RED}[Operator Error] Failed to set controller mode: {e}{RESET}")
        return False


def reset_server_state(token: Optional[str] = None) -> bool:
    """Resets dashboard metrics and controller state to cold start baseline."""
    auth_token = get_operator_token(token)
    url = get_management_url("reset")
    req = urllib.request.Request(
        url,
        data=b"{}",
        headers={
            "Content-Type": "application/json",
            "X-Operator-Token": auth_token
        }
    )
    try:
        with urllib.request.urlopen(req, timeout=5) as resp:
            return True
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8", errors="replace") if hasattr(e, "read") else ""
        print(f"{RED}[Operator Error] Failed to reset server state ({e.code}): {body}{RESET}")
        return False
    except Exception as e:
        print(f"{RED}[Operator Error] Failed to reset server state: {e}{RESET}")
        return False


def run_comparative_benchmark(defense: Optional[str] = None, token: Optional[str] = None) -> bool:
    """
    Authorized Operator Benchmark:
    Local Control-Path Demonstration: Compares Base (fixed intensity) versus Recall-Aware (adaptive controller)
    for the selected defense across an identical 5-flow measurement sequence.
    If defense is None, obtains the current defense from validated dashboard telemetry.
    All flows pass scenario=None to prevent procedural leaks into telemetry.
    """
    c = get_ctx()
    auth_token = get_operator_token(token)

    if defense:
        target_def = defense.lower().strip()
    else:
        try:
            req = urllib.request.Request(get_stats_url())
            with urllib.request.urlopen(req, timeout=3) as resp:
                data = json.loads(resp.read().decode("utf-8"))
            telemetry_def = data.get("afp", {}).get("defense_name")
            if not telemetry_def or telemetry_def.lower() not in ["afp", "rs", "fs", "none"]:
                print(f"{RED}[Error] Current defense telemetry is unavailable or invalid ('{telemetry_def}'). Cannot run benchmark.{RESET}")
                return False
            target_def = telemetry_def.lower()
        except Exception as e:
            print(f"{RED}[Error] Failed to fetch current defense telemetry from server: {e}. Cannot run benchmark.{RESET}")
            return False

    print(f"\n{YELLOW}{BOLD}======================================================================{RESET}")
    print(f"{YELLOW}{BOLD}  OPERATOR BENCHMARK: BASE vs. RECALL-AWARE ({target_def.upper()}){RESET}")
    print(f"{WHITE}  Evaluating identical 5-flow measurement sequence under fixed vs adaptive controller{RESET}")
    print(f"{YELLOW}{BOLD}======================================================================{RESET}\n")

    # Pick 5 fixed measurement attack samples for exact repeatability
    if c.profile == "fixture20":
        fixed_indices = c.attack_indices[:5].tolist()
    else:
        fixed_indices = c.dataset.measurement_attack_indices[:5].tolist()

    print(f"{WHITE}Saved Measurement Target IDs: {CYAN}{fixed_indices}{RESET} (Profile: {c.profile})")
    print(f"{DIM}Note: Demonstrates control path. First adaptive update applies after the 5th attack decision.{RESET}\n")

    # Step 1: Base Mode
    print(f"{WHITE}Step 1: Setting {target_def.upper()} Defense to {YELLOW}BASE MODE (Static Calibrated Intensity){RESET}...")
    if not set_server_defense(target_def, token=auth_token) or not set_server_mode("base", token=auth_token):
        print(f"{RED}[FAIL] Operator setup failed for Base mode.{RESET}")
        return False

    if not reset_server_state(token=auth_token):
        print(f"{RED}[FAIL] Reset failed before Base arm.{RESET}")
        return False
    time.sleep(0.5)

    print(f"{DIM}Sending 5 fixed attack flows under Base {target_def.upper()}...{RESET}")
    results_base = []
    for idx in fixed_indices:
        x_row = c.dataset.X.iloc[idx].values
        fam = str(c.dataset.metadata.iloc[idx].get("attack_family", "Unknown"))
        r = run_single_flow(flow_type=fam, is_attack=True, vector=x_row, sample_id=idx, scenario=None)
        if r is None or r.get("status_code") not in (200, 403):
            status = r.get("status_code") if r else "No response"
            print(f"{RED}[FAIL] Error during Base mode flow transmission (HTTP {status}).{RESET}")
            return False
        results_base.append(r)
        time.sleep(0.2)

    time.sleep(0.8)

    # Step 2: Recall-Aware Mode
    print(f"\n{WHITE}Step 2: Activating {GREEN}RECALL-AWARE CONTROLLER (Dynamic Feedback Window){RESET}...")
    if not set_server_defense(target_def, token=auth_token) or not set_server_mode("recall-aware", token=auth_token):
        print(f"{RED}[FAIL] Operator setup failed for Recall-Aware mode.{RESET}")
        return False

    if not reset_server_state(token=auth_token):
        print(f"{RED}[FAIL] Reset failed before Recall-Aware arm.{RESET}")
        return False
    time.sleep(0.5)

    results_recall = []
    print(f"{DIM}Sending the SAME 5 attack flows under Recall-Aware {target_def.upper()}...{RESET}")
    for idx in fixed_indices:
        x_row = c.dataset.X.iloc[idx].values
        fam = str(c.dataset.metadata.iloc[idx].get("attack_family", "Unknown"))
        r = run_single_flow(flow_type=fam, is_attack=True, vector=x_row, sample_id=idx, scenario=None)
        if r is None or r.get("status_code") not in (200, 403):
            status = r.get("status_code") if r else "No response"
            print(f"{RED}[FAIL] Error during Recall-Aware mode flow transmission (HTTP {status}).{RESET}")
            return False
        results_recall.append(r)
        time.sleep(0.2)

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
    elif target_def == "afp":
        if detected_count == total_count:
            print(f"{WHITE}With all {total_count} attack flows detected (Recall = 1.000 >= Rmin), the controller maintained calibrated baseline intensity={obs_intensity_str} in {obs_state} state.{RESET}")
            print(f"{DIM}Note: When evasion causes lower recall below threshold, the controller adapts downward (Adaptive Feature Poisoning: study-defined bounded noise, not centroid projection).{RESET}\n")
        else:
            print(f"{WHITE}Observed evasion resulted in controller adaptation: intensity={obs_intensity_str}, state={obs_state} (Adaptive Feature Poisoning: study-defined bounded noise, not centroid projection).{RESET}\n")
    elif target_def == "rs":
        print(f"{WHITE}Randomized Smoothing evaluated in {obs_state} mode (sigma parameter: {obs_intensity_str}). Confidence scores reflect 11-member ensemble vote fractions from Gaussian noisy copies with majority voting.{RESET}\n")
    elif target_def == "fs":
        d = max(0, 6 - int(raw_intensity)) if isinstance(raw_intensity, (int, float)) else "N/A"
        print(f"{WHITE}Feature Squeezing evaluated in {obs_state} mode (continuous squeezing intensity: {obs_intensity_str}, retained decimal places d={d} via d = max(0, 6 - int(squeezing_intensity))).{RESET}\n")
    else:
        print(f"{WHITE}Baseline classifier (No defense) evaluated with zero perturbation (intensity={obs_intensity_str}).{RESET}\n")

    return True


def run_cross_defense_comparison(count: int = 5, token: Optional[str] = None) -> bool:
    """
    Authorized Operator Benchmark:
    Replays one saved measurement sequence across AFP, RS, FS, and None in Base mode.
    Sequence contains both benign and attack rows to evaluate TP, FN, FP, TN, Recall, and FPR.
    Note: The comparison sequence is intentionally bounded to 5 flows for rapid inspection.
    """
    c = get_ctx()
    auth_token = get_operator_token(token)

    print(f"\n{CYAN}{BOLD}======================================================================{RESET}")
    print(f"{CYAN}{BOLD}  OPERATOR BENCHMARK: CROSS-DEFENSE REPLAY (BASE MODE){RESET}")
    print(f"{WHITE}  Evaluating AFP, RS, FS, and None across identical saved measurement flows (bounded to 5 flows){RESET}")
    print(f"{CYAN}{BOLD}======================================================================{RESET}\n")

    # Select saved sequence containing both classes (2 benign, 3 attack)
    if c.profile == "fixture20":
        sample_sequence = [0, 1, 10, 11, 12]
    else:
        sample_sequence = list(c.dataset.measurement_benign_indices[:2]) + list(c.dataset.measurement_attack_indices[:3])

    print(f"{WHITE}Replay Sequence IDs: {CYAN}{sample_sequence}{RESET} (Profile: {c.profile}, Seed: {c.seed})")
    print(f"{DIM}Classes in sequence: {sum(1 for sid in sample_sequence if c.dataset.metadata.iloc[sid]['y_binary'] == 0)} Benign, {sum(1 for sid in sample_sequence if c.dataset.metadata.iloc[sid]['y_binary'] == 1)} Attack{RESET}\n")

    defenses = ["afp", "rs", "fs", "none"]
    arm_results = {}
    all_arms_completed = True

    for def_name in defenses:
        print(f"{WHITE}Evaluating Defense: {YELLOW}{def_name.upper()}{RESET} (Base Mode)...")
        if not set_server_defense(def_name, token=auth_token) or not set_server_mode("base", token=auth_token):
            print(f"{RED}[FAIL] Setup failed for {def_name}.{RESET}")
            arm_results[def_name] = {"defense": def_name.upper(), "status": "FAILED", "error": "Setup failed"}
            all_arms_completed = False
            continue

        if not reset_server_state(token=auth_token):
            print(f"{RED}[FAIL] Reset failed for {def_name}.{RESET}")
            arm_results[def_name] = {"defense": def_name.upper(), "status": "FAILED", "error": "Reset failed"}
            all_arms_completed = False
            continue
        time.sleep(0.3)

        flow_verdicts = []
        failed_transmissions = []
        for sid in sample_sequence:
            x_row = c.dataset.X.iloc[sid].values
            true_y = int(c.dataset.metadata.iloc[sid]["y_binary"])
            fam = str(c.dataset.metadata.iloc[sid].get("attack_family", "Normal" if true_y == 0 else "Unknown"))
            r = run_single_flow(flow_type=fam, is_attack=(true_y == 1), vector=x_row, sample_id=sid, scenario=None)
            if r is None or r.get("status_code") not in (200, 403):
                sc = r.get("status_code") if r else "No response"
                failed_transmissions.append((sid, sc))
            else:
                flow_verdicts.append(r)
            time.sleep(0.15)

        if failed_transmissions:
            print(f"{RED}[FAIL] Arm {def_name.upper()} had {len(failed_transmissions)} failed transmissions: {failed_transmissions}. Arm incomplete.{RESET}")
            arm_results[def_name] = {"defense": def_name.upper(), "status": "FAILED", "error": f"{len(failed_transmissions)} transmissions failed"}
            all_arms_completed = False
            continue

        # Retrieve server stats
        try:
            req = urllib.request.Request(get_stats_url())
            with urllib.request.urlopen(req, timeout=3) as resp:
                stats_payload = json.loads(resp.read().decode())
        except Exception as e:
            print(f"{RED}[FAIL] Telemetry retrieval failed for {def_name}: {e}{RESET}")
            arm_results[def_name] = {"defense": def_name.upper(), "status": "FAILED", "error": f"Telemetry failed: {e}"}
            all_arms_completed = False
            continue

        st = stats_payload.get("stats") if isinstance(stats_payload, dict) else None
        afp_info = stats_payload.get("afp") if isinstance(stats_payload, dict) else None

        if not isinstance(st, dict) or "tp" not in st or "fn" not in st:
            print(f"{RED}[FAIL] Malformed or missing telemetry stats for {def_name}.{RESET}")
            arm_results[def_name] = {"defense": def_name.upper(), "status": "FAILED", "error": "Malformed telemetry"}
            all_arms_completed = False
            continue

        tp = st.get("tp", 0)
        fn = st.get("fn", 0)
        fp = st.get("fp", 0)
        tn = st.get("tn", 0)

        positives = tp + fn
        negatives = fp + tn
        recall_val = f"{(tp / positives):.3f}" if positives > 0 else "N/A"
        fpr_val = f"{(fp / negatives):.3f}" if negatives > 0 else "N/A"

        conf_type = "Vote fraction (ensemble)" if def_name == "rs" else "RF probability"
        intensity_val = afp_info.get("intensity", "N/A") if isinstance(afp_info, dict) else "N/A"

        arm_results[def_name] = {
            "defense": def_name.upper(),
            "status": "COMPLETED",
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
    print(f"{BOLD}{'Defense':<10} {'Mode':<8} {'TP':<5} {'FN':<5} {'FP':<5} {'TN':<5} {'Recall':<10} {'FPR':<10} {'Intensity / Decimals':<22} {'Confidence Semantics':<24}{RESET}")
    print(f"{'-'*102}")
    for def_name in defenses:
        if def_name in arm_results:
            r = arm_results[def_name]
            if r.get("status") == "COMPLETED":
                if def_name == "fs" and isinstance(r['intensity'], (int, float)):
                    d = max(0, 6 - int(r['intensity']))
                    int_str = f"{r['intensity']:.5f} (d={d})"
                else:
                    int_str = f"{r['intensity']:.5f}" if isinstance(r['intensity'], (int, float)) else str(r['intensity'])
                print(f"{r['defense']:<10} {'Base':<8} {r['tp']:<5} {r['fn']:<5} {r['fp']:<5} {r['tn']:<5} {r['recall']:<10} {r['fpr']:<10} {int_str:<22} {r['conf_semantics']:<24}")
            else:
                print(f"{r['defense']:<10} {'Base':<8} {'FAIL':<5} {'—':<5} {'—':<5} {'—':<5} {'—':<10} {'—':<10} {'—':<22} {r.get('error', 'Arm failed'):<24}")
        else:
            print(f"{def_name.upper():<10} {'Base':<8} {'FAIL':<5} {'—':<5} {'—':<5} {'—':<5} {'—':<10} {'—':<10} {'—':<22} {'Arm not executed':<24}")
    print(f"{'-'*102}")
    print(f"{DIM}Note: RS confidence reflects ensemble vote fractions from Gaussian noisy copies with majority voting (11 sub-models). FS performs decimal precision reduction with d = max(0, 6 - int(squeezing_intensity)), distinguishing continuous intensity from retained decimal places. AFP (Adaptive Feature Poisoning: study-defined bounded noise, not centroid projection), FS, and None report direct Random Forest probability estimates. Equal verdicts do not imply identical internal representations.{RESET}\n")

    return all_arms_completed


def main():
    parser = argparse.ArgumentParser(description="Operator Benchmark & Automated Comparison Suite (IDS Tool)")
    parser.add_argument(
        "--mode",
        choices=["benchmark", "compare", "compare-all", "set-defense", "set-mode", "reset"],
        default="compare",
        help="Operator action: 'compare'/'benchmark' (Base vs RA), 'compare-all' (cross-defense replay), or management actions"
    )
    parser.add_argument("--defense", choices=["afp", "rs", "fs", "none"], default=None, help="Target defense for benchmark or set-defense")
    parser.add_argument("--controller", choices=["recall-aware", "base"], default=None, help="Target mode for set-mode")
    parser.add_argument("--target", default="http://localhost:8000/api/server/data", help="Target URL for protected server endpoint")
    parser.add_argument("--dataset", choices=["expanded", "fixture20"], default=None, help="Dataset profile (expanded or fixture20)")
    parser.add_argument("--seed", type=int, default=42, help="RNG seed for target sampling without replacement")
    parser.add_argument("--token", default=None, help="Operator authentication token override")
    parser.add_argument("--count", type=int, default=5, help="Number of flows for comparison sequence")

    args = parser.parse_args()

    if args.target:
        set_target_url(args.target)

    active_profile = args.dataset or os.environ.get("IDS_DATA_PROFILE", "expanded").strip().lower()
    try:
        current_ctx = get_ctx(active_profile, seed=args.seed)
    except Exception as e:
        print(f"{RED}[Fatal] Failed to initialize dataset profile '{active_profile}': {e}{RESET}")
        sys.exit(1)

    if not check_backend_compatibility(require_profile=active_profile):
        print(f"{RED}[Fatal] Backend compatibility check failed. Halting.{RESET}")
        sys.exit(1)

    success = True
    token = args.token

    if args.mode in ["benchmark", "compare"]:
        success = run_comparative_benchmark(defense=args.defense, token=token)
    elif args.mode == "compare-all":
        success = run_cross_defense_comparison(count=args.count, token=token)
    elif args.mode == "set-defense":
        if not args.defense:
            print(f"{RED}[Error] --defense is required when mode is 'set-defense'.{RESET}")
            sys.exit(1)
        success = set_server_defense(args.defense, token=token)
    elif args.mode == "set-mode":
        if not args.controller:
            print(f"{RED}[Error] --controller is required when mode is 'set-mode'.{RESET}")
            sys.exit(1)
        success = set_server_mode(args.controller, token=token)
    elif args.mode == "reset":
        success = reset_server_state(token=token)

    if not success:
        sys.exit(1)


if __name__ == "__main__":
    main()
