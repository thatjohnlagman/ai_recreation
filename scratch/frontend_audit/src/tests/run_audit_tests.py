"""
Comprehensive Automated Test Runner for the Master Runtime Audit (Tests A - G).
"""
import sys
import time
import requests
import numpy as np
import pandas as pd
import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "runtime_package"))

from operator_benchmarks import get_operator_token

BASE_URL = "http://localhost:8000"
OPERATOR_HEADERS = {"X-Operator-Token": get_operator_token()}


def run_all_audit_tests():
    results = {}

    print("=================================================================")
    print("  RUNNING CONTROLLED RUNTIME AUDIT TESTS (TESTS A - G)")
    print("=================================================================\n")

    # -------------------------------------------------------------------------
    # TEST A: No Attacker Running
    # -------------------------------------------------------------------------
    print(">> Running Test A: No Attacker Running...")
    requests.post(f"{BASE_URL}/api/simulation/stop")
    stats_a = requests.get(f"{BASE_URL}/api/dashboard/stats").json()
    sim_a = stats_a.get("simulation", {})
    assert sim_a.get("active_scenario") == "None", f"Expected 'None', got {sim_a.get('active_scenario')}"
    assert sim_a.get("is_active") is False, "Expected is_active=False"
    results["Test A"] = {
        "status": "PASS",
        "description": "No Attacker Running",
        "observed_scenario": sim_a.get("active_scenario"),
        "is_active": sim_a.get("is_active")
    }
    print("   [PASS] Scenario is None and is_active is False.\n")

    # -------------------------------------------------------------------------
    # TEST B: Silent Probing Attack Execution (Real SilentProbingAttack.generate)
    # -------------------------------------------------------------------------
    print(">> Running Test B: Silent Probing Active Attack...")
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path("runtime_package").resolve()))
    from attacks.silent_probing import SilentProbingAttack
    with open("runtime_package/model/feature_names.json", "r") as f:
        f_names = json.load(f)
    with open("runtime_package/model/feature_mask.json", "r") as f:
        mask_dict = json.load(f)
        mod_mask = np.array(list(mask_dict.values())[0], dtype=bool)
    bounds_df = pd.read_parquet("runtime_package/model/training_bounds.parquet")
    X_demo = pd.read_parquet("runtime_package/demo_data/X_demo.parquet")
    meta_demo = pd.read_parquet("runtime_package/demo_data/metadata_demo.parquet")

    from runtime_package.data_loader import get_dataset
    stats_info = requests.get(f"{BASE_URL}/api/dashboard/stats").json()
    prof_audit = stats_info.get("data_profile", "fixture20")
    ds_audit = get_dataset(prof_audit)

    start_b = requests.post(
        f"{BASE_URL}/api/simulation/start",
        json={"scenario": "Silent Probing", "session_id": "audit-sim-b"}
    ).json()
    assert start_b["scenario"] == "Silent Probing"

    # Instantiate real attack and generate candidate (makes 0 queries, unchanged copy)
    attack_b = SilentProbingAttack(f_names, mod_mask, bounds_df)
    chosen_idx_b = int(ds_audit.measurement_attack_indices[0])
    x_orig_b = ds_audit.X.iloc[chosen_idx_b].values
    res_b_atk = attack_b.generate(x_orig_b, oracle=None, sample_id=chosen_idx_b, true_label=1)
    assert res_b_atk.query_count == 0
    assert np.array_equal(res_b_atk.X_adv, x_orig_b)

    # Send generated candidate over HTTP
    req_b = {
        "source_ip": "10.0.2.45",
        "destination_ip": "192.168.1.10",
        "attack_scenario": "Silent Probing",
        "sample_id": chosen_idx_b,
        "is_attack": True,
        "feature_vector": res_b_atk.X_adv.tolist()
    }
    res_b = requests.post(f"{BASE_URL}/api/server/data", json=req_b)
    data_b = res_b.json()
    details_b = data_b.get("details", {})
    
    stats_b = requests.get(f"{BASE_URL}/api/dashboard/stats").json()
    sim_b = stats_b.get("simulation", {})
    
    assert sim_b.get("active_scenario") == "Silent Probing"
    assert details_b.get("ids_classification") in ["Attack", "Benign"]
    assert details_b.get("attack_scenario") == "Silent Probing"
    assert details_b.get("ids_classification") != "Silent Probing"

    results["Test B"] = {
        "status": "PASS",
        "description": "Silent Probing Execution & Separation",
        "scenario": sim_b.get("active_scenario"),
        "server_status_code": res_b.status_code,
        "ids_classification": details_b.get("ids_classification"),
        "confidence": details_b.get("confidence")
    }
    print(f"   [PASS] Scenario: {sim_b.get('active_scenario')} | IDS Classification: {details_b.get('ids_classification')} | HTTP: {res_b.status_code}\n")
    requests.post(f"{BASE_URL}/api/simulation/stop")

    # -------------------------------------------------------------------------
    # TEST C: Genuine Adversarial Evasion via DecisionBoundaryAttack.generate over HTTP
    # -------------------------------------------------------------------------
    print(">> Running Test C: Genuine Adversarial Evasion via DecisionBoundaryAttack.generate...")
    requests.post(
        f"{BASE_URL}/api/simulation/start",
        json={"scenario": "Decision-Boundary Attack", "session_id": "audit-sim-c"}
    )
    from attacks.boundary_attack import DecisionBoundaryAttack
    from attacker_sim import TargetOracle
    attack_c = DecisionBoundaryAttack(f_names, mod_mask, bounds_df, max_queries=50, binary_search_steps=10)
    oracle_c = TargetOracle(target_url=f"{BASE_URL}/api/server/data", max_queries_per_sample=50, scenario_name="Decision-Boundary Attack", session_id="audit-sim-c")
    if len(ds_audit.crafting_benign_indices) > 0:
        ref_pool = ds_audit.X.iloc[ds_audit.crafting_benign_indices[:100]].values
    else:
        ref_pool = ds_audit.X.iloc[ds_audit.measurement_benign_indices].values

    # Run genuine DecisionBoundaryAttack.generate querying server over HTTP
    chosen_idx_c = int(ds_audit.measurement_attack_indices[0])
    orig_atk = ds_audit.X.iloc[chosen_idx_c].values
    res_c_atk = attack_c.generate(orig_atk, oracle_c, sample_id=chosen_idx_c, true_label=1, reference_pool=ref_pool)
    print(f"   [INFO] DecisionBoundaryAttack generated: status={res_c_atk.status_code}, queries={oracle_c.get_query_count(chosen_idx_c)}, success={res_c_atk.success}")

    # Now verify the final candidate flow against the server
    req_c = {
        "source_ip": "10.0.3.77",
        "destination_ip": "192.168.1.10",
        "attack_scenario": "Decision-Boundary Attack",
        "sample_id": chosen_idx_c,
        "is_attack": True,  # True attack label / ground truth
        "feature_vector": res_c_atk.X_adv.tolist()
    }
    res_c = requests.post(f"{BASE_URL}/api/server/data", json=req_c)
    assert res_c.status_code in (200, 403), f"Expected 200 or 403, got {res_c.status_code}"
    details_c = res_c.json().get("details", {})
    
    assert details_c.get("attack_scenario") == "Decision-Boundary Attack", "Scenario must remain Decision-Boundary Attack"
    assert "DoS attacks-SlowHTTPTest" in details_c.get("traffic_family")

    results["Test C"] = {
        "status": "PASS",
        "description": "DecisionBoundaryAttack over HTTP Oracle",
        "active_scenario": details_c.get("attack_scenario"),
        "traffic_family": details_c.get("traffic_family"),
        "ids_classification": details_c.get("ids_classification"),
        "verdict": details_c.get("verdict"),
        "attack_success": res_c_atk.success,
        "queries_used": oracle_c.get_query_count(chosen_idx_c)
    }
    print(f"   [PASS] Boundary Search Completed -> Scenario: {details_c.get('attack_scenario')} | Traffic Family: {details_c.get('traffic_family')} | IDS Classification: {details_c.get('ids_classification')} | Verdict: {details_c.get('verdict')}\n")
    requests.post(f"{BASE_URL}/api/simulation/stop")


    # -------------------------------------------------------------------------
    # TEST D: Dataset-Derived Traffic Family
    # -------------------------------------------------------------------------
    print(">> Running Test D: Dataset-Derived Traffic Family...")
    # Sample 12 is DDoS attacks-LOIC-HTTP in metadata_demo (iloc 12)
    req_d = {
        "source_ip": "172.16.0.12",
        "sample_id": 12,
        "is_attack": True
    }
    res_d = requests.post(f"{BASE_URL}/api/server/data", json=req_d)
    feed_list = requests.get(f"{BASE_URL}/api/dashboard/stats").json()["recent_feed"]
    feed_d = next((item for item in feed_list if item["source_ip"] == "172.16.0.12"), feed_list[0])
    
    assert "Dataset-derived" in feed_d.get("traffic_family_source")
    assert feed_d.get("traffic_family") == "DDoS attacks-LOIC-HTTP"

    results["Test D"] = {
        "status": "PASS",
        "description": "Legitimate Dataset Family Metadata",
        "traffic_family": feed_d.get("traffic_family"),
        "traffic_family_source": feed_d.get("traffic_family_source")
    }
    print(f"   [PASS] Traffic Family: {feed_d.get('traffic_family')} | Source: {feed_d.get('traffic_family_source')}\n")

    # -------------------------------------------------------------------------
    # TEST E: Unknown Traffic Family for Synthetic Flows
    # -------------------------------------------------------------------------
    print(">> Running Test E: Unknown Family for Synthetic Flow...")
    req_e = {
        "source_ip": "198.51.100.99",
        "feature_vector": [0.0] * 78,
        "is_attack": True
    }
    res_e = requests.post(f"{BASE_URL}/api/server/data", json=req_e)
    feed_list = requests.get(f"{BASE_URL}/api/dashboard/stats").json()["recent_feed"]
    feed_e = next((item for item in feed_list if item["source_ip"] == "198.51.100.99"), feed_list[0])

    assert feed_e.get("traffic_family") == "Unknown"
    assert feed_e.get("traffic_family_source") == "Synthetic / Non-dataset"

    results["Test E"] = {
        "status": "PASS",
        "description": "Unknown Traffic Family for Non-Dataset Input",
        "traffic_family": feed_e.get("traffic_family"),
        "traffic_family_source": feed_e.get("traffic_family_source")
    }
    print(f"   [PASS] Traffic Family: {feed_e.get('traffic_family')} | Source: {feed_e.get('traffic_family_source')}\n")

    # -------------------------------------------------------------------------
    # TEST F: Private IP vs Public IP Geolocation
    # -------------------------------------------------------------------------
    print(">> Running Test F: Private vs Public IP Geolocation...")
    # 1. Private RFC 1918 IP
    requests.post(f"{BASE_URL}/api/server/data", json={"source_ip": "192.168.1.188", "sample_id": 0})
    feed_list = requests.get(f"{BASE_URL}/api/dashboard/stats").json()["recent_feed"]
    feed_priv = next((item for item in feed_list if item["source_ip"] == "192.168.1.188"), feed_list[0])
    assert feed_priv.get("location") == "Private Network", f"Expected Private Network, got {feed_priv.get('location')}"

    # 2. Public IP with no GeoIP DB
    requests.post(f"{BASE_URL}/api/server/data", json={"source_ip": "203.0.113.250", "sample_id": 0})
    feed_list = requests.get(f"{BASE_URL}/api/dashboard/stats").json()["recent_feed"]
    feed_pub = next((item for item in feed_list if item["source_ip"] == "203.0.113.250"), feed_list[0])
    assert feed_pub.get("location") == "Unknown", f"Expected Unknown, got {feed_pub.get('location')}"

    results["Test F"] = {
        "status": "PASS",
        "description": "Strict IP Geolocation Handling",
        "private_ip_location": feed_priv.get("location"),
        "public_ip_location": feed_pub.get("location")
    }
    print(f"   [PASS] Private IP (192.168.1.188): {feed_priv.get('location')} | Public IP (203.0.113.250): {feed_pub.get('location')}\n")

    # -------------------------------------------------------------------------
    # TEST G: Defense Pipeline & Mode Switching (Base vs Recall-Aware)
    # -------------------------------------------------------------------------
    print(">> Running Test G: Defense Switching (Base vs Recall-Aware)...")
    # Set to Base mode
    requests.post(f"{BASE_URL}/api/dashboard/set-defense", json={"defense": "afp"}, headers=OPERATOR_HEADERS)
    res_base = requests.post(f"{BASE_URL}/api/dashboard/set-mode", json={"mode": "base"}, headers=OPERATOR_HEADERS).json()
    assert res_base["mode"] == "base"
    stats_base = requests.get(f"{BASE_URL}/api/dashboard/stats").json()
    assert stats_base["afp"]["mode"] == "base"

    # Set to Recall-Aware mode
    res_ra = requests.post(f"{BASE_URL}/api/dashboard/set-mode", json={"mode": "recall-aware"}, headers=OPERATOR_HEADERS).json()
    assert res_ra["mode"] == "recall-aware"
    stats_ra = requests.get(f"{BASE_URL}/api/dashboard/stats").json()
    assert stats_ra["afp"]["mode"] == "recall-aware"

    results["Test G"] = {
        "status": "PASS",
        "description": "Base vs Recall-Aware Controller Mode Switching",
        "base_mode_verified": stats_base["afp"]["mode"],
        "ra_mode_verified": stats_ra["afp"]["mode"]
    }
    print(f"   [PASS] Successfully toggled between Base and Recall-Aware modes.\n")

    print("=================================================================")
    print("  ALL 7 AUDIT TESTS (TESTS A - G) PASSED SUCCESSFULLY!")
    print("=================================================================\n")
    return results

if __name__ == "__main__":
    run_all_audit_tests()
