import pytest
import requests
import json
import numpy as np

BASE_URL = "http://localhost:8000"

@pytest.fixture(autouse=True)
def check_live_server():
    try:
        r = requests.get(f"{BASE_URL}/api/dashboard/stats", timeout=1)
        if r.status_code != 200:
            pytest.skip("Live server on port 8000 not responding.")
    except Exception:
        pytest.skip("Live server on port 8000 not reachable.")

def test_simulation_session_lifecycle():
    """Verify session management: start, query status, and stop."""
    # Start session
    start_res = requests.post(
        f"{BASE_URL}/api/simulation/start",
        json={"scenario": "Silent Probing", "session_id": "test-session-001"}
    )
    assert start_res.status_code == 200
    start_data = start_res.json()
    assert start_data["status"] == "started"
    assert start_data["scenario"] == "Silent Probing"
    assert start_data["session_id"] == "test-session-001"

    # Query dashboard telemetry
    stats_res = requests.get(f"{BASE_URL}/api/dashboard/stats")
    assert stats_res.status_code == 200
    sim_data = stats_res.json().get("simulation", {})
    assert sim_data["active_scenario"] == "Silent Probing"
    assert sim_data["is_active"] is True
    assert sim_data["session_id"] == "test-session-001"

    # Stop session
    stop_res = requests.post(f"{BASE_URL}/api/simulation/stop")
    assert stop_res.status_code == 200
    assert stop_res.json()["status"] == "stopped"

    # Verify session is cleared
    stats_after = requests.get(f"{BASE_URL}/api/dashboard/stats").json()
    assert stats_after["simulation"]["active_scenario"] == "None"
    assert stats_after["simulation"]["is_active"] is False

def test_authoritative_binary_ids_classification():
    """Verify that Random Forest output is strictly binary: 'Attack' or 'Benign'."""
    # Test submission
    payload = {
        "source_ip": "10.0.2.45",
        "sample_id": 1,
        "is_attack": True,
        "attack_scenario": "Silent Probing"
    }
    res = requests.post(f"{BASE_URL}/api/server/data", json=payload)
    assert res.status_code in [200, 403]
    details = res.json().get("details", {})
    assert details["ids_classification"] in ["Attack", "Benign"]
    assert details["ids_classification"] != "DDoS"
    assert details["ids_classification"] != "Silent Probing"

def test_traffic_family_separation():
    """Verify Traffic Family comes from dataset metadata or is marked Unknown for synthetic."""
    # 1. Dataset flow (sample_id 0 is Benign in CSE-CIC-IDS2018 demo metadata)
    res_ds = requests.post(
        f"{BASE_URL}/api/server/data",
        json={"source_ip": "10.0.1.5", "sample_id": 0}
    )
    assert res_ds.status_code == 200
    details_ds = res_ds.json().get("details", {})
    assert details_ds["traffic_family"] == "Benign"

    # 2. Synthetic flow without dataset metadata
    res_synth = requests.post(
        f"{BASE_URL}/api/server/data",
        json={"source_ip": "198.51.100.22", "feature_vector": [0.0] * 78}
    )
    assert res_synth.status_code in [200, 403]
    details_synth = res_synth.json().get("details", {})
    assert details_synth["traffic_family"] == "Unknown"

def test_ip_geolocation_strictness():
    """Verify strict IP classification: Private Network vs Unknown (no invented countries)."""
    # RFC 1918 Private IP
    res_priv = requests.post(
        f"{BASE_URL}/api/server/data",
        json={"source_ip": "192.168.1.50", "sample_id": 0}
    )
    assert res_priv.status_code == 200

    # Public IP with no GeoIP DB
    res_pub = requests.post(
        f"{BASE_URL}/api/server/data",
        json={"source_ip": "203.0.113.199", "sample_id": 0}
    )
    assert res_pub.status_code == 200

    feed = requests.get(f"{BASE_URL}/api/dashboard/stats").json()["recent_feed"]
    # Check that private IP received Private Network
    priv_entry = next((item for item in feed if item["source_ip"] == "192.168.1.50"), None)
    if priv_entry:
        assert priv_entry["location"] == "Private Network"

    # Check that public IP received Unknown
    pub_entry = next((item for item in feed if item["source_ip"] == "203.0.113.199"), None)
    if pub_entry:
        assert pub_entry["location"] == "Unknown"

def test_evasion_semantics():
    """Verify genuine adversarial evasion: an actual attack sample perturbed by attack algorithm yields Benign classification."""
    requests.post(
        f"{BASE_URL}/api/simulation/start",
        json={"scenario": "Decision-Boundary Attack", "session_id": "bisection-01"}
    )
    try:
        import pandas as pd
        from pathlib import Path
        X_demo = pd.read_parquet(Path("runtime_package/demo_data/X_demo.parquet"))
        metadata_demo = pd.read_parquet(Path("runtime_package/demo_data/metadata_demo.parquet"))
        
        orig_attack = X_demo.iloc[10].values
        ref_benign = X_demo[metadata_demo["y_binary"] == 0].iloc[0].values
        
        # Bisection candidate crossing boundary into benign prediction region
        alpha = 0.6
        x_adv = ((1 - alpha) * orig_attack + alpha * ref_benign).tolist()
        
        res = requests.post(
            f"{BASE_URL}/api/server/data",
            json={
                "source_ip": "10.0.3.77",
                "sample_id": 10,  # Actual attack sample in dataset
                "is_attack": True,  # True attack label (ground truth)
                "attack_scenario": "Decision-Boundary Attack",
                "feature_vector": x_adv
            }
        )
        assert res.status_code == 200
        details = res.json().get("details", {})
        # Scenario is Decision-Boundary Attack, but IDS classification on feature_vector is Benign
        assert details["attack_scenario"] == "Decision-Boundary Attack"
        assert details["ids_classification"] == "Benign"
        assert details["verdict"] == "FORWARDED"
    finally:
        requests.post(f"{BASE_URL}/api/simulation/stop")

