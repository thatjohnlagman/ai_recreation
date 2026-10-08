import pytest
import requests
import json
import numpy as np
from pathlib import Path

import sys
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
from operator_benchmarks import get_operator_token

BASE_URL = "http://localhost:8000"

def get_operator_headers():
    try:
        return {"X-Operator-Token": get_operator_token()}
    except Exception:
        return {}

@pytest.fixture(autouse=True)
def check_live_server():
    try:
        r = requests.get(f"{BASE_URL}/api/dashboard/stats", timeout=1)
        if r.status_code != 200:
            pytest.skip("Live server on port 8000 not responding.")
    except Exception:
        pytest.skip("Live server on port 8000 not reachable.")

def test_dashboard_stats_endpoint():
    res = requests.get(f"{BASE_URL}/api/dashboard/stats")
    assert res.status_code == 200
    data = res.json()
    assert "stats" in data
    assert "afp" in data
    assert "threat_locations" in data
    assert "recent_feed" in data

def test_set_defense_and_mode():
    headers = get_operator_headers()
    # Test setting defense to RS
    res = requests.post(f"{BASE_URL}/api/dashboard/set-defense", json={"defense": "rs"}, headers=headers)
    assert res.status_code == 200
    assert res.json()["defense"] == "rs"

    # Test setting mode to base
    res = requests.post(f"{BASE_URL}/api/dashboard/set-mode", json={"mode": "base"}, headers=headers)
    assert res.status_code == 200
    assert res.json()["mode"] == "base"

    # Restore to AFP and Recall-Aware
    requests.post(f"{BASE_URL}/api/dashboard/set-defense", json={"defense": "afp"}, headers=headers)
    requests.post(f"{BASE_URL}/api/dashboard/set-mode", json={"mode": "recall-aware"}, headers=headers)

def test_protected_server_benign_traffic():
    payload = {
        "source_ip": "192.168.1.100",
        "sample_id": 0,
        "is_attack": False
    }
    res = requests.post(f"{BASE_URL}/api/server/data", json=payload)
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "success"

def test_protected_server_attack_traffic():
    payload = {
        "source_ip": "185.199.110.23",
        "sample_id": 1,
        "is_attack": True,
        "country": "Ukraine"
    }
    res = requests.post(f"{BASE_URL}/api/server/data", json=payload)
    assert res.status_code in [200, 403]
    data = res.json()
    assert "status" in data

def test_full_78_feature_vector_submission():
    dummy_78 = [0.0] * 78
    payload = {
        "source_ip": "203.0.113.45",
        "flow_type": "Adversarial Probe",
        "is_attack": True,
        "feature_vector": dummy_78
    }
    res = requests.post(f"{BASE_URL}/api/server/data", json=payload)
    assert res.status_code in [200, 403]
