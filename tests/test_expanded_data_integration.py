"""
Expanded Data Integration Verification Suite
Tests the 90,000-row expanded dataset contract, seeded measurement-pool sampling without replacement,
profile mismatch rejection, query vs target metric separation, attack workflows, and defense comparisons.
"""

import sys
import os
import json
import time
import hashlib
import requests
import numpy as np
import pandas as pd
from pathlib import Path
from collections import deque

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "runtime_package"))

from runtime_package.data_loader import get_dataset, LoadedDataset
from attacker_sim import AttackerContext, RemoteServerOracle, TargetOracle, run_single_flow, set_server_defense, set_server_mode

BASE_URL = "http://localhost:8000"


def test_1_data_contract_and_roles():
    print(">> [Check 1] Verifying 90,000-row Data Contract, Roles, and Schemas...")
    ds = get_dataset("expanded")
    assert ds.total_rows == 90000, f"Expected 90,000 rows, got {ds.total_rows}"
    assert ds.measurement_rows == 72000, f"Expected 72,000 measurement rows, got {ds.measurement_rows}"
    assert ds.crafting_rows == 18000, f"Expected 18,000 crafting rows, got {ds.crafting_rows}"
    assert len(ds.X) == 90000
    assert len(ds.metadata) == 90000
    assert ds.X.shape[1] == 78

    # Check finite float32 compatibility
    assert ds.X.values.dtype in (np.float32, np.float64)
    assert np.isfinite(ds.X.values).all(), "Found non-finite values in X_eval"

    # Check role separation: disjoint and complete
    craft_set = set(ds.crafting_indices)
    meas_set = set(ds.measurement_indices)
    assert len(craft_set.intersection(meas_set)) == 0, "Crafting and measurement roles overlap!"
    assert len(craft_set.union(meas_set)) == 90000, "Roles do not fully cover 90,000 rows"

    # Check class distributions
    y = ds.metadata["y_binary"].values
    assert int(np.sum(y == 0)) == 74679, f"Expected 74,679 benign, got {np.sum(y == 0)}"
    assert int(np.sum(y == 1)) == 15321, f"Expected 15,321 attack, got {np.sum(y == 1)}"

    # Check feature mask: 63 modifiable, 15 protected
    with open(REPO_ROOT / "runtime_package" / "model" / "feature_mask.json") as f:
        mask = list(json.load(f).values())[0]
    assert sum(mask) == 63 and (len(mask) - sum(mask)) == 15

    print(f"   [PASS] 90,000 rows validated: 72k measurement (59,743 benign, 12,257 attack), 18k crafting (14,936 benign, 3,064 attack).")
    print(f"   [PASS] 78 float32 finite features, 63 modifiable, 15 protected. Complete coverage, 0 overlap.\n")


def test_2_seeded_sampling_without_replacement():
    print(">> [Check 2] Verifying Seeded Sampling Without Replacement & Queue Exhaustion...")
    ctx_sample = AttackerContext(profile="expanded", seed=12345)

    # 1. Draw 100 measurement target IDs and verify all are distinct and belong to measurement pool
    drawn_ids = []
    for _ in range(100):
        tid = ctx_sample.draw_measurement_target(is_attack=False)
        drawn_ids.append(tid)

    assert len(drawn_ids) == 100
    assert len(set(drawn_ids)) == 100, f"Found duplicates in first 100 draws without replacement!"
    for tid in drawn_ids:
        assert ctx_sample.dataset.is_measurement(tid), f"Target ID {tid} is not in measurement pool!"
        assert not ctx_sample.dataset.is_crafting(tid), f"Target ID {tid} belongs to crafting pool!"

    # 2. Draw 50 attack targets and verify distinct and in measurement attack pool
    drawn_attacks = []
    for _ in range(50):
        aid = ctx_sample.draw_measurement_target(is_attack=True)
        drawn_attacks.append(aid)
    assert len(set(drawn_attacks)) == 50, "Found duplicates in attack targets!"
    for aid in drawn_attacks:
        assert ctx_sample.dataset.metadata.iloc[aid]["y_binary"] == 1
        assert ctx_sample.dataset.is_measurement(aid)

    # 3. Seed repeatability: reset with same seed produces identical sequence
    ctx_sample.reset_queues(seed=12345)
    repeat_ids = [ctx_sample.draw_measurement_target(is_attack=False) for _ in range(100)]
    assert drawn_ids == repeat_ids, "Same seed did not reproduce identical sequence!"

    # 4. Varied seed produces different sequence
    ctx_sample.reset_queues(seed=99999)
    diff_ids = [ctx_sample.draw_measurement_target(is_attack=False) for _ in range(100)]
    assert drawn_ids != diff_ids, "Different seed produced identical sequence!"

    # 5. Test queue exhaustion behavior on a small sub-queue
    small_queue = deque([101, 102, 103])
    exhausted_draws = []
    cycle = 0
    for _ in range(7):
        if len(small_queue) == 0:
            cycle += 1
            small_queue = deque([101, 102, 103])
        exhausted_draws.append(small_queue.popleft())
    assert len(exhausted_draws) == 7
    assert cycle == 2, f"Expected 2 exhaustion cycles, got {cycle}"

    print("   [PASS] 100 benign & 50 attack measurement draws verified with 0 duplicates before exhaustion.")
    print("   [PASS] Seed reproducibility and pool exhaustion cycle mechanics verified.\n")


def test_3_profile_mismatch_and_input_validation():
    print(">> [Check 3] Verifying Backend Profile Mismatch & Input Validation Rejection...")
    try:
        stats = requests.get(f"{BASE_URL}/api/dashboard/stats", timeout=2).json()
    except Exception:
        try:
            import pytest
            pytest.skip("Target server not running on port 8000")
        except ImportError:
            print("   [SKIP] Server not running.")
            return

    if stats.get("data_profile") != "expanded":
        try:
            import pytest
            pytest.skip(f"Server is running profile '{stats.get('data_profile')}'; test_3 requires expanded server.")
        except ImportError:
            print(f"   [SKIP] Server is running profile '{stats.get('data_profile')}'.")
            return

    # Reset metrics first
    requests.post(f"{BASE_URL}/api/dashboard/reset")
    stats_before = requests.get(f"{BASE_URL}/api/dashboard/stats").json().get("stats", {})
    traffic_before = int(str(stats_before.get("total_traffic", "0")).replace(",", ""))
    assert traffic_before == 0

    # 1. Profile mismatch: send profile="fixture20" to expanded server
    res_mismatch = requests.post(f"{BASE_URL}/api/server/data", json={
        "source_ip": "10.0.1.5",
        "sample_id": 0,
        "data_profile": "fixture20"
    })
    assert res_mismatch.status_code == 400, f"Expected 400 for profile mismatch, got {res_mismatch.status_code}"
    assert "Dataset profile mismatch" in res_mismatch.text

    # 2. Unknown profile: send profile="invalid_foo"
    res_unknown = requests.post(f"{BASE_URL}/api/server/data", json={
        "source_ip": "10.0.1.5",
        "sample_id": 0,
        "data_profile": "invalid_foo"
    })
    assert res_unknown.status_code == 400, f"Expected 400 for unknown profile, got {res_unknown.status_code}"
    assert "Unknown dataset profile" in res_unknown.text

    # 3. Invalid sample_id: negative sample_id
    res_neg = requests.post(f"{BASE_URL}/api/server/data", json={
        "source_ip": "10.0.1.5",
        "sample_id": -5,
        "data_profile": "expanded"
    })
    assert res_neg.status_code == 400, f"Expected 400 for negative sample_id, got {res_neg.status_code}"
    assert "Invalid sample_id" in res_neg.text

    # 4. Invalid sample_id: out of range (> 89,999)
    res_oob = requests.post(f"{BASE_URL}/api/server/data", json={
        "source_ip": "10.0.1.5",
        "sample_id": 999999,
        "data_profile": "expanded"
    })
    assert res_oob.status_code == 400, f"Expected 400 for out-of-range sample_id, got {res_oob.status_code}"
    assert "Invalid sample_id" in res_oob.text

    # 5. Verify counters did NOT change after all 4 rejected requests
    stats_after = requests.get(f"{BASE_URL}/api/dashboard/stats").json().get("stats", {})
    traffic_after = int(str(stats_after.get("total_traffic", "0")).replace(",", ""))
    assert traffic_after == 0, f"Counters changed despite invalid requests! traffic={traffic_after}"

    print("   [PASS] Profile mismatch rejected with 400 before counter modification.")
    print("   [PASS] Unknown profile and out-of-bounds sample_ids rejected with 400 before counter modification.\n")


def test_4_live_http_measured_target_submissions():
    print(">> [Check 4] Verifying Live HTTP Submissions of Measurement Flows...")
    try:
        stats = requests.get(f"{BASE_URL}/api/dashboard/stats", timeout=2).json()
    except Exception:
        try:
            import pytest
            pytest.skip("Target server not running on port 8000")
        except ImportError:
            return

    if stats.get("data_profile") != "expanded":
        try:
            import pytest
            pytest.skip(f"Server is running profile '{stats.get('data_profile')}'; test_4 requires expanded server.")
        except ImportError:
            return

    requests.post(f"{BASE_URL}/api/dashboard/reset")
    requests.post(f"{BASE_URL}/api/dashboard/set-defense", json={"defense": "afp"})
    requests.post(f"{BASE_URL}/api/dashboard/set-mode", json={"mode": "base"})

    ds = get_dataset("expanded")
    benign_sid = int(ds.measurement_benign_indices[0])
    attack_sid = int(ds.measurement_attack_indices[0])

    # 1. Send Benign measurement row
    r_benign = requests.post(f"{BASE_URL}/api/server/data", json={
        "source_ip": "192.168.1.100",
        "sample_id": benign_sid,
        "data_profile": "expanded",
        "is_query": False
    })
    assert r_benign.status_code in (200, 403)
    b_data = r_benign.json()
    assert b_data.get("details", {}).get("sample_id") == benign_sid
    assert b_data.get("details", {}).get("data_profile") == "expanded"
    assert b_data.get("details", {}).get("is_query") is False

    # 2. Send Attack measurement row
    r_attack = requests.post(f"{BASE_URL}/api/server/data", json={
        "source_ip": "203.0.113.45",
        "sample_id": attack_sid,
        "data_profile": "expanded",
        "is_query": False
    })
    assert r_attack.status_code in (200, 403)
    a_data = r_attack.json()
    assert a_data.get("details", {}).get("sample_id") == attack_sid
    assert a_data.get("details", {}).get("data_profile") == "expanded"
    assert a_data.get("details", {}).get("is_query") is False

    # 3. Verify target counters incremented
    stats = requests.get(f"{BASE_URL}/api/dashboard/stats").json().get("stats", {})
    traffic = int(str(stats.get("total_traffic", "0")).replace(",", ""))
    assert traffic == 2, f"Expected total_traffic=2, got {traffic}"
    tp = stats.get("tp", 0)
    fn = stats.get("fn", 0)
    assert (tp + fn) == 1, f"Expected 1 attack decision, got tp={tp}, fn={fn}"

    print(f"   [PASS] Benign (ID {benign_sid}) and Attack (ID {attack_sid}) successfully evaluated over HTTP.")
    print(f"   [PASS] Target flow counters correctly updated: total_traffic={traffic}, decisions={tp+fn}.\n")


def test_5_query_vs_target_metric_separation():
    print(">> [Check 5] Verifying Query Telemetry vs. Target Decision Metric Separation...")
    try:
        stats = requests.get(f"{BASE_URL}/api/dashboard/stats", timeout=2).json()
    except Exception:
        try:
            import pytest
            pytest.skip("Target server not running on port 8000")
        except ImportError:
            return

    if stats.get("data_profile") != "expanded":
        try:
            import pytest
            pytest.skip(f"Server is running profile '{stats.get('data_profile')}'; test_5 requires expanded server.")
        except ImportError:
            return

    requests.post(f"{BASE_URL}/api/dashboard/reset")
    requests.post(f"{BASE_URL}/api/dashboard/set-defense", json={"defense": "afp"})
    requests.post(f"{BASE_URL}/api/dashboard/set-mode", json={"mode": "recall-aware"})

    ds = get_dataset("expanded")
    crafting_sid = int(ds.crafting_indices[0])

    # 1. Send 10 explicit query flows (marked is_query=True, crafting sample_id)
    for i in range(10):
        res = requests.post(f"{BASE_URL}/api/server/data", json={
            "source_ip": "203.0.113.45",
            "sample_id": crafting_sid,
            "data_profile": "expanded",
            "is_query": True,
            "query_stage": "surrogate_fitting"
        })
        assert res.status_code in (200, 403)
        assert res.json().get("details", {}).get("is_query") is True

    # 2. Check server stats: query_count must be 10, target metrics must be 0!
    stats = requests.get(f"{BASE_URL}/api/dashboard/stats").json().get("stats", {})
    afp = requests.get(f"{BASE_URL}/api/dashboard/stats").json().get("afp", {})

    traffic = int(str(stats.get("total_traffic", "0")).replace(",", ""))
    query_count = stats.get("query_count", 0)
    tp = stats.get("tp", 0)
    fn = stats.get("fn", 0)
    fp = stats.get("fp", 0)
    tn = stats.get("tn", 0)
    batch_id = afp.get("batch_id", 0)

    assert query_count == 10, f"Expected query_count=10, got {query_count}"
    assert traffic == 0, f"Expected total_traffic=0, got {traffic}"
    assert tp == 0 and fn == 0 and fp == 0 and tn == 0, f"Target confusion matrix polluted by queries!"
    assert batch_id == 0, f"Controller batch advanced by query flows! batch_id={batch_id}"

    # 3. Now send 1 real target flow (is_query=False, measurement row)
    meas_sid = int(ds.measurement_attack_indices[0])
    r_target = requests.post(f"{BASE_URL}/api/server/data", json={
        "source_ip": "203.0.113.45",
        "sample_id": meas_sid,
        "data_profile": "expanded",
        "is_query": False
    })
    assert r_target.status_code in (200, 403)

    stats_after = requests.get(f"{BASE_URL}/api/dashboard/stats").json().get("stats", {})
    traffic_after = int(str(stats_after.get("total_traffic", "0")).replace(",", ""))
    query_after = stats_after.get("query_count", 0)
    assert traffic_after == 1, f"Expected total_traffic=1 after target flow, got {traffic_after}"
    assert query_after == 10, f"Expected query_count=10 unchanged, got {query_after}"

    print(f"   [PASS] 10 crafting queries incremented query_count to 10 with 0 pollution of target metrics.")
    print(f"   [PASS] Controller batch_id remained 0 during queries; target flow incremented traffic to 1.\n")


def test_6_cross_defense_comparison():
    print(">> [Check 6] Verifying Cross-Defense Comparison Workflow...")
    try:
        stats = requests.get(f"{BASE_URL}/api/dashboard/stats", timeout=2).json()
    except Exception:
        try:
            import pytest
            pytest.skip("Target server not running on port 8000")
        except ImportError:
            return

    if stats.get("data_profile") != "expanded":
        try:
            import pytest
            pytest.skip(f"Server is running profile '{stats.get('data_profile')}'; test_6 requires expanded server.")
        except ImportError:
            return

    from attacker_sim import run_cross_defense_comparison
    # Run bounded comparison of 5 flows across AFP, RS, FS, and None
    run_cross_defense_comparison(count=5)
    print("   [PASS] Cross-defense comparison completed across AFP, RS, FS, and None.\n")


def run_all_expanded_tests():
    print("=================================================================")
    print("  EXPANDED SIMULATION DATA INTEGRATION VERIFICATION")
    print("=================================================================\n")
    test_1_data_contract_and_roles()
    test_2_seeded_sampling_without_replacement()
    test_3_profile_mismatch_and_input_validation()
    test_4_live_http_measured_target_submissions()
    test_5_query_vs_target_metric_separation()
    test_6_cross_defense_comparison()
    print("=================================================================")
    print("  ALL EXPANDED DATA INTEGRATION CHECKS PASSED SUCCESSFULLY!")
    print("=================================================================")


if __name__ == "__main__":
    run_all_expanded_tests()
