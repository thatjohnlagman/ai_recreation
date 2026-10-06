"""
Defense Readiness Verification Suite
Tests all P0 (verdicts, truthfulness, input validation, oracle failure propagation, intensity tracking)
and P1 (provenance, RNG seed independence, offline CLI, attack implementations) requirements.
"""
import os
import sys
import hashlib
import json
import time
import subprocess
import requests
import numpy as np
import pandas as pd
import joblib
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "runtime_package"))

BASE_URL = "http://localhost:8000"


def test_1_model_integrity_and_specs():
    """Verify exact model SHA-256 hash, 200 trees, 78 numerical features, binary output."""
    print(">> [Check 1] Verifying Model SHA-256 and Specifications...")
    rf_path = REPO_ROOT / "runtime_package" / "model" / "frozen_rf.joblib"
    assert rf_path.exists(), f"Model file not found at {rf_path}"

    hasher = hashlib.sha256()
    with open(rf_path, "rb") as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    model_hash = hasher.hexdigest()
    expected_hash = "9608672c5d5e38a9272c560679cdd2291399de0e0917c8c9dff373bfd200f51d"
    assert model_hash == expected_hash, f"Model SHA-256 mismatch! Got {model_hash}, expected {expected_hash}"

    model = joblib.load(rf_path)
    assert len(model.estimators_) == 200, f"Expected 200 trees, got {len(model.estimators_)}"
    assert model.n_features_in_ == 78, f"Expected 78 input features, got {model.n_features_in_}"
    assert list(model.classes_) == [0, 1], f"Expected classes [0, 1], got {model.classes_}"
    print(f"   [PASS] Model SHA-256: {model_hash}")
    print(f"   [PASS] 200 trees, 78 features, classes {list(model.classes_)}\n")


def test_2_feature_mask_and_bounds():
    """Verify feature mask (63 modifiable, 15 protected) and 78 training bounds."""
    print(">> [Check 2] Verifying Feature Mask and Training Bounds...")
    mask_path = REPO_ROOT / "runtime_package" / "model" / "feature_mask.json"
    with open(mask_path, "r") as f:
        mask_dict = json.load(f)
    mask = list(mask_dict.values())[0]
    assert len(mask) == 78, f"Expected 78 mask elements, got {len(mask)}"
    modifiable_count = sum(1 for m in mask if m is True)
    protected_count = sum(1 for m in mask if m is False)
    assert modifiable_count == 63, f"Expected 63 modifiable features, got {modifiable_count}"
    assert protected_count == 15, f"Expected 15 protected features, got {protected_count}"

    bounds_path = REPO_ROOT / "runtime_package" / "model" / "training_bounds.parquet"
    bounds_df = pd.read_parquet(bounds_path)
    assert len(bounds_df) == 78, f"Expected 78 bounds rows, got {len(bounds_df)}"
    assert "train_min" in bounds_df.columns and "train_max" in bounds_df.columns
    print(f"   [PASS] Feature mask: {modifiable_count} modifiable, {protected_count} protected (total {len(mask)})")
    print(f"   [PASS] Training bounds: {len(bounds_df)} features verified.\n")


def test_3_offline_cli_multi_batch():
    """Verify runtime_package/run.py crosses batch boundary without pending-decision error."""
    print(">> [Check 3] Verifying Standalone Offline CLI (run.py)...")
    python_exe = sys.executable
    cmd = [
        python_exe,
        str(REPO_ROOT / "runtime_package" / "run.py"),
        "--attack", "silent_probing",
        "--defense", "afp",
        "--controller", "recall-aware"
    ]
    res = subprocess.run(cmd, capture_output=True, text=True, cwd=str(REPO_ROOT))
    assert res.returncode == 0, f"run.py failed with returncode {res.returncode}:\n{res.stderr}"
    assert "Demo completed." in res.stdout, "run.py did not complete successfully"
    assert "Final Perturbation Intensity:" in res.stdout, "run.py did not print perturbation intensity"
    assert "RuntimeError" not in res.stderr and "RuntimeError" not in res.stdout
    print("   [PASS] Offline CLI run.py executed across batch boundary successfully.\n")


def test_4_target_oracle_error_handling():
    """Verify TargetOracle strictly rejects non-200/403 responses and propagates errors."""
    print(">> [Check 4] Verifying TargetOracle Error Handling...")
    from attacker_sim import TargetOracle

    oracle = TargetOracle(target_url=f"{BASE_URL}/api/server/data", max_queries_per_sample=50)

    # 1. Invalid input should trigger 400 and raise RuntimeError in oracle
    invalid_vector = np.array([float('nan')] * 78, dtype=np.float32)
    error_raised = False
    try:
        oracle.predict(invalid_vector, sample_ids=[0])
    except RuntimeError as e:
        error_raised = True
        assert "Target oracle failure" in str(e)
    assert error_raised, "TargetOracle failed to raise RuntimeError on HTTP 400"
    assert oracle.get_query_count(0) == 0, "Oracle must NOT increment query count on error"

    # 2. Wrong length vector should trigger 400 and raise RuntimeError
    wrong_len_vector = np.zeros(77, dtype=np.float32)
    error_raised = False
    try:
        oracle.predict(wrong_len_vector, sample_ids=[0])
    except RuntimeError:
        error_raised = True
    assert error_raised, "TargetOracle failed to raise RuntimeError on wrong-length vector"
    assert oracle.get_query_count(0) == 0

    print("   [PASS] TargetOracle correctly propagates errors and does NOT count failed queries.\n")


def test_5_cold_start_and_reset_truthfulness():
    """Verify dashboard stats show honest empty states and no fabricated data."""
    print(">> [Check 5] Verifying Cold Start & Reset Truthfulness...")
    # Reset state
    res_reset = requests.post(f"{BASE_URL}/api/dashboard/reset")
    assert res_reset.status_code == 200

    stats_res = requests.get(f"{BASE_URL}/api/dashboard/stats")
    assert stats_res.status_code == 200
    data = stats_res.json()
    stats = data.get("stats", {})

    assert str(stats.get("total_traffic")) == "0", f"Expected 0 total traffic, got {stats.get('total_traffic')}"
    assert str(stats.get("detected_attacks")) == "0", f"Expected 0 detected attacks, got {stats.get('detected_attacks')}"
    # Recall and FPR must be None / '—' (zero denominator / unavailable), not 0.962 or 0.05!
    assert stats.get("recall_numeric") is None or stats.get("recall") is None, f"Expected None for recall, got {stats}"
    assert stats.get("fpr_numeric") is None or stats.get("fpr") is None, f"Expected None for fpr, got {stats}"
    assert stats.get("detection_recall") == "—"
    assert stats.get("false_positive_rate") == "—"
    assert len(data.get("threat_locations", [])) == 0, "Threat locations must be empty on reset"
    assert len(data.get("recent_feed", [])) == 0, "Recent feed must be empty on reset"
    print("   [PASS] Total traffic: 0 | Detected attacks: 0 | Recall: None | FPR: None")
    print("   [PASS] Threat locations: [] | Recent feed: []\n")


def test_6_input_validation_and_inference_independence():
    """Verify strict validation and that feature_vector takes precedence over sample_id."""
    print(">> [Check 6] Verifying Input Validation & Inference Independence...")

    # 1. Missing both vector and sample_id -> 400 Bad Request
    res = requests.post(f"{BASE_URL}/api/server/data", json={"source_ip": "1.2.3.4"})
    assert res.status_code == 400, f"Expected 400, got {res.status_code}"

    # 2. Vector with wrong length (77 items) -> 400 Bad Request
    res = requests.post(f"{BASE_URL}/api/server/data", json={"source_ip": "1.2.3.4", "feature_vector": [0.0] * 77})
    assert res.status_code == 400, f"Expected 400, got {res.status_code}"

    # 3. Vector with nonfinite value -> 400 Bad Request
    raw_bad_json = '{"source_ip": "1.2.3.4", "feature_vector": [' + ', '.join(['NaN'] + ['0.0'] * 77) + ']}'
    res = requests.post(f"{BASE_URL}/api/server/data", data=raw_bad_json, headers={"Content-Type": "application/json"})
    assert res.status_code == 400, f"Expected 400, got {res.status_code}"

    # 4. Out of range sample_id -> 400 Bad Request
    res = requests.post(f"{BASE_URL}/api/server/data", json={"source_ip": "1.2.3.4", "sample_id": 999999})
    assert res.status_code == 400, f"Expected 400, got {res.status_code}"

    # 5. Valid vector with sample_id -> vector is classified, sample_id used strictly for provenance
    # Load demo data
    X_demo = pd.read_parquet(REPO_ROOT / "runtime_package" / "demo_data" / "X_demo.parquet")
    meta_demo = pd.read_parquet(REPO_ROOT / "runtime_package" / "demo_data" / "metadata_demo.parquet")

    # Sample 0 is benign. We submit sample 10's attack vector while claiming sample_id 0.
    sample_10_vec = X_demo.iloc[10].values.tolist()
    res = requests.post(
        f"{BASE_URL}/api/server/data",
        json={"source_ip": "10.0.1.20", "sample_id": 0, "feature_vector": sample_10_vec}
    )
    assert res.status_code in (200, 403)
    details = res.json().get("details", {})
    # Classification must be Attack (from the feature_vector), NOT Benign (from sample_id 0)
    assert details.get("ids_classification") == "Attack", f"Inference must evaluate vector! Got {details.get('ids_classification')}"
    # Provenance provenance label shows original sample family
    assert "Benign" in details.get("traffic_family"), "Metadata provenance should reflect sample_id"
    print("   [PASS] Malformed/missing inputs return 400. Valid feature_vector takes precedence for classification.\n")


def test_7_used_vs_next_intensity_tracking():
    """Verify flow response records used_intensity and exposes next_intensity on 5th decision boundary."""
    print(">> [Check 7] Verifying Used vs Next Intensity & 5-Decision Transition...")
    requests.post(f"{BASE_URL}/api/dashboard/reset")
    requests.post(f"{BASE_URL}/api/dashboard/set-defense", json={"defense": "afp"})
    requests.post(f"{BASE_URL}/api/dashboard/set-mode", json={"mode": "recall-aware"})

    X_demo = pd.read_parquet(REPO_ROOT / "runtime_package" / "demo_data" / "X_demo.parquet")

    # Submit 4 labeled attack decisions (flows 1-4)
    for i in range(4):
        vec = X_demo.iloc[10 + i].values.tolist()
        res = requests.post(
            f"{BASE_URL}/api/server/data",
            json={"source_ip": f"10.0.1.{i+1}", "sample_id": 10 + i, "is_attack": True, "feature_vector": vec}
        )
        assert res.status_code in (200, 403)
        details = res.json().get("details", {})
        assert details.get("used_intensity") == 0.0003, f"Flow {i+1} used_intensity expected 0.0003, got {details.get('used_intensity')}"
        assert details.get("next_intensity") == 0.0003, f"Flow {i+1} next_intensity expected 0.0003, got {details.get('next_intensity')}"

    # Flow 5: The 5th labeled attack decision triggers controller evaluation
    vec_5 = X_demo.iloc[14].values.tolist()
    res_5 = requests.post(
        f"{BASE_URL}/api/server/data",
        json={"source_ip": "10.0.1.5", "sample_id": 14, "is_attack": True, "feature_vector": vec_5}
    )
    details_5 = res_5.json().get("details", {})
    # Flow 5 itself must report the intensity actually USED during inference (0.0003)
    assert details_5.get("used_intensity") == 0.0003, f"Flow 5 used_intensity must be 0.0003, got {details_5.get('used_intensity')}"
    # The next_intensity should be updated for subsequent flows
    next_int = details_5.get("next_intensity")
    assert next_int is not None, "next_intensity must be provided"
    print(f"   [PASS] 5th Decision: used_intensity={details_5.get('used_intensity')}, next_intensity={next_int}\n")


def test_8_scenario_seed_independence():
    """Verify caller req.attack_scenario does not alter defense perturbation RNG."""
    print(">> [Check 8] Verifying Scenario Seed Independence...")
    X_demo = pd.read_parquet(REPO_ROOT / "runtime_package" / "demo_data" / "X_demo.parquet")
    test_vec = X_demo.iloc[10].values.tolist()

    # Reset and configure static Base AFP mode
    requests.post(f"{BASE_URL}/api/dashboard/set-defense", json={"defense": "afp"})
    requests.post(f"{BASE_URL}/api/dashboard/set-mode", json={"mode": "base"})

    res_a = requests.post(
        f"{BASE_URL}/api/server/data",
        json={"source_ip": "10.0.9.1", "sample_id": 10, "attack_scenario": "Scenario_ALPHA", "feature_vector": test_vec}
    )
    details_a = res_a.json().get("details", {})

    res_b = requests.post(
        f"{BASE_URL}/api/server/data",
        json={"source_ip": "10.0.9.2", "sample_id": 10, "attack_scenario": "Scenario_BETA", "feature_vector": test_vec}
    )
    details_b = res_b.json().get("details", {})

    assert details_a.get("ids_classification") == details_b.get("ids_classification"), "Verdict altered by scenario string!"
    assert details_a.get("confidence") == details_b.get("confidence"), "Confidence altered by scenario string!"
    print("   [PASS] Attack scenario string does not manipulate defense RNG or verdict.\n")


def test_9_provenance_and_geolocation_truthfulness():
    """Verify strict IP location (Private Network vs Unknown) and unknown family for synthetic vectors."""
    print(">> [Check 9] Verifying Provenance and Geolocation Truthfulness...")
    # Public IP with untrusted country claim
    requests.post(
        f"{BASE_URL}/api/server/data",
        json={"source_ip": "203.0.113.88", "sample_id": 0, "country": "Germany"}
    )
    feed = requests.get(f"{BASE_URL}/api/dashboard/stats").json().get("recent_feed", [])
    entry_pub = next((item for item in feed if item["source_ip"] == "203.0.113.88"), None)
    assert entry_pub is not None
    assert entry_pub.get("location") == "Unknown", f"Expected Unknown for public IP, got {entry_pub.get('location')}"

    # Private IP
    requests.post(
        f"{BASE_URL}/api/server/data",
        json={"source_ip": "192.168.1.55", "sample_id": 0}
    )
    feed = requests.get(f"{BASE_URL}/api/dashboard/stats").json().get("recent_feed", [])
    entry_priv = next((item for item in feed if item["source_ip"] == "192.168.1.55"), None)
    assert entry_priv is not None
    assert entry_priv.get("location") == "Private Network"

    # Synthetic vector without sample_id
    requests.post(
        f"{BASE_URL}/api/server/data",
        json={"source_ip": "10.0.5.1", "feature_vector": [0.0] * 78}
    )
    feed = requests.get(f"{BASE_URL}/api/dashboard/stats").json().get("recent_feed", [])
    entry_synth = next((item for item in feed if item["source_ip"] == "10.0.5.1"), None)
    assert entry_synth is not None
    assert entry_synth.get("traffic_family") == "Unknown"
    assert entry_synth.get("traffic_family_source") in ("Synthetic / Non-dataset", "Unbound / Synthetic")
    print("   [PASS] Public IP location: Unknown | Private IP: Private Network")
    print("   [PASS] Synthetic flow provenance: Unknown (Synthetic / Non-dataset)\n")


def test_10_attack_classes_smoke():
    """Verify SilentProbingAttack and DecisionBoundaryAttack execution."""
    print(">> [Check 10] Verifying Attack Classes Smoke Execution...")
    from attacks.silent_probing import SilentProbingAttack
    from attacks.boundary_attack import DecisionBoundaryAttack
    from attacker_sim import TargetOracle

    with open(REPO_ROOT / "runtime_package" / "model" / "feature_names.json") as f:
        f_names = json.load(f)
    with open(REPO_ROOT / "runtime_package" / "model" / "feature_mask.json") as f:
        mod_mask = np.array(list(json.load(f).values())[0], dtype=bool)
    bounds_df = pd.read_parquet(REPO_ROOT / "runtime_package" / "model" / "training_bounds.parquet")
    X_demo = pd.read_parquet(REPO_ROOT / "runtime_package" / "demo_data" / "X_demo.parquet")
    meta_demo = pd.read_parquet(REPO_ROOT / "runtime_package" / "demo_data" / "metadata_demo.parquet")

    # 1. Silent Probing Attack
    silent = SilentProbingAttack(f_names, mod_mask, bounds_df)
    x_sample = X_demo.iloc[10].values
    res_silent = silent.generate(x_sample, oracle=None, sample_id=10, true_label=1)
    assert res_silent.query_count == 0
    assert np.array_equal(res_silent.X_adv, x_sample)
    print("   [PASS] SilentProbingAttack: 0 queries, exact unperturbed copy verified.")

    # 2. Decision Boundary Attack against live server
    boundary = DecisionBoundaryAttack(f_names, mod_mask, bounds_df, max_queries=50, binary_search_steps=10)
    oracle = TargetOracle(target_url=f"{BASE_URL}/api/server/data", max_queries_per_sample=50, scenario_name="Decision-Boundary Attack")
    ref_pool = X_demo[meta_demo["y_binary"] == 0].values

    res_boundary = boundary.generate(x_sample, oracle, sample_id=10, true_label=1, reference_pool=ref_pool)
    assert oracle.get_query_count(10) <= 50, f"Query count exceeded budget: {oracle.get_query_count(10)}"
    print(f"   [PASS] DecisionBoundaryAttack completed: status={res_boundary.status_code}, queries_used={oracle.get_query_count(10)}/50\n")


def run_all_checks():
    print("=================================================================")
    print("  STARTING DEFENSE READINESS VERIFICATION SUITE")
    print("=================================================================\n")
    test_1_model_integrity_and_specs()
    test_2_feature_mask_and_bounds()
    test_3_offline_cli_multi_batch()
    test_4_target_oracle_error_handling()
    test_5_cold_start_and_reset_truthfulness()
    test_6_input_validation_and_inference_independence()
    test_7_used_vs_next_intensity_tracking()
    test_8_scenario_seed_independence()
    test_9_provenance_and_geolocation_truthfulness()
    test_10_attack_classes_smoke()
    print("=================================================================")
    print("  ALL 10 DEFENSE READINESS CHECKS PASSED SUCCESSFULLY!")
    print("=================================================================")


if __name__ == "__main__":
    run_all_checks()
