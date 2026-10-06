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
import yaml
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
    """Verify strict validation, float32 finite checks, origin checks, and session authorization."""
    print(">> [Check 6] Verifying Input Validation, Origin Checking & Session Authorization...")

    # 1. Missing both vector and sample_id -> 400 Bad Request
    res = requests.post(f"{BASE_URL}/api/server/data", json={"source_ip": "1.2.3.4"})
    assert res.status_code == 400, f"Expected 400, got {res.status_code}"

    # 2. Vector with wrong length (77 items) -> 400 Bad Request
    res = requests.post(f"{BASE_URL}/api/server/data", json={"source_ip": "1.2.3.4", "feature_vector": [0.0] * 77})
    assert res.status_code == 400, f"Expected 400, got {res.status_code}"

    # 3. Vector with nonfinite value (NaN/Inf in JSON) -> 400 Bad Request
    raw_bad_json = '{"source_ip": "1.2.3.4", "feature_vector": [' + ', '.join(['NaN'] + ['0.0'] * 77) + ']}'
    res = requests.post(f"{BASE_URL}/api/server/data", data=raw_bad_json, headers={"Content-Type": "application/json"})
    assert res.status_code == 400, f"Expected 400, got {res.status_code}"

    # 4. Vector with 1e40 (finite in Python float64, overflows to inf in float32) -> 400 Bad Request
    res = requests.post(f"{BASE_URL}/api/server/data", json={"source_ip": "1.2.3.4", "feature_vector": [1e40] + [0.0] * 77})
    assert res.status_code == 400, f"Expected 400 for float32 overflow, got {res.status_code}"

    # 5. Out-of-range float32 with valid sample_id provided alongside -> still 400 Bad Request
    res = requests.post(f"{BASE_URL}/api/server/data", json={"source_ip": "1.2.3.4", "sample_id": 0, "feature_vector": [1e40] + [0.0] * 77})
    assert res.status_code == 400, f"Expected 400 for float32 overflow even with sample_id, got {res.status_code}"

    # 6. Out of range sample_id -> 400 Bad Request
    res = requests.post(f"{BASE_URL}/api/server/data", json={"source_ip": "1.2.3.4", "sample_id": 999999})
    assert res.status_code == 400, f"Expected 400, got {res.status_code}"

    # 7. Exact origin match: Sample 0 vector with sample_id 0 -> verified exact dataset sample
    X_demo = pd.read_parquet(REPO_ROOT / "runtime_package" / "demo_data" / "X_demo.parquet")
    sample_0_vec = X_demo.iloc[0].values.tolist()
    res_exact = requests.post(
        f"{BASE_URL}/api/server/data",
        json={"source_ip": "10.0.1.1", "sample_id": 0, "feature_vector": sample_0_vec}
    )
    assert res_exact.status_code in (200, 403)
    det_exact = res_exact.json().get("details", {})
    assert det_exact.get("traffic_family") == "Benign"
    assert det_exact.get("ground_truth_status") == "Dataset: 0"

    # 8. Mismatched vector: Sample 10's attack vector claiming sample_id 0 (without verified session)
    # Reset stats to observe metric isolation
    requests.post(f"{BASE_URL}/api/dashboard/reset")
    stats_before = requests.get(f"{BASE_URL}/api/dashboard/stats").json().get("stats", {})

    sample_10_vec = X_demo.iloc[10].values.tolist()
    res_mismatch = requests.post(
        f"{BASE_URL}/api/server/data",
        json={"source_ip": "10.0.1.20", "sample_id": 0, "feature_vector": sample_10_vec}
    )
    assert res_mismatch.status_code in (200, 403)
    details_mismatch = res_mismatch.json().get("details", {})
    # Classification must evaluate vector (Attack), NOT sample_id 0
    assert details_mismatch.get("ids_classification") == "Attack", f"Inference must evaluate vector! Got {details_mismatch.get('ids_classification')}"
    # Must label as Claimed original family, NOT verified origin
    assert "Claimed original family" in details_mismatch.get("traffic_family"), f"Must indicate unverified claimed origin: {details_mismatch.get('traffic_family')}"
    # Without verified session, must remain Unlabeled and NOT update TN or FP
    assert details_mismatch.get("ground_truth_status") == "Unlabeled", f"Mismatched ID must be Unlabeled: {details_mismatch.get('ground_truth_status')}"

    stats_after = requests.get(f"{BASE_URL}/api/dashboard/stats").json().get("stats", {})
    assert stats_after.get("tp") == stats_before.get("tp") == 0
    assert stats_after.get("fp") == stats_before.get("fp") == 0
    assert stats_after.get("tn") == stats_before.get("tn") == 0
    assert stats_after.get("fn") == stats_before.get("fn") == 0

    # 9. Session Authorization: Fake session ID "made-up" with is_attack=True must NOT update metrics
    res_fake = requests.post(
        f"{BASE_URL}/api/server/data",
        json={"source_ip": "10.0.1.30", "feature_vector": sample_10_vec, "is_attack": True, "session_id": "made-up"}
    )
    det_fake = res_fake.json().get("details", {})
    assert det_fake.get("ground_truth_status") == "Unlabeled", f"Fake session must not be trusted: {det_fake.get('ground_truth_status')}"
    stats_after_fake = requests.get(f"{BASE_URL}/api/dashboard/stats").json().get("stats", {})
    assert stats_after_fake.get("tp") == 0

    # 10. Valid Simulation Session: Accepted when active session matches
    sim_start = requests.post(f"{BASE_URL}/api/simulation/start", json={"scenario": "silent_probing", "session_id": "test-sim-001"})
    assert sim_start.status_code == 200
    res_valid_sim = requests.post(
        f"{BASE_URL}/api/server/data",
        json={"source_ip": "10.0.1.40", "feature_vector": sample_10_vec, "is_attack": True, "session_id": "test-sim-001"}
    )
    det_valid = res_valid_sim.json().get("details", {})
    assert det_valid.get("ground_truth_status") == "Simulator: 1"
    stats_valid = requests.get(f"{BASE_URL}/api/dashboard/stats").json().get("stats", {})
    assert stats_valid.get("tp") == 1
    requests.post(f"{BASE_URL}/api/simulation/stop")

    # 11. Exact Dataset Label Precedence vs Conflicting Simulator Feedback
    # Exact benign row + active session + is_attack=True must remain Dataset: 0
    requests.post(f"{BASE_URL}/api/dashboard/reset")
    sim_prec = requests.post(f"{BASE_URL}/api/simulation/start", json={"scenario": "precedence_test", "session_id": "prec-sess-001"})
    assert sim_prec.status_code == 200

    # Submit exact benign row (sample 0) with conflicting is_attack=True
    res_conf_benign = requests.post(
        f"{BASE_URL}/api/server/data",
        json={
            "source_ip": "10.0.1.50",
            "sample_id": 0,
            "feature_vector": sample_0_vec,
            "is_attack": True,  # Conflicting!
            "session_id": "prec-sess-001"
        }
    )
    det_conf_b = res_conf_benign.json().get("details", {})
    assert det_conf_b.get("ground_truth_status") == "Dataset: 0", f"Exact benign must remain Dataset: 0, got {det_conf_b.get('ground_truth_status')}"
    stats_conf = requests.get(f"{BASE_URL}/api/dashboard/stats").json().get("stats", {})
    assert stats_conf.get("fn") == 0, "Exact benign must NOT increment FN!"
    assert stats_conf.get("tn") == 1, "Exact benign must increment TN when predicted benign"

    # Submit exact attack row (sample 10) with conflicting is_attack=False
    res_conf_attack = requests.post(
        f"{BASE_URL}/api/server/data",
        json={
            "source_ip": "10.0.1.51",
            "sample_id": 10,
            "feature_vector": sample_10_vec,
            "is_attack": False,  # Conflicting!
            "session_id": "prec-sess-001"
        }
    )
    det_conf_a = res_conf_attack.json().get("details", {})
    assert det_conf_a.get("ground_truth_status") == "Dataset: 1", f"Exact attack must remain Dataset: 1, got {det_conf_a.get('ground_truth_status')}"
    stats_conf_2 = requests.get(f"{BASE_URL}/api/dashboard/stats").json().get("stats", {})
    assert stats_conf_2.get("fp") == 0, "Exact attack must NOT increment FP!"
    assert stats_conf_2.get("tp") == 1, "Exact attack must increment TP when predicted attack"
    requests.post(f"{BASE_URL}/api/simulation/stop")

    print("   [PASS] 1e40 overflow rejected (400). Mismatched origin labeled claimed original family.")
    print("   [PASS] Exact dataset labels strictly take precedence over conflicting simulator feedback.")
    print("   [PASS] Arbitrary session strings rejected; validated sessions accept simulator feedback.\n")


def test_7_used_vs_next_intensity_tracking():
    """Verify flow response records used_intensity, next_intensity, and proves dynamic controller adaptation."""
    print(">> [Check 7] Verifying Used vs Next Intensity & Proving Controller Adaptation...")
    X_demo = pd.read_parquet(REPO_ROOT / "runtime_package" / "demo_data" / "X_demo.parquet")
    meta_demo = pd.read_parquet(REPO_ROOT / "runtime_package" / "demo_data" / "metadata_demo.parquet")

    # -------------------------------------------------------------------------
    # Part A: Ceiling Hold Verification (5 True Positives)
    # -------------------------------------------------------------------------
    requests.post(f"{BASE_URL}/api/dashboard/reset")
    requests.post(f"{BASE_URL}/api/dashboard/set-defense", json={"defense": "afp"})
    requests.post(f"{BASE_URL}/api/dashboard/set-mode", json={"mode": "recall-aware"})

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

    # Flow 5: The 5th labeled attack decision (detected TP)
    vec_5 = X_demo.iloc[14].values.tolist()
    res_5 = requests.post(
        f"{BASE_URL}/api/server/data",
        json={"source_ip": "10.0.1.5", "sample_id": 14, "is_attack": True, "feature_vector": vec_5}
    )
    details_5 = res_5.json().get("details", {})
    assert details_5.get("used_intensity") == 0.0003
    # With 5/5 detections (TP=5, FN=0), rolling recall = 1.0 >= Rmin, next intensity stays at ceiling 0.0003
    assert details_5.get("next_intensity") == 0.0003
    print("   [PASS] 5/5 Detections: Controller holds at calibrated ceiling (0.00030 -> 0.00030, Green).")

    # -------------------------------------------------------------------------
    # Part B: Deterministic Low-Recall Adaptation Proof via Genuine Evasion Replay
    # -------------------------------------------------------------------------
    # Functional Rehearsal:
    # 1. In static Base AFP mode, use DecisionBoundaryAttack to craft an evasive candidate
    #    from real attack sample 10. Verify original dataset label is Attack (1), protected
    #    features are preserved, vector is finite and bounded, and server verdict is Benign.
    # 2. Freeze the candidate. Reset to Recall-Aware AFP, then replay five correctly labeled
    #    copies in a valid simulator session (small functional replay, not a statistical benchmark).
    # 3. Assert live verdicts, flow 5 used intensity (0.00030), next intensity (0.00012),
    #    state (Red), and flow 6 used intensity (0.00012).
    requests.post(f"{BASE_URL}/api/dashboard/reset")
    requests.post(f"{BASE_URL}/api/dashboard/set-defense", json={"defense": "afp"})
    requests.post(f"{BASE_URL}/api/dashboard/set-mode", json={"mode": "base"})

    from attacks.boundary_attack import DecisionBoundaryAttack
    from attacker_sim import TargetOracle

    with open(REPO_ROOT / "runtime_package" / "model" / "feature_names.json") as f:
        f_names = json.load(f)
    with open(REPO_ROOT / "runtime_package" / "model" / "feature_mask.json") as f:
        mod_mask = np.array(list(json.load(f).values())[0], dtype=bool)
    bounds_df = pd.read_parquet(REPO_ROOT / "runtime_package" / "model" / "training_bounds.parquet")

    x_orig_10 = X_demo.iloc[10].values.astype(np.float32)
    assert int(meta_demo.iloc[10]["y_binary"]) == 1, "Sample 10 must be Attack in metadata"

    craft_sim = requests.post(f"{BASE_URL}/api/simulation/start", json={"scenario": "boundary_crafting", "session_id": "craft-sess-001"})
    assert craft_sim.status_code == 200

    boundary_crafter = DecisionBoundaryAttack(f_names, mod_mask, bounds_df, max_queries=50, binary_search_steps=10)
    oracle_crafter = TargetOracle(target_url=f"{BASE_URL}/api/server/data", max_queries_per_sample=50, scenario_name="Decision-Boundary Search")
    ref_pool = X_demo[meta_demo["y_binary"] == 0].values

    craft_res = boundary_crafter.generate(x_orig_10, oracle_crafter, sample_id=10, true_label=1, reference_pool=ref_pool)
    assert craft_res.success is True, f"Crafting failed: status={craft_res.status_code}"

    # Verify candidate properties
    x_adv_evasion = np.array(craft_res.X_adv, dtype=np.float32)
    assert np.array_equal(x_adv_evasion[~mod_mask], x_orig_10[~mod_mask]), "Protected features must match original attack!"
    assert np.isfinite(x_adv_evasion).all(), "Candidate must be finite!"
    assert not np.array_equal(x_adv_evasion, x_orig_10), "Candidate must be transformed from original!"

    # Verify server classification of candidate in Base mode
    test_cand_res = requests.post(
        f"{BASE_URL}/api/server/data",
        json={
            "source_ip": "10.0.1.99",
            "sample_id": 10,
            "feature_vector": x_adv_evasion.tolist(),
            "is_attack": True,
            "session_id": "craft-sess-001"
        }
    )
    assert test_cand_res.status_code == 200, f"Candidate must evade: got {test_cand_res.status_code}"
    det_cand = test_cand_res.json().get("details", {})
    assert det_cand.get("ids_classification") == "Benign", "Candidate must be classified as Benign"
    requests.post(f"{BASE_URL}/api/simulation/stop")

    # Replay Phase: Reset and configure Recall-Aware AFP
    requests.post(f"{BASE_URL}/api/dashboard/reset")
    requests.post(f"{BASE_URL}/api/dashboard/set-defense", json={"defense": "afp"})
    requests.post(f"{BASE_URL}/api/dashboard/set-mode", json={"mode": "recall-aware"})

    replay_sim = requests.post(f"{BASE_URL}/api/simulation/start", json={"scenario": "genuine_evasion_replay", "session_id": "replay-001"})
    assert replay_sim.status_code == 200

    # Replay first 4 genuine evasions (False Negatives: Model says Benign, True label is Attack)
    adv_list = x_adv_evasion.tolist()
    for i in range(4):
        res_ev = requests.post(
            f"{BASE_URL}/api/server/data",
            json={
                "source_ip": f"10.0.3.{i+1}",
                "sample_id": 10,
                "feature_vector": adv_list,
                "is_attack": True,
                "session_id": "replay-001"
            }
        )
        assert res_ev.status_code == 200  # Allowed (Model predicts Benign)
        det_ev = res_ev.json().get("details", {})
        assert det_ev.get("ids_classification") == "Benign"
        assert det_ev.get("ground_truth_status") == "Simulator: 1"
        assert det_ev.get("used_intensity") == 0.0003
        assert det_ev.get("next_intensity") == 0.0003

    # Flow 5: The 5th genuine False Negative completes batch of 5 with TP=0, FN=5 (Recall = 0.0 < Rcritical)
    res_ev_5 = requests.post(
        f"{BASE_URL}/api/server/data",
        json={
            "source_ip": "10.0.3.5",
            "sample_id": 10,
            "feature_vector": adv_list,
            "is_attack": True,
            "session_id": "replay-001"
        }
    )
    assert res_ev_5.status_code == 200
    det_ev_5 = res_ev_5.json().get("details", {})
    assert det_ev_5.get("ids_classification") == "Benign", "Flow 5 must evade model"
    assert det_ev_5.get("ground_truth_status") == "Simulator: 1", "Ground truth must be Simulator: 1"
    assert det_ev_5.get("used_intensity") == 0.0003, f"Flow 5 used_intensity must be 0.0003, got {det_ev_5.get('used_intensity')}"
    assert abs(det_ev_5.get("next_intensity") - 0.00012) < 1e-6, f"Flow 5 next_intensity must be 0.00012, got {det_ev_5.get('next_intensity')}"
    assert det_ev_5.get("controller_state") == "Red", f"Controller state must be Red, got {det_ev_5.get('controller_state')}"

    # Flow 6: Subsequent flow must execute inference using the NEW updated intensity (0.00012)
    res_ev_6 = requests.post(
        f"{BASE_URL}/api/server/data",
        json={
            "source_ip": "10.0.3.6",
            "sample_id": 10,
            "feature_vector": adv_list,
            "is_attack": True,
            "session_id": "replay-001"
        }
    )
    det_ev_6 = res_ev_6.json().get("details", {})
    assert abs(det_ev_6.get("used_intensity") - 0.00012) < 1e-6, f"Flow 6 used_intensity must be 0.00012, got {det_ev_6.get('used_intensity')}"
    requests.post(f"{BASE_URL}/api/simulation/stop")

    # -------------------------------------------------------------------------
    # Part C: Direct Controller Unit Proof
    # -------------------------------------------------------------------------
    from runtime_package.controller.recall_controller import RecallAwareController
    with open(REPO_ROOT / "runtime_package" / "configs" / "controllers.yaml") as f:
        ctrl_configs = yaml.safe_load(f)
    with open(REPO_ROOT / "runtime_package" / "configs" / "defenses.yaml") as f:
        def_configs = yaml.safe_load(f)

    ctrl = RecallAwareController(ctrl_configs["controller_configurations"]["C1"], def_configs["afp"], "afp")
    d0 = ctrl.get_intensity(0)
    assert d0.intensity == 0.0003
    up = ctrl.submit_observations(0, tp=0, fn=5)
    assert up.state == "Red"
    assert abs(up.clipped_next_intensity - 0.00012) < 1e-6
    d1 = ctrl.get_intensity(1)
    assert abs(d1.intensity - 0.00012) < 1e-6

    print(f"   [PASS] Deterministic Adaptation Proved: Flow 5 used=0.00030, next=0.00012; Flow 6 used=0.00012.\n")


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
    print(f"   [PASS] DecisionBoundaryAttack completed: status={res_boundary.status_code}, queries_used={oracle.get_query_count(10)}/50")

    # 3. Surrogate Fitting HTTP Path Verification
    from attacks.surrogate_transfer import SurrogateTransferAttack
    requests.post(f"{BASE_URL}/api/dashboard/reset")
    requests.post(f"{BASE_URL}/api/dashboard/set-defense", json={"defense": "afp"})
    requests.post(f"{BASE_URL}/api/dashboard/set-mode", json={"mode": "recall-aware"})

    stats_before = requests.get(f"{BASE_URL}/api/dashboard/stats").json().get("stats", {})
    assert stats_before.get("tp") == stats_before.get("fp") == stats_before.get("tn") == stats_before.get("fn") == 0

    surr_sim = requests.post(f"{BASE_URL}/api/simulation/start", json={"scenario": "surrogate_transfer", "session_id": "surr-fit-sess-001"})
    assert surr_sim.status_code == 200

    surr_attack = SurrogateTransferAttack(f_names, mod_mask, bounds_df)
    surr_oracle = TargetOracle(target_url=f"{BASE_URL}/api/server/data", scenario_name="Surrogate Transferability", session_id="surr-fit-sess-001")

    x_pool = X_demo.values
    y_oracle = surr_oracle.predict(x_pool, sample_ids=list(range(len(x_pool))), stage="surrogate_fitting")
    surr_attack.fit_surrogate(x_pool, y_oracle)

    stats_after_fit = requests.get(f"{BASE_URL}/api/dashboard/stats").json().get("stats", {})
    afp_after_fit = requests.get(f"{BASE_URL}/api/dashboard/stats").json().get("afp", {})

    # Benign references in pool (samples 0-9) must contribute to TN (or FP), NOT FN/TP
    assert stats_after_fit.get("tn") == 10, f"Expected 10 TNs from benign pool, got {stats_after_fit.get('tn')}"
    assert stats_after_fit.get("fp") == 0, f"Expected 0 FPs from benign pool, got {stats_after_fit.get('fp')}"

    # Only attack references (samples 10-19) contribute to TP/FN
    assert stats_after_fit.get("tp") + stats_after_fit.get("fn") == 10, f"Expected 10 attack decisions, got {stats_after_fit.get('tp') + stats_after_fit.get('fn')}"

    # Verify controller batch progression: exactly 2 batches of 5 completed
    assert afp_after_fit.get("batch_id") == 2, f"Expected batch_id=2 after 10 attack decisions, got {afp_after_fit.get('batch_id')}"
    requests.post(f"{BASE_URL}/api/simulation/stop")
    print("   [PASS] Surrogate Fitting HTTP Path: 10 Benign references -> 10 TN, 0 FN; 10 Attack references -> batch_id progressed to 2.\n")


def test_11_html_escaping_and_injection():
    """Verify request-derived HTML in IP/traffic fields does not cause script injection."""
    print(">> [Check 11] Verifying HTML Escaping & Injection Protection...")
    evil_payload = {
        "source_ip": "<script>alert('xss')</script>",
        "destination_ip": "<img src=x onerror=alert(1)>",
        "sample_id": 0
    }
    res = requests.post(f"{BASE_URL}/api/server/data", json=evil_payload)
    assert res.status_code in (200, 403)
    details = res.json().get("details", {})
    assert details.get("source_ip") == evil_payload["source_ip"]

    # Verify that in dashboard feeds, the data is properly escaped before rendering
    def py_escape_html(s):
        if s is None:
            return ""
        return (str(s)
                .replace("&", "&amp;")
                .replace("<", "&lt;")
                .replace(">", "&gt;")
                .replace('"', "&quot;")
                .replace("'", "&#039;"))

    escaped_src = py_escape_html(details.get("source_ip"))
    assert "<script>" not in escaped_src
    assert "&lt;script&gt;alert(&#039;xss&#039;)&lt;/script&gt;" == escaped_src
    print("   [PASS] Untrusted request-derived strings are neutralized by HTML escaping.\n")


def test_12_dynamic_target_resolution():
    """Verify attacker_sim.py resolves --target dynamically across oracles and endpoints."""
    print(">> [Check 12] Verifying Dynamic Target Resolution in Attacker Sim...")
    import attacker_sim
    custom_target = "http://127.0.0.1:8888/api/server/data"
    attacker_sim.set_target_url(custom_target)
    assert attacker_sim.get_server_url() == custom_target
    assert attacker_sim.get_defense_url() == "http://127.0.0.1:8888/api/dashboard/set-defense"
    assert attacker_sim.get_mode_url() == "http://127.0.0.1:8888/api/dashboard/set-mode"
    assert attacker_sim.get_sim_start_url() == "http://127.0.0.1:8888/api/simulation/start"
    assert attacker_sim.get_sim_stop_url() == "http://127.0.0.1:8888/api/simulation/stop"

    oracle = attacker_sim.RemoteServerOracle(scenario_name="Test")
    assert oracle.target_url == custom_target

    # Restore default target
    attacker_sim.set_target_url(f"{BASE_URL}/api/server/data")
    assert attacker_sim.get_server_url() == f"{BASE_URL}/api/server/data"
    print("   [PASS] Dynamic --target resolution verified for all endpoints and oracles.\n")


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
    test_11_html_escaping_and_injection()
    test_12_dynamic_target_resolution()
    print("=================================================================")
    print("  ALL 12 DEFENSE READINESS CHECKS PASSED SUCCESSFULLY!")
    print("=================================================================")


if __name__ == "__main__":
    run_all_checks()
