import pytest
import numpy as np
import pandas as pd
from pathlib import Path
from recall_aware_ids.experiment.policies import FixedIntensityPolicy
from recall_aware_ids.experiment.metrics import calculate_metrics
from recall_aware_ids.experiment.caching import validate_cache_manifest

def test_fixed_intensity_policy():
    policy = FixedIntensityPolicy(config_id="Base", fixed_intensity=0.0003)
    dec = policy.get_intensity(0)
    assert dec.batch_id == 0
    assert dec.intensity == 0.0003
    
    upd = policy.submit_observations(0, 10, 5)
    assert upd.state == "Base"
    assert upd.used_intensity == 0.0003
    
    # test reset
    policy.reset()
    dec2 = policy.get_intensity(1)
    assert dec2.intensity == 0.0003

def test_calculate_metrics():
    y_true = np.array([1, 1, 0, 0, 1])
    y_pred = np.array([1, 0, 0, 1, 1])
    positive_scores = np.array([0.9, 0.4, 0.2, 0.8, 0.95])
    
    # 2 TP, 1 FN, 1 TN, 1 FP
    eligible = np.array([True, True, False, False, True])
    attempted = np.array([True, False, False, False, True])
    successful = np.array([True, False, False, False, False])
    
    metrics = calculate_metrics(y_true, y_pred, positive_scores, eligible, attempted, successful)
    
    assert metrics["tp"] == 2
    assert metrics["fn"] == 1
    assert metrics["tn"] == 1
    assert metrics["fp"] == 1
    
    assert metrics["eligible_count"] == 3
    assert metrics["attempted_count"] == 2
    assert metrics["successful_count"] == 1
    assert metrics["asr"] == 0.5
    
    assert metrics["pr_auc_average_precision"] >= 0.0
    assert metrics["pr_auc_average_precision"] <= 1.0

def test_metrics_zero_division():
    # Test zero division explicitly
    metrics = calculate_metrics(np.array([1]), np.array([0]), np.array([0.1]))
    assert metrics["precision"] == 0.0
    assert metrics["pr_auc_average_precision"] == 0.0

def test_cache_validation(tmp_path):
    manifest_path = tmp_path / "manifest.json"
    import json
    
    valid_data = {
        "frozen_rf_hash": "abc",
        "attack_script_hashes": {"main.py": "def"}
    }
    
    with open(manifest_path, "w") as f:
        json.dump(valid_data, f)
        
    expected_good = {"frozen_rf_hash": "abc", "attack_script_hashes": {"main.py": "def"}}
    assert validate_cache_manifest(manifest_path, expected_good) is True
    
    expected_bad = {"frozen_rf_hash": "xyz"}
    with pytest.raises(ValueError, match="Cache invalid: Hash mismatch"):
        validate_cache_manifest(manifest_path, expected_bad)
        
    expected_bad_script = {"attack_script_hashes": {"main.py": "xyz"}}
    with pytest.raises(ValueError, match="Cache invalid: Hash mismatch"):
        validate_cache_manifest(manifest_path, expected_bad_script)
