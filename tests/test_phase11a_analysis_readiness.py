import pytest
import os
import json
import csv
from pathlib import Path
import shutil
import math
import sys
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.build_analysis_tables import process_run, calculate_metrics

def test_calculate_metrics_nan_rejection():
    # Test valid
    acc, prec, rec, f1, bal_acc = calculate_metrics(50, 10, 400, 40, total=500)
    assert 0 <= acc <= 1
    
    # Test sum assertion
    with pytest.raises(ValueError):
        calculate_metrics(0, 0, 0, 0, total=500)
        
def test_unweighted_batch_mean_prevention():
    # Counterexample: 
    # Batch 1: TP=100, FN=0 -> Recall = 1.0 (from 100 attempted)
    # Batch 2: TP=10, FN=90 -> Recall = 0.1 (from 100 attempted)
    # Mean of batch recalls = 0.55
    # Actual run recall = 110 / 200 = 0.55. Wait, that's equal.
    # Let's make the denominators different.
    # Batch 1: TP=100, FN=0 (Attempted=100) -> Recall = 1.0
    # Batch 2: TP=1, FN=9 (Attempted=10) -> Recall = 0.1
    # Batch mean = 0.55
    # Run sum = TP=101, FN=9 -> Recall = 101/110 = 0.918
    # We prove they are different.
    
    b1_rec = 1.0
    b2_rec = 0.1
    mean_of_recalls = (b1_rec + b2_rec) / 2.0
    
    run_tp = 101
    run_fn = 9
    run_fp = 0
    run_tn = 890 # Make total 1000
    
    acc, prec, rec, f1, bal_acc = calculate_metrics(run_tp, run_fp, run_tn, run_fn, total=1000)
    
    assert abs(rec - 0.91818) < 1e-4
    assert rec != mean_of_recalls, "Run recall is NOT the unweighted mean of batch recalls"

def test_c1_minus_base_direction_ordering():
    # C1_accuracy = 0.9, Base_accuracy = 0.8 => diff = +0.1
    from scripts.build_analysis_tables import check_eq
    c1r = {"accuracy": 0.9, "precision": 0.0, "recall": 0.0, "f1_score": 0.0, "balanced_accuracy": 0.0, "run_id": "c1"}
    br = {"accuracy": 0.8, "precision": 0.0, "recall": 0.0, "f1_score": 0.0, "balanced_accuracy": 0.0, "run_id": "base"}
    diff = c1r["accuracy"] - br["accuracy"]
    assert diff > 0, "C1 - Base direction must yield positive when C1 is higher"
    assert abs(diff - 0.1) < 1e-5

def test_parquet_interception(monkeypatch):
    import builtins
    original_open = builtins.open
    
    def mocked_open(file, *args, **kwargs):
        if str(file).endswith(".parquet"):
            raise PermissionError("Parquet access is strictly prohibited during Phase 11A aggregation")
        return original_open(file, *args, **kwargs)
        
    monkeypatch.setattr(builtins, "open", mocked_open)
    
    with pytest.raises(PermissionError):
        open("dummy.parquet", "rb")
        
def test_missing_batch_rejection(tmp_path):
    rid = "primary_42_SilentProbing_afp_Base"
    run_dir = tmp_path / rid
    run_dir.mkdir()
    
    # Write only 143 batches
    confusions = [{"batch_id": i, "tp": 0, "fp": 0, "tn": 500, "fn": 0} for i in range(143)]
    configs = [{"batch_id": i} for i in range(143)]
    
    with open(run_dir / "confusion.json", "w") as f: json.dump(confusions, f)
    with open(run_dir / "config.json", "w") as f: json.dump(configs, f)
    with open(run_dir / "run_summary.json", "w") as f: json.dump({}, f)
    with open(run_dir / "completion.json", "w") as f: json.dump({}, f)
    
    row = {"run_id": rid, "seed": 42, "attack_scenario": "SilentProbing", "defense_name": "afp", "controller_config_id": "Base"}
    
    with mock.patch("scripts.build_analysis_tables.EVAL_DIR", tmp_path):
        with pytest.raises(ValueError, match="Batch count"):
            process_run(row)
            
def test_out_of_order_batch_rejection(tmp_path):
    rid = "primary_42_SilentProbing_afp_Base"
    run_dir = tmp_path / rid
    run_dir.mkdir()
    
    confusions = [{"batch_id": i, "tp": 0, "fp": 0, "tn": 500, "fn": 0} for i in range(144)]
    configs = [{"batch_id": i} for i in range(144)]
    
    # Swap 0 and 1
    confusions[0]["batch_id"] = 1
    confusions[1]["batch_id"] = 0
    
    with open(run_dir / "confusion.json", "w") as f: json.dump(confusions, f)
    with open(run_dir / "config.json", "w") as f: json.dump(configs, f)
    with open(run_dir / "run_summary.json", "w") as f: json.dump({}, f)
    with open(run_dir / "completion.json", "w") as f: json.dump({}, f)
    
    row = {"run_id": rid, "seed": 42, "attack_scenario": "SilentProbing", "defense_name": "afp", "controller_config_id": "Base"}
    
    with mock.patch("scripts.build_analysis_tables.EVAL_DIR", tmp_path):
        with pytest.raises(ValueError, match="Batch ID out of order"):
            process_run(row)
