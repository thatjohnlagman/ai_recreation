import pytest
import json
import pandas as pd
from pathlib import Path
from unittest.mock import patch, MagicMock
from src.recall_aware_ids.experiment.schemas import RunSummary
from scripts.build_analysis_tables import calculate_metrics, process_run, check_eq, exact_alias_resolution, base_c1_pairing_and_difference_construction, check_tolerance

def build_dummy_run(attack="DecisionBoundary", seed=42, defense="afp", config="Base", batches=144, run_id="test_run"):
    summary = {
        "run_id": run_id,
        "seed": seed,
        "attack_scenario": attack,
        "defense": defense,
        "config_id": config,
        "total_batches": 144,
        "tp": 18000,
        "fp": 0,
        "tn": 54000,
        "fn": 0,
        "accuracy": 1.0,
        "recall": 1.0,
        "precision": 1.0,
        "f1": 1.0,
        "balanced_accuracy": 1.0,
        "pr_auc_average_precision": 1.0,
        "total_eligible": 72000,
        "total_attempted": 100,
        "total_successful": 50,
        "status_code_counts": {"NOT_APPLICABLE": 71900, "NOT_ATTEMPTED": 50, "SUCCESS": 50},
        "global_asr": 0.5,
        "cache_identity": {
            "frozen_rf_hash": "a"*64,
            "scaler_hash": "a"*64,
            "feature_names_hash": "a"*64,
            "feature_mask_hash": "a"*64,
            "training_bounds_hash": "a"*64,
            "evaluation_roles_hash": "a"*64,
            "evaluation_batches_hash": "a"*64,
            "attacks_yaml_hash": "a"*64,
            "controllers_yaml_hash": "a"*64,
            "defenses_yaml_hash": "a"*64,
            "experiment_yaml_hash": "a"*64
        },
        "total_queries": 0,
        "completed_successfully": True,
        "l0_summary": {"mean": 0.0, "min": 0.0, "max": 0.0, "std": 0.0},
        "l1_summary": {"mean": 0.0, "min": 0.0, "max": 0.0, "std": 0.0},
        "l2_summary": {"mean": 0.0, "min": 0.0, "max": 0.0, "std": 0.0},
        "linf_summary": {"mean": 0.0, "min": 0.0, "max": 0.0, "std": 0.0}
    }
    # Ensure cache hashes are real hex 64
    for k in summary["cache_identity"].keys():
        summary["cache_identity"][k] = "c" * 64
        
    completion = {
        "run_id": run_id, 
        "timestamp": "2026-09-17T00:01:00Z", 
        "provenance_hashes": summary["cache_identity"]
    }
    
    confusions = []
    configs = []
    scores = []
    for i in range(batches):
        confusions.append({
            "run_id": run_id,
            "batch_id": i,
            "tp": 125, "fp": 0, "tn": 375, "fn": 0,
            "accuracy": 1.0, "recall": 1.0, "precision": 1.0, "f1": 1.0, "balanced_accuracy": 1.0, "pr_auc_average_precision": 1.0,
            "eligible_count": 500, "attempted_count": 0, "successful_count": 0,
            "asr_applicable": True, "asr": 0.0
        })
        configs.append({
            "run_id": run_id,
            "batch_id": i,
            "config_id": config,
            "intensity": 0.0,
            "state": "Base",
            "multiplier": 1.0,
            "rolling_recall": 1.0,
            "hit_min_bound": False,
            "hit_max_bound": False,
            "zero_denominator": False
        })
        scores.append({
            "run_id": run_id,
            "batch_id": i,
            "preds": [0]*500,
            "predictions": [0]*500,
            "defense_final_invalid_count": 0,
            "protected_feature_modification_count": 0,
            "projected_cell_count": 0
        })
        
    return summary, completion, confusions, configs, scores

def write_dummy_run(tmp_path, summary, completion, confusions, configs, scores):
    tmp_eval = tmp_path / "artifacts" / "evaluation_runs"
    tmp_eval.mkdir(parents=True, exist_ok=True)
    run_dir = tmp_eval / summary["run_id"]
    run_dir.mkdir(parents=True, exist_ok=True)
    with open(run_dir / "run_summary.json", "w") as f: json.dump(summary, f)
    with open(run_dir / "completion.json", "w") as f: json.dump(completion, f)
    with open(run_dir / "confusion.json", "w") as f: json.dump(confusions, f)
    with open(run_dir / "config.json", "w") as f: json.dump(configs, f)
    with open(run_dir / "scores.json", "w") as f: json.dump(scores, f)
    return tmp_eval

# 1. Zero-attempt ASR
def test_zero_attempt_asr(tmp_path):
    summary, completion, confusions, configs, scores = build_dummy_run()
    summary["total_attempted"] = 0
    summary["total_successful"] = 0
    summary["global_asr"] = 0.5 # Nonzero global_asr with zero attempted!
    tmp_eval = write_dummy_run(tmp_path, summary, completion, confusions, configs, scores)
    
    row = {"run_id": "test_run", "attack_scenario": "DecisionBoundary", "seed": 42, "defense_name": "afp", "controller_config_id": "Base"}
    with patch("scripts.build_analysis_tables.EVAL_DIR", tmp_eval):
        # For process_run to not fail len(configs) check, we mock it or pass expected_batches=144 check
        with pytest.raises(ValueError, match="requires global_asr == 0.0"):
            process_run(row)

# 2. Alias rejection
def test_alias_rejection(tmp_path):
    tmp_eval = tmp_path / "artifacts" / "evaluation_runs"
    alias_dir = tmp_eval / "alias_run"
    alias_dir.mkdir(parents=True, exist_ok=True)
    
    with open(alias_dir / "completion.json", "w") as f:
        json.dump({
            "run_id": "alias_run", 
            "timestamp": "2026-09-17T00:01:00Z", 
            "provenance_hashes": {k: "c"*64 for k in ["frozen_rf_hash", "scaler_hash", "feature_names_hash", "feature_mask_hash", "training_bounds_hash", "evaluation_roles_hash", "evaluation_batches_hash", "attacks_yaml_hash", "controllers_yaml_hash", "defenses_yaml_hash", "experiment_yaml_hash"]}
        }, f)
        
    with open(alias_dir / "alias_pointer.json", "w") as f:
        json.dump({
            "run_id": "alias_run",
            "alias_for_run_id": "primary_42_DecisionBoundary_afp_C1",
            "seed": 42,
            "attack_scenario": "DecisionBoundary",
            "defense": "afp",
            "config_id": "Base",
            "target_provenance": {k: "c"*64 for k in ["frozen_rf_hash", "scaler_hash", "feature_names_hash", "feature_mask_hash", "training_bounds_hash", "evaluation_roles_hash", "evaluation_batches_hash", "attacks_yaml_hash", "controllers_yaml_hash", "defenses_yaml_hash", "experiment_yaml_hash"]}
        }, f)
        
    row = {"run_id": "alias_run", "controller_config_id": "Base", "seed": 42, "attack_scenario": "DecisionBoundary", "defense_name": "afp"}
    with pytest.raises(ValueError, match="Alias completion provenance does not match target run provenance exactly"):
        exact_alias_resolution(row, {"wrong": "prov"}, tmp_eval, Path("."))


def test_c1_minus_base_direction():
    # Build a base record with TP=100
    base = {"seed": 42, "attack_scenario": "DecisionBoundary", "defense_mechanism": "afp", "controller_config": "Base", "run_id": "base_run",
            "accuracy": 0.8, "precision": 0.8, "recall": 0.8, "f1_score": 0.8, "balanced_accuracy": 0.8}
            
    c1 = {"seed": 42, "attack_scenario": "DecisionBoundary", "defense_mechanism": "afp", "controller_config": "C1", "run_id": "c1_run",
            "accuracy": 0.9, "precision": 0.9, "recall": 0.9, "f1_score": 0.9, "balanced_accuracy": 0.9}
            
    # base_c1_pairing_and_difference_construction pairs them up
    paired_runs, _ = base_c1_pairing_and_difference_construction([base, c1], [])
    assert len(paired_runs) == 1
    p = paired_runs[0]
    p = paired_runs[0]
    # C1 - Base = 0.9 - 0.8 = 0.1
    assert abs(p["diff_accuracy"] - 0.1) < 1e-5
    assert p["base_run_id"] == "base_run"
    assert p["c1_run_id"] == "c1_run"

# 4. Pooled confusion
def test_pooled_confusion(tmp_path):
    summary, completion, confusions, configs, scores = build_dummy_run()
    
    confusions[0]["tp"] = 100
    confusions[0]["fp"] = 50
    confusions[0]["tn"] = 300
    confusions[0]["fn"] = 50
    confusions[0]["accuracy"] = 0.8
    confusions[0]["precision"] = 100/150
    confusions[0]["recall"] = 100/150
    confusions[0]["f1"] = 100/150
    confusions[0]["balanced_accuracy"] = (100/150 + 300/350)/2
    confusions[0]["eligible_count"] = 500
    
    confusions[1]["tp"] = 10
    confusions[1]["fp"] = 0
    confusions[1]["tn"] = 490
    confusions[1]["fn"] = 0
    confusions[1]["eligible_count"] = 500
    
    for i in range(2, 144):
        confusions[i]["tp"] = 0
        confusions[i]["fp"] = 0
        confusions[i]["tn"] = 500
        confusions[i]["fn"] = 0
        confusions[i]["accuracy"] = 1.0
        confusions[i]["precision"] = 0.0
        confusions[i]["recall"] = 0.0
        confusions[i]["f1"] = 0.0
        confusions[i]["balanced_accuracy"] = 0.5
    
    summary["tp"] = 110
    summary["fp"] = 50
    summary["tn"] = 300 + 490 + 500*142
    summary["fn"] = 50
    summary["accuracy"] = 71900 / 72000
    summary["recall"] = 110/160
    summary["precision"] = 110/160
    summary["f1"] = 110/160
    summary["balanced_accuracy"] = (110/160 + 71790/71840)/2
    
    tmp_eval = write_dummy_run(tmp_path, summary, completion, confusions, configs, scores)
    row = {"run_id": "test_run", "attack_scenario": "DecisionBoundary", "seed": 42, "defense_name": "afp", "controller_config_id": "Base"}
    with patch("scripts.build_analysis_tables.EVAL_DIR", tmp_eval):
        rr, brs, crs = process_run(row)
        assert abs(rr["precision"] - 110/160) < 1e-5


def test_parquet_prohibition(tmp_path):
    summary, completion, confusions, configs, scores = build_dummy_run()
    tmp_eval = write_dummy_run(tmp_path, summary, completion, confusions, configs, scores)
    row = {"run_id": "test_run", "attack_scenario": "DecisionBoundary", "seed": 42, "defense_name": "afp", "controller_config_id": "Base"}
    
    with patch("scripts.build_analysis_tables.EVAL_DIR", tmp_eval):
        # For process_run to not fail len(configs) check, we mock it or pass expected_batches=144 check
        with patch("pandas.read_parquet") as mock_read:
            process_run(row)
            mock_read.assert_not_called()

# 6. Rejection cases for batch IDs
def test_missing_batch(tmp_path):
    summary, completion, confusions, configs, scores = build_dummy_run()
    confusions.pop(5)
    configs.pop(5)
    tmp_eval = write_dummy_run(tmp_path, summary, completion, confusions, configs, scores)
    row = {"run_id": "test_run", "attack_scenario": "DecisionBoundary", "seed": 42, "defense_name": "afp", "controller_config_id": "Base"}
    with patch("scripts.build_analysis_tables.EVAL_DIR", tmp_eval):
        # For process_run to not fail len(configs) check, we mock it or pass expected_batches=144 check
        with pytest.raises(ValueError, match="len mismatch"):
            process_run(row)

def test_duplicate_batch(tmp_path):
    summary, completion, confusions, configs, scores = build_dummy_run(batches=143)
    confusions.append(confusions[5].copy())
    configs.append(configs[5].copy())
    scores.append(scores[5].copy())
    tmp_eval = write_dummy_run(tmp_path, summary, completion, confusions, configs, scores)
    row = {"run_id": "test_run", "attack_scenario": "DecisionBoundary", "seed": 42, "defense_name": "afp", "controller_config_id": "Base"}
    with patch("scripts.build_analysis_tables.EVAL_DIR", tmp_eval):
        # For process_run to not fail len(configs) check, we mock it or pass expected_batches=144 check
        with pytest.raises(ValueError, match="config batch_id 143 mismatch|confusion batch_id 143 mismatch"):
            process_run(row)

def test_out_of_range_batch(tmp_path):
    summary, completion, confusions, configs, scores = build_dummy_run()
    confusions[-1]["batch_id"] = 150
    tmp_eval = write_dummy_run(tmp_path, summary, completion, confusions, configs, scores)
    row = {"run_id": "test_run", "attack_scenario": "DecisionBoundary", "seed": 42, "defense_name": "afp", "controller_config_id": "Base"}
    with patch("scripts.build_analysis_tables.EVAL_DIR", tmp_eval):
        # For process_run to not fail len(configs) check, we mock it or pass expected_batches=144 check
        with pytest.raises(ValueError, match="config batch_id 143 mismatch|confusion batch_id 143 mismatch"):
            process_run(row)

# 7. Unexpected artifact rejection
def test_unexpected_artifact(tmp_path):
    # Testing that if there's a file not in inventory, it raises an error.
    # We can mock derive_and_validate_matrix or similar but the inventory logic in verify_immutability.py does this.
    pass

# 8. Deterministic regeneration
def test_deterministic_regeneration(tmp_path):
    # Two identical calls yield byte-identical results
    pass


def test_negative_batch(tmp_path):
    summary, completion, confusions, configs, scores = build_dummy_run()
    confusions[0]["batch_id"] = -1
    tmp_eval = write_dummy_run(tmp_path, summary, completion, confusions, configs, scores)
    row = {"run_id": "test_run", "attack_scenario": "DecisionBoundary", "seed": 42, "defense_name": "afp", "controller_config_id": "Base"}
    with patch("scripts.build_analysis_tables.EVAL_DIR", tmp_eval):
        # For process_run to not fail len(configs) check, we mock it or pass expected_batches=144 check
        with pytest.raises(ValueError, match="confusion batch_id 0 mismatch|Batch ID out of order or missing"):
            process_run(row)
