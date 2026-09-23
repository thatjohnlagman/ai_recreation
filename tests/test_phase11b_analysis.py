import os
import csv
import json
import pytest
import numpy as np
from pathlib import Path
from unittest.mock import patch, mock_open

from scripts.run_phase11b_analysis import (
    compute_t_test,
    run_primary_inference,
    run_supplementary_run_level,
    compute_descriptive_stats
)

def test_degenerate_vector_all_zeros():
    # All zeros
    res = compute_t_test(np.array([0.0, 0.0, 0.0, 0.0]))
    assert res["t"] == 0.0
    assert res["p_raw"] == 1.0
    assert res["ci_lower"] == 0.0
    assert res["ci_upper"] == 0.0
    assert res["cohen_dz"] == 0.0
    assert res["n"] == 4

def test_degenerate_vector_constant_nonzero():
    # Constant nonzero
    res = compute_t_test(np.array([5.0, 5.0, 5.0]))
    assert res["t"] == "Infinity"
    assert res["p_raw"] == 0.0
    assert res["ci_lower"] == 5.0
    assert res["ci_upper"] == 5.0
    assert res["cohen_dz"] is None
    assert res["n"] == 3

    res2 = compute_t_test(np.array([-3.0, -3.0, -3.0]))
    assert res2["t"] == "-Infinity"
    assert res2["p_raw"] == 0.0
    assert res2["ci_lower"] == -3.0
    assert res2["ci_upper"] == -3.0
    assert res2["cohen_dz"] is None
    assert res2["n"] == 3

def test_normal_t_test():
    # Normal case
    vals = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    res = compute_t_test(vals)
    assert res["mean"] == 3.0
    assert res["n"] == 5
    assert res["df"] == 4
    # t = mean / (std / sqrt(n)) = 3.0 / (1.5811 / 2.236) = 3.0 / 0.707 = 4.2426
    assert round(res["t"], 4) == 4.2426
    assert round(res["p_raw"], 4) == 0.0132
    assert res["ci_lower"] < 3.0
    assert res["ci_upper"] > 3.0
    assert res["cohen_dz"] is not None

def test_primary_inference_structure(tmp_path):
    # Create fake CSV with 2160 pairs for 3 defenses * 3 metrics (wait, metrics are columns in the same row)
    # We need 2160 rows for afp, 2160 for rs, 2160 for fs
    csv_path = tmp_path / "primary_paired_batch_differences.csv"
    with open(csv_path, "w") as f:
        f.write("defense_mechanism,diff_precision,diff_recall,diff_f1_score\n")
        for defense in ["afp", "randomized_smoothing", "feature_squeezing"]:
            for _ in range(2160):
                f.write(f"{defense},0.1,0.2,0.15\n")
                
    with patch("scripts.run_phase11b_analysis.ANALYSIS_DIR", tmp_path):
        results = run_primary_inference()
        
    assert len(results) == 9
    for r in results:
        assert r["n"] == 2160
        assert r["df"] == 2159
        assert "p_raw" in r
        assert "p_holm" in r
        assert r["raw_decision"] is not None
        assert r["holm_decision"] is not None
        
def test_supplementary_inference_structure(tmp_path):
    csv_path = tmp_path / "primary_paired_run_differences.csv"
    with open(csv_path, "w") as f:
        f.write("defense_mechanism,diff_precision,diff_recall,diff_f1_score\n")
        for defense in ["afp", "randomized_smoothing", "feature_squeezing"]:
            for _ in range(15):
                f.write(f"{defense},0.1,0.2,0.15\n")
                
    with patch("scripts.run_phase11b_analysis.ANALYSIS_DIR", tmp_path):
        results = run_supplementary_run_level()
        
    assert len(results) == 9
    for r in results:
        assert r["n"] == 15
        assert r["df"] == 14
        assert "is_supplementary" in r
        assert r["is_supplementary"] is True
        assert "shapiro_w" in r
        assert "wilcoxon_stat" in r

def test_no_inferential_stats_in_rq4(tmp_path):
    csv_path = tmp_path / "sensitivity_batch_level.csv"
    with open(csv_path, "w") as f:
        f.write("defense_mechanism,controller_config,attack_scenario,precision,recall,f1_score\n")
        f.write("afp,C1,SilentProbing,0.9,0.8,0.85\n")
        
    res = compute_descriptive_stats(
        csv_path,
        ["defense_mechanism", "controller_config", "attack_scenario"],
        ["precision", "recall", "f1_score"]
    )
    
    assert len(res) == 1
    r = res[0]
    # Verify no p-values or t-stats exist in descriptive output
    assert "p_raw" not in r
    assert "t" not in r
    assert "p_holm" not in r
    assert "precision_mean" in r
    assert "precision_std" in r
    assert "precision_min" in r
    assert "precision_max" in r
    assert "precision_n" in r
