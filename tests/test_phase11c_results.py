import os
import csv
import pytest
from pathlib import Path
from scripts.run_phase11c_results import format_p_value, format_p_value_raw

REPO_ROOT = Path(__file__).resolve().parent.parent
ANALYSIS_DIR = REPO_ROOT / "artifacts" / "analysis"

def test_p_value_formatting():
    assert format_p_value(0.0) == "p < .001"
    assert format_p_value(0.0001) == "p < .001"
    assert format_p_value(0.05) == "p = 0.050"
    
    assert format_p_value_raw(0.0) == "<.001"
    assert format_p_value_raw(0.0001) == "<.001"
    assert format_p_value_raw(0.05) == "0.050"

def test_phase11c_table_dimensions():
    # If the files exist (they will be generated before bundle), verify them
    rq1_file = ANALYSIS_DIR / "phase11c_rq1_base_performance.csv"
    if rq1_file.exists():
        with open(rq1_file) as f:
            lines = f.readlines()
            assert len(lines) == 10 # 9 rows + header
            
    rq3_file = ANALYSIS_DIR / "phase11c_rq3_statistical_decisions.csv"
    if rq3_file.exists():
        with open(rq3_file) as f:
            lines = f.readlines()
            assert len(lines) == 10 # 9 rows + header

def test_rq3_nine_test_mapping():
    rq3_file = ANALYSIS_DIR / "phase11c_rq3_statistical_decisions.csv"
    if not rq3_file.exists():
        return
        
    with open(rq3_file) as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        
    assert len(rows) == 9
    defenses = [r["defense"] for r in rows]
    metrics = [r["metric"] for r in rows]
    assert set(defenses) == {"afp", "feature_squeezing", "randomized_smoothing"}
    assert set(metrics) == {"precision", "recall", "f1_score"}
    
    for row in rows:
        c1 = float(row["c1_mean"])
        base = float(row["base_mean"])
        diff = float(row["c1_minus_base_mean"])
        # Check C1 - Base direction
        assert round(c1 - base, 4) == round(diff, 4)
        
def test_no_forbidden_access():
    # We assert that the analysis script does not import or read parquets
    # This is a static check
    script_path = REPO_ROOT / "scripts" / "run_phase11c_results.py"
    with open(script_path) as f:
        content = f.read()
    assert ".parquet" not in content
    assert ".joblib" not in content
    assert "scores" not in content.lower()
