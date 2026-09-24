import os
import csv
import pytest
from pathlib import Path
from scripts.run_phase11c_results import format_p_value_text, format_p_value_raw, read_csv, calculate_file_hash

REPO_ROOT = Path(__file__).resolve().parent.parent
ANALYSIS_DIR = REPO_ROOT / "artifacts" / "analysis"
REPORTS_DIR = REPO_ROOT / "artifacts" / "reports"

def test_p_value_formatting():
    assert format_p_value_text(0.0) == "p < .001"
    assert format_p_value_text(0.0001) == "p < .001"
    assert format_p_value_text(0.05) == "p = 0.050"
    
    assert format_p_value_raw(0.0) == "<.001"
    assert format_p_value_raw(0.0001) == "<.001"
    assert format_p_value_raw(0.05) == "0.050"

def test_phase11c_table_dimensions():
    # Require files to exist
    rq1_file = ANALYSIS_DIR / "phase11c_rq1_base_performance.csv"
    rq2_file = ANALYSIS_DIR / "phase11c_rq2_controller_performance.csv"
    rq3_file = ANALYSIS_DIR / "phase11c_rq3_statistical_decisions.csv"
    rq4_file = ANALYSIS_DIR / "phase11c_rq4_sensitivity_summary.csv"
    
    assert rq1_file.exists(), "RQ1 file missing"
    assert rq2_file.exists(), "RQ2 file missing"
    assert rq3_file.exists(), "RQ3 file missing"
    assert rq4_file.exists(), "RQ4 file missing"

    rq1 = read_csv(rq1_file)
    rq2 = read_csv(rq2_file)
    rq3 = read_csv(rq3_file)
    rq4 = read_csv(rq4_file)
    
    # 3 defenses * 3 attacks = 9 rows
    assert len(rq1) == 9
    assert len(rq2) == 9
    # 3 defenses * 3 metrics = 9 rows
    assert len(rq3) == 9
    # 3 defenses * 3 attacks * 7 configs = 63 rows
    assert len(rq4) == 63
    
    # Check keys
    for r in rq1:
        assert set(r.keys()) == {"defense", "attack_scenario", "precision", "recall", "f1_score"}
    for r in rq2:
        assert set(r.keys()) == {"defense", "attack_scenario", "precision", "recall", "f1_score"}

def test_rq3_nine_test_mapping():
    rq3_file = ANALYSIS_DIR / "phase11c_rq3_statistical_decisions.csv"
    assert rq3_file.exists()
    rows = read_csv(rq3_file)
    
    assert len(rows) == 9
    defenses = [r["defense"] for r in rows]
    metrics = [r["metric"] for r in rows]
    assert set(defenses) == {"afp", "feature_squeezing", "randomized_smoothing"}
    assert set(metrics) == {"precision", "recall", "f1_score"}
    
    for row in rows:
        c1 = float(row["c1_mean"])
        base = float(row["base_mean"])
        diff = float(row["c1_minus_base_mean"])
        assert round(c1 - base, 4) == round(diff, 4)

def test_no_forbidden_access():
    script_path = REPO_ROOT / "scripts" / "run_phase11c_results.py"
    with open(script_path) as f:
        content = f.read()
    # Must not contain forbidden extensions
    assert ".parquet" not in content
    assert ".joblib" not in content
    assert "scores" not in content.lower()

def test_traceability_exact_match():
    # Read traceability report
    trace_file = REPORTS_DIR / "phase11c_traceability_report.csv"
    assert trace_file.exists()
    traces = read_csv(trace_file)
    
    for t in traces:
        assert t["match_status"] == "PASS"
        assert abs(float(t["source_value"]) - float(t["generated_value"])) < 1e-7

def test_seven_phase11a_hashes():
    summary_path = REPORTS_DIR / "phase11a_analysis_tables_summary.txt"
    assert summary_path.exists()
    
    expected_hashes = {}
    with open(summary_path, "r") as f:
        current_table = None
        for line in f:
            line = line.strip()
            if line.startswith("Table: "):
                current_table = line.replace("Table: ", "")
            elif line.startswith("SHA-256: ") and current_table:
                expected_hashes[current_table] = line.replace("SHA-256: ", "")
                current_table = None
                
    assert len(expected_hashes) == 7
    for table, expected_hash in expected_hashes.items():
        table_path = ANALYSIS_DIR / table
        actual_hash = calculate_file_hash(table_path)
        assert actual_hash == expected_hash

def test_chapter_4_values():
    draft_path = REPO_ROOT / "docs" / "PHASE11C_CHAPTER4_DRAFT.md"
    assert draft_path.exists()
    with open(draft_path, "r") as f:
        content = f.read()
        
    assert "Ennaji et al. (2025)" in content
    assert "seeds 42–44" in content or "seeds 42-44" in content
    assert "432 batches" in content
    assert "valid internal matched comparison" in content
    
    # Check that Silent Probing C1 Recall is ~93.70%
    assert "93.70%" in content
    assert "80.66%" in content
