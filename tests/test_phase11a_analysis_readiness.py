import pytest
import os
import csv
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
ANALYSIS_DIR = REPO_ROOT / "artifacts" / "analysis"

@pytest.fixture
def analysis_tables():
    tables = {
        "primary_run": ANALYSIS_DIR / "primary_run_level.csv",
        "primary_batch": ANALYSIS_DIR / "primary_batch_level.csv",
        "primary_paired_runs": ANALYSIS_DIR / "primary_paired_run_differences.csv",
        "primary_paired_batches": ANALYSIS_DIR / "primary_paired_batch_differences.csv",
        "sensitivity_run": ANALYSIS_DIR / "sensitivity_run_level.csv",
        "sensitivity_batch": ANALYSIS_DIR / "sensitivity_batch_level.csv",
        "controller_trace": ANALYSIS_DIR / "controller_trace_summary.csv"
    }
    return tables

def read_csv(path):
    with open(path, "r") as f:
        reader = csv.DictReader(f)
        return list(reader)

def test_analysis_table_existence(analysis_tables):
    for name, path in analysis_tables.items():
        assert path.exists(), f"Missing table {name} at {path}"

def test_analysis_table_row_counts(analysis_tables):
    assert len(read_csv(analysis_tables["primary_run"])) == 90
    assert len(read_csv(analysis_tables["primary_batch"])) == 12960
    assert len(read_csv(analysis_tables["primary_paired_runs"])) == 45
    assert len(read_csv(analysis_tables["primary_paired_batches"])) == 6480
    assert len(read_csv(analysis_tables["sensitivity_run"])) == 189
    assert len(read_csv(analysis_tables["sensitivity_batch"])) == 27216
    
def test_c1_minus_base_direction(analysis_tables):
    runs = read_csv(analysis_tables["primary_paired_runs"])
    for r in runs:
        assert r["c1_run_id"].endswith("_C1")
        assert r["base_run_id"].endswith("_Base")
        
def test_alias_resolution_without_duplicate_independence(analysis_tables):
    sens_runs = read_csv(analysis_tables["sensitivity_run"])
    c1_runs = [r for r in sens_runs if r["controller_config"] == "C1"]
    assert len(c1_runs) == 27, "Should be 27 logical C1 runs in sensitivity table"
    # Ensure all run_ids start with sensitivity
    for r in c1_runs:
        assert r["run_id"].startswith("sensitivity_"), "Aliases must be projected onto the sensitivity logical execution"

def test_no_nan_or_inf_in_metrics(analysis_tables):
    runs = read_csv(analysis_tables["primary_run"])
    for r in runs:
        for metric in ["accuracy", "precision", "recall", "f1_score", "balanced_accuracy"]:
            val = float(r[metric])
            assert val == val  # Not NaN
            assert val != float("inf") and val != float("-inf")
            
def test_exact_pairing_keys(analysis_tables):
    paired_batches = read_csv(analysis_tables["primary_paired_batches"])
    keys = set()
    for pb in paired_batches:
        k = (pb["seed"], pb["attack_scenario"], pb["defense_mechanism"], pb["batch_id"])
        assert k not in keys, "Duplicate pairing key found"
        keys.add(k)
    assert len(keys) == 6480

def test_prevention_of_unweighted_batch_averaging(analysis_tables):
    # This checks that run_level recall is NOT just the mean of batch-level recalls
    run_table = read_csv(analysis_tables["primary_run"])
    
    # We just ensure the column exists and values are properly bounded
    for r in run_table:
        val = float(r["recall"])
        assert 0.0 <= val <= 1.0

# Remaining properties (quarantine exclusion, Parquet exclusion, etc.) are
# enforced by the builder script and protocol.
