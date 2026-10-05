import pytest
import subprocess
import sys
from pathlib import Path

DEMO_DIR = Path(__file__).parents[1]

def test_imports():
    try:
        import attacks.silent_probing
        import defenses.afp
        import controller.recall_controller
    except ImportError as e:
        pytest.fail(f"Import failed: {e}")

def test_model_and_stats_load():
    import joblib
    import json
    import pandas as pd
    
    model = joblib.load(DEMO_DIR / "model" / "frozen_rf.joblib")
    assert model is not None
    
    with open(DEMO_DIR / "model" / "feature_names.json", "r") as f:
        feature_names = json.load(f)
    assert len(feature_names) == 78
    
    bounds = pd.read_parquet(DEMO_DIR / "model" / "training_bounds.parquet")
    assert not bounds.empty

def test_base_and_recall_aware_cli():
    # Test Base Mode
    res_base = subprocess.run(
        [sys.executable, str(DEMO_DIR / "run.py"), "--attack", "silent_probing", "--defense", "afp", "--controller", "base"],
        cwd=DEMO_DIR, capture_output=True, text=True
    )
    assert res_base.returncode == 0
    assert "Base AFP" in res_base.stdout
    
    # Test Recall-Aware Mode
    res_ra = subprocess.run(
        [sys.executable, str(DEMO_DIR / "run.py"), "--attack", "silent_probing", "--defense", "afp", "--controller", "recall-aware"],
        cwd=DEMO_DIR, capture_output=True, text=True
    )
    assert res_ra.returncode == 0
    assert "Recall-aware AFP" in res_ra.stdout or "Recall-aware AFP" in res_ra.stdout.lower() or "Recall-Aware AFP" in res_ra.stdout

def test_comparison_mode():
    res = subprocess.run(
        [sys.executable, str(DEMO_DIR / "run.py"), "--attack", "decision_boundary", "--defense", "afp", "--compare"],
        cwd=DEMO_DIR, capture_output=True, text=True
    )
    assert res.returncode == 0
    assert "No Defense" in res.stdout
    assert "Base AFP" in res.stdout
    assert "Recall-Aware AFP" in res.stdout

def test_all_defenses():
    for d in ["afp", "rs", "fs"]:
        res = subprocess.run(
            [sys.executable, str(DEMO_DIR / "run.py"), "--attack", "none", "--defense", d],
            cwd=DEMO_DIR, capture_output=True, text=True
        )
        assert res.returncode == 0

def test_all_attacks():
    for a in ["silent_probing", "surrogate_transfer", "decision_boundary"]:
        res = subprocess.run(
            [sys.executable, str(DEMO_DIR / "run.py"), "--attack", a, "--defense", "none"],
            cwd=DEMO_DIR, capture_output=True, text=True
        )
        assert res.returncode == 0
