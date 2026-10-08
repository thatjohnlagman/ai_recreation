import os
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import joblib

PROJECT_ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from recall_aware_ids.utils.config import _resolve_config_dir, load_yaml

def test_frozen_model_consistency():
    cfg = load_yaml(_resolve_config_dir() / "model.yaml")
    rf_cfg = cfg["random_forest"]
    ser_cfg = cfg["serialization"]
    
    model_path = PROJECT_ROOT / ser_cfg["model_path"]
    train_x_path = PROJECT_ROOT / "data" / "processed" / "X_train.parquet"
    train_y_path = PROJECT_ROOT / "data" / "processed" / "metadata_train.parquet"
    
    assert model_path.exists(), "Frozen RF model not serialized!"
    assert train_x_path.exists(), "Training data missing"
    
    # 1. Load the model
    rf = joblib.load(model_path)
    
    # 2. Check frozen parameters
    assert rf.n_estimators == rf_cfg["n_estimators"], "n_estimators mismatch"
    assert rf.max_depth == rf_cfg["max_depth"], "max_depth mismatch"
    assert rf.criterion == rf_cfg["criterion"], "criterion mismatch"
    assert rf.n_jobs == rf_cfg["n_jobs"], "n_jobs mismatch"
    assert rf.random_state == rf_cfg["random_state"], "random_state mismatch"
    
    # 3. Reload consistency check
    # Take a tiny sample to ensure predict gives exact same results as expected
    X_train = pd.read_parquet(train_x_path)
    y_train_meta = pd.read_parquet(train_y_path)
    
    # Sample 100 rows deterministically
    np.random.seed(42)
    idx = np.random.choice(len(X_train), 100, replace=False)
    X_sub = X_train.values[idx]
    y_sub = y_train_meta["y_binary"].values[idx]
    
    pred1 = rf.predict(X_sub)
    pred_proba1 = rf.predict_proba(X_sub)
    
    # Reload the model
    rf_reloaded = joblib.load(model_path)
    pred2 = rf_reloaded.predict(X_sub)
    pred_proba2 = rf_reloaded.predict_proba(X_sub)
    
    np.testing.assert_array_equal(pred1, pred2, err_msg="Hard predictions inconsistent across reload!")
    np.testing.assert_allclose(pred_proba1, pred_proba2, err_msg="Probabilities inconsistent across reload!")
    
    print("All Phase 5 reload consistency tests passed successfully!")

if __name__ == "__main__":
    test_frozen_model_consistency()
