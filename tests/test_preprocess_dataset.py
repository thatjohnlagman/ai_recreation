import os
import sys
import tempfile
import json
from pathlib import Path
import pandas as pd
import numpy as np
import joblib
import shutil

PROJECT_ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

import preprocess_dataset

import sys
from pathlib import Path
import pandas as pd
import numpy as np

PROJECT_ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from recall_aware_ids.utils.config import get_experiment_config
import preprocess_dataset

def test_processed_dataset_integrity():
    PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
    
    # Load metadata (training & calibration partitions + evaluation manifest)
    # (Phase 10A tests must never load official metadata_eval.parquet)
    meta_train = pd.read_parquet(PROCESSED_DIR / "metadata_train.parquet")
    meta_cal = pd.read_parquet(PROCESSED_DIR / "metadata_calibration.parquet")
    meta_eval = pd.read_csv(PROJECT_ROOT / "data" / "manifests" / "evaluation_roles.csv")
    
    # 1. Exact sizes
    len_train = len(meta_train)
    len_eval = len(meta_eval)
    len_cal = len(meta_cal)
    
    # Train final (178500) + Cal (31500) = 210000 exactly
    assert len_train == 178500, f"Expected 178,500 Train, got {len_train}"
    assert len_cal == 31500, f"Expected 31,500 Calibration, got {len_cal}"
    assert len_eval == 90000, f"Expected 90,000 Eval, got {len_eval}"
    assert len_train + len_cal == 210000, "Train + Calibration is not exactly 210,000"
    
    # 2. Complete evaluation batches of 500
    assert len_eval % 500 == 0, "Eval size is not a perfect multiple of 500"
    assert len_eval // 500 == 180, f"Expected exactly 180 complete batches, got {len_eval // 500}"
    
    # 3. Composite-identifier disjointness
    train_identities = set(zip(meta_train["_source_file"], meta_train["_raw_row_idx"]))
    eval_identities = set(zip(meta_eval["_source_file"], meta_eval["_raw_row_idx"]))
    cal_identities = set(zip(meta_cal["_source_file"], meta_cal["_raw_row_idx"]))
    
    assert len(train_identities.intersection(eval_identities)) == 0, "Composite identity overlap: Train/Eval"
    assert len(train_identities.intersection(cal_identities)) == 0, "Composite identity overlap: Train/Cal"
    assert len(eval_identities.intersection(cal_identities)) == 0, "Composite identity overlap: Eval/Cal"

    # 4. Correct feature exclusions and retention
    X_train = pd.read_parquet(PROCESSED_DIR / "X_train.parquet")
    features = list(X_train.columns)
    assert "Src IP" not in features, "Failed to exclude Src IP"
    assert "Label" not in features, "Failed to exclude Label"
    assert "Timestamp" not in features, "Failed to exclude Timestamp"
    assert "Src Port" not in features, "Failed to exclude Src Port (leakage)"
    assert "Dst Port" in features, "Failed to retain Dst Port"
    
    # 5. Check for NaN/Inf across all features
    assert not X_train.isna().any().any(), "NaNs found in X_train"
    assert not np.isinf(X_train.values).any(), "Inf found in X_train"

    print("All final dataset validation tests passed successfully!")

if __name__ == "__main__":
    test_processed_dataset_integrity()
