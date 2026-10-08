"""
scripts/construct_evaluation_batches.py
Deterministic Evaluation Roles & Matched Batch Construction
"""

import sys
import hashlib
import json
import time
from pathlib import Path
from datetime import datetime
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

PROJECT_ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from recall_aware_ids.utils.config import load_yaml, _resolve_config_dir

PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
MANIFESTS_DIR = PROJECT_ROOT / "data" / "manifests"
REPORTS_DIR = PROJECT_ROOT / "artifacts" / "reports"
MODELS_DIR = PROJECT_ROOT / "artifacts" / "models"

EXPECTED_RF_HASH = "9608672c5d5e38a9272c560679cdd2291399de0e0917c8c9dff373bfd200f51d"

def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(1024 * 1024):
            h.update(chunk)
    return h.hexdigest()

def check_partition_integrity(name, x_path, y_path, expected_len, feature_names):
    X = pd.read_parquet(x_path)
    y = pd.read_parquet(y_path)
    
    assert X.shape == (expected_len, 78), f"{name} X shape mismatch"
    assert y.shape[0] == expected_len, f"{name} y shape mismatch"
    
    assert list(X.columns) == feature_names, f"{name} feature names mismatch"
    assert not X.isna().any().any(), f"{name} contains NaNs"
    assert np.isfinite(X.values).all(), f"{name} contains Infs"
    
    y["composite_id"] = list(zip(y["_source_file"], y["_raw_row_idx"]))
    assert y["composite_id"].nunique() == expected_len, f"{name} duplicate composite IDs"
    
    return X, y

def preflight_check(cfg_exp, cfg_atk):
    print("Running Preflight Integrity Gate...")
    
    # Assert configurations match expectations
    ws_size = cfg_exp["dataset"]["working_sample_size"]
    eval_frac = cfg_exp["split"]["eval_fraction"]
    eval_rows = int(ws_size * eval_frac)
    batch_size = cfg_exp["evaluation"]["batch_size"]
    split_seed = cfg_exp["split"]["split_seed"]
    crafting_frac = cfg_atk["shared"]["crafting_pool_fraction"]
    crafting_rows = int(eval_rows * crafting_frac)
    measure_rows = eval_rows - crafting_rows
    batch_count = measure_rows // batch_size
    
    assert eval_rows == 90000
    assert crafting_frac == 0.20
    assert crafting_rows == 18000
    assert measure_rows == 72000
    assert batch_size == 500
    assert batch_count == 144
    assert split_seed == 42
    
    # Verify RF model hash directly
    rf_joblib_path = MODELS_DIR / "frozen_rf.joblib"
    rf_sha256_path = MODELS_DIR / "frozen_rf.sha256"
    
    actual_rf_hash = sha256_file(rf_joblib_path)
    assert actual_rf_hash == EXPECTED_RF_HASH, f"Actual RF hash mismatch: {actual_rf_hash}"
    
    with open(rf_sha256_path, "r") as f:
        sidecar_hash = f.read().strip()
    assert sidecar_hash == EXPECTED_RF_HASH, "Sidecar hash mismatch"
    
    # Load feature names
    with open(PROJECT_ROOT / "artifacts/preprocessors/feature_names.json", "r") as f:
        feature_names = json.load(f)
        
    X_train, y_train = check_partition_integrity("Train", PROCESSED_DIR / "X_train.parquet", PROCESSED_DIR / "metadata_train.parquet", 178500, feature_names)
    X_cal, y_cal = check_partition_integrity("Calibration", PROCESSED_DIR / "X_calibration.parquet", PROCESSED_DIR / "metadata_calibration.parquet", 31500, feature_names)
    X_eval, y_eval = check_partition_integrity("Evaluation", PROCESSED_DIR / "X_eval.parquet", PROCESSED_DIR / "metadata_eval.parquet", 90000, feature_names)
    
    train_ids = set(y_train["composite_id"])
    cal_ids = set(y_cal["composite_id"])
    eval_ids = set(y_eval["composite_id"])
    
    assert len(train_ids.intersection(eval_ids)) == 0, "Train/Eval overlap!"
    assert len(cal_ids.intersection(eval_ids)) == 0, "Cal/Eval overlap!"
    assert len(train_ids.intersection(cal_ids)) == 0, "Train/Cal overlap!"
    
    print("Preflight check passed.")
    return y_eval, actual_rf_hash, sidecar_hash

def deterministic_role_split(y_eval, split_seed=42):
    y_eval = y_eval.copy()
    y_eval["eval_position"] = np.arange(len(y_eval))
    
    idx_crafting, idx_measurement = train_test_split(
        y_eval["eval_position"].values,
        test_size=72000,
        stratify=y_eval["y_binary"].values,
        random_state=split_seed
    )
    
    y_eval.loc[idx_crafting, "role"] = "crafting"
    y_eval.loc[idx_measurement, "role"] = "measurement"
    
    return y_eval

def construct_batches(y_eval, batch_size=500, batch_seed=42):
    y_eval = y_eval.copy()
    measure_df = y_eval[y_eval["role"] == "measurement"].copy()
    
    meas_attacks = measure_df[measure_df["y_binary"] == 1].copy()
    meas_benign = measure_df[measure_df["y_binary"] == 0].copy()
    
    meas_attacks = meas_attacks.sample(frac=1.0, random_state=batch_seed)
    meas_benign = meas_benign.sample(frac=1.0, random_state=batch_seed)
    
    batches = []
    att_idx = 0
    ben_idx = 0
    
    for b in range(1, 145):
        num_att = 86 if b <= 17 else 85
        num_ben = batch_size - num_att
        
        b_att = meas_attacks.iloc[att_idx : att_idx + num_att]
        b_ben = meas_benign.iloc[ben_idx : ben_idx + num_ben]
        
        att_idx += num_att
        ben_idx += num_ben
        
        b_df = pd.concat([b_att, b_ben]).sample(frac=1.0, random_state=batch_seed + b)
        b_df["batch_id"] = b
        b_df["within_batch_position"] = np.arange(batch_size)
        
        batches.append(b_df)
        
    final_batches_df = pd.concat(batches)
    
    y_eval = y_eval.merge(
        final_batches_df[["eval_position", "batch_id", "within_batch_position"]], 
        on="eval_position", 
        how="left"
    )
    return y_eval

def run():
    print("="*60)
    print("Evaluation Roles and Batch Construction")
    print("="*60)
    
    cfg_dir = _resolve_config_dir()
    cfg_exp = load_yaml(cfg_dir / "experiment.yaml")
    cfg_atk = load_yaml(cfg_dir / "attacks.yaml")
    
    y_eval, actual_rf_hash, sidecar_hash = preflight_check(cfg_exp, cfg_atk)
    
    print("\nExecuting role split and batch construction...")
    split_seed = cfg_exp["split"]["split_seed"]
    y_eval_with_roles = deterministic_role_split(y_eval, split_seed=split_seed)
    y_eval_final = construct_batches(y_eval_with_roles, batch_size=500, batch_seed=split_seed)
    
    crafting_ids = set(y_eval_final[y_eval_final["role"] == "crafting"]["composite_id"])
    measure_ids = set(y_eval_final[y_eval_final["role"] == "measurement"]["composite_id"])
    assert len(crafting_ids.intersection(measure_ids)) == 0, "Crafting/Measurement overlap!"
    
    cols_to_save = [
        "eval_position", "_source_file", "_raw_row_idx", 
        "y_binary", "attack_family", "role", "batch_id", "within_batch_position"
    ]
    roles_df = y_eval_final[cols_to_save].sort_values("eval_position")
    
    batches_df = roles_df[roles_df["role"] == "measurement"].sort_values(["batch_id", "within_batch_position"])
    batches_df["batch_id"] = batches_df["batch_id"].astype(int)
    batches_df["within_batch_position"] = batches_df["within_batch_position"].astype(int)
    
    # Save to temp and hash to verify if they match authoritative exactly
    temp_roles = MANIFESTS_DIR / "temp_roles.csv"
    temp_batches = MANIFESTS_DIR / "temp_batches.csv"
    roles_df.to_csv(temp_roles, index=False)
    batches_df.to_csv(temp_batches, index=False)
    
    roles_hash = sha256_file(temp_roles)
    batches_hash = sha256_file(temp_batches)
    
    expected_roles_hash = "cbf650879aa1369fa26b803777f30d4b0add9e7aff2c05d399ba7f42563f7b45"
    expected_batches_hash = "4084017e5593732e455763416f7fc38254fc9dab2466eca289b66e68dc0ff79a"
    
    if roles_hash != expected_roles_hash or batches_hash != expected_batches_hash:
        print(f"Roles Hash: {roles_hash}")
        print(f"Batches Hash: {batches_hash}")
        raise ValueError("Reproduced CSV hashes do not match authoritative hashes!")
        
    # Since they match, replace the old ones
    temp_roles.replace(MANIFESTS_DIR / "evaluation_roles.csv")
    temp_batches.replace(MANIFESTS_DIR / "evaluation_batches.csv")
    
    print("\nAuthoritative CSVs successfully reproduced.")
    
    manifest_data = {
        "creation_time": datetime.utcnow().isoformat() + "Z",
        "seeds": {
            "split_seed": split_seed,
            "batch_seed": split_seed
        },
        "configured_values": {
            "eval_rows": 90000,
            "crafting_fraction": 0.20,
            "crafting_rows": 18000,
            "measurement_rows": 72000,
            "batch_size": 500,
            "batch_count": 144
        },
        "methods": {
            "role_splitting": "sklearn.model_selection.train_test_split (stratified on y_binary)",
            "batching": "stratified chunking with randomized within-batch order"
        },
        "hashes": {
            "X_eval_parquet": sha256_file(PROCESSED_DIR / "X_eval.parquet"),
            "metadata_eval_parquet": sha256_file(PROCESSED_DIR / "metadata_eval.parquet"),
            "evaluation_roles_csv": roles_hash,
            "evaluation_batches_csv": batches_hash,
            "experiment_yaml": sha256_file(cfg_dir / "experiment.yaml"),
            "attacks_yaml": sha256_file(cfg_dir / "attacks.yaml"),
            "previous_protocol_hash": "c10055f2759dce8ac7b65218902f44e334b35167b8d530be73831244de342ec4",
            "current_protocol_hash": sha256_file(PROJECT_ROOT / "docs" / "EXPERIMENT_PROTOCOL.md"),
            "actual_rf_joblib": actual_rf_hash,
            "rf_sidecar": sidecar_hash
        },
        "nan_inf_counts": {
            "Train": 0,
            "Calibration": 0,
            "Evaluation": 0
        },
        "composite_overlap_counts": {
            "Train_Eval": 0,
            "Cal_Eval": 0,
            "Train_Cal": 0,
            "Crafting_Measurement": 0
        },
        "counts": {
            "total_eval_records": 90000,
            "crafting_pool": {
                "total": 18000,
                "benign": 14936,
                "attack": 3064
            },
            "measurement_pool": {
                "total": 72000,
                "benign": 59743,
                "attack": 12257,
                "total_batches": 144,
                "batch_size": 500,
                "batches_with_86_attacks": 17,
                "batches_with_85_attacks": 127
            }
        },
        "reconstruction_status": "Deterministic reproduction byte-for-byte exact."
    }
    
    with open(MANIFESTS_DIR / "evaluation_partition_manifest.json", "w") as f:
        json.dump(manifest_data, f, indent=2)
        
    print("Expanded JSON manifest saved.")

if __name__ == "__main__":
    run()
