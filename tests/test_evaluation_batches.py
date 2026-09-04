import sys
import hashlib
from pathlib import Path
import pandas as pd
import numpy as np

PROJECT_ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from recall_aware_ids.utils.config import load_yaml, _resolve_config_dir
from construct_evaluation_batches import deterministic_role_split, construct_batches

PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
MANIFESTS_DIR = PROJECT_ROOT / "data" / "manifests"
MODELS_DIR = PROJECT_ROOT / "artifacts" / "models"

EXPECTED_RF_HASH = "9608672c5d5e38a9272c560679cdd2291399de0e0917c8c9dff373bfd200f51d"

def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(1024 * 1024):
            h.update(chunk)
    return h.hexdigest()

def test_evaluation_roles_and_batches():
    cfg_dir = _resolve_config_dir()
    cfg_exp = load_yaml(cfg_dir / "experiment.yaml")
    cfg_atk = load_yaml(cfg_dir / "attacks.yaml")
    split_seed = cfg_exp["split"]["split_seed"]
    
    # 1. Verify RF hash directly matches sidecar and expected
    rf_joblib_path = MODELS_DIR / "frozen_rf.joblib"
    rf_sha256_path = MODELS_DIR / "frozen_rf.sha256"
    assert sha256_file(rf_joblib_path) == EXPECTED_RF_HASH
    with open(rf_sha256_path, "r") as f:
        assert f.read().strip() == EXPECTED_RF_HASH
        
    # 2. Partitions integrity (finite, identical order)
    X_train = pd.read_parquet(PROCESSED_DIR / "X_train.parquet")
    X_cal = pd.read_parquet(PROCESSED_DIR / "X_calibration.parquet")
    X_eval = pd.read_parquet(PROCESSED_DIR / "X_eval.parquet")
    
    assert list(X_train.columns) == list(X_cal.columns) == list(X_eval.columns)
    
    for X in [X_train, X_cal, X_eval]:
        assert not X.isna().any().any()
        assert np.isfinite(X.values).all()
        
    # 3. Load persisted manifests
    roles_df = pd.read_csv(MANIFESTS_DIR / "evaluation_roles.csv")
    batches_df = pd.read_csv(MANIFESTS_DIR / "evaluation_batches.csv")
    
    # 4. Independent Reconstruction
    y_eval = pd.read_parquet(PROCESSED_DIR / "metadata_eval.parquet")
    y_eval_with_roles = deterministic_role_split(y_eval, split_seed=split_seed)
    y_eval_final = construct_batches(y_eval_with_roles, batch_size=500, batch_seed=split_seed)
    
    cols = ["eval_position", "role", "batch_id", "within_batch_position"]
    rec_roles = y_eval_final[cols].sort_values("eval_position").reset_index(drop=True)
    persisted_roles = roles_df[cols].sort_values("eval_position").reset_index(drop=True)
    
    # Fill NAs to compare safely
    rec_roles = rec_roles.fillna(-1)
    persisted_roles = persisted_roles.fillna(-1)
    
    pd.testing.assert_frame_equal(rec_roles, persisted_roles, check_dtype=False)
    
    # 5. Exact Counts
    assert len(roles_df) == 90000
    crafting_df = roles_df[roles_df["role"] == "crafting"]
    measure_df = roles_df[roles_df["role"] == "measurement"]
    assert len(crafting_df) == 18000
    assert len(measure_df) == 72000
    
    assert (crafting_df["y_binary"] == 0).sum() == 14936
    assert (crafting_df["y_binary"] == 1).sum() == 3064
    assert (measure_df["y_binary"] == 0).sum() == 59743
    assert (measure_df["y_binary"] == 1).sum() == 12257
    
    # 6. Check batches_df strictly matches measure_df
    pd.testing.assert_frame_equal(
        batches_df.sort_values("eval_position").reset_index(drop=True)[["eval_position", "batch_id", "within_batch_position"]],
        measure_df.sort_values("eval_position").reset_index(drop=True)[["eval_position", "batch_id", "within_batch_position"]],
        check_dtype=False
    )
    
    # 7. Batch ID and Position Constraints
    assert set(batches_df["batch_id"].unique()) == set(range(1, 145))
    
    # Each batch has exactly 0 through 499
    for b_id, grp in batches_df.groupby("batch_id"):
        assert set(grp["within_batch_position"]) == set(range(500)), f"Batch {b_id} missing positions"
        
    # 8. No baseline artifacts
    baseline_metrics = PROJECT_ROOT / "artifacts" / "reports" / "baseline_metrics.json"
    assert not baseline_metrics.exists(), "Phase 6 improperly generated model predictions!"

    print("All Strengthened Phase 6 Integrity Tests Passed successfully!")

if __name__ == "__main__":
    test_evaluation_roles_and_batches()
