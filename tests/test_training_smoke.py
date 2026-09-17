"""
tests/test_training_smoke.py

Separated training-derived smoke test for Phase 10B Attack-Cache Orchestration.
Uses ONLY X_train.parquet and metadata_train.parquet (small deterministic subset).
All outputs are strictly written to a temporary output directory outside artifacts/caches.
EVALUATION DATA (X_eval.parquet, metadata_eval.parquet) MUST NEVER BE ACCESSED.
OFFICIAL CACHES (artifacts/caches) MUST NEVER BE CREATED OR PUBLISHED.
"""
import json
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from recall_aware_ids.experiment.caching import calculate_file_hash
from scripts.build_evaluation_caches import (
    build_expected_cache_identity,
    build_single_cache,
    validate_completed_cache,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIGS_DIR = REPO_ROOT / "configs"
DATA_DIR = REPO_ROOT / "data"
ARTIFACTS_DIR = REPO_ROOT / "artifacts"

# Valid 64-char hex strings for testing
_SYNTH_PROV = {
    "attacks_yaml_hash": "1111111111111111111111111111111111111111111111111111111111111111",
    "controllers_yaml_hash": "1111111111111111111111111111111111111111111111111111111111111111",
    "defenses_yaml_hash": "1111111111111111111111111111111111111111111111111111111111111111",
    "experiment_yaml_hash": "1111111111111111111111111111111111111111111111111111111111111111",
    "evaluation_roles_hash": "1111111111111111111111111111111111111111111111111111111111111111",
    "evaluation_batches_hash": "1111111111111111111111111111111111111111111111111111111111111111",
    "frozen_rf_hash": "1111111111111111111111111111111111111111111111111111111111111111",
    "scaler_hash": "1111111111111111111111111111111111111111111111111111111111111111",
    "feature_names_hash": "1111111111111111111111111111111111111111111111111111111111111111",
    "feature_mask_hash": "1111111111111111111111111111111111111111111111111111111111111111",
    "training_bounds_hash": "1111111111111111111111111111111111111111111111111111111111111111",
    "X_eval_hash": "1111111111111111111111111111111111111111111111111111111111111111",
    "metadata_eval_hash": "1111111111111111111111111111111111111111111111111111111111111111",
    "crafting_identity_hash": "1111111111111111111111111111111111111111111111111111111111111111",
    "measurement_identity_hash": "1111111111111111111111111111111111111111111111111111111111111111",
}


def test_training_smoke_uses_only_training_partition_and_tmp_dir(tmp_path):
    """
    Separated smoke test:
      - Uses only data/processed/X_train.parquet and metadata_train.parquet.
      - Never touches X_eval.parquet or metadata_eval.parquet.
      - Output directory is outside artifacts/caches (in tmp_path).
      - Never publishes an official cache.
    """
    x_train_path = DATA_DIR / "processed/X_train.parquet"
    meta_train_path = DATA_DIR / "processed/metadata_train.parquet"

    if not x_train_path.exists() or not meta_train_path.exists():
        pytest.skip("Training partition files not present on machine")

    # Guard: Assert that evaluation paths are NEVER opened
    original_read_parquet = pd.read_parquet

    def guarded_read_parquet(filepath, *args, **kwargs):
        fp_str = str(filepath)
        if "X_eval" in fp_str or "metadata_eval" in fp_str:
            raise PermissionError(f"CRITICAL SECURITY VIOLATION: Smoke test attempted to access evaluation data: {fp_str}")
        return original_read_parquet(filepath, *args, **kwargs)

    with patch("pandas.read_parquet", side_effect=guarded_read_parquet):
        # Load small deterministic 60-row slice from training data
        df_x = pd.read_parquet(x_train_path).head(60)
        df_meta = pd.read_parquet(meta_train_path).head(60)

        with open(ARTIFACTS_DIR / "preprocessors/feature_mask.json") as f:
            fnames = json.load(f)["feature_columns"]

        X_train_slice = df_x[fnames].values.astype(np.float32)
        y_train_slice = df_meta["y_binary"].values.astype(int)

        X_craft = X_train_slice[:20]
        y_craft = y_train_slice[:20]
        X_meas = X_train_slice[20:60]
        y_meas = y_train_slice[20:60]
        eps_meas = np.arange(len(X_meas), dtype=np.int64)

        batches_df = pd.DataFrame({
            "batch_id": [0] * len(X_meas),
            "eval_position": eps_meas,
            "within_batch_position": np.arange(len(X_meas)),
        })

        import joblib
        model = joblib.load(ARTIFACTS_DIR / "models/frozen_rf.joblib")

        smoke_out_dir = tmp_path / "smoke_caches_outside_official"
        official_cache_root = REPO_ROOT / "artifacts/caches"

        # Record official cache root state before smoke test
        official_before = list(official_cache_root.glob("*")) if official_cache_root.exists() else []

        # Run smoke execution for SilentProbing
        c_path = build_single_cache(
            scenario="Silent Probing",
            seed=42,
            output_dir=smoke_out_dir,
            configs_dir=CONFIGS_DIR,
            data_dir=DATA_DIR,
            artifacts_dir=ARTIFACTS_DIR,
            resolved_batches=batches_df,
            official_mode=False,
            override_X_craft=X_craft,
            override_y_craft=y_craft,
            override_X_meas=X_meas,
            override_y_meas=y_meas,
            override_eps=eps_meas,
            override_predict_fn=model.predict,
            override_provenance=_SYNTH_PROV,
            override_row_count=len(X_meas),
        )

        # 1. Output exists in temporary directory
        assert c_path.exists()
        assert c_path.parent == smoke_out_dir
        assert c_path.name == "SilentProbing_42"

        # 2. Strict completion validation succeeds with independent expected identity
        attacks_src = REPO_ROOT / "src/recall_aware_ids/attacks"
        script_hashes = {
            "base.py": calculate_file_hash(attacks_src / "base.py"),
            "silent_probing.py": calculate_file_hash(attacks_src / "silent_probing.py"),
            "surrogate_transfer.py": calculate_file_hash(attacks_src / "surrogate_transfer.py"),
            "boundary_attack.py": calculate_file_hash(attacks_src / "boundary_attack.py"),
            "oracle.py": calculate_file_hash(attacks_src / "oracle.py"),
            "caching.py": calculate_file_hash(REPO_ROOT / "src/recall_aware_ids/experiment/caching.py"),
            "build_evaluation_caches.py": calculate_file_hash(REPO_ROOT / "scripts/build_evaluation_caches.py"),
        }
        x_hash = calculate_file_hash(c_path / "X_attacked.parquet")
        s_hash = calculate_file_hash(c_path / "status.parquet")

        expected_identity = build_expected_cache_identity(
            scenario="SilentProbing",
            seed=42,
            expected_row_count=len(X_meas),
            prov_hashes=_SYNTH_PROV,
            script_hashes=script_hashes,
            attack_parameters={"modifies_samples": False},
            query_budgets={"max_queries_per_sample": 0},
            x_attacked_sha256=x_hash,
            status_sha256=s_hash,
        )

        is_valid, reason = validate_completed_cache(
            cache_dir=c_path,
            expected_scenario="SilentProbing",
            expected_seed=42,
            expected_cache_identity=expected_identity,
            expected_feature_names=fnames,
            resolved_batches=batches_df,
            official_mode=False,
            expected_row_count=len(X_meas),
        )
        assert is_valid, f"Validation failed: {reason}"

        # 3. Cache artifacts are present and correct
        assert (c_path / "X_attacked.parquet").exists()
        assert (c_path / "status.parquet").exists()
        assert (c_path / "manifest.json").exists()
        assert (c_path / "completion.json").exists()

        # 4. Official cache directory was completely untouched
        official_after = list(official_cache_root.glob("*")) if official_cache_root.exists() else []
        assert official_before == official_after, "Official artifacts/caches was modified by smoke test!"
