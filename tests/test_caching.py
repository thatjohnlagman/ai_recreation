import pytest
import json
import pandas as pd
import numpy as np
from pathlib import Path
from recall_aware_ids.experiment.caching import validate_cache_manifest, ConcreteAttackCacheProvider

def test_concrete_cache_provider(tmp_path):
    # Setup dummy parquet and manifest
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    
    # 72000 rows
    X_attacked = np.zeros((72000, 78))
    pd.DataFrame(X_attacked).to_parquet(cache_dir / "X_attacked.parquet")
    
    status_df = pd.DataFrame({
        'eval_position': np.arange(72000),
        'eligible': np.ones(72000, dtype=bool),
        'attempted': np.ones(72000, dtype=bool),
        'successful': np.zeros(72000, dtype=bool),
        'status_code': ['success'] * 72000,
        'queries_used': np.zeros(72000, dtype=int),
        'l0': 0.0, 'l1': 0.0, 'l2': 0.0, 'linf': 0.0
    })
    status_df.to_parquet(cache_dir / "status.parquet")
    
    resolved_batches = pd.DataFrame({
        'measurement_idx': np.arange(500),
        'batch_id': np.zeros(500, dtype=int),
        'eval_position': np.arange(500)
    })
    
    provider = ConcreteAttackCacheProvider(cache_dir, resolved_batches)
    data = provider.get_batch_data(0)
    
    assert data['X_attacked'].shape == (500, 78)
    assert len(data['eligible']) == 500
    assert data['eligible'].dtype == bool

def test_cache_validation_corruption(tmp_path):
    manifest_path = tmp_path / "manifest.json"
    artifact_path = tmp_path / "X_attacked.parquet"
    
    # Write dummy artifact
    with open(artifact_path, "wb") as f:
        f.write(b"dummy")
        
    from recall_aware_ids.experiment.caching import calculate_file_hash
    ahash = calculate_file_hash(artifact_path)
    
    valid_manifest = {
        "X_eval_hash": "a", "metadata_eval_hash": "b", "evaluation_roles_hash": "c", "evaluation_batches_hash": "d",
        "crafting_identity_hash": "e", "measurement_identity_hash": "f",
        "attacks_yaml_hash": "g", "attack_script_hashes": {},
        "frozen_rf_hash": "h", "scaler_hash": "i", "feature_names_hash": "j", "feature_mask_hash": "k", "training_bounds_hash": "l",
        "attack_parameters": {}, "query_budgets": {},
        "attack_scenario": "m", "effective_seed": 42,
        "schema_version": "1.0", "row_count": 72000, "output_sha256": ahash
    }
    
    with open(manifest_path, "w") as f:
        json.dump(valid_manifest, f)
        
    expected_hashes = {
        "X_eval_hash": "a", "metadata_eval_hash": "b", "evaluation_roles_hash": "c", "evaluation_batches_hash": "d",
        "crafting_identity_hash": "e", "measurement_identity_hash": "f",
        "attacks_yaml_hash": "g", "attack_script_hashes": {},
        "frozen_rf_hash": "h", "scaler_hash": "i", "feature_names_hash": "j", "feature_mask_hash": "k", "training_bounds_hash": "l",
        "attack_parameters": {}, "query_budgets": {},
        "attack_scenario": "m", "effective_seed": 42,
        "schema_version": "1.0", "row_count": 72000, "output_sha256": ahash
    }
    
    # Should pass
    assert validate_cache_manifest(manifest_path, artifact_path, expected_hashes)
    
    # Test corruption
    with open(artifact_path, "wb") as f:
        f.write(b"corrupted")
        
    with pytest.raises(ValueError, match="does not match manifest output_sha256"):
        validate_cache_manifest(manifest_path, artifact_path, expected_hashes)
        
    # Re-fix artifact, corrupt JSON
    with open(artifact_path, "wb") as f:
        f.write(b"dummy")
        
    with open(manifest_path, "w") as f:
        f.write("{invalid json")
        
    with pytest.raises(ValueError, match="Corrupted JSON"):
        validate_cache_manifest(manifest_path, artifact_path, expected_hashes)
