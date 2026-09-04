import pytest
import numpy as np
import pandas as pd
from pathlib import Path
import json

from recall_aware_ids.experiment.policies import FixedIntensityPolicy
from recall_aware_ids.experiment.metrics import calculate_metrics
from recall_aware_ids.experiment.caching import validate_cache_manifest
from recall_aware_ids.experiment.role_resolution import validate_and_resolve_roles
from recall_aware_ids.experiment.boundary_selection import select_boundary_targets
from recall_aware_ids.experiment.schemas import BatchConfigLog, BatchConfusionLog
from recall_aware_ids.experiment.runner import ExperimentRunner, AttackCacheProvider, LabelProvider

def test_role_joins_and_exact_counts():
    # Construct synthetic valid
    metadata = pd.DataFrame({"eval_position": np.arange(90000)})
    
    # 18000 crafting, 72000 measurement
    roles = pd.DataFrame({
        "eval_position": np.arange(90000),
        "role": ["crafting"]*18000 + ["measurement"]*72000
    })
    
    # 144 batches of 500 for measurement
    batches = pd.DataFrame({
        "eval_position": np.arange(18000, 90000),
        "batch_id": np.repeat(np.arange(144), 500)
    })
    
    resolved = validate_and_resolve_roles(metadata, roles, batches)
    # validate_and_resolve_roles returns canonical measurement-only mapping (72000 rows)
    assert len(resolved) == 72000
    
def test_role_overlap_rejection():
    metadata = pd.DataFrame({"eval_position": np.arange(90000)})
    roles = pd.DataFrame({
        "eval_position": np.arange(90000),
        "role": ["crafting"]*18000 + ["measurement"]*72000
    })
    
    # Put a crafting eval_position into batches
    batches = pd.DataFrame({
        "eval_position": np.concatenate([np.array([0]), np.arange(18001, 90000)]),
        "batch_id": np.repeat(np.arange(144), 500)
    })
    
    with pytest.raises(ValueError, match="Cross-role identity"):
        validate_and_resolve_roles(metadata, roles, batches)

def test_immutable_fixed_policy():
    policy = FixedIntensityPolicy(config_id="Base", fixed_intensity=0.0003, intensity_min=0.0001, intensity_max=0.0005)
    dec = policy.get_intensity(0)
    assert dec.intensity == 0.0003
    
    with pytest.raises(Exception): # Frozen dataclass
        policy.fixed_intensity = 0.0004
        
def test_metrics_strict_validation():
    # Value error on malformed
    with pytest.raises(ValueError, match="Inputs must be 1D arrays"):
        calculate_metrics(np.array([[1]]), np.array([1]), np.array([0.9]))
        
    with pytest.raises(ValueError, match="Labels and predictions must be strictly binary"):
        calculate_metrics(np.array([2]), np.array([1]), np.array([0.9]))
        
    # None for Silent Probing ASR
    metrics = calculate_metrics(np.array([1]), np.array([1]), np.array([0.9]), asr_applicable=False)
    assert metrics["asr"] is None
    assert metrics["asr_applicable"] is False

def test_complete_cache_validation(tmp_path):
    manifest_path = tmp_path / "manifest.json"
    artifact_path = tmp_path / "X_attacked.parquet"
    
    with open(artifact_path, "wb") as f:
        f.write(b"dummy")
        
    from recall_aware_ids.experiment.caching import calculate_file_hash
    ahash = calculate_file_hash(artifact_path)
    
    good_manifest = {
        "X_eval_hash": "a", "metadata_eval_hash": "b", "evaluation_roles_hash": "c", "evaluation_batches_hash": "d",
        "crafting_identity_hash": "e", "measurement_identity_hash": "f",
        "attacks_yaml_hash": "g", "attack_script_hashes": {"script.py": "h"},
        "frozen_rf_hash": "i", "scaler_hash": "j", "feature_names_hash": "k", "feature_mask_hash": "l", "training_bounds_hash": "m",
        "attack_parameters": {"param": 1}, "query_budgets": {"queries": 50},
        "attack_scenario": "Surrogate", "effective_seed": 42,
        "schema_version": "1.0", "row_count": 72000, "output_sha256": ahash
    }
    
    with open(manifest_path, "w") as f:
        json.dump(good_manifest, f)
        
    # Validation succeeds
    assert validate_cache_manifest(manifest_path, artifact_path, good_manifest) is True
    
    # Missing required key
    bad = good_manifest.copy()
    del bad["X_eval_hash"]
    with open(manifest_path, "w") as f:
        json.dump(bad, f)
        
    with pytest.raises(ValueError, match="Missing required key"):
         validate_cache_manifest(manifest_path, artifact_path, good_manifest)
