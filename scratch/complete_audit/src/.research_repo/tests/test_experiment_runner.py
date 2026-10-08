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

def test_fixed_intensity_policy_rejects_bool_config_id():
    with pytest.raises((ValueError, TypeError), match="config_id"):
        FixedIntensityPolicy(config_id=True, fixed_intensity=0.0003, intensity_min=0.0001, intensity_max=0.0005)

def test_fixed_intensity_policy_rejects_bool_batch_id():
    policy = FixedIntensityPolicy(config_id="Base", fixed_intensity=0.0003, intensity_min=0.0001, intensity_max=0.0005)
    # Right now this might just pass because bool is an int. Red test to ensure it fails.
    with pytest.raises(TypeError, match="strictly int, not bool"):
        policy.get_intensity(True)

def test_fixed_intensity_policy_rejects_malformed_bounds():
    with pytest.raises(ValueError):
        FixedIntensityPolicy(config_id="Base", fixed_intensity=0.5, intensity_min=float('inf'), intensity_max=0.6)
    with pytest.raises(ValueError):
        FixedIntensityPolicy(config_id="Base", fixed_intensity=0.5, intensity_min=0.1, intensity_max=float('nan'))
    with pytest.raises(ValueError):
        FixedIntensityPolicy(config_id="Base", fixed_intensity=0.5, intensity_min=0.6, intensity_max=0.4) # min > max

def test_fixed_intensity_policy_batch_id_sequences():
    policy = FixedIntensityPolicy(config_id="Base", fixed_intensity=0.0003, intensity_min=0.0001, intensity_max=0.0005)
    
    # Negative batch id
    with pytest.raises(ValueError, match="non-negative"):
        policy.get_intensity(-1)
        
    # Skipped, repeated, out-of-order shouldn't matter for FIXED intensity policy 
    # BUT wait, the prompt says: "skipped, repeated, negative, and out-of-order batch IDs as applicable to the interface."
    # Since FixedIntensityPolicy is stateless, skipped/repeated/out-of-order are perfectly valid and should succeed.
    assert policy.get_intensity(5).intensity == 0.0003
    assert policy.get_intensity(5).intensity == 0.0003
    assert policy.get_intensity(2).intensity == 0.0003

        
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
    status_path = tmp_path / "status.parquet"
    
    with open(artifact_path, "wb") as f:
        f.write(b"dummy_x_data")
    with open(status_path, "wb") as f:
        f.write(b"dummy_status_data")
        
    from recall_aware_ids.experiment.caching import calculate_file_hash
    x_hash = calculate_file_hash(artifact_path)
    s_hash = calculate_file_hash(status_path)
    
    import hashlib
    def _d(name: str) -> str:
        return hashlib.sha256(name.encode()).hexdigest()

    good_manifest = {
        "X_eval_hash": _d("X_eval"),
        "metadata_eval_hash": _d("metadata_eval"),
        "evaluation_roles_hash": _d("evaluation_roles"),
        "evaluation_batches_hash": _d("evaluation_batches"),
        "crafting_identity_hash": _d("crafting_identity"),
        "measurement_identity_hash": _d("measurement_identity"),
        "attacks_yaml_hash": _d("attacks_yaml"),
        "attack_script_hashes": {"script.py": calculate_file_hash(Path(__file__))},
        "frozen_rf_hash": _d("frozen_rf"),
        "scaler_hash": _d("scaler"),
        "feature_names_hash": _d("feature_names"),
        "feature_mask_hash": _d("feature_mask"),
        "training_bounds_hash": _d("training_bounds"),
        "attack_parameters": {"param": 1},
        "query_budgets": {"queries": 50},
        "attack_scenario": "SurrogateTransfer",
        "effective_seed": 42,
        "schema_version": "1.0",
        "row_count": 72000,
        "X_attacked_sha256": x_hash,
        "status_sha256": s_hash,
        "output_sha256": x_hash,
    }
    
    with open(manifest_path, "w") as f:
        json.dump(good_manifest, f)
        
    # Validation succeeds
    assert validate_cache_manifest(manifest_path, artifact_path, good_manifest, status_path=status_path) is True
    
    # Missing required key
    bad = good_manifest.copy()
    del bad["X_eval_hash"]
    with open(manifest_path, "w") as f:
        json.dump(bad, f)
        
    with pytest.raises(ValueError, match="Missing required key"):
         validate_cache_manifest(manifest_path, artifact_path, good_manifest, status_path=status_path)
