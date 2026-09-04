import hashlib
import json
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Any

def calculate_file_hash(filepath: Path) -> str:
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            h.update(chunk)
    return h.hexdigest()

def validate_cache_manifest(manifest_path: Path, cache_artifact_path: Path, expected_hashes: Dict[str, Any]) -> bool:
    if not manifest_path.exists():
        raise ValueError(f"Cache invalid: Manifest missing at {manifest_path}")
        
    if not cache_artifact_path.exists():
        raise ValueError(f"Cache invalid: Artifact missing at {cache_artifact_path}")
        
    # Recompute and compare output_sha256
    artifact_hash = calculate_file_hash(cache_artifact_path)
    
    try:
        with open(manifest_path, "r") as f:
            manifest = json.load(f)
            
        if manifest.get("output_sha256") != artifact_hash:
            raise ValueError(f"Cache invalid: Artifact hash {artifact_hash} does not match manifest output_sha256 {manifest.get('output_sha256')}")
            
        required_keys = [
            "X_eval_hash", "metadata_eval_hash", "evaluation_roles_hash", "evaluation_batches_hash",
            "crafting_identity_hash", "measurement_identity_hash",
            "attacks_yaml_hash", "attack_script_hashes",
            "frozen_rf_hash", "scaler_hash", "feature_names_hash", "feature_mask_hash", "training_bounds_hash",
            "attack_parameters", "query_budgets",
            "attack_scenario", "effective_seed",
            "schema_version", "row_count", "output_sha256"
        ]
        
        for k in required_keys:
            if k not in manifest:
                raise ValueError(f"Cache invalid: Missing required key '{k}' in manifest")
                
        # Check all expected hashes match the manifest EXACTLY
        for key, expected_val in expected_hashes.items():
            if key not in manifest:
                raise ValueError(f"Cache invalid: Expected key '{key}' not found in manifest")
                
            if key == "attack_script_hashes" or key == "attack_parameters" or key == "query_budgets":
                if not isinstance(manifest[key], dict) or not isinstance(expected_val, dict):
                    raise ValueError(f"Cache invalid: '{key}' must be a dictionary")
                for k2, v2 in expected_val.items():
                    if manifest[key].get(k2) != v2:
                        raise ValueError(f"Cache invalid: Mismatch in {key}[{k2}]. Expected {v2}, got {manifest[key].get(k2)}")
                for k2 in manifest[key]:
                    if k2 not in expected_val:
                         raise ValueError(f"Cache invalid: Unexpected key in {key}: {k2}")
            else:
                if manifest[key] != expected_val:
                    raise ValueError(f"Cache invalid: Mismatch for {key}. Expected {expected_val}, got {manifest[key]}")
                    
        if manifest["row_count"] != 72000:
            raise ValueError(f"Cache invalid: row_count must be exactly 72000, got {manifest['row_count']}")
            
        return True
    except json.JSONDecodeError as e:
        raise ValueError(f"Cache invalid: Corrupted JSON - {str(e)}")
    except Exception as e:
        raise ValueError(f"Cache validation failed: {str(e)}")

class ConcreteAttackCacheProvider:
    def __init__(self, cache_dir: Path, resolved_batches: pd.DataFrame):
        self.cache_dir = cache_dir
        self.resolved_batches = resolved_batches
        
        artifact_path = self.cache_dir / "X_attacked.parquet"
        status_path = self.cache_dir / "status.parquet"
        
        if not artifact_path.exists() or not status_path.exists():
            raise FileNotFoundError("Cache artifacts not found")
            
        self.X_attacked = pd.read_parquet(artifact_path).values
        self.status = pd.read_parquet(status_path)
        
        if len(self.X_attacked) != 72000:
            raise ValueError(f"X_attacked has {len(self.X_attacked)} rows, expected 72000")
        if len(self.status) != 72000:
            raise ValueError(f"status has {len(self.status)} rows, expected 72000")

    def get_batch_data(self, batch_id: int) -> Dict[str, np.ndarray]:
        batch_df = self.resolved_batches[self.resolved_batches['batch_id'] == batch_id]
        batch_df = batch_df.sort_values('eval_position')
        
        if len(batch_df) != 500:
            raise ValueError(f"Batch {batch_id} must have exactly 500 records")
            
        batch_indices = batch_df['measurement_idx'].values
        
        X_batch = self.X_attacked[batch_indices]
        status_batch = self.status.iloc[batch_indices]
        
        # Enforce exactly boolean masks (pandas booleans or cast to strict bool arrays)
        eligible = np.asarray(status_batch['eligible'], dtype=bool)
        attempted = np.asarray(status_batch['attempted'], dtype=bool)
        successful = np.asarray(status_batch['successful'], dtype=bool)
        
        return {
            'X_attacked': X_batch,
            'eligible': eligible,
            'attempted': attempted,
            'successful': successful,
            'queries': np.asarray(status_batch['queries_used'], dtype=int)
        }

class CacheOrchestrator:
    """Future orchestration interface. Not executed on eval data in Phase 10."""
    def generate_cache(self):
        raise NotImplementedError("Cache generation not allowed during execution phase")
