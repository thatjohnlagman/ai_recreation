"""
Tests for AttackCacheBuilder and ConcreteAttackCacheProvider.
Uses synthetic data only — no evaluation records accessed.
"""
import pytest
import json
import pandas as pd
import numpy as np
from pathlib import Path

from recall_aware_ids.experiment.caching import (
    validate_cache_manifest, ConcreteAttackCacheProvider,
    AttackCacheBuilder, calculate_file_hash,
)

import hashlib

# 64-char hex used for provenance in tests
_H = hashlib.sha256(b"authoritative_test_seed").hexdigest()
_SCRIPT_HASHES = {"generator.py": _H}
_PARAMS = {"modifies_samples": False}
_BUDGETS = {"max_queries_per_sample": 0}

_SYNTH_PROV = {
    "frozen_rf_hash": _H,
    "scaler_hash": _H,
    "feature_names_hash": _H,
    "feature_mask_hash": _H,
    "training_bounds_hash": _H,
    "evaluation_roles_hash": _H,
    "evaluation_batches_hash": _H,
    "attacks_yaml_hash": _H,
    "controllers_yaml_hash": _H,
    "defenses_yaml_hash": _H,
    "experiment_yaml_hash": _H,
    # Extra keys used by builder's manifest
    "X_eval_hash": _H,
    "metadata_eval_hash": _H,
    "crafting_identity_hash": _H,
    "measurement_identity_hash": _H,
}

N = 100  # small synthetic size for speed


def make_synthetic_resolved_batches(n_rows=N, n_batches=None):
    """n_rows must be divisible by n_batches (default all in batch 0)."""
    if n_batches is None:
        return pd.DataFrame({
            "measurement_idx": np.arange(n_rows),
            "batch_id": np.zeros(n_rows, dtype=int),
            "eval_position": np.arange(n_rows),
        })
    batch_size = n_rows // n_batches
    return pd.DataFrame({
        "measurement_idx": np.arange(n_rows),
        "batch_id": np.repeat(np.arange(n_batches), batch_size),
        "eval_position": np.arange(n_rows),
    })


# ─── validate_cache_manifest ─────────────────────────────────────────────────

def test_validate_cache_manifest_passes(tmp_path):
    x_path = tmp_path / "X_attacked.parquet"
    s_path = tmp_path / "status.parquet"
    m_path = tmp_path / "manifest.json"

    with open(x_path, "wb") as f:
        f.write(b"data")
    with open(s_path, "wb") as f:
        f.write(b"status")

    x_hash = calculate_file_hash(x_path)
    s_hash = calculate_file_hash(s_path)

    manifest = {
        "X_eval_hash": _H, "metadata_eval_hash": _H,
        "evaluation_roles_hash": _H, "evaluation_batches_hash": _H,
        "crafting_identity_hash": _H, "measurement_identity_hash": _H,
        "attacks_yaml_hash": _H, "attack_script_hashes": _SCRIPT_HASHES,
        "frozen_rf_hash": _H, "scaler_hash": _H,
        "feature_names_hash": _H, "feature_mask_hash": _H, "training_bounds_hash": _H,
        "controllers_yaml_hash": _H, "defenses_yaml_hash": _H, "experiment_yaml_hash": _H,
        "attack_parameters": _PARAMS, "query_budgets": _BUDGETS,
        "attack_scenario": "SilentProbing", "effective_seed": 42,
        "schema_version": "1.0", "row_count": N,
        "X_attacked_sha256": x_hash, "status_sha256": s_hash, "output_sha256": x_hash,
    }
    with open(m_path, "w") as f:
        json.dump(manifest, f)

    manifest_expected = dict(manifest)
    assert validate_cache_manifest(m_path, x_path, manifest_expected, status_path=s_path, expected_row_count=N) is True


def test_validate_cache_manifest_fails_corrupted_artifact(tmp_path):
    x_path = tmp_path / "X_attacked.parquet"
    s_path = tmp_path / "status.parquet"
    m_path = tmp_path / "manifest.json"

    with open(x_path, "wb") as f:
        f.write(b"original")
    with open(s_path, "wb") as f:
        f.write(b"status")

    x_hash = calculate_file_hash(x_path)
    s_hash = calculate_file_hash(s_path)
    manifest = {
        "X_eval_hash": _H, "metadata_eval_hash": _H,
        "evaluation_roles_hash": _H, "evaluation_batches_hash": _H,
        "crafting_identity_hash": _H, "measurement_identity_hash": _H,
        "attacks_yaml_hash": _H, "attack_script_hashes": _SCRIPT_HASHES,
        "frozen_rf_hash": _H, "scaler_hash": _H,
        "feature_names_hash": _H, "feature_mask_hash": _H, "training_bounds_hash": _H,
        "attack_parameters": _PARAMS, "query_budgets": _BUDGETS,
        "attack_scenario": "SilentProbing", "effective_seed": 42,
        "schema_version": "1.0", "row_count": 72000,
        "X_attacked_sha256": x_hash, "status_sha256": s_hash, "output_sha256": x_hash,
    }
    with open(m_path, "w") as f:
        json.dump(manifest, f)

    with open(x_path, "wb") as f:
        f.write(b"corrupted")

    with pytest.raises(ValueError, match="X_attacked hash"):
        validate_cache_manifest(m_path, x_path, {})


def test_validate_cache_manifest_fails_missing_key(tmp_path):
    x_path = tmp_path / "X_attacked.parquet"
    m_path = tmp_path / "manifest.json"
    with open(x_path, "wb") as f:
        f.write(b"data")
    x_hash = calculate_file_hash(x_path)
    # Missing "row_count"
    manifest = {"X_attacked_sha256": x_hash, "output_sha256": x_hash}
    with open(m_path, "w") as f:
        json.dump(manifest, f)
    with pytest.raises(ValueError, match="Missing required key"):
        validate_cache_manifest(m_path, x_path, {})


def test_validate_cache_manifest_fails_corrupted_json(tmp_path):
    x_path = tmp_path / "X_attacked.parquet"
    m_path = tmp_path / "manifest.json"
    with open(x_path, "wb") as f:
        f.write(b"data")
    with open(m_path, "w") as f:
        f.write("{broken json")
    with pytest.raises(ValueError, match="Corrupted JSON"):
        validate_cache_manifest(m_path, x_path, {})


# ─── ConcreteAttackCacheProvider ─────────────────────────────────────────────

def _write_valid_cache(cache_dir: Path, n_rows: int = N):
    feature_names_path = Path(__file__).resolve().parents[1] / "artifacts/preprocessors/feature_mask.json"
    if feature_names_path.exists():
        with open(feature_names_path) as f:
            cols = json.load(f)["feature_columns"]
    else:
        cols = [str(i) for i in range(78)]

    X = np.zeros((n_rows, len(cols)), dtype=np.float32)
    pd.DataFrame(X, columns=cols).to_parquet(cache_dir / "X_attacked.parquet", index=False)

    status_df = pd.DataFrame({
        "eval_position": np.arange(n_rows, dtype=int),
        "eligible": np.zeros(n_rows, dtype=bool),
        "attempted": np.zeros(n_rows, dtype=bool),
        "successful": np.zeros(n_rows, dtype=bool),
        "status_code": ["NOT_APPLICABLE"] * n_rows,
        "queries_used": np.zeros(n_rows, dtype=int),
        "l0": 0.0, "l1": 0.0, "l2": 0.0, "linf": 0.0,
    })
    status_df.to_parquet(cache_dir / "status.parquet", index=False)


def test_concrete_cache_provider_valid(tmp_path):
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    # Must use exactly 500 rows for batch 0 since get_batch_data enforces 500
    _write_valid_cache(cache_dir, 500)

    resolved = make_synthetic_resolved_batches(500)
    
    # Must write manifest to be valid
    x_hash = calculate_file_hash(cache_dir / "X_attacked.parquet")
    s_hash = calculate_file_hash(cache_dir / "status.parquet")
    manifest = {
        "X_eval_hash": _H, "metadata_eval_hash": _H,
        "evaluation_roles_hash": _H, "evaluation_batches_hash": _H,
        "crafting_identity_hash": _H, "measurement_identity_hash": _H,
        "attacks_yaml_hash": _H, "attack_script_hashes": _SCRIPT_HASHES,
        "frozen_rf_hash": _H, "scaler_hash": _H,
        "feature_names_hash": _H, "feature_mask_hash": _H, "training_bounds_hash": _H,
        "controllers_yaml_hash": _H, "defenses_yaml_hash": _H, "experiment_yaml_hash": _H,
        "attack_parameters": _PARAMS, "query_budgets": _BUDGETS,
        "attack_scenario": "SilentProbing", "effective_seed": 42,
        "schema_version": "1.0", "row_count": 500,
        "X_attacked_sha256": x_hash, "status_sha256": s_hash, "output_sha256": x_hash,
    }
    with open(cache_dir / "manifest.json", "w") as f:
        json.dump(manifest, f)
        
    feature_names_path = Path(__file__).resolve().parents[1] / "artifacts/preprocessors/feature_mask.json"
    with open(feature_names_path) as f:
        real_features = json.load(f)["feature_columns"]
        
    with open(cache_dir / "manifest.json") as f:
        full_manifest = json.load(f)

    provider = ConcreteAttackCacheProvider(
        cache_dir, resolved, 
        expected_cache_identity=full_manifest, 
        expected_feature_names=real_features, official_mode=False,
        expected_row_count=500
    )

    data = provider.get_batch_data(0)
    assert data["X_attacked"].shape == (500, 78)
    assert data["eligible"].dtype == bool
    assert data["attempted"].dtype == bool
    assert data["successful"].dtype == bool


def test_concrete_cache_provider_rejects_non_bool_status(tmp_path):
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    _write_valid_cache(cache_dir, 500)

    # Overwrite status with integer eligible column
    status_df = pd.read_parquet(cache_dir / "status.parquet")
    status_df["eligible"] = status_df["eligible"].astype(int)  # int, not bool
    status_df.to_parquet(cache_dir / "status.parquet", index=False)

    resolved = make_synthetic_resolved_batches(500)
    
    x_hash = calculate_file_hash(cache_dir / "X_attacked.parquet")
    s_hash = calculate_file_hash(cache_dir / "status.parquet")
    manifest = {
        "X_eval_hash": _H, "metadata_eval_hash": _H,
        "evaluation_roles_hash": _H, "evaluation_batches_hash": _H,
        "crafting_identity_hash": _H, "measurement_identity_hash": _H,
        "attacks_yaml_hash": _H, "attack_script_hashes": _SCRIPT_HASHES,
        "frozen_rf_hash": _H, "scaler_hash": _H,
        "feature_names_hash": _H, "feature_mask_hash": _H, "training_bounds_hash": _H,
        "controllers_yaml_hash": _H, "defenses_yaml_hash": _H, "experiment_yaml_hash": _H,
        "attack_parameters": _PARAMS, "query_budgets": _BUDGETS,
        "attack_scenario": "SilentProbing", "effective_seed": 42,
        "schema_version": "1.0", "row_count": 500,
        "X_attacked_sha256": x_hash, "status_sha256": s_hash, "output_sha256": x_hash,
    }
    with open(cache_dir / "manifest.json", "w") as f:
        json.dump(manifest, f)
        
    feature_names_path = Path(__file__).resolve().parents[1] / "artifacts/preprocessors/feature_mask.json"
    with open(feature_names_path) as f:
        real_features = json.load(f)["feature_columns"]

    with open(cache_dir / "manifest.json") as f:
        full_manifest = json.load(f)

    with pytest.raises(TypeError, match="strictly bool"):
        ConcreteAttackCacheProvider(
            cache_dir, resolved, 
            expected_cache_identity=full_manifest,
            expected_feature_names=real_features, official_mode=False,
            expected_row_count=500
        )


def test_concrete_cache_provider_rejects_wrong_row_count(tmp_path):
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    _write_valid_cache(cache_dir, 50)  # 50 rows but expecting 100

    resolved = make_synthetic_resolved_batches(N)
    
    x_hash = calculate_file_hash(cache_dir / "X_attacked.parquet")
    s_hash = calculate_file_hash(cache_dir / "status.parquet")
    manifest = {
        "X_eval_hash": _H, "metadata_eval_hash": _H,
        "evaluation_roles_hash": _H, "evaluation_batches_hash": _H,
        "crafting_identity_hash": _H, "measurement_identity_hash": _H,
        "attacks_yaml_hash": _H, "attack_script_hashes": _SCRIPT_HASHES,
        "frozen_rf_hash": _H, "scaler_hash": _H,
        "feature_names_hash": _H, "feature_mask_hash": _H, "training_bounds_hash": _H,
        "controllers_yaml_hash": _H, "defenses_yaml_hash": _H, "experiment_yaml_hash": _H,
        "attack_parameters": _PARAMS, "query_budgets": _BUDGETS,
        "attack_scenario": "SilentProbing", "effective_seed": 42,
        "schema_version": "1.0", "row_count": 50,
        "X_attacked_sha256": x_hash, "status_sha256": s_hash, "output_sha256": x_hash,
    }
    with open(cache_dir / "manifest.json", "w") as f:
        json.dump(manifest, f)
        
    feature_names_path = Path(__file__).resolve().parents[1] / "artifacts/preprocessors/feature_mask.json"
    with open(feature_names_path) as f:
        real_features = json.load(f)["feature_columns"]

    with open(cache_dir / "manifest.json") as f:
        full_manifest = json.load(f)

    with pytest.raises(ValueError, match="row_count must be"):
        ConcreteAttackCacheProvider(
            cache_dir, resolved, 
            expected_cache_identity=full_manifest,
            expected_feature_names=real_features, official_mode=False,
            expected_row_count=N
        )


def test_concrete_cache_provider_rejects_wrong_column_count(tmp_path):
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    # Write with 10 columns instead of 78
    X = np.zeros((N, 10), dtype=np.float32)
    pd.DataFrame(X).to_parquet(cache_dir / "X_attacked.parquet", index=False)
    status_df = pd.DataFrame({
        "eval_position": np.arange(N, dtype=int),
        "eligible": np.ones(N, dtype=bool),
        "attempted": np.zeros(N, dtype=bool),
        "successful": np.zeros(N, dtype=bool),
        "status_code": ["NOT_APPLICABLE"] * N,
        "queries_used": np.zeros(N, dtype=int),
        "l0": 0.0, "l1": 0.0, "l2": 0.0, "linf": 0.0,
    })
    status_df.to_parquet(cache_dir / "status.parquet", index=False)

    resolved = make_synthetic_resolved_batches(N)
    
    x_hash = calculate_file_hash(cache_dir / "X_attacked.parquet")
    s_hash = calculate_file_hash(cache_dir / "status.parquet")
    manifest = {
        "X_eval_hash": _H, "metadata_eval_hash": _H,
        "evaluation_roles_hash": _H, "evaluation_batches_hash": _H,
        "crafting_identity_hash": _H, "measurement_identity_hash": _H,
        "attacks_yaml_hash": _H, "attack_script_hashes": _SCRIPT_HASHES,
        "frozen_rf_hash": _H, "scaler_hash": _H,
        "feature_names_hash": _H, "feature_mask_hash": _H, "training_bounds_hash": _H,
        "controllers_yaml_hash": _H, "defenses_yaml_hash": _H, "experiment_yaml_hash": _H,
        "attack_parameters": _PARAMS, "query_budgets": _BUDGETS,
        "attack_scenario": "SilentProbing", "effective_seed": 42,
        "schema_version": "1.0", "row_count": N,
        "X_attacked_sha256": x_hash, "status_sha256": s_hash, "output_sha256": x_hash,
    }
    with open(cache_dir / "manifest.json", "w") as f:
        json.dump(manifest, f)
        
    feature_names_path = Path(__file__).resolve().parents[1] / "artifacts/preprocessors/feature_mask.json"
    with open(feature_names_path) as f:
        real_features = json.load(f)["feature_columns"]

    with open(cache_dir / "manifest.json") as f:
        full_manifest = json.load(f)

    with pytest.raises(ValueError, match="match expected_feature_names in order"):
        ConcreteAttackCacheProvider(
            cache_dir, resolved, 
            expected_cache_identity=full_manifest,
            expected_feature_names=real_features, official_mode=False,
            expected_row_count=N
        )


def test_concrete_cache_provider_rejects_missing_manifest(tmp_path):
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    _write_valid_cache(cache_dir, 500)
    resolved = make_synthetic_resolved_batches(500)
    # Right now this passes, but it should FAIL if it doesn't find manifest.json
    # The user wants this to be a red test. We expect it to fail (raise FileNotFoundError or ValueError)
    with pytest.raises((FileNotFoundError, ValueError), match="manifest"):
        ConcreteAttackCacheProvider(
            cache_dir, resolved, expected_row_count=500,
            expected_cache_identity=_SYNTH_PROV, expected_feature_names=[str(i) for i in range(78)], official_mode=False
        )

def test_concrete_cache_provider_rejects_mismatched_provenance(tmp_path):
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    _write_valid_cache(cache_dir, 500)
    
    # Write a valid manifest
    m_path = cache_dir / "manifest.json"
    x_hash = calculate_file_hash(cache_dir / "X_attacked.parquet")
    s_hash = calculate_file_hash(cache_dir / "status.parquet")
    manifest = {
        "X_eval_hash": _H, "metadata_eval_hash": _H,
        "evaluation_roles_hash": _H, "evaluation_batches_hash": _H,
        "crafting_identity_hash": _H, "measurement_identity_hash": _H,
        "attacks_yaml_hash": _H, "attack_script_hashes": _SCRIPT_HASHES,
        "frozen_rf_hash": _H, "scaler_hash": _H,
        "feature_names_hash": _H, "feature_mask_hash": _H, "training_bounds_hash": _H,
        "controllers_yaml_hash": _H, "defenses_yaml_hash": _H, "experiment_yaml_hash": _H,
        "attack_parameters": _PARAMS, "query_budgets": _BUDGETS,
        "attack_scenario": "SilentProbing", "effective_seed": 42,
        "schema_version": "1.0", "row_count": 500,
        "X_attacked_sha256": x_hash, "status_sha256": s_hash, "output_sha256": x_hash,
    }
    import json
    with open(m_path, "w") as f:
        json.dump(manifest, f)
        
    resolved = make_synthetic_resolved_batches(500)
    mismatched_prov = dict(_SYNTH_PROV)
    mismatched_prov["frozen_rf_hash"] = "c" * 64
    
    feature_names_path = Path(__file__).resolve().parents[1] / "artifacts/preprocessors/feature_mask.json"
    with open(feature_names_path) as f:
        real_features = json.load(f)["feature_columns"]

    with pytest.raises(ValueError, match="Mismatch"):
        ConcreteAttackCacheProvider(
            cache_dir, resolved, expected_row_count=500,
            expected_cache_identity=mismatched_prov, expected_feature_names=real_features, official_mode=False
        )

def test_concrete_cache_provider_cache_identity_exposure(tmp_path):
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    _write_valid_cache(cache_dir, 500)
    
    m_path = cache_dir / "manifest.json"
    x_hash = calculate_file_hash(cache_dir / "X_attacked.parquet")
    s_hash = calculate_file_hash(cache_dir / "status.parquet")
    manifest = {
        "X_eval_hash": _H, "metadata_eval_hash": _H,
        "evaluation_roles_hash": _H, "evaluation_batches_hash": _H,
        "crafting_identity_hash": _H, "measurement_identity_hash": _H,
        "attacks_yaml_hash": _H, "attack_script_hashes": _SCRIPT_HASHES,
        "frozen_rf_hash": _H, "scaler_hash": _H,
        "feature_names_hash": _H, "feature_mask_hash": _H, "training_bounds_hash": _H,
        "controllers_yaml_hash": _H, "defenses_yaml_hash": _H, "experiment_yaml_hash": _H,
        "attack_parameters": _PARAMS, "query_budgets": _BUDGETS,
        "attack_scenario": "SilentProbing", "effective_seed": 42,
        "schema_version": "1.0", "row_count": 500,
        "X_attacked_sha256": x_hash, "status_sha256": s_hash, "output_sha256": x_hash,
    }
    import json
    with open(m_path, "w") as f:
        json.dump(manifest, f)
        
    resolved = make_synthetic_resolved_batches(500)
    feature_names_path = Path(__file__).resolve().parents[1] / "artifacts/preprocessors/feature_mask.json"
    with open(feature_names_path) as f:
        real_features = json.load(f)["feature_columns"]

    provider = ConcreteAttackCacheProvider(
        cache_dir, resolved, expected_row_count=500,
        expected_cache_identity=manifest, expected_feature_names=real_features
    )
    
    assert provider.cache_identity == manifest
    batch_data = provider.get_batch_data(0)
    assert "status_codes" in batch_data
    assert "l0" in batch_data



# ─── AttackCacheBuilder round-trip ────────────────────────────────────────────

def _make_builder(seed=42):
    import json as _j
    from pathlib import Path as _P
    root = _P(__file__).parents[1]
    with open(root / "artifacts/preprocessors/feature_mask.json") as _f:
        mask_data = _j.load(_f)
    import pandas as pd
    bounds = pd.read_parquet(root / "artifacts/preprocessors/training_bounds.parquet")
    return AttackCacheBuilder(
        feature_names=mask_data["feature_columns"],
        modifiable_mask=mask_data["feature_mask"],
        training_bounds=bounds,
        provenance_hashes=_SYNTH_PROV,
        seed=seed,
    )


def test_cache_builder_silent_probing_round_trip(tmp_path):
    builder = _make_builder()
    rng = np.random.RandomState(42)
    X = rng.rand(N, 78).astype(np.float32)
    y = rng.randint(0, 2, N)
    eps = np.arange(N)

    out_dir = builder.build(
        scenario=AttackCacheBuilder.SCENARIO_SILENT_PROBING,
        X_measurement=X, y_measurement=y, eval_positions=eps,
        output_dir=tmp_path / "cache_sp",
    )

    # X_attacked must be identical to X_input for silent probing
    df_x = pd.read_parquet(out_dir / "X_attacked.parquet")
    assert np.allclose(df_x.values.astype(np.float32), X)

    # All status must be NOT_APPLICABLE
    df_s = pd.read_parquet(out_dir / "status.parquet")
    assert (df_s["status_code"] == "NOT_APPLICABLE").all()
    assert not df_s["eligible"].any()
    assert not df_s["attempted"].any()
    assert df_s["queries_used"].sum() == 0

    # Manifest exists and has both hash fields
    with open(out_dir / "manifest.json") as f:
        manifest = json.load(f)
    assert "X_attacked_sha256" in manifest
    assert "status_sha256" in manifest
    assert manifest["attack_scenario"] == AttackCacheBuilder.SCENARIO_SILENT_PROBING


def test_cache_builder_silent_probing_determinism(tmp_path):
    builder = _make_builder()
    rng = np.random.RandomState(7)
    X = rng.rand(N, 78).astype(np.float32)
    y = np.zeros(N, dtype=int)
    eps = np.arange(N)

    out1 = builder.build(
        scenario=AttackCacheBuilder.SCENARIO_SILENT_PROBING,
        X_measurement=X, y_measurement=y, eval_positions=eps,
        output_dir=tmp_path / "det1",
    )
    out2 = builder.build(
        scenario=AttackCacheBuilder.SCENARIO_SILENT_PROBING,
        X_measurement=X, y_measurement=y, eval_positions=eps,
        output_dir=tmp_path / "det2",
    )
    h1 = calculate_file_hash(out1 / "X_attacked.parquet")
    h2 = calculate_file_hash(out2 / "X_attacked.parquet")
    assert h1 == h2


def test_cache_builder_provider_integration(tmp_path):
    """Build cache → load via provider → verify batch alignment.
    Uses 500-row synthetic dataset so get_batch_data(0) works correctly.
    """
    builder = _make_builder()
    rng = np.random.RandomState(0)
    N500 = 500
    X = rng.rand(N500, 78).astype(np.float32)
    y = np.zeros(N500, dtype=int)
    eps = np.arange(N500)  # eval_positions 0..499

    out_dir = builder.build(
        scenario=AttackCacheBuilder.SCENARIO_SILENT_PROBING,
        X_measurement=X, y_measurement=y, eval_positions=eps,
        output_dir=tmp_path / "cache",
    )

    resolved = make_synthetic_resolved_batches(N500)
    with open(out_dir / "manifest.json") as f:
        full_manifest = json.load(f)

    provider = ConcreteAttackCacheProvider(
        out_dir, resolved, 
        expected_cache_identity=full_manifest,
        expected_feature_names=builder.feature_names, official_mode=False,
        expected_row_count=N500
    )

    batch_data = provider.get_batch_data(0)
    assert batch_data["X_attacked"].shape == (N500, 78)
    # eval_position-based alignment: sorted by eval_position, should match X
    assert np.allclose(batch_data["X_attacked"], X)


def test_cache_builder_rejects_placeholder_provenance():
    import json as _j
    from pathlib import Path as _P
    root = _P(__file__).parents[1]
    with open(root / "artifacts/preprocessors/feature_mask.json") as _f:
        mask_data = _j.load(_f)
    import pandas as pd
    bounds = pd.read_parquet(root / "artifacts/preprocessors/training_bounds.parquet")
    bad_prov = {"dummy": "placeholder"}
    with pytest.raises(ValueError, match="placeholder"):
        AttackCacheBuilder(
            feature_names=mask_data["feature_columns"],
            modifiable_mask=mask_data["feature_mask"],
            training_bounds=bounds,
            provenance_hashes=bad_prov,
            seed=42,
        )

class MockSurrogateAttack:
    def fit_surrogate(self, X, y):
        self.X_fit = X
        self.y_fit = y
        
    def generate_candidate(self, x):
        return x + 1.0, {"l0": 1.0, "l1": 1.0, "l2": 1.0, "linf": 1.0}

    def evaluate_transfer(self, X_cand, X_orig, oracle, sample_id, true_label):
        oracle.predict(X_cand.reshape(1, -1), sample_ids=[sample_id])
        from recall_aware_ids.attacks.base import AttackResult
        return AttackResult(
            sample_id=sample_id, X_adv=X_cand, eligible=True, attempted=True, success=True,
            status_code="SUCCESS", message="Mock success", query_count=1,
            magnitudes={"l0": 1.0, "l1": 1.0, "l2": 1.0, "linf": 1.0}
        )

class MockBoundaryAttack:
    def generate(self, x, oracle, sample_id, true_label, reference_pool):
        oracle.predict(x.reshape(1, -1), sample_ids=[sample_id])
        from recall_aware_ids.attacks.base import AttackResult
        return AttackResult(
            sample_id=sample_id, X_adv=x + 2.0, eligible=True, attempted=True, success=True,
            status_code="SUCCESS", message="Mock success", query_count=1,
            magnitudes={"l0": 2.0, "l1": 2.0, "l2": 2.0, "linf": 2.0}
        )

def test_cache_builder_surrogate_semantics(tmp_path):
    builder = _make_builder()
    rng = np.random.RandomState(0)
    X_meas = rng.rand(10, 78).astype(np.float32)
    y_meas = np.ones(10, dtype=int)
    eps = np.arange(10)
    
    X_craft = rng.rand(5, 78).astype(np.float32)
    y_craft = np.ones(5, dtype=int)
    
    def mock_predict(x, **kwargs):
        # Always return 1 to make them eligible, except for candidate evaluation
        return np.ones(len(x), dtype=int)
        
    surrogate = MockSurrogateAttack()
    
    out_dir = builder.build(
        scenario=AttackCacheBuilder.SCENARIO_SURROGATE,
        X_measurement=X_meas, y_measurement=y_meas, eval_positions=eps,
        output_dir=tmp_path / "surrogate_cache",
        X_crafting=X_craft, y_crafting=y_craft,
        oracle_or_predict_fn=mock_predict,
        surrogate_attack=surrogate,
        attack_script_hashes=_SCRIPT_HASHES,
        attack_parameters=_PARAMS,
        query_budgets=_BUDGETS,
    )
    
    # Prove it fits only on crafting pool
    assert np.allclose(surrogate.X_fit, X_craft)
    
    import pandas as pd
    status = pd.read_parquet(out_dir / "status.parquet")
    # All 10 attempted because they were eligible
    assert status["attempted"].all()
    # Oracle queries tracked per candidate eval = 1
    assert (status["queries_used"] == 1).all()


def test_cache_builder_boundary_semantics(tmp_path):
    builder = _make_builder()
    rng = np.random.RandomState(0)
    X_meas = rng.rand(100, 78).astype(np.float32)
    y_meas = np.ones(100, dtype=int)
    eps = np.arange(100)
    
    def mock_predict(x, **kwargs):
        return np.ones(len(x), dtype=int)
        
    boundary = MockBoundaryAttack()
    
    out_dir = builder.build(
        scenario=AttackCacheBuilder.SCENARIO_BOUNDARY,
        X_measurement=X_meas, y_measurement=y_meas, eval_positions=eps,
        output_dir=tmp_path / "boundary_cache",
        oracle_or_predict_fn=mock_predict,
        boundary_attack=boundary,
        benign_reference_pool=X_meas,
        n_boundary_targets=20,
        max_queries=50,
        attack_script_hashes=_SCRIPT_HASHES,
        attack_parameters=_PARAMS,
        query_budgets=_BUDGETS,
    )
    
    import pandas as pd
    status = pd.read_parquet(out_dir / "status.parquet")
    # Exactly 20 attempted
    assert status["attempted"].sum() == 20
    assert status["eligible"].sum() == 100
    # Unattempted have 0 queries
    assert status.loc[~status["attempted"], "queries_used"].sum() == 0
    # Attempted have 1 query from our mock
    assert status.loc[status["attempted"], "queries_used"].sum() == 20

def test_cache_builder_atomic_write(tmp_path):
    builder = _make_builder()
    out_dir = tmp_path / "cache"
    out_dir.mkdir()
    
    rng = np.random.RandomState(0)
    X_meas = rng.rand(10, 78).astype(np.float32)
    y_meas = np.ones(10, dtype=int)
    eps = np.arange(10)
    
    with pytest.raises(FileExistsError):
        builder.build(
            scenario=AttackCacheBuilder.SCENARIO_SILENT_PROBING,
            X_measurement=X_meas, y_measurement=y_meas, eval_positions=eps,
            output_dir=out_dir
        )
