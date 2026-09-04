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

# 64-char hex used for provenance in tests
_H = "b" * 64

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
        "attacks_yaml_hash": _H, "attack_script_hashes": {},
        "frozen_rf_hash": _H, "scaler_hash": _H,
        "feature_names_hash": _H, "feature_mask_hash": _H, "training_bounds_hash": _H,
        "attack_parameters": {}, "query_budgets": {},
        "attack_scenario": "SilentProbing", "effective_seed": 42,
        "schema_version": "1.0", "row_count": 72000,
        "X_attacked_sha256": x_hash, "status_sha256": s_hash, "output_sha256": x_hash,
    }
    with open(m_path, "w") as f:
        json.dump(manifest, f)

    assert validate_cache_manifest(m_path, x_path, {}, status_path=s_path) is True


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
        "attacks_yaml_hash": _H, "attack_script_hashes": {},
        "frozen_rf_hash": _H, "scaler_hash": _H,
        "feature_names_hash": _H, "feature_mask_hash": _H, "training_bounds_hash": _H,
        "attack_parameters": {}, "query_budgets": {},
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
    X = np.zeros((n_rows, 78), dtype=np.float32)
    pd.DataFrame(X).to_parquet(cache_dir / "X_attacked.parquet", index=False)

    status_df = pd.DataFrame({
        "eval_position": np.arange(n_rows, dtype=int),
        "eligible": np.ones(n_rows, dtype=bool),
        "attempted": np.ones(n_rows, dtype=bool),
        "successful": np.zeros(n_rows, dtype=bool),
        "status_code": ["TARGET_REJECTION"] * n_rows,
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
    provider = ConcreteAttackCacheProvider(cache_dir, resolved, expected_row_count=500)

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
    with pytest.raises(TypeError, match="strictly bool"):
        ConcreteAttackCacheProvider(cache_dir, resolved, expected_row_count=500)


def test_concrete_cache_provider_rejects_wrong_row_count(tmp_path):
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    _write_valid_cache(cache_dir, 50)  # 50 rows but expecting 100

    resolved = make_synthetic_resolved_batches(N)
    with pytest.raises(ValueError, match="rows"):
        ConcreteAttackCacheProvider(cache_dir, resolved, expected_row_count=N)


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
    with pytest.raises(ValueError, match="columns"):
        ConcreteAttackCacheProvider(cache_dir, resolved, expected_row_count=N)


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
        X_input=X, y_input=y, eval_positions=eps,
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
        X_input=X, y_input=y, eval_positions=eps,
        output_dir=tmp_path / "det1",
    )
    out2 = builder.build(
        scenario=AttackCacheBuilder.SCENARIO_SILENT_PROBING,
        X_input=X, y_input=y, eval_positions=eps,
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
        X_input=X, y_input=y, eval_positions=eps,
        output_dir=tmp_path / "cache",
    )

    resolved = make_synthetic_resolved_batches(N500)
    provider = ConcreteAttackCacheProvider(out_dir, resolved, expected_row_count=N500)

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
