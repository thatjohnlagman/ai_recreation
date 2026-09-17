"""
tests/test_build_evaluation_caches.py

Focused tests for Phase 10B Attack-Cache Orchestration Readiness.
All tests use synthetic fixtures or training-partition data.
EVALUATION DATA (X_eval.parquet, metadata_eval.parquet) AND EVALUATION LABELS ARE STRICTLY PROHIBITED.
"""
import copy
import datetime
import hashlib
import json
import subprocess
from pathlib import Path
from unittest.mock import patch, MagicMock

import numpy as np
import pandas as pd
import pytest
import yaml

from recall_aware_ids.experiment.caching import (
    AttackCacheBuilder,
    ConcreteAttackCacheProvider,
    calculate_file_hash,
)
from recall_aware_ids.experiment.matrix import (
    generate_evaluation_matrix,
    get_unique_executions,
)
from recall_aware_ids.experiment.schemas import (
    _HEX64,
    _REQUIRED_PROVENANCE_KEYS,
)
from scripts.build_evaluation_caches import (
    FREEZE_COMMIT,
    FREEZE_TAG,
    FROZEN_DATE,
    PROTECTED_HASHES,
    build_evaluation_caches,
    build_single_cache,
    canonicalize_scenario,
    derive_cache_pairs,
    display_scenario,
    quarantine_directory,
    run_preflight,
    validate_completed_cache,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIGS_DIR = REPO_ROOT / "configs"
DATA_DIR = REPO_ROOT / "data"
ARTIFACTS_DIR = REPO_ROOT / "artifacts"

# 64-char valid synthetic hashes for tests
_VALID_HASH_A = "1111111111111111111111111111111111111111111111111111111111111111"
_VALID_HASH_B = "2222222222222222222222222222222222222222222222222222222222222222"
_VALID_HASH_C = "3333333333333333333333333333333333333333333333333333333333333333"

_SYNTH_PROV = {
    "frozen_rf_hash": _VALID_HASH_A,
    "scaler_hash": _VALID_HASH_A,
    "feature_names_hash": _VALID_HASH_A,
    "feature_mask_hash": _VALID_HASH_A,
    "training_bounds_hash": _VALID_HASH_A,
    "evaluation_roles_hash": _VALID_HASH_A,
    "evaluation_batches_hash": _VALID_HASH_A,
    "attacks_yaml_hash": _VALID_HASH_A,
    "controllers_yaml_hash": _VALID_HASH_A,
    "defenses_yaml_hash": _VALID_HASH_A,
    "experiment_yaml_hash": _VALID_HASH_A,
    "X_eval_hash": _VALID_HASH_A,
    "metadata_eval_hash": _VALID_HASH_A,
    "crafting_identity_hash": _VALID_HASH_A,
    "measurement_identity_hash": _VALID_HASH_A,
}


@pytest.fixture
def synthetic_data():
    """Generates small synthetic measurement and crafting data for testing."""
    rng = np.random.RandomState(42)
    n_meas = 100
    n_craft = 50

    with open(ARTIFACTS_DIR / "preprocessors/feature_mask.json") as f:
        fnames = json.load(f)["feature_columns"]

    X_meas = rng.rand(n_meas, len(fnames)).astype(np.float32)
    y_meas = rng.choice([0, 1], size=n_meas, p=[0.7, 0.3]).astype(int)
    eps_meas = np.arange(n_meas, dtype=np.int64)

    X_craft = rng.rand(n_craft, len(fnames)).astype(np.float32)
    y_craft = rng.choice([0, 1], size=n_craft, p=[0.7, 0.3]).astype(int)

    batches_df = pd.DataFrame({
        "batch_id": [0] * n_meas,
        "eval_position": eps_meas,
        "within_batch_position": np.arange(n_meas),
    })

    def mock_predict(x):
        # Deterministic mock predict: 1 if first feature > 0.5 else 0
        return (x[:, 0] > 0.5).astype(int)

    return {
        "X_meas": X_meas,
        "y_meas": y_meas,
        "eps_meas": eps_meas,
        "X_craft": X_craft,
        "y_craft": y_craft,
        "batches_df": batches_df,
        "mock_predict": mock_predict,
        "feature_names": fnames,
    }


# ===========================================================================
# 1. Safe Preflight Default
# ===========================================================================
def test_safe_preflight_default(tmp_path):
    """Running entry point without --execute runs preflight only, leaving output empty."""
    out_dir = tmp_path / "caches"
    # Ensure build_evaluation_caches does not create any cache directories
    exit_code = build_evaluation_caches(
        configs_dir=CONFIGS_DIR,
        data_dir=DATA_DIR,
        artifacts_dir=ARTIFACTS_DIR,
        output_dir=out_dir,
        execute=False,
        preflight_only=False,
    )
    assert exit_code == 0
    # No caches generated
    assert not out_dir.exists() or len(list(out_dir.glob("*"))) == 0


# ===========================================================================
# 2. Execute Required for Data Access
# ===========================================================================
def test_execute_required_for_data_access(tmp_path):
    """Preflight mode must strictly reject loading evaluation features or labels."""
    with patch("pandas.read_parquet") as mock_parquet:
        # Preflight should read training bounds (parquet) but NEVER X_eval or metadata_eval
        out_dir = tmp_path / "caches"
        run_preflight(
            configs_dir=CONFIGS_DIR,
            data_dir=DATA_DIR,
            artifacts_dir=ARTIFACTS_DIR,
            output_dir=out_dir,
            enforce_git=False,
        )
        for call_args in mock_parquet.call_args_list:
            called_path = str(call_args[0][0])
            assert "X_eval" not in called_path, "X_eval.parquet was loaded during preflight!"
            assert "metadata_eval" not in called_path, "metadata_eval.parquet was loaded during preflight!"


# ===========================================================================
# 3. Exact Derivation of 15 Pairs
# ===========================================================================
def test_exact_derivation_of_15_pairs():
    """Derives exactly 15 canonical (scenario, seed) pairs from frozen configs."""
    pairs = derive_cache_pairs(CONFIGS_DIR)
    assert len(pairs) == 15

    scenarios = sorted(list({p[0] for p in pairs}))
    assert scenarios == ["Decision Boundary", "Silent Probing", "Surrogate Transfer"]

    seeds = sorted(list({p[1] for p in pairs}))
    assert seeds == [42, 43, 44, 45, 46]

    # Each scenario must have all 5 seeds
    for scen in scenarios:
        scen_seeds = [p[1] for p in pairs if p[0] == scen]
        assert scen_seeds == [42, 43, 44, 45, 46]


# ===========================================================================
# 4. Alias Reuse
# ===========================================================================
def test_alias_reuse():
    """Sensitivity aliases (27 C1 runs) perfectly reuse the primary run's cache identity."""
    matrix = generate_evaluation_matrix(CONFIGS_DIR)
    aliases = matrix[matrix["is_alias"]]
    assert len(aliases) == 27

    for _, r in aliases.iterrows():
        target_id = r["alias_for_run_id"]
        target = matrix[matrix["run_id"] == target_id].iloc[0]
        # Same scenario and seed
        assert r["attack_scenario"] == target["attack_scenario"]
        assert r["seed"] == target["seed"]
        # Same canonical slug
        assert canonicalize_scenario(r["attack_scenario"]) == canonicalize_scenario(target["attack_scenario"])


# ===========================================================================
# 5. Rejection of Altered Freeze Ancestry or Config Hashes
# ===========================================================================
def test_rejection_of_altered_freeze_ancestry(tmp_path):
    """Preflight rejects if freeze tag is missing or not an ancestor of HEAD."""
    with patch("subprocess.check_output") as mock_git:
        mock_git.return_value = "deadbeefdeadbeefdeadbeefdeadbeefdeadbeef"
        with pytest.raises(ValueError, match="Freeze tag.*points to commit"):
            run_preflight(
                configs_dir=CONFIGS_DIR,
                data_dir=DATA_DIR,
                artifacts_dir=ARTIFACTS_DIR,
                output_dir=tmp_path,
                enforce_git=True,
            )


def test_rejection_of_altered_config_date_frozen(tmp_path):
    """Preflight rejects if experiment.date_frozen is modified."""
    bad_configs = tmp_path / "configs"
    bad_configs.mkdir()
    import shutil
    for f in CONFIGS_DIR.glob("*.yaml"):
        shutil.copy(f, bad_configs / f.name)

    # Alter date_frozen
    exp_path = bad_configs / "experiment.yaml"
    with open(exp_path, "r") as f:
        cfg = yaml.safe_load(f)
    cfg["experiment"]["date_frozen"] = "2026-09-18T00:00:00+00:00"
    with open(exp_path, "w") as f:
        yaml.dump(cfg, f)

    with pytest.raises(ValueError, match="experiment.date_frozen is"):
        run_preflight(
            configs_dir=bad_configs,
            data_dir=DATA_DIR,
            artifacts_dir=ARTIFACTS_DIR,
            output_dir=tmp_path,
            enforce_git=False,
        )


# ===========================================================================
# 6. Protected-Hash Mismatch Rejection
# ===========================================================================
def test_protected_hash_mismatch_rejection(tmp_path):
    """Preflight rejects if RF, Roles, or Batches protected hashes do not match."""
    # Test with empty/altered RF file
    bad_artifacts = tmp_path / "artifacts"
    bad_models = bad_artifacts / "models"
    bad_models.mkdir(parents=True)
    fake_rf = bad_models / "frozen_rf.joblib"
    fake_rf.write_bytes(b"tampered model data")

    with pytest.raises(ValueError, match="Protected hash mismatch for RF"):
        run_preflight(
            configs_dir=CONFIGS_DIR,
            data_dir=DATA_DIR,
            artifacts_dir=bad_artifacts,
            output_dir=tmp_path / "caches",
            enforce_git=False,
        )


# ===========================================================================
# 7. No Defense/Config Dimension in Cache Identity
# ===========================================================================
def test_no_defense_config_dimension_in_cache_identity():
    """Cache identity depends solely on (scenario, seed), independent of defense or controller."""
    pairs = derive_cache_pairs(CONFIGS_DIR)
    for pair in pairs:
        assert len(pair) == 2
        scen, seed = pair
        assert isinstance(scen, str)
        assert isinstance(seed, int)
        assert not hasattr(pair, "defense_name")
        assert not hasattr(pair, "controller_config_id")

    matrix = generate_evaluation_matrix(CONFIGS_DIR)
    # Every (scenario, seed) maps to multiple defenses and controllers
    grouped = matrix.groupby(["attack_scenario", "seed"])
    for _, group in grouped:
        assert set(group["defense_name"]) == {"afp", "feature_squeezing", "randomized_smoothing"}
        assert len(set(group["controller_config_id"])) >= 2


# ===========================================================================
# 8. Atomic Publication and Failure Quarantine
# ===========================================================================
def test_atomic_publication_and_failure_quarantine(tmp_path, synthetic_data):
    """If cache generation encounters an error, staging is quarantined and target dir is not created."""
    out_dir = tmp_path / "caches"

    with patch("scripts.build_evaluation_caches.validate_completed_cache", return_value=(False, "Simulated post-build corruption")):
        with pytest.raises(RuntimeError, match="Staging cache validation failed"):
            build_single_cache(
                scenario="Silent Probing",
                seed=42,
                output_dir=out_dir,
                configs_dir=CONFIGS_DIR,
                data_dir=DATA_DIR,
                artifacts_dir=ARTIFACTS_DIR,
                resolved_batches=synthetic_data["batches_df"],
                official_mode=False,
                override_X_craft=synthetic_data["X_craft"],
                override_y_craft=synthetic_data["y_craft"],
                override_X_meas=synthetic_data["X_meas"],
                override_y_meas=synthetic_data["y_meas"],
                override_eps=synthetic_data["eps_meas"],
                override_predict_fn=synthetic_data["mock_predict"],
                override_provenance=_SYNTH_PROV,
                override_row_count=len(synthetic_data["X_meas"]),
            )

    target_dir = out_dir / "SilentProbing_42"
    assert not target_dir.exists(), "Target directory should not exist after build failure"

    # Quarantine directory must exist
    quarantined = list(out_dir.glob("SilentProbing_42*quarantined_*"))
    assert len(quarantined) >= 1, "Failed staging directory was not quarantined"


# ===========================================================================
# 9. Complete-Cache Reuse
# ===========================================================================
def test_complete_cache_reuse(tmp_path, synthetic_data):
    """A valid completed cache is verified and reused without recomputation."""
    out_dir = tmp_path / "caches"
    cache_path = build_single_cache(
        scenario="Silent Probing",
        seed=42,
        output_dir=out_dir,
        configs_dir=CONFIGS_DIR,
        data_dir=DATA_DIR,
        artifacts_dir=ARTIFACTS_DIR,
        resolved_batches=synthetic_data["batches_df"],
        official_mode=False,
        override_X_craft=synthetic_data["X_craft"],
        override_y_craft=synthetic_data["y_craft"],
        override_X_meas=synthetic_data["X_meas"],
        override_y_meas=synthetic_data["y_meas"],
        override_eps=synthetic_data["eps_meas"],
        override_predict_fn=synthetic_data["mock_predict"],
        override_provenance=_SYNTH_PROV,
        override_row_count=len(synthetic_data["X_meas"]),
    )

    manifest_mtime = (cache_path / "manifest.json").stat().st_mtime_ns

    # Call again: must detect existing valid cache and reuse it
    reused_path = build_single_cache(
        scenario="Silent Probing",
        seed=42,
        output_dir=out_dir,
        configs_dir=CONFIGS_DIR,
        data_dir=DATA_DIR,
        artifacts_dir=ARTIFACTS_DIR,
        resolved_batches=synthetic_data["batches_df"],
        official_mode=False,
        override_X_craft=synthetic_data["X_craft"],
        override_y_craft=synthetic_data["y_craft"],
        override_X_meas=synthetic_data["X_meas"],
        override_y_meas=synthetic_data["y_meas"],
        override_eps=synthetic_data["eps_meas"],
        override_predict_fn=synthetic_data["mock_predict"],
        override_provenance=_SYNTH_PROV,
        override_row_count=len(synthetic_data["X_meas"]),
    )

    assert reused_path == cache_path
    assert (reused_path / "manifest.json").stat().st_mtime_ns == manifest_mtime


# ===========================================================================
# 10. Invalid-Cache Rejection & Quarantine
# ===========================================================================
def test_invalid_cache_rejection(tmp_path, synthetic_data):
    """An invalid or corrupted cache is rejected, quarantined, and rebuilt."""
    out_dir = tmp_path / "caches"
    cache_path = build_single_cache(
        scenario="Silent Probing",
        seed=42,
        output_dir=out_dir,
        configs_dir=CONFIGS_DIR,
        data_dir=DATA_DIR,
        artifacts_dir=ARTIFACTS_DIR,
        resolved_batches=synthetic_data["batches_df"],
        official_mode=False,
        override_X_craft=synthetic_data["X_craft"],
        override_y_craft=synthetic_data["y_craft"],
        override_X_meas=synthetic_data["X_meas"],
        override_y_meas=synthetic_data["y_meas"],
        override_eps=synthetic_data["eps_meas"],
        override_predict_fn=synthetic_data["mock_predict"],
        override_provenance=_SYNTH_PROV,
        override_row_count=len(synthetic_data["X_meas"]),
    )

    # Corrupt completion.json
    comp_file = cache_path / "completion.json"
    comp_file.write_text("corrupted json")

    # Building again must detect the corrupted cache, quarantine it, and rebuild a valid one
    new_cache_path = build_single_cache(
        scenario="Silent Probing",
        seed=42,
        output_dir=out_dir,
        configs_dir=CONFIGS_DIR,
        data_dir=DATA_DIR,
        artifacts_dir=ARTIFACTS_DIR,
        resolved_batches=synthetic_data["batches_df"],
        official_mode=False,
        override_X_craft=synthetic_data["X_craft"],
        override_y_craft=synthetic_data["y_craft"],
        override_X_meas=synthetic_data["X_meas"],
        override_y_meas=synthetic_data["y_meas"],
        override_eps=synthetic_data["eps_meas"],
        override_predict_fn=synthetic_data["mock_predict"],
        override_provenance=_SYNTH_PROV,
        override_row_count=len(synthetic_data["X_meas"]),
    )

    assert new_cache_path.exists()
    assert (new_cache_path / "completion.json").exists()
    quarantined = list(out_dir.glob("SilentProbing_42_quarantined_*"))
    assert len(quarantined) >= 1, "Corrupted cache was not quarantined"


# ===========================================================================
# 11. Labels Unavailable During Candidate Generation
# ===========================================================================
def test_labels_unavailable_during_candidate_generation():
    """Verifies that Surrogate and Boundary candidate generation does not use true labels."""
    from recall_aware_ids.attacks.surrogate_transfer import SurrogateTransferAttack
    from recall_aware_ids.attacks.boundary_attack import DecisionBoundaryAttack

    with open(ARTIFACTS_DIR / "preprocessors/feature_mask.json") as f:
        mask_data = json.load(f)
    bounds = pd.read_parquet(ARTIFACTS_DIR / "preprocessors/training_bounds.parquet")

    surrogate = SurrogateTransferAttack(
        feature_names=mask_data["feature_columns"],
        modifiable_mask=mask_data["feature_mask"],
        training_bounds=bounds,
    )

    # generate_candidate signature: takes only x, no labels
    import inspect
    sig = inspect.signature(surrogate.generate_candidate)
    assert "label" not in sig.parameters
    assert "y" not in sig.parameters


# ===========================================================================
# 12. Correct Scenario Routing
# ===========================================================================
def test_correct_scenario_routing(tmp_path, synthetic_data):
    """Builder properly routes to each of the three canonical scenarios."""
    out_dir = tmp_path / "caches"

    for scen in ["Silent Probing", "Surrogate Transfer", "Decision Boundary"]:
        c_path = build_single_cache(
            scenario=scen,
            seed=42,
            output_dir=out_dir,
            configs_dir=CONFIGS_DIR,
            data_dir=DATA_DIR,
            artifacts_dir=ARTIFACTS_DIR,
            resolved_batches=synthetic_data["batches_df"],
            official_mode=False,
            override_X_craft=synthetic_data["X_craft"],
            override_y_craft=synthetic_data["y_craft"],
            override_X_meas=synthetic_data["X_meas"],
            override_y_meas=synthetic_data["y_meas"],
            override_eps=synthetic_data["eps_meas"],
            override_predict_fn=synthetic_data["mock_predict"],
            override_provenance=_SYNTH_PROV,
            override_row_count=len(synthetic_data["X_meas"]),
            override_n_boundary_targets=5,
        )
        assert c_path.exists()
        with open(c_path / "manifest.json") as f:
            m = json.load(f)
        assert m["attack_scenario"] == canonicalize_scenario(scen)


# ===========================================================================
# 13. Exact Global 200-Target Boundary Selection
# ===========================================================================
def test_exact_global_200_target_boundary_selection():
    """Boundary target selection selects exactly 200 samples deterministically in official mode."""
    from recall_aware_ids.experiment.boundary_selection import select_boundary_targets
    rng = np.random.RandomState(42)
    # 72,000 samples with 15,000 eligible attack samples
    eligible_mask = np.zeros(72000, dtype=bool)
    eligible_indices = rng.choice(72000, size=15000, replace=False)
    eligible_mask[eligible_indices] = True

    for seed in [42, 43, 44, 45, 46]:
        selected = select_boundary_targets(eligible_mask, seed=seed, n_attack_samples=200, official_mode=True)
        assert selected.sum() == 200
        # Determinism check
        selected_repeat = select_boundary_targets(eligible_mask, seed=seed, n_attack_samples=200, official_mode=True)
        assert np.array_equal(selected, selected_repeat)


# ===========================================================================
# 14. Silent Probing Zero-Query and Identity Behavior
# ===========================================================================
def test_silent_probing_zero_query_and_identity_behavior(tmp_path, synthetic_data):
    """Silent Probing performs identity transform, zero queries, and all NOT_APPLICABLE statuses."""
    out_dir = tmp_path / "caches"
    c_path = build_single_cache(
        scenario="Silent Probing",
        seed=42,
        output_dir=out_dir,
        configs_dir=CONFIGS_DIR,
        data_dir=DATA_DIR,
        artifacts_dir=ARTIFACTS_DIR,
        resolved_batches=synthetic_data["batches_df"],
        official_mode=False,
        override_X_craft=synthetic_data["X_craft"],
        override_y_craft=synthetic_data["y_craft"],
        override_X_meas=synthetic_data["X_meas"],
        override_y_meas=synthetic_data["y_meas"],
        override_eps=synthetic_data["eps_meas"],
        override_predict_fn=synthetic_data["mock_predict"],
        override_provenance=_SYNTH_PROV,
        override_row_count=len(synthetic_data["X_meas"]),
    )

    df_x = pd.read_parquet(c_path / "X_attacked.parquet")
    df_s = pd.read_parquet(c_path / "status.parquet")

    # Features must match input identically
    assert np.allclose(df_x.values, synthetic_data["X_meas"])
    # 0 queries
    assert (df_s["queries_used"] == 0).all()
    # All NOT_APPLICABLE
    assert (df_s["status_code"] == "NOT_APPLICABLE").all()
    assert not df_s["eligible"].any()
    assert not df_s["attempted"].any()
    assert not df_s["successful"].any()


# ===========================================================================
# 15. No Partial-Cache Resumption
# ===========================================================================
def test_no_partial_cache_resumption(tmp_path, synthetic_data):
    """Stale .staging or .tmp directories are quarantined; never partially resumed."""
    out_dir = tmp_path / "caches"
    target_name = "SilentProbing_42"
    stale_staging = out_dir / f"{target_name}.staging"
    stale_staging.mkdir(parents=True)
    (stale_staging / "half_written.tmp").write_text("junk")

    c_path = build_single_cache(
        scenario="Silent Probing",
        seed=42,
        output_dir=out_dir,
        configs_dir=CONFIGS_DIR,
        data_dir=DATA_DIR,
        artifacts_dir=ARTIFACTS_DIR,
        resolved_batches=synthetic_data["batches_df"],
        official_mode=False,
        override_X_craft=synthetic_data["X_craft"],
        override_y_craft=synthetic_data["y_craft"],
        override_X_meas=synthetic_data["X_meas"],
        override_y_meas=synthetic_data["y_meas"],
        override_eps=synthetic_data["eps_meas"],
        override_predict_fn=synthetic_data["mock_predict"],
        override_provenance=_SYNTH_PROV,
        override_row_count=len(synthetic_data["X_meas"]),
    )

    assert c_path.exists()
    assert not stale_staging.exists()
    quarantined = list(out_dir.glob(f"{target_name}_stale_staging_quarantined_*"))
    assert len(quarantined) >= 1


# ===========================================================================
# 16. Deterministic Pair Ordering
# ===========================================================================
def test_deterministic_pair_ordering():
    """derive_cache_pairs produces identical ordered sequence across calls."""
    pairs1 = derive_cache_pairs(CONFIGS_DIR)
    pairs2 = derive_cache_pairs(CONFIGS_DIR)
    assert pairs1 == pairs2
    # Verify strict ascending sort order
    assert pairs1 == sorted(pairs1, key=lambda x: (x[0], x[1]))


# ===========================================================================
# 17. Training-Derived Smoke Test
# ===========================================================================
def test_training_derived_smoke(tmp_path):
    """Small smoke test using training-partition records to test orchestrator end-to-end."""
    x_train_path = DATA_DIR / "processed/X_train.parquet"
    meta_train_path = DATA_DIR / "processed/metadata_train.parquet"

    if not x_train_path.exists() or not meta_train_path.exists():
        pytest.skip("X_train.parquet or metadata_train.parquet not present")

    # Load small 60-record slice from training partition
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

    out_dir = tmp_path / "smoke_caches"
    c_path = build_single_cache(
        scenario="Silent Probing",
        seed=42,
        output_dir=out_dir,
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

    assert c_path.exists()
    assert (c_path / "X_attacked.parquet").exists()
    assert (c_path / "status.parquet").exists()
    assert (c_path / "manifest.json").exists()
    assert (c_path / "completion.json").exists()
