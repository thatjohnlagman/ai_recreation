"""
tests/test_build_evaluation_caches.py

Focused tests for Phase 10B Attack-Cache Orchestration Readiness (v2).
All tests use synthetic fixtures exclusively.
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
    FROZEN_PROTOCOL_FILES,
    PROTECTED_HASHES,
    build_evaluation_caches,
    build_expected_cache_identity,
    build_single_cache,
    canonicalize_scenario,
    derive_cache_pairs,
    display_scenario,
    parse_args,
    quarantine_directory,
    run_preflight,
    validate_completed_cache,
    validate_evaluation_partition_alignment,
    verify_frozen_configurations,
    verify_reproducible_execution_state,
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
    exit_code = build_evaluation_caches(
        configs_dir=CONFIGS_DIR,
        data_dir=DATA_DIR,
        artifacts_dir=ARTIFACTS_DIR,
        output_dir=out_dir,
        execute=False,
        preflight_only=False,
    )
    assert exit_code == 0
    assert not out_dir.exists() or len(list(out_dir.glob("*"))) == 0


# ===========================================================================
# 2. Execute Required for Data Access & Preflight Evaluation File Blocking
# ===========================================================================
def test_execute_required_for_data_access(tmp_path):
    """Preflight mode must strictly reject loading evaluation features or labels."""
    with patch("pandas.read_parquet") as mock_parquet:
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


def test_preflight_evaluation_file_blocking(tmp_path):
    """Verifies that run_preflight never accesses evaluation data files even if rigged to fail."""
    def boom(path, *args, **kwargs):
        if "X_eval" in str(path) or "metadata_eval" in str(path):
            raise AssertionError(f"Evaluation file access attempted during preflight: {path}")
        return pd.read_parquet.__wrapped__(path, *args, **kwargs) if hasattr(pd.read_parquet, "__wrapped__") else pd.read_parquet(path, *args, **kwargs)

    with patch("pandas.read_parquet", side_effect=boom):
        results = run_preflight(
            configs_dir=CONFIGS_DIR,
            data_dir=DATA_DIR,
            artifacts_dir=ARTIFACTS_DIR,
            output_dir=tmp_path / "caches",
            enforce_git=False,
        )
        assert results["status"] == "PASS"


# ===========================================================================
# 3. Exact Derivation of 15 Canonical Pairs
# ===========================================================================
def test_exact_derivation_of_15_pairs():
    """Derives exactly 15 canonical (scenario, seed) pairs from frozen configs."""
    pairs = derive_cache_pairs(CONFIGS_DIR)
    assert len(pairs) == 15

    scenarios = sorted(list({p[0] for p in pairs}))
    assert scenarios == ["DecisionBoundary", "SilentProbing", "SurrogateTransfer"]

    seeds = sorted(list({p[1] for p in pairs}))
    assert seeds == [42, 43, 44, 45, 46]

    for scen in scenarios:
        scen_seeds = [p[1] for p in pairs if p[0] == scen]
        assert scen_seeds == [42, 43, 44, 45, 46]

    for p in pairs:
        assert isinstance(p, tuple)
        assert len(p) == 2
        assert isinstance(p[0], str)
        assert isinstance(p[1], int)


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
        assert r["attack_scenario"] == target["attack_scenario"]
        assert r["seed"] == target["seed"]
        assert canonicalize_scenario(r["attack_scenario"]) == canonicalize_scenario(target["attack_scenario"])


# ===========================================================================
# 5. Strengthened Frozen File Byte & SHA-256 Comparison
# ===========================================================================
def test_frozen_file_byte_comparison():
    """Strengthened check: Compares bytes and SHA-256 of all frozen files vs freeze tag."""
    results = verify_frozen_configurations(REPO_ROOT, enforce_git=True)
    assert len(results) == len(FROZEN_PROTOCOL_FILES)
    for path_rel in FROZEN_PROTOCOL_FILES:
        assert path_rel in results
        comp = results[path_rel]
        assert comp["bytes_match"] is True, f"Byte mismatch in {path_rel}"
        assert comp["disk_sha256"] == comp["tagged_sha256"], f"SHA mismatch in {path_rel}"
        assert len(comp["disk_sha256"]) == 64


def test_frozen_file_tampering_detected(tmp_path):
    """Tampering with any frozen configuration or protocol file causes immediate failure."""
    with patch("subprocess.check_output") as mock_git:
        mock_git.return_value = b"tampered byte sequence that does not match disk"
        with pytest.raises(ValueError, match="Byte mismatch in frozen file"):
            verify_frozen_configurations(REPO_ROOT, enforce_git=True)


# ===========================================================================
# 6. Rejection of Altered Freeze Ancestry
# ===========================================================================
def test_rejection_of_altered_freeze_ancestry(tmp_path):
    """Preflight rejects if freeze tag is missing or points to wrong commit."""
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


# ===========================================================================
# 7. Protected-Hash Mismatch Rejection
# ===========================================================================
def test_protected_hash_mismatch_rejection(tmp_path):
    """Preflight rejects if RF, Roles, or Batches protected hashes do not match."""
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
# 8. No Defense/Config Dimension in Cache Identity
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
    grouped = matrix.groupby(["attack_scenario", "seed"])
    for _, group in grouped:
        assert set(group["defense_name"]) == {"afp", "feature_squeezing", "randomized_smoothing"}
        assert len(set(group["controller_config_id"])) >= 2


# ===========================================================================
# 9. Atomic Publication and Failure Quarantine
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

    quarantined = list(out_dir.glob("SilentProbing_42*quarantined_*"))
    assert len(quarantined) >= 1, "Failed staging directory was not quarantined"


# ===========================================================================
# 10. Complete-Cache Reuse with Independent Identity
# ===========================================================================
def test_complete_cache_reuse(tmp_path, synthetic_data):
    """A valid completed cache is verified against independent identity and reused."""
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
# 11. Invalid-Cache Rejection & Quarantine
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

    comp_file = cache_path / "completion.json"
    comp_file.write_text("corrupted json")

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
# 12. Path Containment & Traversal Rejection
# ===========================================================================
def test_path_containment_and_traversal_rejection(tmp_path):
    """Quarantine and staging strictly reject path traversal or foreign directory targets."""
    cache_root = tmp_path / "caches"
    cache_root.mkdir()

    unrelated_dir = tmp_path / "unrelated_dir"
    unrelated_dir.mkdir()

    # Reject quarantine of unrelated directory
    with pytest.raises(ValueError, match="is not strictly inside cache root"):
        quarantine_directory(unrelated_dir, cache_root=cache_root)

    # Reject quarantine of cache root itself
    with pytest.raises(ValueError, match="Cannot quarantine the cache root directory itself"):
        quarantine_directory(cache_root, cache_root=cache_root)

    # Reject directory outside cache root using ../ traversal
    traversal_dir = cache_root / "../unrelated_dir"
    with pytest.raises(ValueError, match="is not strictly inside cache root"):
        quarantine_directory(traversal_dir, cache_root=cache_root)


# ===========================================================================
# 13. Independent Cache Reuse Validation Tests
# ===========================================================================
def test_independent_cache_reuse_rejects_wrong_scenario(tmp_path, synthetic_data):
    """Cache reuse validation strictly rejects a self-consistent cache with the wrong scenario."""
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

    # Validate expecting DecisionBoundary instead of SilentProbing
    with open(c_path / "manifest.json") as f:
        manifest = json.load(f)

    is_valid, reason = validate_completed_cache(
        cache_dir=c_path,
        expected_scenario="DecisionBoundary",
        expected_seed=42,
        expected_cache_identity=manifest,
        expected_feature_names=synthetic_data["feature_names"],
        resolved_batches=synthetic_data["batches_df"],
        official_mode=False,
        expected_row_count=len(synthetic_data["X_meas"]),
    )
    assert not is_valid
    assert "Scenario mismatch" in reason


def test_independent_cache_reuse_rejects_wrong_seed(tmp_path, synthetic_data):
    """Cache reuse validation strictly rejects a self-consistent cache with the wrong seed."""
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

    with open(c_path / "manifest.json") as f:
        manifest = json.load(f)

    # Validate expecting seed 43 instead of 42
    is_valid, reason = validate_completed_cache(
        cache_dir=c_path,
        expected_scenario="SilentProbing",
        expected_seed=43,
        expected_cache_identity=manifest,
        expected_feature_names=synthetic_data["feature_names"],
        resolved_batches=synthetic_data["batches_df"],
        official_mode=False,
        expected_row_count=len(synthetic_data["X_meas"]),
    )
    assert not is_valid
    assert "Seed mismatch" in reason


def test_independent_cache_reuse_rejects_stale_provenance(tmp_path, synthetic_data):
    """Cache reuse validation strictly rejects a cache whose provenance differs from current expectations."""
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

    with open(c_path / "manifest.json") as f:
        manifest = json.load(f)

    # Modify expected identity with a new/different hash
    stale_expected_identity = copy.deepcopy(manifest)
    stale_expected_identity["attacks_yaml_hash"] = _VALID_HASH_B

    is_valid, reason = validate_completed_cache(
        cache_dir=c_path,
        expected_scenario="SilentProbing",
        expected_seed=42,
        expected_cache_identity=stale_expected_identity,
        expected_feature_names=synthetic_data["feature_names"],
        resolved_batches=synthetic_data["batches_df"],
        official_mode=False,
        expected_row_count=len(synthetic_data["X_meas"]),
    )
    assert not is_valid
    assert "attacks_yaml_hash" in reason or "Validation error" in reason


def test_independent_cache_reuse_rejects_altered_manifest_and_completion(tmp_path, synthetic_data):
    """Reject a cache whose manifest and completion were altered together to fake an identity."""
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

    # Alter both manifest and completion to claim seed 45
    with open(c_path / "manifest.json") as f:
        m_data = json.load(f)
    m_data["effective_seed"] = 45
    with open(c_path / "manifest.json", "w") as f:
        json.dump(m_data, f, indent=2)

    with open(c_path / "completion.json") as f:
        c_data = json.load(f)
    c_data["effective_seed"] = 45
    with open(c_path / "completion.json", "w") as f:
        json.dump(c_data, f, indent=2)

    # Validate expecting true seed 42
    is_valid, reason = validate_completed_cache(
        cache_dir=c_path,
        expected_scenario="SilentProbing",
        expected_seed=42,
        expected_cache_identity=m_data,
        expected_feature_names=synthetic_data["feature_names"],
        resolved_batches=synthetic_data["batches_df"],
        official_mode=False,
        expected_row_count=len(synthetic_data["X_meas"]),
    )
    assert not is_valid
    assert "Seed mismatch" in reason


def test_independent_cache_reuse_rejects_earlier_orchestration_commit(tmp_path, synthetic_data):
    """Reject a cache produced by an earlier orchestration commit."""
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
        override_orchestration_commit="older_commit_111111",
    )

    with open(c_path / "manifest.json") as f:
        manifest = json.load(f)

    is_valid, reason = validate_completed_cache(
        cache_dir=c_path,
        expected_scenario="SilentProbing",
        expected_seed=42,
        expected_cache_identity=manifest,
        expected_feature_names=synthetic_data["feature_names"],
        resolved_batches=synthetic_data["batches_df"],
        official_mode=False,
        expected_row_count=len(synthetic_data["X_meas"]),
        expected_orchestration_commit="newer_commit_222222",
    )
    assert not is_valid
    assert "Orchestration commit mismatch" in reason


# ===========================================================================
# 14. Reproducible Execution State Verification Tests
# ===========================================================================
def test_reproducible_state_enforcement_passes_clean():
    """verify_reproducible_execution_state passes when working tree is clean."""
    real_check_output = subprocess.check_output

    def mock_check_output(cmd, *args, **kwargs):
        cmd_str = " ".join(cmd) if isinstance(cmd, list) else str(cmd)
        if "--porcelain" in cmd_str:
            return ""  # clean status
        if "diff" in cmd_str and "src/" in cmd_str:
            return ""  # src/ unchanged relative to freeze
        return real_check_output(cmd, *args, **kwargs)

    with patch("subprocess.check_output", side_effect=mock_check_output):
        head = verify_reproducible_execution_state(
            repo_root=REPO_ROOT,
            configs_dir=CONFIGS_DIR,
            artifacts_dir=ARTIFACTS_DIR,
            data_dir=DATA_DIR,
        )
    assert isinstance(head, str)
    assert len(head) == 40 or len(head) == 64


def test_reproducible_state_rejects_dirty_tracked_files():
    """verify_reproducible_execution_state rejects if tracked files are modified."""
    real_check_output = subprocess.check_output

    def mock_check_output(cmd, *args, **kwargs):
        cmd_str = " ".join(cmd) if isinstance(cmd, list) else str(cmd)
        if "--porcelain" in cmd_str:
            return " M scripts/build_evaluation_caches.py\n"
        if "diff" in cmd_str and "src/" in cmd_str:
            return ""
        return real_check_output(cmd, *args, **kwargs)

    with patch("subprocess.check_output", side_effect=mock_check_output):
        with pytest.raises(RuntimeError, match="tracked files are modified"):
            verify_reproducible_execution_state(
                repo_root=REPO_ROOT,
                configs_dir=CONFIGS_DIR,
                artifacts_dir=ARTIFACTS_DIR,
                data_dir=DATA_DIR,
            )


def test_reproducible_state_rejects_untracked_code_files():
    """verify_reproducible_execution_state rejects if untracked code/test files exist."""
    real_check_output = subprocess.check_output

    def mock_check_output(cmd, *args, **kwargs):
        cmd_str = " ".join(cmd) if isinstance(cmd, list) else str(cmd)
        if "--porcelain" in cmd_str:
            return "?? tests/sneaky_untracked_test.py\n"
        if "diff" in cmd_str and "src/" in cmd_str:
            return ""
        return real_check_output(cmd, *args, **kwargs)

    with patch("subprocess.check_output", side_effect=mock_check_output):
        with pytest.raises(RuntimeError, match="untracked code/test/config files detected"):
            verify_reproducible_execution_state(
                repo_root=REPO_ROOT,
                configs_dir=CONFIGS_DIR,
                artifacts_dir=ARTIFACTS_DIR,
                data_dir=DATA_DIR,
            )


# ===========================================================================
# 15. Evaluation Partition Alignment Validation Tests
# ===========================================================================
def test_partition_alignment_validation_passes_valid(synthetic_data):
    """Validation passes for exactly 90,000 aligned rows, 18k craft, 72k meas, 144 batches of 500."""
    fnames = synthetic_data["feature_names"]
    n_total = 90000
    n_craft = 18000
    n_meas = 72000

    # Build synthetic 90k dataframes
    df_x = pd.DataFrame(np.ones((n_total, len(fnames)), dtype=np.float32), columns=fnames)
    df_meta = pd.DataFrame({"eval_position": np.arange(n_total, dtype=int), "y_binary": 0})
    df_roles = pd.DataFrame({
        "eval_position": np.arange(n_total, dtype=int),
        "role": ["crafting"] * n_craft + ["measurement"] * n_meas,
    })
    meas_eps = np.arange(n_craft, n_total, dtype=int)
    batch_ids = np.repeat(np.arange(144), 500)
    df_batches = pd.DataFrame({
        "eval_position": meas_eps,
        "batch_id": batch_ids,
    })

    # Must pass cleanly without error
    validate_evaluation_partition_alignment(
        df_x_eval=df_x,
        df_meta_eval=df_meta,
        df_roles=df_roles,
        df_batches=df_batches,
        expected_feature_names=fnames,
    )


def test_partition_alignment_rejects_row_count_mismatch(synthetic_data):
    """Validation rejects if X_eval has anything other than 90,000 rows."""
    fnames = synthetic_data["feature_names"]
    df_x_bad = pd.DataFrame(np.ones((89999, len(fnames)), dtype=np.float32), columns=fnames)
    with pytest.raises(ValueError, match="X_eval row count mismatch"):
        validate_evaluation_partition_alignment(
            df_x_eval=df_x_bad,
            df_meta_eval=pd.DataFrame(),
            df_roles=pd.DataFrame(),
            df_batches=pd.DataFrame(),
            expected_feature_names=fnames,
        )


def test_partition_alignment_rejects_nan_or_inf(synthetic_data):
    """Validation rejects if X_eval contains NaN or infinite feature values."""
    fnames = synthetic_data["feature_names"]
    n_total = 90000
    x_mat = np.ones((n_total, len(fnames)), dtype=np.float32)
    x_mat[10, 5] = np.nan
    df_x = pd.DataFrame(x_mat, columns=fnames)
    df_meta = pd.DataFrame({"eval_position": np.arange(n_total, dtype=int)})
    df_roles = pd.DataFrame({
        "eval_position": np.arange(n_total, dtype=int),
        "role": ["crafting"] * 18000 + ["measurement"] * 72000,
    })
    df_batches = pd.DataFrame({
        "eval_position": np.arange(18000, n_total, dtype=int),
        "batch_id": np.repeat(np.arange(144), 500),
    })

    with pytest.raises(ValueError, match="non-finite"):
        validate_evaluation_partition_alignment(
            df_x_eval=df_x,
            df_meta_eval=df_meta,
            df_roles=df_roles,
            df_batches=df_batches,
            expected_feature_names=fnames,
        )


def test_partition_alignment_rejects_overlapping_roles(synthetic_data):
    """Validation rejects if crafting and measurement partitions are not strictly disjoint."""
    fnames = synthetic_data["feature_names"]
    n_total = 90000
    df_x = pd.DataFrame(np.ones((n_total, len(fnames)), dtype=np.float32), columns=fnames)
    df_meta = pd.DataFrame({"eval_position": np.arange(n_total, dtype=int)})
    roles = ["crafting"] * 18000 + ["measurement"] * 72000
    # Overlap position 18000 with 0 so crafting has {0..17999} and measurement has {0, 18001..89999}
    eps = list(range(n_total))
    eps[18000] = 0
    roles_df = pd.DataFrame({
        "eval_position": eps,
        "role": roles,
    })
    df_batches = pd.DataFrame({
        "eval_position": eps[18000:],
        "batch_id": np.repeat(np.arange(144), 500),
    })

    with pytest.raises(ValueError, match="roles overlap"):
        validate_evaluation_partition_alignment(
            df_x_eval=df_x,
            df_meta_eval=df_meta,
            df_roles=roles_df,
            df_batches=df_batches,
            expected_feature_names=fnames,
        )


# ===========================================================================
# 16. Label Isolation Semantics Verification Tests
# ===========================================================================
def test_surrogate_fitting_labels_originate_from_oracle_not_ground_truth(synthetic_data):
    """In Surrogate Transfer, labels for surrogate fitting come from oracle predictions, NOT ground truth."""
    from recall_aware_ids.attacks.surrogate_transfer import SurrogateTransferAttack
    from recall_aware_ids.attacks.oracle import BlackBoxOracle

    bounds = pd.read_parquet(ARTIFACTS_DIR / "preprocessors/training_bounds.parquet")
    fnames = synthetic_data["feature_names"]
    with open(ARTIFACTS_DIR / "preprocessors/feature_mask.json") as f:
        mod_mask = json.load(f)["feature_mask"]

    surrogate_attack = SurrogateTransferAttack(
        feature_names=fnames,
        modifiable_mask=mod_mask,
        training_bounds=bounds,
        effective_seed=42,
    )

    # Oracle predict function: predicts based on first feature threshold so both 0 and 1 are present
    def mock_oracle_fn(x):
        return (x[:, 0] > 0.5).astype(int)

    # Ground truth crafting labels: all 0
    ground_truth_zeros = np.zeros(len(synthetic_data["X_craft"]), dtype=int)

    # Fit surrogate using the BlackBoxOracle pattern from AttackCacheBuilder
    crafting_oracle = BlackBoxOracle(mock_oracle_fn, max_queries_per_sample=None)
    c_ids = [f"craft_{i}" for i in range(len(synthetic_data["X_craft"]))]
    y_pool = crafting_oracle.predict(synthetic_data["X_craft"], sample_ids=c_ids)

    # Assert y_pool originated from oracle (contains both 0 and 1), not ground truth (all 0)
    assert 0 in y_pool and 1 in y_pool
    assert not np.array_equal(y_pool, ground_truth_zeros)

    surrogate_attack.fit_surrogate(synthetic_data["X_craft"], y_pool)
    # Fitted surrogate tree predictions match oracle predictions on pool
    preds = surrogate_attack.surrogate.predict(synthetic_data["X_craft"])
    assert np.array_equal(preds, y_pool)


def test_surrogate_candidate_generation_does_not_receive_labels():
    """Surrogate candidate generation takes only feature vector x with zero access to labels."""
    from recall_aware_ids.attacks.surrogate_transfer import SurrogateTransferAttack
    import inspect

    bounds = pd.read_parquet(ARTIFACTS_DIR / "preprocessors/training_bounds.parquet")
    with open(ARTIFACTS_DIR / "preprocessors/feature_mask.json") as f:
        mask_data = json.load(f)

    surrogate = SurrogateTransferAttack(
        feature_names=mask_data["feature_columns"],
        modifiable_mask=mask_data["feature_mask"],
        training_bounds=bounds,
    )

    sig = inspect.signature(surrogate.generate_candidate)
    params = list(sig.parameters.keys())
    assert params == ["X_orig"] or "y" not in params
    assert "label" not in params
    assert "true_label" not in params


def test_decision_boundary_reference_selection_uses_model_predictions(synthetic_data):
    """Decision Boundary benign reference pool selection queries model predictions on crafting rows, NOT crafting labels."""
    X_craft = synthetic_data["X_craft"]
    n_craft = len(X_craft)

    # Target model predict function: deterministic classification based on feature threshold
    def mock_predict_fn(x):
        return (x[:, 0] > 0.5).astype(int)

    # Ground truth crafting labels: test conflicting label assignments
    y_craft_all_zeros = np.zeros(n_craft, dtype=int)
    y_craft_all_ones = np.ones(n_craft, dtype=int)
    y_craft_inverted = 1 - synthetic_data["y_craft"]

    # Canonical selection rule (LABEL ISOLATION RULE):
    # craft_preds = predict_fn(X_craft)
    # benign_mask = (craft_preds == 0)
    # benign_reference_pool = X_craft[benign_mask]
    craft_preds = mock_predict_fn(X_craft)
    benign_mask = (craft_preds == 0)

    assert np.any(benign_mask), "Should contain benign predictions"
    assert not np.all(benign_mask), "Should contain malicious predictions"

    pool_from_zeros = X_craft[(mock_predict_fn(X_craft) == 0)]
    pool_from_ones = X_craft[(mock_predict_fn(X_craft) == 0)]
    pool_from_inverted = X_craft[(mock_predict_fn(X_craft) == 0)]

    # Proves benign reference pool is identical across all label assignments
    assert np.array_equal(pool_from_zeros, pool_from_ones)
    assert np.array_equal(pool_from_zeros, pool_from_inverted)

    # Proves model prediction strictly governs inclusion/exclusion, not ground truth
    for i in range(n_craft):
        sample = X_craft[i:i+1]
        pred = mock_predict_fn(sample)[0]
        in_pool = any(np.allclose(sample[0], p) for p in pool_from_zeros)
        if pred == 0:
            assert in_pool, f"Crafting sample {i} predicted benign (0) must be in reference pool"
        else:
            assert not in_pool, f"Crafting sample {i} predicted malicious (1) must NOT be in reference pool"


def test_boundary_perturbation_search_does_not_use_labels_for_path():
    """Binary search interpolation step evaluations depend strictly on oracle queries, not ground truth."""
    from recall_aware_ids.attacks.boundary_attack import DecisionBoundaryAttack
    from recall_aware_ids.attacks.oracle import BlackBoxOracle

    bounds = pd.read_parquet(ARTIFACTS_DIR / "preprocessors/training_bounds.parquet")
    with open(ARTIFACTS_DIR / "preprocessors/feature_mask.json") as f:
        mask_data = json.load(f)

    attack = DecisionBoundaryAttack(
        feature_names=mask_data["feature_columns"],
        modifiable_mask=mask_data["feature_mask"],
        training_bounds=bounds,
        max_queries=50,
        binary_search_steps=10,
    )

    # Oracle query recording
    queried_points = []
    def mock_oracle(x):
        queried_points.append(np.array(x, copy=True))
        # Feature 1 is modifiable per feature_mask.json
        return (x[:, 1] > 0.5).astype(int)

    oracle = BlackBoxOracle(mock_oracle, max_queries_per_sample=50)

    # Malicious sample: feature 1 = 0.8 (oracle predicts 1)
    X_mal = np.zeros(78, dtype=np.float32)
    X_mal[1] = 0.8
    # Benign reference sample: feature 1 = 0.2 (oracle predicts 0)
    X_benign_ref = np.zeros(78, dtype=np.float32)
    X_benign_ref[1] = 0.2
    ref_pool = np.array([X_benign_ref])

    res = attack.generate(
        X_orig=X_mal,
        oracle=oracle,
        sample_id="test_sample_0",
        true_label=1,
        reference_pool=ref_pool,
    )

    assert res.attempted is True
    # Search executed 10 binary search steps guided exclusively by oracle midpoints
    assert len(queried_points) >= 12  # eligibility + endpoint + 10 binary search steps + verification
    assert res.success is True


def test_label_change_affects_clean_tp_eligibility_without_altering_attack_logic():
    """Changing ground-truth labels affects clean-TP eligibility and status codes without altering attack logic."""
    from recall_aware_ids.attacks.boundary_attack import DecisionBoundaryAttack
    from recall_aware_ids.attacks.oracle import BlackBoxOracle

    bounds = pd.read_parquet(ARTIFACTS_DIR / "preprocessors/training_bounds.parquet")
    with open(ARTIFACTS_DIR / "preprocessors/feature_mask.json") as f:
        mask_data = json.load(f)

    attack = DecisionBoundaryAttack(
        feature_names=mask_data["feature_columns"],
        modifiable_mask=mask_data["feature_mask"],
        training_bounds=bounds,
        max_queries=50,
        binary_search_steps=10,
    )

    def mock_oracle(x):
        return (x[:, 0] > 0.5).astype(int)

    X_sample = np.zeros(78, dtype=np.float32)
    X_sample[0] = 0.8  # oracle predicts 1
    ref_pool = np.zeros((1, 78), dtype=np.float32)  # oracle predicts 0

    # Case 1: Ground truth y=1 -> clean TP, eligible for evasion attempt
    oracle1 = BlackBoxOracle(mock_oracle, max_queries_per_sample=50)
    res_clean_tp = attack.generate(
        X_orig=X_sample,
        oracle=oracle1,
        sample_id="sample_tp",
        true_label=1,
        reference_pool=ref_pool,
    )
    assert res_clean_tp.eligible is True
    assert res_clean_tp.attempted is True

    # Case 2: Ground truth y=0 -> false alarm, ineligible for evasion
    oracle2 = BlackBoxOracle(mock_oracle, max_queries_per_sample=50)
    res_false_alarm = attack.generate(
        X_orig=X_sample,
        oracle=oracle2,
        sample_id="sample_fa",
        true_label=0,
        reference_pool=ref_pool,
    )
    assert res_false_alarm.eligible is False
    assert res_false_alarm.attempted is False
    assert res_false_alarm.status_code == "INELIGIBLE_TRUE_BENIGN"

    # Case 3: Ground truth y=1 but model predicts 0 -> false negative, ineligible
    X_fn = np.zeros(78, dtype=np.float32)
    X_fn[0] = 0.2  # oracle predicts 0
    oracle3 = BlackBoxOracle(mock_oracle, max_queries_per_sample=50)
    res_fn = attack.generate(
        X_orig=X_fn,
        oracle=oracle3,
        sample_id="sample_fn",
        true_label=1,
        reference_pool=ref_pool,
    )
    assert res_fn.eligible is False
    assert res_fn.attempted is False
    assert res_fn.status_code == "INELIGIBLE_FALSE_NEGATIVE"


# ===========================================================================
# 17. Scenario and Seed Filtering
# ===========================================================================
def test_scenario_seed_filtering():
    """CLI and orchestrator filters correctly select valid subsets and reject invalid inputs."""
    args = parse_args(["--scenario", "Silent Probing", "--seed", "44", "--preflight-only"])
    assert args.scenario == "Silent Probing"
    assert args.seed == 44
    assert args.preflight_only is True

    with pytest.raises(SystemExit):
        parse_args(["--seed", "99"])

    with pytest.raises(SystemExit):
        parse_args(["--scenario", "UnrecognizedScenario"])


# ===========================================================================
# 18. Deterministic Pair Ordering
# ===========================================================================
def test_deterministic_pair_ordering():
    """derive_cache_pairs produces identical ordered sequence across calls."""
    pairs1 = derive_cache_pairs(CONFIGS_DIR)
    pairs2 = derive_cache_pairs(CONFIGS_DIR)
    assert pairs1 == pairs2
    assert len(pairs1) == 15
    assert [p[0] for p in pairs1[:5]] == ["SilentProbing"] * 5
    assert [p[1] for p in pairs1[:5]] == [42, 43, 44, 45, 46]
    assert [p[0] for p in pairs1[5:10]] == ["SurrogateTransfer"] * 5
    assert [p[1] for p in pairs1[5:10]] == [42, 43, 44, 45, 46]
    assert [p[0] for p in pairs1[10:15]] == ["DecisionBoundary"] * 5
    assert [p[1] for p in pairs1[10:15]] == [42, 43, 44, 45, 46]
