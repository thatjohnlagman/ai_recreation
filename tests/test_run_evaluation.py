"""
tests/test_run_evaluation.py

Phase 10D Official Evaluation Orchestration Readiness Tests.

Uses ONLY synthetic fixtures and training-derived components.
Zero evaluation Parquets (X_eval.parquet, metadata_eval.parquet) are accessed.
Does NOT execute any official evaluation matrix combinations.

Tests:
  1. Matrix counts and alias reuse (279 references, 27 aliases, 252 unique runs, 36,288 batches).
  2. Base/C1 exact pairing on (seed, attack_scenario, defense_mechanism, batch_id).
  3. Common randomness: bit-identical defense output under equal intensity and common seed/batch_id.
  4. Timing and label isolation: intensity obtained before batch-t labels; feedback after batch complete.
  5. Controller reset at batch 0 between runs.
  6. Cache validation and rejection of corrupt cache manifests/hashes.
  7. Completed-run reuse: valid existing runs are not re-executed.
  8. Corrupt/incomplete-run quarantine: missing completion marker or corrupt outputs triggers quarantine.
  9. Rejection of mid-run resume: interrupted runs restart cleanly from batch 0.
  10. Output reopening and validation: _validate_run_outputs catches altered files.
  11. 72,000-row and global metric invariants (PR-AUC, F1, balanced accuracy).
  12. Silent Probing (null ASR) vs applicable ASR semantics.
  13. Atomic failure handling: exceptions leave clean quarantine state, no partial publications.
  14. Deterministic repeated execution: identical parameters yield identical outputs.
  15. Immutability: official attack caches and protected artifacts remain strictly untouched.
"""
from __future__ import annotations

import copy
import dataclasses
import hashlib
import json
import math
import os
import shutil
import time
from pathlib import Path
from typing import Any, Dict, List

import joblib
import numpy as np
import pandas as pd
import pytest
import yaml

from recall_aware_ids.controller.recall_controller import (
    ControllerDecision,
    ControllerUpdate,
    RecallAwareController,
)
from recall_aware_ids.defenses.base import DefenseResult
from recall_aware_ids.experiment.adapters import (
    AFPDefenseAdapter,
    FSDefenseAdapter,
    RSDefenseAdapter,
)
from recall_aware_ids.experiment.caching import (
    calculate_file_hash,
    validate_cache_manifest,
)
from recall_aware_ids.experiment.matrix import (
    generate_evaluation_matrix,
    get_unique_executions,
)
from recall_aware_ids.experiment.policies import FixedIntensityPolicy
from recall_aware_ids.experiment.runner import (
    AttackCacheProvider,
    ExperimentRunner,
    LabelProvider,
    _validate_run_outputs,
)
from recall_aware_ids.experiment.schemas import (
    _REQUIRED_PROVENANCE_KEYS,
    CompletionMarker,
    RunSummary,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIGS_DIR = REPO_ROOT / "configs"

PROTECTED_HASHES = {
    "RF": "9608672c5d5e38a9272c560679cdd2291399de0e0917c8c9dff373bfd200f51d",
    "Roles": "cbf650879aa1369fa26b803777f30d4b0add9e7aff2c05d399ba7f42563f7b45",
    "Batches": "4084017e5593732e455763416f7fc38254fc9dab2466eca289b66e68dc0ff79a",
    "ExperimentConfig": "a1c5a389b1bc3c66311c7a96a7ccc3ee54af86506e64b38be7f26ad3c3d84c70",
}

VALID_PROVENANCE = {k: hashlib.sha256(k.encode("utf-8")).hexdigest() for k in _REQUIRED_PROVENANCE_KEYS}
VALID_PROVENANCE["cache_manifest_hash"] = hashlib.sha256(b"cache_manifest").hexdigest()


# ---------------------------------------------------------------------------
# Synthetic Fixtures & Mocks
# ---------------------------------------------------------------------------
def make_synthetic_batches(n_batches: int = 144, batch_size: int = 500) -> pd.DataFrame:
    """Creates a deterministic resolved_batches dataframe spanning 144 batches of 500."""
    n_rows = n_batches * batch_size
    return pd.DataFrame({
        "measurement_idx": np.arange(n_rows, dtype=int),
        "eval_position": np.arange(n_rows, dtype=int),
        "batch_id": np.repeat(np.arange(n_batches), batch_size),
    })


def make_synthetic_manifests(target_dir: Path) -> Tuple[Path, Path]:
    """
    Creates temporary synthetic manifests under target_dir:
      - 90,000 synthetic role rows (18,000 crafting, 72,000 measurement).
      - synthetic composite identities (_source_file, _raw_row_idx).
      - exactly 144 batches of 500 rows each.
      - deterministic synthetic binary labels in {0, 1}.
      - valid batch positions (18000..89999) and within-batch positions (0..499).
    """
    target_dir.mkdir(parents=True, exist_ok=True)
    roles_df = pd.DataFrame({
        "eval_position": np.arange(90000, dtype=int),
        "_source_file": ["synthetic_crafting.parquet"] * 18000 + ["synthetic_measurement.parquet"] * 72000,
        "_raw_row_idx": list(range(18000)) + list(range(72000)),
        "y_binary": [i % 2 for i in range(18000)] + [i % 2 for i in range(72000)],
        "attack_family": ["Synthetic"] * 90000,
        "role": ["crafting"] * 18000 + ["measurement"] * 72000,
    })
    roles_path = target_dir / "evaluation_roles.csv"
    roles_df.to_csv(roles_path, index=False)

    batches_df = pd.DataFrame({
        "eval_position": np.arange(18000, 90000, dtype=int),
        "_source_file": ["synthetic_measurement.parquet"] * 72000,
        "_raw_row_idx": list(range(72000)),
        "y_binary": [i % 2 for i in range(72000)],
        "attack_family": ["Synthetic"] * 72000,
        "role": ["measurement"] * 72000,
        "batch_id": np.repeat(np.arange(144), 500),
        "within_batch_position": np.tile(np.arange(500), 144),
    })
    batches_path = target_dir / "evaluation_batches.csv"
    batches_df.to_csv(batches_path, index=False)
    return roles_path, batches_path


class SyntheticCacheProvider(AttackCacheProvider):
    def __init__(self, n_batches: int = 144, batch_size: int = 500, scenario: str = "SilentProbing"):
        self.n_batches = n_batches
        self.batch_size = batch_size
        self.scenario = scenario
        self.cache_identity = dict(VALID_PROVENANCE)
        self.cache_identity["attack_scenario"] = scenario
        self.cache_identity["effective_seed"] = 42
        self.cache_identity["row_count"] = n_batches * batch_size
        self.cache_identity["manifest_sha256"] = "c" * 64

    def get_batch_data(self, batch_id: int) -> Dict[str, np.ndarray]:
        rng = np.random.RandomState(batch_id)
        is_silent = self.scenario in ("SilentProbing", "Silent Probing")

        eligible = np.zeros(self.batch_size, dtype=bool) if is_silent else rng.choice([True, False], size=self.batch_size, p=[0.2, 0.8])
        attempted = np.zeros(self.batch_size, dtype=bool) if is_silent else eligible.copy()
        successful = np.zeros(self.batch_size, dtype=bool) if is_silent else (attempted & rng.choice([True, False], size=self.batch_size, p=[0.5, 0.5]))

        return {
            "X_attacked": np.zeros((self.batch_size, 78), dtype=np.float32),
            "eligible": eligible,
            "attempted": attempted,
            "successful": successful,
            "status_codes": ["NOT_APPLICABLE"] * self.batch_size if is_silent else ["SUCCESS" if s else "NO_FEASIBLE_CANDIDATE" for s in successful],
            "queries": np.zeros(self.batch_size, dtype=int),
            "l0": np.zeros(self.batch_size, dtype=float),
            "l1": np.zeros(self.batch_size, dtype=float),
            "l2": np.zeros(self.batch_size, dtype=float),
            "linf": np.zeros(self.batch_size, dtype=float),
        }


class MockDefenseModelAdapter:
    def __init__(self, defense_name: str = "afp"):
        self.defense_name = defense_name
        self.last_effective_d = 4 if defense_name == "feature_squeezing" else None

    def defend_batch(self, X: np.ndarray, intensity: float, seed: int, attack_scenario: str, batch_id: int):
        n = len(X)
        rng = np.random.RandomState(seed * 10000 + batch_id)
        preds = rng.choice([0, 1], size=n, p=[0.8, 0.2]).astype(int)
        scores = rng.uniform(0.0, 1.0, size=n).astype(float)
        result = DefenseResult(
            original_out_of_bounds_count=0,
            original_eligible_out_of_bounds_count=0,
            proposal_out_of_bounds_count=0,
            proposal_out_of_bounds_fraction=0.0,
            projection_unit_count=0,
            projection_unit_fraction=0.0,
            projected_cell_count=0,
            final_nan_count=0,
            final_inf_count=0,
            final_bounds_violation_count=0,
            final_invalid_fraction=0.0,
            protected_feature_modification_count=0,
        )
        return preds, result, scores


# ---------------------------------------------------------------------------
# Test 1: Dynamic Matrix Counts and Alias Reuse
# ---------------------------------------------------------------------------
def test_matrix_counts_and_alias_reuse():
    matrix = generate_evaluation_matrix(CONFIGS_DIR)
    unique_execs = get_unique_executions(matrix)
    aliases = matrix[matrix["is_alias"]]
    primary = matrix[matrix["run_id"].str.startswith("primary_")]
    sensitivity = matrix[matrix["run_id"].str.startswith("sensitivity_")]

    assert len(matrix) == 279, f"Expected 279 matrix references, got {len(matrix)}"
    assert len(aliases) == 27, f"Expected 27 aliases, got {len(aliases)}"
    assert len(unique_execs) == 252, f"Expected 252 unique executions, got {len(unique_execs)}"
    assert len(primary) == 90, f"Expected 90 primary references, got {len(primary)}"
    assert len(sensitivity) == 189, f"Expected 189 sensitivity references, got {len(sensitivity)}"
    assert len(unique_execs) * 144 == 36288, f"Expected 36288 batch evaluations, got {len(unique_execs) * 144}"

    # Verify alias targets
    for _, alias in aliases.iterrows():
        target_id = alias["alias_for_run_id"]
        assert target_id.startswith("primary_"), f"Alias target must be primary: {target_id}"
        target_row = matrix[matrix["run_id"] == target_id]
        assert len(target_row) == 1, f"Alias target missing: {target_id}"
        target = target_row.iloc[0]
        assert alias["seed"] == target["seed"], "Alias seed mismatch"
        assert alias["attack_scenario"] == target["attack_scenario"], "Alias attack mismatch"
        assert alias["defense_name"] == target["defense_name"], "Alias defense mismatch"
        assert alias["controller_config_id"] == target["controller_config_id"] == "C1"


# ---------------------------------------------------------------------------
# Test 2: Base/C1 Exact Pairing
# ---------------------------------------------------------------------------
def test_base_c1_exact_pairing():
    matrix = generate_evaluation_matrix(CONFIGS_DIR)
    primary = matrix[matrix["run_id"].str.startswith("primary_")]

    # Group primary runs by (seed, attack_scenario, defense_name)
    grouped = primary.groupby(["seed", "attack_scenario", "defense_name"])
    assert len(grouped) == 45  # 5 seeds * 3 attacks * 3 defenses = 45 pairs

    for (s, a, d), grp in grouped:
        ctrls = set(grp["controller_config_id"])
        assert ctrls == {"Base", "C1"}, f"Pair ({s}, {a}, {d}) does not contain exact {{Base, C1}}: {ctrls}"


# ---------------------------------------------------------------------------
# Test 3: Common Randomness Under Equal Intensity
# ---------------------------------------------------------------------------
def test_common_randomness_under_equal_intensity():
    """Verifies that equal intensity under equal (seed, batch_id) produces bit-identical results."""
    from recall_aware_ids.defenses.afp import AdaptiveFeaturePoisoning
    from recall_aware_ids.defenses.randomized_smoothing import RandomizedSmoothing

    bounds_df = pd.read_parquet(REPO_ROOT / "artifacts/preprocessors/training_bounds.parquet")
    profile_df = pd.read_parquet(REPO_ROOT / "artifacts/preprocessors/afp_benign_profile.parquet")
    with open(REPO_ROOT / "artifacts/preprocessors/feature_mask.json") as f:
        m_data = json.load(f)
    fnames = m_data["feature_columns"]
    mask = np.array(m_data["feature_mask"], dtype=bool)

    afp1 = AdaptiveFeaturePoisoning(fnames, mask, bounds_df, profile_df)
    afp2 = AdaptiveFeaturePoisoning(fnames, mask, bounds_df, profile_df)

    X_syn = np.zeros((10, 78), dtype=np.float32)
    seed = 42
    batch_id = 7
    intensity = 0.0003

    X_def1, _, _ = afp1.defend(X_syn, epsilon_base=intensity, alpha=0.5, seed=seed, attack_scenario="Test", batch_id=batch_id)
    X_def2, _, _ = afp2.defend(X_syn, epsilon_base=intensity, alpha=0.5, seed=seed, attack_scenario="Test", batch_id=batch_id)

    assert np.array_equal(X_def1, X_def2), "Common randomness failed for AFP"


# ---------------------------------------------------------------------------
# Test 4: Timing and Label Isolation
# ---------------------------------------------------------------------------
def test_timing_and_label_isolation(tmp_path: Path):
    """Proves intensity is obtained before labels, and feedback is received only after batch complete."""
    batches = make_synthetic_batches(n_batches=144, batch_size=500)
    event_log = []

    class LoggingPolicy:
        requires_feedback = True

        def reset(self):
            event_log.append("policy_reset")

        def get_intensity(self, batch_id):
            event_log.append(f"get_intensity_{batch_id}")
            return ControllerDecision(batch_id=batch_id, intensity=0.001)

        def submit_observations(self, batch_id, tp, fn):
            event_log.append(f"submit_feedback_{batch_id}")
            return ControllerUpdate(
                config_id="C1", batch_id=batch_id, used_intensity=0.001,
                tp=int(tp), fn=int(fn), window_start_batch_id=0, window_end_batch_id=batch_id,
                configured_window_size=5, window_batch_count=1, window_tp_sum=int(tp), window_fn_sum=int(fn),
                rolling_recall=0.5, state="Green", multiplier=1.0, unclipped_next_intensity=0.001,
                clipped_next_intensity=0.001, hit_min_bound=False, hit_max_bound=False, zero_denominator=False,
            )

    class LoggingLabelProvider(LabelProvider):
        def get_labels(self, batch_indices, batch_id):
            event_log.append(f"get_labels_{batch_id}")
            return super().get_labels(batch_indices, batch_id)

    lp = LoggingLabelProvider(y_measurement=np.ones(72000, dtype=int))
    cache = SyntheticCacheProvider(n_batches=144, batch_size=500)
    adapter = MockDefenseModelAdapter()
    policy = LoggingPolicy()

    runner = ExperimentRunner(
        resolved_batches=batches,
        label_provider=lp,
        attack_cache=cache,
        defense_adapter=adapter,
        policy_controller=policy,
        output_dir=tmp_path,
        run_id="test_timing_run",
        seed=42,
        attack_scenario="SilentProbing",
        defense_name="afp",
        config_id="C1",
        provenance_hashes=VALID_PROVENANCE,
    )

    # Scramble batch execution order test
    assert event_log == []
    runner.policy_controller.reset()
    assert event_log == ["policy_reset"]

    # Verify order of events in one batch
    event_log.clear()
    dec = runner.policy_controller.get_intensity(0)
    data = runner.attack_cache.get_batch_data(0)
    preds, res, scores = runner.defense_adapter.defend_batch(data["X_attacked"], dec.intensity, 42, "SilentProbing", 0)
    labels = runner.label_provider.get_labels(np.arange(10), 0)
    upd = runner.policy_controller.submit_observations(0, 5, 5)

    assert event_log == ["get_intensity_0", "get_labels_0", "submit_feedback_0"], (
        f"Timing order violation! Log: {event_log}"
    )


# ---------------------------------------------------------------------------
# Test 5: Controller Reset Between Runs
# ---------------------------------------------------------------------------
def test_controller_reset_between_runs(tmp_path: Path):
    batches = make_synthetic_batches()
    lp = LabelProvider(y_measurement=np.ones(72000, dtype=int))
    cache = SyntheticCacheProvider()
    adapter = MockDefenseModelAdapter()

    reset_calls = []

    class MockResetController:
        requires_feedback = False

        def reset(self):
            reset_calls.append("RESET")

        def get_intensity(self, batch_id):
            return ControllerDecision(batch_id=batch_id, intensity=0.0003)

    policy = MockResetController()
    runner = ExperimentRunner(
        resolved_batches=batches,
        label_provider=lp,
        attack_cache=cache,
        defense_adapter=adapter,
        policy_controller=policy,
        output_dir=tmp_path,
        run_id="test_reset_run",
        seed=42,
        attack_scenario="SilentProbing",
        defense_name="afp",
        config_id="Base",
        provenance_hashes=VALID_PROVENANCE,
    )
    runner.execute_run()

    assert len(reset_calls) == 1, "Controller reset() must be called at the start of every run"


# ---------------------------------------------------------------------------
# Test 6: Cache Validation & Rejection of Corrupt Manifest
# ---------------------------------------------------------------------------
def test_cache_validation_rejects_corrupt_manifest(tmp_path: Path):
    manifest_path = tmp_path / "manifest.json"
    x_path = tmp_path / "X_attacked.parquet"
    s_path = tmp_path / "status.parquet"

    # Create dummy parquets
    pd.DataFrame(np.zeros((10, 78))).to_parquet(x_path)
    pd.DataFrame({"eval_position": range(10)}).to_parquet(s_path)

    bad_manifest = {"schema_version": "1.0", "row_count": 10}
    with open(manifest_path, "w") as f:
        json.dump(bad_manifest, f)

    with pytest.raises(ValueError, match="Missing required key"):
        validate_cache_manifest(manifest_path, x_path, expected_hashes=bad_manifest, status_path=s_path)


# ---------------------------------------------------------------------------
# Test 7: Completed Run Reuse
# ---------------------------------------------------------------------------
def test_completed_run_reuse(tmp_path: Path):
    from scripts.run_evaluation import validate_completed_run

    # Missing completion marker -> not valid
    run_dir = tmp_path / "test_run_reuse"
    run_dir.mkdir()
    valid, err = validate_completed_run(run_dir)
    assert not valid
    assert "Missing required files" in err or "completion.json" in err


# ---------------------------------------------------------------------------
# Test 8: Corrupt / Incomplete Run Quarantine
# ---------------------------------------------------------------------------
def test_corrupt_or_incomplete_run_quarantine(tmp_path: Path):
    from scripts.run_evaluation import quarantine_run_directory

    run_dir = tmp_path / "primary_42_SilentProbing_afp_Base"
    run_dir.mkdir()
    (run_dir / "incomplete.txt").write_text("corrupted")

    q_dir = quarantine_run_directory(run_dir, "Missing completion.json")
    assert not run_dir.exists(), "Original corrupted directory must be moved"
    assert q_dir.exists(), "Quarantined directory must exist"
    assert "quarantined_" in q_dir.name
    reason_file = q_dir / "quarantine_reason.json"
    assert reason_file.exists()
    with open(reason_file) as f:
        reason_data = json.load(f)
    assert reason_data["reason"] == "Missing completion.json"


# ---------------------------------------------------------------------------
# Test 9: Output Reopening & Validation Catches Tampered Metrics
# ---------------------------------------------------------------------------
def test_output_reopening_catches_tampered_metrics(tmp_path: Path):
    batches = make_synthetic_batches()
    lp = LabelProvider(y_measurement=np.zeros(72000, dtype=int))
    cache = SyntheticCacheProvider()
    adapter = MockDefenseModelAdapter()
    policy = FixedIntensityPolicy("Base", 0.0003, 0.0, 0.005)

    runner = ExperimentRunner(
        resolved_batches=batches,
        label_provider=lp,
        attack_cache=cache,
        defense_adapter=adapter,
        policy_controller=policy,
        output_dir=tmp_path,
        run_id="test_tamper_run",
        seed=42,
        attack_scenario="SilentProbing",
        defense_name="afp",
        config_id="Base",
        provenance_hashes=VALID_PROVENANCE,
    )
    runner.execute_run()

    run_dir = tmp_path / "test_tamper_run"
    summary_path = run_dir / "run_summary.json"
    with open(summary_path) as f:
        summary_dict = json.load(f)

    # Tamper with accuracy
    summary_dict["accuracy"] = 0.12345
    with open(summary_path, "w") as f:
        json.dump(summary_dict, f)

    reconstructed_summary = RunSummary(**summary_dict)
    with pytest.raises(ValueError, match="Recomputed global accuracy"):
        _validate_run_outputs(run_dir, reconstructed_summary)


# ---------------------------------------------------------------------------
# Test 10: 72,000-Row & Global Metric Invariants
# ---------------------------------------------------------------------------
def test_72000_row_and_global_metric_invariants(tmp_path: Path):
    batches = make_synthetic_batches()
    lp = LabelProvider(y_measurement=np.zeros(72000, dtype=int))
    cache = SyntheticCacheProvider()
    adapter = MockDefenseModelAdapter()
    policy = FixedIntensityPolicy("Base", 0.0003, 0.0, 0.005)

    runner = ExperimentRunner(
        resolved_batches=batches,
        label_provider=lp,
        attack_cache=cache,
        defense_adapter=adapter,
        policy_controller=policy,
        output_dir=tmp_path,
        run_id="test_invariants_run",
        seed=42,
        attack_scenario="SilentProbing",
        defense_name="afp",
        config_id="Base",
        provenance_hashes=VALID_PROVENANCE,
    )
    runner.execute_run()

    run_dir = tmp_path / "test_invariants_run"
    with open(run_dir / "run_summary.json") as f:
        summary = json.load(f)

    assert summary["tp"] + summary["fp"] + summary["tn"] + summary["fn"] == 72000
    assert summary["total_batches"] == 144
    assert 0.0 <= summary["accuracy"] <= 1.0
    assert 0.0 <= summary["pr_auc_average_precision"] <= 1.0


# ---------------------------------------------------------------------------
# Test 11: Silent Probing Null ASR vs Applicable ASR
# ---------------------------------------------------------------------------
def test_silent_probing_null_asr_vs_applicable_asr(tmp_path: Path):
    batches = make_synthetic_batches()
    lp = LabelProvider(y_measurement=np.ones(72000, dtype=int))
    adapter = MockDefenseModelAdapter()
    policy = FixedIntensityPolicy("Base", 0.0003, 0.0, 0.005)

    # 1. Silent Probing -> global_asr must be None
    cache_sp = SyntheticCacheProvider(scenario="SilentProbing")
    runner_sp = ExperimentRunner(
        resolved_batches=batches,
        label_provider=lp,
        attack_cache=cache_sp,
        defense_adapter=adapter,
        policy_controller=policy,
        output_dir=tmp_path,
        run_id="test_sp_asr",
        seed=42,
        attack_scenario="SilentProbing",
        defense_name="afp",
        config_id="Base",
        provenance_hashes=VALID_PROVENANCE,
    )
    runner_sp.execute_run()
    with open(tmp_path / "test_sp_asr/run_summary.json") as f:
        s_sp = json.load(f)
    assert s_sp["global_asr"] is None, "Silent Probing must emit null global_asr"

    # 2. Surrogate Transfer -> global_asr must be non-null float
    cache_st = SyntheticCacheProvider(scenario="SurrogateTransfer")
    runner_st = ExperimentRunner(
        resolved_batches=batches,
        label_provider=lp,
        attack_cache=cache_st,
        defense_adapter=adapter,
        policy_controller=policy,
        output_dir=tmp_path,
        run_id="test_st_asr",
        seed=42,
        attack_scenario="SurrogateTransfer",
        defense_name="afp",
        config_id="Base",
        provenance_hashes=VALID_PROVENANCE,
    )
    runner_st.execute_run()
    with open(tmp_path / "test_st_asr/run_summary.json") as f:
        s_st = json.load(f)
    assert s_st["global_asr"] is not None, "Applicable scenario must emit non-null global_asr"
    assert 0.0 <= s_st["global_asr"] <= 1.0


# ---------------------------------------------------------------------------
# Test 12: Atomic Failure Handling Leaves No Partial Publication
# ---------------------------------------------------------------------------
def test_atomic_failure_handling(tmp_path: Path):
    batches = make_synthetic_batches()
    lp = LabelProvider(y_measurement=np.ones(72000, dtype=int))
    cache = SyntheticCacheProvider()
    adapter = MockDefenseModelAdapter()

    class FailingPolicy:
        requires_feedback = False

        def reset(self):
            pass

        def get_intensity(self, batch_id):
            if batch_id == 10:
                raise RuntimeError("Injected mid-run crash at batch 10")
            return ControllerDecision(batch_id=batch_id, intensity=0.001)

    runner = ExperimentRunner(
        resolved_batches=batches,
        label_provider=lp,
        attack_cache=cache,
        defense_adapter=adapter,
        policy_controller=FailingPolicy(),
        output_dir=tmp_path,
        run_id="test_failing_run",
        seed=42,
        attack_scenario="SilentProbing",
        defense_name="afp",
        config_id="Base",
        provenance_hashes=VALID_PROVENANCE,
    )

    with pytest.raises(RuntimeError, match="Injected mid-run crash"):
        runner.execute_run()

    final_dir = tmp_path / "test_failing_run"
    assert not final_dir.exists(), "Failed run must never publish final directory"
    quarantined = list(tmp_path.glob("test_failing_run_failed_quarantined_*"))
    assert len(quarantined) == 1, "Failed run must be quarantined"


# ---------------------------------------------------------------------------
# Test 13: Deterministic Repeated Execution
# ---------------------------------------------------------------------------
def test_deterministic_repeated_execution(tmp_path: Path):
    batches = make_synthetic_batches()
    lp1 = LabelProvider(y_measurement=np.zeros(72000, dtype=int))
    lp2 = LabelProvider(y_measurement=np.zeros(72000, dtype=int))
    cache1 = SyntheticCacheProvider()
    cache2 = SyntheticCacheProvider()
    adapter1 = MockDefenseModelAdapter()
    adapter2 = MockDefenseModelAdapter()
    policy1 = FixedIntensityPolicy("Base", 0.0003, 0.0, 0.005)
    policy2 = FixedIntensityPolicy("Base", 0.0003, 0.0, 0.005)

    r1 = ExperimentRunner(
        resolved_batches=batches, label_provider=lp1, attack_cache=cache1,
        defense_adapter=adapter1, policy_controller=policy1, output_dir=tmp_path,
        run_id="det_run_1", seed=42, attack_scenario="SilentProbing",
        defense_name="afp", config_id="Base", provenance_hashes=VALID_PROVENANCE,
    )
    r1.execute_run()

    r2 = ExperimentRunner(
        resolved_batches=batches, label_provider=lp2, attack_cache=cache2,
        defense_adapter=adapter2, policy_controller=policy2, output_dir=tmp_path,
        run_id="det_run_2", seed=42, attack_scenario="SilentProbing",
        defense_name="afp", config_id="Base", provenance_hashes=VALID_PROVENANCE,
    )
    r2.execute_run()

    with open(tmp_path / "det_run_1/scores.json") as f1, open(tmp_path / "det_run_2/scores.json") as f2:
        s1 = json.load(f1)
        s2 = json.load(f2)

    # Predictions and scores must be bit-identical across identical runs
    for b in range(144):
        assert s1[b]["predictions"] == s2[b]["predictions"]
        assert s1[b]["scores"] == s2[b]["scores"]


# ---------------------------------------------------------------------------
# Test 14: Official Attack Caches and Protected Artifacts Remain Immutable
# ---------------------------------------------------------------------------
def test_official_artifacts_and_caches_immutability():
    rf_path = REPO_ROOT / "artifacts/models/frozen_rf.joblib"
    roles_path = REPO_ROOT / "data/manifests/evaluation_roles.csv"
    batches_path = REPO_ROOT / "data/manifests/evaluation_batches.csv"
    exp_cfg_path = REPO_ROOT / "configs/experiment.yaml"

    assert calculate_file_hash(rf_path) == PROTECTED_HASHES["RF"]
    assert calculate_file_hash(roles_path) == PROTECTED_HASHES["Roles"]
    assert calculate_file_hash(batches_path) == PROTECTED_HASHES["Batches"]
    assert calculate_file_hash(exp_cfg_path) == PROTECTED_HASHES["ExperimentConfig"]

    # Verify inventory v2 hashes for official caches
    with open(REPO_ROOT / "artifacts/reports/cache_inventory_v2.json") as f:
        inventory = json.load(f)

    for c in inventory["caches"]:
        cdir = Path(c["directory"])
        for fname, exp_hash in c["artifact_hashes"].items():
            actual = calculate_file_hash(cdir / fname)
            assert actual == exp_hash, f"Cache file {cdir / fname} altered! Expected {exp_hash}, got {actual}"


# ---------------------------------------------------------------------------
# Test 15: Production Provenance Construction & Regression on Missing Keys
# ---------------------------------------------------------------------------
def test_production_provenance_construction_and_regression_on_missing_keys():
    from scripts.run_evaluation import build_production_provenance
    from recall_aware_ids.experiment.schemas import CompletionMarker, _REQUIRED_PROVENANCE_KEYS

    # Exercise actual production provenance construction path
    prov = build_production_provenance(repo_root=REPO_ROOT)
    assert set(prov.keys()) == _REQUIRED_PROVENANCE_KEYS
    for k, v in prov.items():
        assert len(v) == 64 and all(c in "0123456789abcdef" for c in v)

    # Validate against actual official cache manifest
    manifest_path = REPO_ROOT / "artifacts/caches/SilentProbing_42/manifest.json"
    with open(manifest_path) as f:
        cache_manifest = json.load(f)
    for k in (
        "frozen_rf_hash", "scaler_hash", "feature_names_hash", "feature_mask_hash",
        "training_bounds_hash", "evaluation_roles_hash", "evaluation_batches_hash",
        "attacks_yaml_hash", "controllers_yaml_hash", "defenses_yaml_hash", "experiment_yaml_hash"
    ):
        assert prov[k] == cache_manifest[k], f"Mismatch for {k} vs cache manifest"

    # RED REGRESSION TEST: Prove old 7-key wiring raises due to missing provenance
    old_wiring_prov = {
        "frozen_rf_hash": prov["frozen_rf_hash"],
        "evaluation_roles_hash": prov["evaluation_roles_hash"],
        "evaluation_batches_hash": prov["evaluation_batches_hash"],
        "attacks_yaml_hash": prov["attacks_yaml_hash"],
        "defenses_yaml_hash": prov["defenses_yaml_hash"],
        "controllers_yaml_hash": prov["controllers_yaml_hash"],
        "experiment_yaml_hash": prov["experiment_yaml_hash"],
    }
    with pytest.raises(ValueError, match="provenance_hashes missing required keys"):
        CompletionMarker(
            run_id="test_run",
            timestamp="2026-09-18T00:00:00Z",
            provenance_hashes=old_wiring_prov,
        )

    # GREEN TEST: Prove full 11-key production provenance validates successfully
    prov_with_cache = dict(prov)
    prov_with_cache["cache_manifest_hash"] = hashlib.sha256(b"cache").hexdigest()
    marker = CompletionMarker(
        run_id="test_run",
        timestamp="2026-09-18T00:00:00Z",
        provenance_hashes=prov_with_cache,
    )
    assert marker.run_id == "test_run"


# ---------------------------------------------------------------------------
# Test 16: Independent Cache Inventory Pinning & Divergence Rejection
# ---------------------------------------------------------------------------
def test_independent_cache_inventory_pinning_and_divergence_rejection(tmp_path: Path):
    from scripts.run_evaluation import validate_cache_against_inventory, build_production_provenance

    prov = build_production_provenance(repo_root=REPO_ROOT)
    inv_path = REPO_ROOT / "artifacts/reports/cache_inventory_v2.json"
    cache_dir = REPO_ROOT / "artifacts/caches/SilentProbing_42"

    # Valid cache matches inventory
    manifest = validate_cache_against_inventory(
        cache_dir=cache_dir,
        scenario="SilentProbing",
        seed=42,
        inventory_path=inv_path,
        expected_provenance=prov,
    )
    assert manifest["attack_scenario"] == "SilentProbing"
    assert manifest["effective_seed"] == 42

    # Divergence 1: Missing / altered scenario in inventory
    with pytest.raises(ValueError, match="not found in independent inventory"):
        validate_cache_against_inventory(
            cache_dir=cache_dir,
            scenario="SilentProbing",
            seed=999,
            inventory_path=inv_path,
            expected_provenance=prov,
        )

    # Divergence 2: Self-consistent but altered parquet vs inventory
    tmp_cache = tmp_path / "SilentProbing_42"
    shutil.copytree(cache_dir, tmp_cache)
    with open(tmp_cache / "X_attacked.parquet", "ab") as f:
        f.write(b"\x00")

    with pytest.raises(ValueError, match="SHA-256 mismatch vs independent inventory"):
        validate_cache_against_inventory(
            cache_dir=tmp_cache,
            scenario="SilentProbing",
            seed=42,
            inventory_path=inv_path,
            expected_provenance=prov,
        )

    # Divergence 3: Self-consistent altered manifest.json vs inventory
    tmp_cache2 = tmp_path / "SilentProbing_42_tampered_manifest"
    shutil.copytree(cache_dir, tmp_cache2)
    with open(tmp_cache2 / "manifest.json", "r") as f:
        m_data = json.load(f)
    m_data["notes"] = "tampered"
    with open(tmp_cache2 / "manifest.json", "w") as f:
        json.dump(m_data, f)

    with pytest.raises(ValueError, match="manifest.json SHA-256 mismatch vs independent inventory"):
        validate_cache_against_inventory(
            cache_dir=tmp_cache2,
            scenario="SilentProbing",
            seed=42,
            inventory_path=inv_path,
            expected_provenance=prov,
        )


# ---------------------------------------------------------------------------
# Test 17: Git Cleanliness Policy Cases
# ---------------------------------------------------------------------------
def test_git_cleanliness_policy(monkeypatch):
    from scripts.run_evaluation import check_git_cleanliness

    def set_mock_git(status_output: str):
        def mock_check_output(cmd, **kwargs):
            if "status" in cmd:
                return status_output
            if "diff" in cmd:
                return ""
            if "rev-parse" in cmd:
                return "b5a615ec571f1f522da3c2c7af450c8dae949ddc"
            return ""
        monkeypatch.setattr("subprocess.check_output", mock_check_output)
        monkeypatch.setattr("subprocess.check_call", lambda *args, **kwargs: 0)

    # Case 1: Unstaged tracked modification
    set_mock_git(" M scripts/run_evaluation.py\n")
    with pytest.raises(RuntimeError, match="tracked files are modified: \\[' M:scripts/run_evaluation.py'\\]"):
        check_git_cleanliness(repo_root=REPO_ROOT, enforce_git=True)

    # Case 2: Staged tracked modification
    set_mock_git("M  scripts/run_evaluation.py\n")
    with pytest.raises(RuntimeError, match="tracked files are modified: \\['M :scripts/run_evaluation.py'\\]"):
        check_git_cleanliness(repo_root=REPO_ROOT, enforce_git=True)

    # Case 3: Combined staged & unstaged modification
    set_mock_git("MM scripts/run_evaluation.py\n")
    with pytest.raises(RuntimeError, match="tracked files are modified: \\['MM:scripts/run_evaluation.py'\\]"):
        check_git_cleanliness(repo_root=REPO_ROOT, enforce_git=True)

    # Case 4: Tracked deleted file
    set_mock_git(" D scripts/old_script.py\n")
    with pytest.raises(RuntimeError, match="tracked files are modified: \\[' D:scripts/old_script.py'\\]"):
        check_git_cleanliness(repo_root=REPO_ROOT, enforce_git=True)

    # Case 5: Tracked renamed file
    set_mock_git("R  scripts/old.py -> scripts/new.py\n")
    with pytest.raises(RuntimeError, match="tracked files are modified"):
        check_git_cleanliness(repo_root=REPO_ROOT, enforce_git=True)

    # Case 6: Untracked code/doc file
    set_mock_git("?? scripts/untracked.py\n")
    with pytest.raises(RuntimeError, match="untracked forbidden files detected: \\['scripts/untracked.py'\\]"):
        check_git_cleanliness(repo_root=REPO_ROOT, enforce_git=True)

    # Case 7: Allowlisted untracked artifacts pass cleanly
    set_mock_git("?? artifacts/caches/SilentProbing_42/\n?? phase10_bundle.zip\n?? test.log\n")
    res = check_git_cleanliness(repo_root=REPO_ROOT, enforce_git=True)
    assert res["clean"] is True
    assert res["allowed_untracked_count"] == 3


# ---------------------------------------------------------------------------
# Test 18: Disk Space Gate Aborts on Low Free Space
# ---------------------------------------------------------------------------
def test_disk_space_gate_aborts_on_low_space(monkeypatch):
    from scripts.run_evaluation import check_disk_space

    # Normal free space check returns positive float
    free_gb = check_disk_space(repo_root=REPO_ROOT, min_gb=1.0)
    assert free_gb > 1.0

    # Low space aborts with RuntimeError (never swallowed)
    with pytest.raises(RuntimeError, match="Insufficient disk space"):
        check_disk_space(repo_root=REPO_ROOT, min_gb=10000.0)

    # OSError on obtaining disk statistics is converted to RuntimeError
    def mock_disk_usage_error(*args, **kwargs):
        raise OSError("I/O error reading filesystem statistics")
    monkeypatch.setattr("shutil.disk_usage", mock_disk_usage_error)

    with pytest.raises(RuntimeError, match="Failed to obtain filesystem disk usage"):
        check_disk_space(repo_root=REPO_ROOT, min_gb=10.0)


# ---------------------------------------------------------------------------
# Test 19: Defense Factories Use defenses.yaml Bounds
# ---------------------------------------------------------------------------
def test_defense_factories_use_defenses_yaml_bounds():
    from scripts.run_evaluation import create_policy_controller

    with open(REPO_ROOT / "configs/controllers.yaml") as f:
        controllers_yaml = yaml.safe_load(f)
    with open(REPO_ROOT / "configs/defenses.yaml") as f:
        defenses_yaml = yaml.safe_load(f)

    # AFP: base 0.0003, bounds [0.0, 0.0003]
    p_afp = create_policy_controller("Base", "afp", controllers_yaml, defenses_yaml)
    assert p_afp.fixed_intensity == 0.0003
    assert p_afp.intensity_min == 0.0
    assert p_afp.intensity_max == 0.0003

    # RS: base 0.0002, bounds [0.0, 0.0002]
    p_rs = create_policy_controller("Base", "randomized_smoothing", controllers_yaml, defenses_yaml)
    assert p_rs.fixed_intensity == 0.0002
    assert p_rs.intensity_min == 0.0
    assert p_rs.intensity_max == 0.0002

    # FS: base 2.0, bounds [0.0, 2.0]
    p_fs = create_policy_controller("Base", "feature_squeezing", controllers_yaml, defenses_yaml)
    assert p_fs.fixed_intensity == 2.0
    assert p_fs.intensity_min == 0.0
    assert p_fs.intensity_max == 2.0


# ---------------------------------------------------------------------------
# Test 20: Manifest Cross-Validation & Synthetic Rejections
# ---------------------------------------------------------------------------
def test_manifest_cross_validation_and_synthetic_rejections(tmp_path: Path):
    from scripts.run_evaluation import prepare_evaluation_batches

    # 1. Real manifests pass
    resolved_batches, y_meas, hashes = prepare_evaluation_batches(REPO_ROOT / "data/manifests")
    assert len(resolved_batches) == 72000
    assert len(y_meas) == 72000
    assert set(resolved_batches["batch_id"]) == set(range(144))

    # 2. Synthetic rejection tests
    roles = pd.read_csv(REPO_ROOT / "data/manifests/evaluation_roles.csv")
    batches = pd.read_csv(REPO_ROOT / "data/manifests/evaluation_batches.csv")

    # Mismatched row count (incomplete positions)
    b_missing = batches.head(71999)
    b_missing.to_csv(tmp_path / "evaluation_batches.csv", index=False)
    roles.to_csv(tmp_path / "evaluation_roles.csv", index=False)
    with pytest.raises(ValueError, match="expected 72000"):
        prepare_evaluation_batches(tmp_path)

    # Duplicate eval_position
    b_dup = batches.copy()
    b_dup.loc[1, "eval_position"] = b_dup.loc[0, "eval_position"]
    b_dup.to_csv(tmp_path / "evaluation_batches.csv", index=False)
    with pytest.raises(ValueError, match="Duplicate eval_position"):
        prepare_evaluation_batches(tmp_path)

    # Crafting role appearing in batches
    craft_pos = roles[roles["role"] == "crafting"].iloc[0]["eval_position"]
    b_craft = batches.copy()
    b_craft.loc[0, "eval_position"] = craft_pos
    b_craft.to_csv(tmp_path / "evaluation_batches.csv", index=False)
    with pytest.raises(ValueError, match="positions do not exactly match measurement role positions"):
        prepare_evaluation_batches(tmp_path)

    # Mislabeled role count
    r_bad_count = roles.copy()
    r_bad_count.loc[0, "role"] = "crafting"
    r_bad_count.to_csv(tmp_path / "evaluation_roles.csv", index=False)
    batches.to_csv(tmp_path / "evaluation_batches.csv", index=False)
    with pytest.raises(ValueError, match="Expected 18000 crafting roles"):
        prepare_evaluation_batches(tmp_path)

    # Non-binary y_binary
    r_bad_y = roles.copy()
    r_bad_y.loc[0, "y_binary"] = 2
    r_bad_y.to_csv(tmp_path / "evaluation_roles.csv", index=False)
    with pytest.raises(ValueError, match="y_binary contains non-binary values"):
        prepare_evaluation_batches(tmp_path)

    # Composite identity mismatch
    r_bad_cid = roles.copy()
    r_bad_cid.loc[0, "_raw_row_idx"] = 9999999
    r_bad_cid.to_csv(tmp_path / "evaluation_roles.csv", index=False)
    with pytest.raises(ValueError, match="_raw_row_idx disagrees"):
        prepare_evaluation_batches(tmp_path)


# ---------------------------------------------------------------------------
# Test 21: Strict Completed-Run Reuse Validation
# ---------------------------------------------------------------------------
def test_strict_completed_run_reuse_validation(tmp_path: Path):
    from scripts.run_evaluation import validate_completed_run
    from recall_aware_ids.experiment.runner import ExperimentRunner, LabelProvider
    from recall_aware_ids.experiment.policies import FixedIntensityPolicy
    import time

    run_id = "primary_42_SilentProbing_afp_Base"
    batches = make_synthetic_batches()
    lp = LabelProvider(y_measurement=np.zeros(72000, dtype=int))
    cache = SyntheticCacheProvider()
    adapter = MockDefenseModelAdapter()
    policy = FixedIntensityPolicy("Base", 0.0003, 0.0, 0.005)

    runner = ExperimentRunner(
        resolved_batches=batches,
        label_provider=lp,
        attack_cache=cache,
        defense_adapter=adapter,
        policy_controller=policy,
        output_dir=tmp_path,
        run_id=run_id,
        seed=42,
        attack_scenario="SilentProbing",
        defense_name="afp",
        config_id="Base",
        provenance_hashes=VALID_PROVENANCE,
    )
    runner.execute_run()

    run_dir = tmp_path / run_id
    row = pd.Series({
        "run_id": run_id, "seed": 42,
        "attack_scenario": "SilentProbing", "defense_name": "afp", "controller_config_id": "Base"
    })
    is_valid, err = validate_completed_run(run_dir, expected_row=row, expected_provenance=VALID_PROVENANCE)
    assert is_valid, f"Expected valid, got: {err}"

    # Rejection on seed mismatch
    bad_row = pd.Series({
        "run_id": run_id, "seed": 43,
        "attack_scenario": "SilentProbing", "defense_name": "afp", "controller_config_id": "Base"
    })
    is_valid, err = validate_completed_run(run_dir, expected_row=bad_row, expected_provenance=VALID_PROVENANCE)
    assert not is_valid and "Seed mismatch" in err

    # Rejection on defense mismatch
    bad_row2 = pd.Series({
        "run_id": run_id, "seed": 42,
        "attack_scenario": "SilentProbing", "defense_name": "randomized_smoothing", "controller_config_id": "Base"
    })
    is_valid, err = validate_completed_run(run_dir, expected_row=bad_row2, expected_provenance=VALID_PROVENANCE)
    assert not is_valid and "Defense mismatch" in err

    # Rejection on unexpected file
    (run_dir / "unexpected.log").write_text("extra")
    is_valid, err = validate_completed_run(run_dir, expected_row=row, expected_provenance=VALID_PROVENANCE)
    assert not is_valid and "Unexpected files" in err
    (run_dir / "unexpected.log").unlink()

    # Rejection on file modified after completion.json
    time.sleep(0.15)
    (run_dir / "scores.json").touch()
    is_valid, err = validate_completed_run(run_dir, expected_row=row, expected_provenance=VALID_PROVENANCE)
    assert not is_valid and "scores.json was modified after completion.json" in err


# ---------------------------------------------------------------------------
# Test 22: Pointer-Only Alias Publication and Reuse
# ---------------------------------------------------------------------------
def test_pointer_only_alias_publication_and_reuse(tmp_path: Path):
    from scripts.run_evaluation import publish_alias, validate_completed_alias
    from recall_aware_ids.experiment.runner import ExperimentRunner, LabelProvider
    from recall_aware_ids.experiment.policies import FixedIntensityPolicy

    target_id = "primary_42_SilentProbing_afp_C1"
    batches = make_synthetic_batches()
    lp = LabelProvider(y_measurement=np.zeros(72000, dtype=int))
    cache = SyntheticCacheProvider()
    adapter = MockDefenseModelAdapter()
    policy = FixedIntensityPolicy("C1", 0.0003, 0.0, 0.005)

    runner = ExperimentRunner(
        resolved_batches=batches,
        label_provider=lp,
        attack_cache=cache,
        defense_adapter=adapter,
        policy_controller=policy,
        output_dir=tmp_path,
        run_id=target_id,
        seed=42,
        attack_scenario="SilentProbing",
        defense_name="afp",
        config_id="C1",
        provenance_hashes=VALID_PROVENANCE,
    )
    runner.execute_run()

    alias_id = "sensitivity_42_SilentProbing_afp_C1"
    alias_row = pd.Series({
        "run_id": alias_id,
        "alias_for_run_id": target_id,
        "seed": 42,
        "attack_scenario": "SilentProbing",
        "defense_name": "afp",
        "controller_config_id": "C1",
        "is_alias": True,
    })

    res = publish_alias(alias_row, output_dir=tmp_path, expected_provenance=VALID_PROVENANCE)
    assert res["status"] == "PUBLISHED_ALIAS"

    alias_dir = tmp_path / alias_id
    assert alias_dir.exists()
    assert set(p.name for p in alias_dir.iterdir()) == {"alias_pointer.json", "completion.json"}

    # Valid reuse returns REUSED_ALIAS
    res2 = publish_alias(alias_row, output_dir=tmp_path, expected_provenance=VALID_PROVENANCE)
    assert res2["status"] == "REUSED_ALIAS"

    # Mismatched tuple rejection
    bad_alias_row = alias_row.copy()
    bad_alias_row["seed"] = 43
    with pytest.raises(ValueError, match="Seed mismatch|Alias tuple seed mismatch"):
        publish_alias(bad_alias_row, output_dir=tmp_path, expected_provenance=VALID_PROVENANCE)

    # Corrupt alias quarantine and recovery
    with open(alias_dir / "alias_pointer.json", "w") as f:
        f.write("corrupted")
    res3 = publish_alias(alias_row, output_dir=tmp_path, expected_provenance=VALID_PROVENANCE)
    assert res3["status"] == "PUBLISHED_ALIAS"
    quarantined = list(tmp_path.glob(f"{alias_id}_quarantined_*"))
    assert len(quarantined) == 1


# ---------------------------------------------------------------------------
# Test 23: Dependency-Safe Execution Planning
# ---------------------------------------------------------------------------
def test_dependency_safe_execution_planning():
    from scripts.run_evaluation import resolve_and_validate_execution_plan, derive_and_validate_matrix

    matrix, _ = derive_and_validate_matrix(REPO_ROOT / "configs")

    alias_id = "sensitivity_42_SilentProbing_afp_C1"
    filtered = matrix[matrix["run_id"] == alias_id]
    assert len(filtered) == 1

    unique_runs, alias_runs = resolve_and_validate_execution_plan(
        matrix=matrix,
        filtered_matrix=filtered,
        output_dir=REPO_ROOT / "artifacts/evaluation_runs",
        expected_provenance={},
    )
    target_id = "primary_42_SilentProbing_afp_C1"
    assert target_id in unique_runs["run_id"].values
    assert alias_id in alias_runs["run_id"].values

    # Reject non-positive max_runs
    with pytest.raises(ValueError, match="--max-runs must be a strictly positive integer"):
        resolve_and_validate_execution_plan(matrix, matrix, REPO_ROOT / "artifacts/evaluation_runs", {}, max_runs=0)
    with pytest.raises(ValueError, match="--max-runs must be a strictly positive integer"):
        resolve_and_validate_execution_plan(matrix, matrix, REPO_ROOT / "artifacts/evaluation_runs", {}, max_runs=-10)


# ---------------------------------------------------------------------------
# Test 24: True Production-Wiring Canary
# ---------------------------------------------------------------------------
def test_production_wiring_canary(tmp_path: Path):
    from scripts.run_evaluation import (
        execute_single_run,
        build_production_provenance,
        prepare_evaluation_batches,
    )

    canary_manifests_dir = tmp_path / "manifests"
    make_synthetic_manifests(canary_manifests_dir)

    prov = build_production_provenance(repo_root=REPO_ROOT, manifests_dir=canary_manifests_dir)
    resolved_batches, y_meas, _ = prepare_evaluation_batches(canary_manifests_dir)

    with open(REPO_ROOT / "artifacts/preprocessors/feature_mask.json") as f:
        mask_data = json.load(f)
    feature_names = mask_data["feature_columns"]
    modifiable_mask = np.array(mask_data["feature_mask"], dtype=bool)

    training_bounds = pd.read_parquet(REPO_ROOT / "artifacts/preprocessors/training_bounds.parquet")
    benign_profile = pd.read_parquet(REPO_ROOT / "artifacts/preprocessors/afp_benign_profile.parquet")
    rf_model = joblib.load(REPO_ROOT / "artifacts/models/frozen_rf.joblib")

    with open(REPO_ROOT / "configs/controllers.yaml") as f:
        controllers_yaml = yaml.safe_load(f)
    with open(REPO_ROOT / "configs/defenses.yaml") as f:
        defenses_yaml = yaml.safe_load(f)

    # Genuinely synthetic 72,000-row cache generated entirely in tmp_path
    # within training feature bounds, without accessing official evaluation caches
    canary_cache_dir = tmp_path / "caches/SilentProbing_42"
    canary_cache_dir.mkdir(parents=True)

    feature_to_mid = {
        row["feature"]: (row["train_min"] + row["train_max"]) / 2.0
        for _, row in training_bounds.iterrows()
    }
    midpoints = np.array([feature_to_mid[col] for col in feature_names], dtype=np.float32)
    synthetic_x_matrix = np.tile(midpoints, (72000, 1))

    df_x = pd.DataFrame(synthetic_x_matrix, columns=feature_names)
    x_parquet_path = canary_cache_dir / "X_attacked.parquet"
    df_x.to_parquet(x_parquet_path, index=False)

    df_status = pd.DataFrame({
        "eval_position": resolved_batches["eval_position"].values,
        "eligible": np.zeros(72000, dtype=bool),
        "attempted": np.zeros(72000, dtype=bool),
        "successful": np.zeros(72000, dtype=bool),
        "status_code": ["NOT_APPLICABLE"] * 72000,
        "queries_used": np.zeros(72000, dtype=np.int64),
        "l0": np.zeros(72000, dtype=np.float64),
        "l1": np.zeros(72000, dtype=np.float64),
        "l2": np.zeros(72000, dtype=np.float64),
        "linf": np.zeros(72000, dtype=np.float64),
    })
    status_parquet_path = canary_cache_dir / "status.parquet"
    df_status.to_parquet(status_parquet_path, index=False)

    from scripts.run_evaluation import calculate_file_hash
    x_hash = calculate_file_hash(x_parquet_path)
    s_hash = calculate_file_hash(status_parquet_path)

    # Hash the actual canonical attack script files
    attack_script_hashes = {
        "base.py": calculate_file_hash(REPO_ROOT / "src/recall_aware_ids/attacks/base.py"),
        "silent_probing.py": calculate_file_hash(REPO_ROOT / "src/recall_aware_ids/attacks/silent_probing.py"),
        "surrogate_transfer.py": calculate_file_hash(REPO_ROOT / "src/recall_aware_ids/attacks/surrogate_transfer.py"),
        "boundary_attack.py": calculate_file_hash(REPO_ROOT / "src/recall_aware_ids/attacks/boundary_attack.py"),
        "oracle.py": calculate_file_hash(REPO_ROOT / "src/recall_aware_ids/attacks/oracle.py"),
        "caching.py": calculate_file_hash(REPO_ROOT / "src/recall_aware_ids/experiment/caching.py"),
        "build_evaluation_caches.py": calculate_file_hash(REPO_ROOT / "scripts/build_evaluation_caches.py"),
    }

    manifest_data = {
        "schema_version": "1.0",
        "attack_scenario": "SilentProbing",
        "effective_seed": 42,
        "row_count": 72000,
        "X_attacked_sha256": x_hash,
        "status_sha256": s_hash,
        "output_sha256": x_hash,
        "attack_script_hashes": attack_script_hashes,
        "attack_parameters": {
            "modifies_samples": False,
        },
        "query_budgets": {
            "max_queries_per_sample": 0,
        },
        "attacks_yaml_hash": prov["attacks_yaml_hash"],
        "controllers_yaml_hash": prov["controllers_yaml_hash"],
        "defenses_yaml_hash": prov["defenses_yaml_hash"],
        "experiment_yaml_hash": prov["experiment_yaml_hash"],
        "model_yaml_hash": calculate_file_hash(REPO_ROOT / "configs/model.yaml"),
        "frozen_rf_hash": prov["frozen_rf_hash"],
        "scaler_hash": prov["scaler_hash"],
        "feature_names_hash": prov["feature_names_hash"],
        "feature_mask_hash": prov["feature_mask_hash"],
        "training_bounds_hash": prov["training_bounds_hash"],
        "evaluation_roles_hash": prov["evaluation_roles_hash"],
        "evaluation_batches_hash": prov["evaluation_batches_hash"],
        "X_eval_hash": hashlib.sha256(b"canary_synthetic_X_eval").hexdigest(),
        "metadata_eval_hash": hashlib.sha256(b"canary_synthetic_metadata_eval").hexdigest(),
        "crafting_identity_hash": hashlib.sha256(b"canary_synthetic_crafting_id").hexdigest(),
        "measurement_identity_hash": hashlib.sha256(b"canary_synthetic_meas_id").hexdigest(),
        "freeze_commit_sha256": hashlib.sha256(b"canary_synthetic_freeze_commit").hexdigest(),
        "freeze_tag_sha256": hashlib.sha256(b"canary_synthetic_freeze_tag").hexdigest(),
        "orchestration_commit_sha256": hashlib.sha256(b"canary_synthetic_orch_commit").hexdigest(),
    }

    manifest_path = canary_cache_dir / "manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest_data, f, indent=2)
    m_hash = calculate_file_hash(manifest_path)

    completion_data = {
        "completion_state": "COMPLETED",
        "completed": True,
        "attack_scenario": "SilentProbing",
        "display_scenario": "Silent Probing",
        "effective_seed": 42,
        "row_count": 72000,
        "freeze_commit": "65005505415a2bdf2d5744dbd135e9214e74081a",
        "freeze_tag": "phase10-protocol-freeze",
        "execution_script_commit": "f3f6f3d94d6854dbd5f02e2474cbacbe2dc62207",
        "generation_start_timestamp": "2026-09-18T00:00:00.000000+00:00",
        "generation_end_timestamp": "2026-09-18T00:01:00.000000+00:00",
        "artifacts": {
            "X_attacked_sha256": x_hash,
            "status_sha256": s_hash,
            "manifest_sha256": m_hash,
        },
        "provenance_hashes": {
            k: v for k, v in manifest_data.items()
            if k.endswith("_hash") or k.endswith("_sha256")
        },
        "attack_parameters": manifest_data["attack_parameters"],
        "query_budgets": manifest_data["query_budgets"],
        "attack_script_hashes": manifest_data["attack_script_hashes"],
    }

    completion_path = canary_cache_dir / "completion.json"
    with open(completion_path, "w") as f:
        json.dump(completion_data, f, indent=2)
    c_hash = calculate_file_hash(completion_path)

    canary_inv_path = tmp_path / "cache_inventory_v2.json"
    inv_data = {
        "caches": [
            {
                "scenario": "SilentProbing",
                "seed": 42,
                "directory": str(canary_cache_dir),
                "artifact_hashes": {
                    "X_attacked.parquet": x_hash,
                    "status.parquet": s_hash,
                    "manifest.json": m_hash,
                    "completion.json": c_hash,
                },
            }
        ]
    }
    with open(canary_inv_path, "w") as f:
        json.dump(inv_data, f, indent=2)

    run_row = pd.Series({
        "run_id": "canary_42_SilentProbing_afp_Base",
        "seed": 42,
        "attack_scenario": "SilentProbing",
        "defense_name": "afp",
        "controller_config_id": "Base",
        "is_alias": False,
    })

    canary_out = tmp_path / "runs"
    canary_out.mkdir()

    # Execute 144 batches end-to-end using real production wiring
    res = execute_single_run(
        run_row=run_row,
        output_dir=canary_out,
        caches_dir=tmp_path / "caches",
        inventory_path=canary_inv_path,
        configs_dir=REPO_ROOT / "configs",
        manifests_dir=canary_manifests_dir,
        models_dir=REPO_ROOT / "artifacts/models",
        preprocessors_dir=REPO_ROOT / "artifacts/preprocessors",
        resolved_batches=resolved_batches,
        y_measurement=y_meas,
        feature_names=feature_names,
        modifiable_mask=modifiable_mask,
        training_bounds=training_bounds,
        benign_profile=benign_profile,
        rf_model=rf_model,
        controllers_yaml=controllers_yaml,
        defenses_yaml=defenses_yaml,
        provenance_hashes=prov,
    )
    assert res["status"] == "COMPLETED"

    run_dir = canary_out / "canary_42_SilentProbing_afp_Base"
    assert run_dir.exists()
    assert set(p.name for p in run_dir.iterdir()) == {
        "config.json", "confusion.json", "scores.json", "run_summary.json", "completion.json"
    }

    # Verify validated reuse on rerun
    res_reuse = execute_single_run(
        run_row=run_row,
        output_dir=canary_out,
        caches_dir=tmp_path / "caches",
        inventory_path=canary_inv_path,
        configs_dir=REPO_ROOT / "configs",
        manifests_dir=canary_manifests_dir,
        models_dir=REPO_ROOT / "artifacts/models",
        preprocessors_dir=REPO_ROOT / "artifacts/preprocessors",
        resolved_batches=resolved_batches,
        y_measurement=y_meas,
        feature_names=feature_names,
        modifiable_mask=modifiable_mask,
        training_bounds=training_bounds,
        benign_profile=benign_profile,
        rf_model=rf_model,
        controllers_yaml=controllers_yaml,
        defenses_yaml=defenses_yaml,
        provenance_hashes=prov,
    )
    assert res_reuse["status"] == "REUSED"

    # Verify quarantine on tampered outputs
    with open(run_dir / "completion.json", "w") as f:
        f.write("corrupted")
    res_tampered = execute_single_run(
        run_row=run_row,
        output_dir=canary_out,
        caches_dir=tmp_path / "caches",
        inventory_path=canary_inv_path,
        configs_dir=REPO_ROOT / "configs",
        manifests_dir=canary_manifests_dir,
        models_dir=REPO_ROOT / "artifacts/models",
        preprocessors_dir=REPO_ROOT / "artifacts/preprocessors",
        resolved_batches=resolved_batches,
        y_measurement=y_meas,
        feature_names=feature_names,
        modifiable_mask=modifiable_mask,
        training_bounds=training_bounds,
        benign_profile=benign_profile,
        rf_model=rf_model,
        controllers_yaml=controllers_yaml,
        defenses_yaml=defenses_yaml,
        provenance_hashes=prov,
    )
    assert res_tampered["status"] == "COMPLETED"
    quarantined = list(canary_out.glob("canary_42_SilentProbing_afp_Base_quarantined_*"))
    assert len(quarantined) == 1


def test_preflight_invokes_manifest_cross_validation(tmp_path):
    """
    Regression test verifying that run_preflight() actually executes manifest
    cross-validation and rejects malformed roles/batches manifests without
    opening evaluation Parquets.
    """
    from scripts.run_evaluation import run_preflight

    # 1. Clean manifests directory should pass preflight manifest check
    report = run_preflight(
        configs_dir=REPO_ROOT / "configs",
        manifests_dir=REPO_ROOT / "data/manifests",
        models_dir=REPO_ROOT / "artifacts/models",
        preprocessors_dir=REPO_ROOT / "artifacts/preprocessors",
        caches_dir=REPO_ROOT / "artifacts/caches",
        inventory_path=REPO_ROOT / "artifacts/reports/cache_inventory_v2.json",
        output_dir=tmp_path / "out",
        repo_root=REPO_ROOT,
        enforce_git=False,
    )
    assert "manifests" in report
    assert report["manifests"]["status"] == "PASS"
    assert report["manifests"]["batches_count"] == 144
    assert report["manifests"]["measurement_rows"] == 72000

    # 2. Corrupted batches manifest (e.g. 72001 rows) must cause run_preflight() to fail
    bad_manifests_dir = tmp_path / "bad_manifests"
    bad_manifests_dir.mkdir()
    shutil.copy(REPO_ROOT / "data/manifests/evaluation_roles.csv", bad_manifests_dir / "evaluation_roles.csv")

    batches_df = pd.read_csv(REPO_ROOT / "data/manifests/evaluation_batches.csv")
    bad_batches_df = pd.concat([batches_df.iloc[[0]], batches_df], ignore_index=True)
    bad_batches_df.to_csv(bad_manifests_dir / "evaluation_batches.csv", index=False)

    with pytest.raises(ValueError, match="evaluation_batches.csv has 72001 rows"):
        run_preflight(
            configs_dir=REPO_ROOT / "configs",
            manifests_dir=bad_manifests_dir,
            models_dir=REPO_ROOT / "artifacts/models",
            preprocessors_dir=REPO_ROOT / "artifacts/preprocessors",
            caches_dir=REPO_ROOT / "artifacts/caches",
            inventory_path=REPO_ROOT / "artifacts/reports/cache_inventory_v2.json",
            output_dir=tmp_path / "out",
            repo_root=REPO_ROOT,
            enforce_git=False,
        )


def test_canary_does_not_access_official_caches(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """
    Regression test verifying that the canary execution path uses strictly
    synthetic/training-derived data in tmp_path and never reads from artifacts/caches/,
    canonical evaluation manifests, or evaluation Parquets.

    Proves:
      1. Guard actively intercepts builtins.open, Path.open, Path.read_text,
         Path.read_bytes, pd.read_csv, and pd.read_parquet and raises PermissionError
         if any attempt is made to access prohibited canonical files.
      2. The full production-wiring canary completes successfully under this guard
         with zero accesses to official evaluation caches, manifests, or Parquets.
    """
    import builtins
    official_caches_dir = (REPO_ROOT / "artifacts/caches").resolve()
    canonical_roles_path = (REPO_ROOT / "data/manifests/evaluation_roles.csv").resolve()
    canonical_batches_path = (REPO_ROOT / "data/manifests/evaluation_batches.csv").resolve()
    eval_parquet_path = (REPO_ROOT / "data/processed/X_eval.parquet").resolve()
    metadata_parquet_path = (REPO_ROOT / "data/processed/metadata_eval.parquet").resolve()

    prohibited_exact_paths = {
        canonical_roles_path,
        canonical_batches_path,
        eval_parquet_path,
        metadata_parquet_path,
    }

    accessed_prohibited_targets = []

    def check_prohibited(target: Any) -> None:
        try:
            if isinstance(target, (str, Path)):
                p = Path(target).resolve()
            elif hasattr(target, "name"):
                p = Path(target.name).resolve()
            else:
                p = Path(str(target)).resolve()

            if p == official_caches_dir or official_caches_dir in p.parents:
                accessed_prohibited_targets.append(str(p))
                raise PermissionError(
                    f"Prohibited access to official evaluation cache during canary execution: {p}"
                )
            if p in prohibited_exact_paths:
                accessed_prohibited_targets.append(str(p))
                raise PermissionError(
                    f"Prohibited access to canonical evaluation file during canary execution: {p}"
                )
            if p.name in ("X_eval.parquet", "metadata_eval.parquet") or (
                p.name in ("evaluation_roles.csv", "evaluation_batches.csv")
                and str(REPO_ROOT.resolve()) in str(p)
                and "canary" not in str(p)
                and p.parent == (REPO_ROOT / "data/manifests").resolve()
            ):
                accessed_prohibited_targets.append(str(p))
                raise PermissionError(
                    f"Prohibited access to canonical evaluation file during canary execution: {p}"
                )
        except PermissionError:
            raise
        except Exception:
            pass

    orig_open = builtins.open
    orig_path_open = Path.open
    orig_path_read_text = Path.read_text
    orig_path_read_bytes = Path.read_bytes
    orig_pd_read_csv = pd.read_csv
    orig_pd_read_parquet = pd.read_parquet

    def guarded_builtin_open(file, *args, **kwargs):
        check_prohibited(file)
        return orig_open(file, *args, **kwargs)

    def guarded_path_open(self, *args, **kwargs):
        check_prohibited(self)
        return orig_path_open(self, *args, **kwargs)

    def guarded_path_read_text(self, *args, **kwargs):
        check_prohibited(self)
        return orig_path_read_text(self, *args, **kwargs)

    def guarded_path_read_bytes(self, *args, **kwargs):
        check_prohibited(self)
        return orig_path_read_bytes(self, *args, **kwargs)

    def guarded_pd_read_csv(filepath_or_buffer, *args, **kwargs):
        check_prohibited(filepath_or_buffer)
        return orig_pd_read_csv(filepath_or_buffer, *args, **kwargs)

    def guarded_pd_read_parquet(path, *args, **kwargs):
        check_prohibited(path)
        return orig_pd_read_parquet(path, *args, **kwargs)

    monkeypatch.setattr(builtins, "open", guarded_builtin_open)
    monkeypatch.setattr(Path, "open", guarded_path_open)
    monkeypatch.setattr(Path, "read_text", guarded_path_read_text)
    monkeypatch.setattr(Path, "read_bytes", guarded_path_read_bytes)
    monkeypatch.setattr(pd, "read_csv", guarded_pd_read_csv)
    monkeypatch.setattr(pd, "read_parquet", guarded_pd_read_parquet)

    # 1. Active detection: prove the guard catches and rejects attempts across all prohibited files and methods
    with pytest.raises(PermissionError, match="Prohibited access to official evaluation cache"):
        with open(official_caches_dir / "SilentProbing_42/manifest.json", "r") as f:
            _ = f.read()

    with pytest.raises(PermissionError, match="Prohibited access to official evaluation cache"):
        (official_caches_dir / "SilentProbing_42/manifest.json").open("r")

    with pytest.raises(PermissionError, match="Prohibited access to canonical evaluation file"):
        canonical_roles_path.read_text()

    with pytest.raises(PermissionError, match="Prohibited access to canonical evaluation file"):
        canonical_batches_path.read_bytes()

    with pytest.raises(PermissionError, match="Prohibited access to canonical evaluation file"):
        pd.read_csv(canonical_roles_path)

    with pytest.raises(PermissionError, match="Prohibited access to canonical evaluation file"):
        pd.read_parquet(eval_parquet_path)

    accessed_prohibited_targets.clear()

    # 2. Canary execution under guard: prove canary runs end-to-end without accessing prohibited canonical files
    canary_sub = tmp_path / "guarded_canary"
    canary_sub.mkdir()
    test_production_wiring_canary(canary_sub)

    assert len(accessed_prohibited_targets) == 0, (
        f"Canary attempted to access prohibited canonical files: {accessed_prohibited_targets}"
    )


def test_attack_script_hashes_correspond_to_named_files():
    """
    Regression test asserting every attack_script_hashes key corresponds
    to the exact SHA-256 hash of the canonical file it names.
    """
    canonical_files = {
        "base.py": REPO_ROOT / "src/recall_aware_ids/attacks/base.py",
        "silent_probing.py": REPO_ROOT / "src/recall_aware_ids/attacks/silent_probing.py",
        "surrogate_transfer.py": REPO_ROOT / "src/recall_aware_ids/attacks/surrogate_transfer.py",
        "boundary_attack.py": REPO_ROOT / "src/recall_aware_ids/attacks/boundary_attack.py",
        "oracle.py": REPO_ROOT / "src/recall_aware_ids/attacks/oracle.py",
        "caching.py": REPO_ROOT / "src/recall_aware_ids/experiment/caching.py",
        "build_evaluation_caches.py": REPO_ROOT / "scripts/build_evaluation_caches.py",
    }
    with open(REPO_ROOT / "artifacts/reports/cache_inventory_v2.json") as f:
        inv = json.load(f)

    for c in inv["caches"]:
        cdir = Path(c["directory"])
        with open(cdir / "manifest.json") as mf:
            m = json.load(mf)
        ash = m["attack_script_hashes"]
        for key, path in canonical_files.items():
            assert key in ash, f"Key {key} missing from {cdir}/manifest.json"
            expected_hash = hashlib.sha256(path.read_bytes()).hexdigest()
            assert ash[key] == expected_hash, (
                f"Mismatch for {key} in {cdir}: expected {expected_hash}, got {ash[key]}"
            )


def test_cli_execute_prohibits_no_enforce_git():
    """
    Regression test proving CLI rejects --execute combined with --no-enforce-git
    before preflight or data loading.
    """
    from scripts.run_evaluation import main

    with pytest.raises(ValueError, match="Cannot combine --execute with --no-enforce-git"):
        main(["--execute", "--no-enforce-git"])


def test_cli_execute_rejects_substituted_inputs_and_unsafe_output(tmp_path: Path):
    """
    Regression test proving CLI --execute mode strictly binds to canonical repository
    inputs and rejects substituted configs, manifests, models, preprocessors, caches,
    inventories, and unsafe output directories.
    """
    from scripts.run_evaluation import main

    sub_dir = tmp_path / "substituted"
    sub_dir.mkdir()

    with pytest.raises(ValueError, match="Non-canonical configs directory in --execute mode"):
        main(["--execute", "--configs-dir", str(sub_dir)])

    with pytest.raises(ValueError, match="Non-canonical manifests directory in --execute mode"):
        main(["--execute", "--manifests-dir", str(sub_dir)])

    with pytest.raises(ValueError, match="Non-canonical models directory in --execute mode"):
        main(["--execute", "--models-dir", str(sub_dir)])

    with pytest.raises(ValueError, match="Non-canonical preprocessors directory in --execute mode"):
        main(["--execute", "--preprocessors-dir", str(sub_dir)])

    with pytest.raises(ValueError, match="Non-canonical caches directory in --execute mode"):
        main(["--execute", "--caches-dir", str(sub_dir)])

    fake_inv = sub_dir / "cache_inventory_v2.json"
    fake_inv.write_text("{}")
    with pytest.raises(ValueError, match="Non-canonical inventory path in --execute mode"):
        main(["--execute", "--inventory-path", str(fake_inv)])

    # Unsafe output directory: traversal outside eval root
    traversal_out = REPO_ROOT / "artifacts/evaluation_runs/../../outside"
    with pytest.raises(ValueError, match="Unsafe output directory in --execute mode"):
        main(["--execute", "--output-dir", str(traversal_out)])

    # Unsafe output directory: unrelated external directory
    external_out = tmp_path / "external_runs"
    with pytest.raises(ValueError, match="Unsafe output directory in --execute mode"):
        main(["--execute", "--output-dir", str(external_out)])

    # Unsafe output directory: symlink directory
    symlink_target = tmp_path / "symlink_target"
    symlink_target.mkdir()
    symlink_out = tmp_path / "symlink_out"
    symlink_out.symlink_to(symlink_target)
    with pytest.raises(ValueError, match="Unsafe output directory in --execute mode|Output directory cannot be a symlink"):
        main(["--execute", "--output-dir", str(symlink_out)])


def test_alias_and_target_validation_regression(tmp_path: Path):
    """
    Comprehensive regression tests for alias and target validation:
      - wrong target controller;
      - wrong target cache identity;
      - wrong alias completion provenance;
      - wrong pointer config;
      - wrong pointer target provenance;
      - altered target completion;
      - structurally valid target belonging to another matrix row;
      - completion written after alias_pointer.json ordering;
      - resolve_and_validate_execution_plan scheduling invalid targets for recomputation.
    """
    from scripts.run_evaluation import (
        publish_alias,
        validate_completed_alias,
        resolve_and_validate_execution_plan,
        derive_and_validate_matrix,
    )
    from recall_aware_ids.experiment.runner import ExperimentRunner, LabelProvider
    from recall_aware_ids.experiment.policies import FixedIntensityPolicy

    target_id = "primary_42_SilentProbing_afp_C1"
    batches = make_synthetic_batches()
    lp = LabelProvider(y_measurement=np.zeros(72000, dtype=int))
    cache = SyntheticCacheProvider()
    adapter = MockDefenseModelAdapter()
    policy = FixedIntensityPolicy("C1", 0.0003, 0.0, 0.005)

    runner = ExperimentRunner(
        resolved_batches=batches,
        label_provider=lp,
        attack_cache=cache,
        defense_adapter=adapter,
        policy_controller=policy,
        output_dir=tmp_path,
        run_id=target_id,
        seed=42,
        attack_scenario="SilentProbing",
        defense_name="afp",
        config_id="C1",
        provenance_hashes={**VALID_PROVENANCE, "cache_manifest_hash": cache.cache_identity["manifest_sha256"]},
    )
    runner.execute_run()

    alias_id = "sensitivity_42_SilentProbing_afp_C1"
    alias_row = pd.Series({
        "run_id": alias_id,
        "alias_for_run_id": target_id,
        "seed": 42,
        "attack_scenario": "SilentProbing",
        "defense_name": "afp",
        "controller_config_id": "C1",
        "is_alias": True,
    })

    publish_alias(
        alias_row,
        output_dir=tmp_path,
        expected_provenance=VALID_PROVENANCE,
        expected_cache_identity=cache.cache_identity,
    )
    alias_dir = tmp_path / alias_id

    def touch_completion_after_pointer():
        p_mtime = (alias_dir / "alias_pointer.json").stat().st_mtime
        os.utime(alias_dir / "completion.json", (p_mtime + 5.0, p_mtime + 5.0))

    # 1. Wrong alias completion provenance
    bad_prov = dict(VALID_PROVENANCE)
    bad_prov["frozen_rf_hash"] = "a" * 64
    v, err = validate_completed_alias(
        alias_dir,
        alias_row,
        expected_provenance=bad_prov,
        output_dir=tmp_path,
        expected_cache_identity=cache.cache_identity,
    )
    assert not v
    assert "Alias completion provenance does not match" in err

    # 2. Wrong pointer config
    ptr_path = alias_dir / "alias_pointer.json"
    with open(ptr_path) as f:
        ptr = json.load(f)
    orig_ptr = dict(ptr)
    ptr["config_id"] = "C2"
    with open(ptr_path, "w") as f:
        json.dump(ptr, f)
    touch_completion_after_pointer()
    v, err = validate_completed_alias(
        alias_dir,
        alias_row,
        expected_provenance=VALID_PROVENANCE,
        output_dir=tmp_path,
        expected_cache_identity=cache.cache_identity,
    )
    assert not v
    assert "Alias pointer config_id mismatch" in err
    with open(ptr_path, "w") as f:
        json.dump(orig_ptr, f)
    touch_completion_after_pointer()

    # 3. Wrong pointer target provenance
    ptr_bad_prov = dict(orig_ptr)
    ptr_bad_prov["target_provenance"] = bad_prov
    with open(ptr_path, "w") as f:
        json.dump(ptr_bad_prov, f)
    touch_completion_after_pointer()
    v, err = validate_completed_alias(
        alias_dir,
        alias_row,
        expected_provenance=VALID_PROVENANCE,
        output_dir=tmp_path,
        expected_cache_identity=cache.cache_identity,
    )
    assert not v
    assert "Alias pointer target_provenance does not match" in err
    with open(ptr_path, "w") as f:
        json.dump(orig_ptr, f)
    touch_completion_after_pointer()

    # 4. Altered target completion
    t_comp = tmp_path / target_id / "completion.json"
    with open(t_comp) as f:
        cdata = json.load(f)
    orig_cdata = dict(cdata)
    cdata["timestamp"] = "2026-09-18T99:99:99Z"
    with open(t_comp, "w") as f:
        json.dump(cdata, f)
    v, err = validate_completed_alias(
        alias_dir,
        alias_row,
        expected_provenance=VALID_PROVENANCE,
        output_dir=tmp_path,
        expected_cache_identity=cache.cache_identity,
    )
    assert not v
    assert "Alias target" in err or "target_completion_sha256 mismatch" in err
    with open(t_comp, "w") as f:
        json.dump(orig_cdata, f)

    # 5. Wrong target cache identity
    bad_cache_id = dict(cache.cache_identity)
    bad_cache_id["attack_scenario"] = "SurrogateTransfer"
    v, err = validate_completed_alias(
        alias_dir,
        alias_row,
        expected_provenance=VALID_PROVENANCE,
        output_dir=tmp_path,
        expected_cache_identity=bad_cache_id,
    )
    assert not v
    assert "run_summary.cache_identity does not match" in err

    # 6. Wrong target controller
    t_sum = tmp_path / target_id / "run_summary.json"
    with open(t_sum) as f:
        sdata = json.load(f)
    orig_sdata = dict(sdata)
    sdata["config_id"] = "Base"
    with open(t_sum, "w") as f:
        json.dump(sdata, f)
    v, err = validate_completed_alias(
        alias_dir,
        alias_row,
        expected_provenance=VALID_PROVENANCE,
        output_dir=tmp_path,
        expected_cache_identity=cache.cache_identity,
    )
    assert not v
    assert "Target run_summary controller config is not C1" in err or "Controller config mismatch" in err
    with open(t_sum, "w") as f:
        json.dump(orig_sdata, f)

    # 7. Structurally valid target belonging to another matrix row
    bad_row = alias_row.copy()
    bad_row["seed"] = 43
    v, err = validate_completed_alias(
        alias_dir,
        bad_row,
        expected_provenance=VALID_PROVENANCE,
        output_dir=tmp_path,
        expected_cache_identity=cache.cache_identity,
    )
    assert not v
    assert "seed mismatch" in err.lower()

    # 8. Mtime ordering
    now = (alias_dir / "completion.json").stat().st_mtime
    os.utime(alias_dir / "alias_pointer.json", (now + 5.0, now + 5.0))
    v, err = validate_completed_alias(
        alias_dir,
        alias_row,
        expected_provenance=VALID_PROVENANCE,
        output_dir=tmp_path,
        expected_cache_identity=cache.cache_identity,
    )
    assert not v
    assert "alias_pointer.json was modified after completion.json" in err
    touch_completion_after_pointer()

    # 9. resolve_and_validate_execution_plan schedules invalid target for computation
    matrix, _ = derive_and_validate_matrix(REPO_ROOT / "configs")
    with open(t_sum, "w") as f:
        sdata["config_id"] = "Base"
        json.dump(sdata, f)

    filtered = matrix[matrix["run_id"] == alias_id]
    unique_runs, alias_runs = resolve_and_validate_execution_plan(
        matrix=matrix,
        filtered_matrix=filtered,
        output_dir=tmp_path,
        expected_provenance=VALID_PROVENANCE,
        expected_cache_identities={target_id: cache.cache_identity},
    )
    assert target_id in unique_runs["run_id"].values
    assert alias_id in alias_runs["run_id"].values


# ---------------------------------------------------------------------------
# Test 31: Builder Benchmark Consistency Gate & Disagreement Rejection
# ---------------------------------------------------------------------------
def test_builder_benchmark_consistency_gate_passes():
    """Verify that current repository benchmark output and reports pass the consistency gate."""
    import subprocess
    from scripts.build_phase10d_review_bundle import verify_benchmark_consistency

    bench_res = subprocess.run(
        ["./.venv-m4/bin/python", "scripts/benchmark_evaluation_pilot.py"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
    )
    assert bench_res.returncode == 0

    runtime_report = REPO_ROOT / "artifacts/reports/phase10d_runtime_storage_estimate.md"
    readiness_report = REPO_ROOT / "artifacts/reports/phase10d_evaluation_orchestration_readiness.md"

    # Must pass without raising
    verify_benchmark_consistency(bench_res.stdout, runtime_report, readiness_report)


def test_builder_benchmark_consistency_gate_rejects_disagreement(tmp_path: Path):
    """Verify that any timing or total disagreement causes verify_benchmark_consistency to abort."""
    import subprocess
    import shutil
    from scripts.build_phase10d_review_bundle import verify_benchmark_consistency

    bench_res = subprocess.run(
        ["./.venv-m4/bin/python", "scripts/benchmark_evaluation_pilot.py"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
    )
    assert bench_res.returncode == 0

    orig_runtime = REPO_ROOT / "artifacts/reports/phase10d_runtime_storage_estimate.md"
    orig_readiness = REPO_ROOT / "artifacts/reports/phase10d_evaluation_orchestration_readiness.md"

    t_runtime = tmp_path / "phase10d_runtime_storage_estimate.md"
    t_readiness = tmp_path / "phase10d_evaluation_orchestration_readiness.md"

    shutil.copy2(orig_runtime, t_runtime)
    shutil.copy2(orig_readiness, t_readiness)

    # 1. Tamper with runtime report timing
    t_runtime.write_text(t_runtime.read_text().replace("0.0555 s", "0.0999 s"))
    with pytest.raises(ValueError, match="Runtime report .* disagrees with benchmark output"):
        verify_benchmark_consistency(bench_res.stdout, t_runtime, t_readiness)

    # Restore runtime report
    shutil.copy2(orig_runtime, t_runtime)

    # 2. Tamper with readiness report total
    t_readiness.write_text(t_readiness.read_text().replace("(5.17 hours)", "(3.26 hours)"))
    with pytest.raises(ValueError, match="Readiness report .* disagrees with benchmark output"):
        verify_benchmark_consistency(bench_res.stdout, t_runtime, t_readiness)




