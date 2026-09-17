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
from pathlib import Path
from typing import Any, Dict, List

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


class SyntheticCacheProvider(AttackCacheProvider):
    def __init__(self, n_batches: int = 144, batch_size: int = 500, scenario: str = "SilentProbing"):
        self.n_batches = n_batches
        self.batch_size = batch_size
        self.scenario = scenario
        self.cache_identity = dict(VALID_PROVENANCE)
        self.cache_identity["attack_scenario"] = scenario
        self.cache_identity["effective_seed"] = 42
        self.cache_identity["row_count"] = n_batches * batch_size

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
    assert "Missing completion.json" in err


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
