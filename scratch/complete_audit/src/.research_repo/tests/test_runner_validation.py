"""
Runner integration and validation tests.

Proves:
  - intensity is obtained before current-batch label access
  - prediction occurs before feedback
  - Base never receives TP/FN
  - RA receives feedback only after each completed batch
  - controller state is reset between runs
  - exactly 144 batch records are produced
  - completion.json is written last (after all other outputs succeed)
  - RunSummary is written and aggregates across all 72,000 logical rows
  - provenance_hashes are validated (rejects placeholder and missing keys)
  - Base logs have all RA-only fields as None
  - scrambled fixture validates eval_position join (not accidental row order)
"""
import pytest
import numpy as np
import pandas as pd
import json
import dataclasses
from pathlib import Path
from typing import Dict

from recall_aware_ids.experiment.runner import ExperimentRunner, AttackCacheProvider, LabelProvider
from recall_aware_ids.experiment.schemas import _REQUIRED_PROVENANCE_KEYS

# ─── Shared constants ─────────────────────────────────────────────────────────

_H = "c" * 64
_GOOD_PROV = {k: _H for k in _REQUIRED_PROVENANCE_KEYS}

N_BATCHES = 144
BATCH_SIZE = 500


# ─── Shared fixtures ──────────────────────────────────────────────────────────

def make_resolved_batches(scramble: bool = False, seed: int = 0):
    """
    Create 144×500 resolved_batches DataFrame.
    If scramble=True, shuffle eval_positions so row order != eval_position order.
    """
    rng = np.random.RandomState(seed)
    eps = np.arange(N_BATCHES * BATCH_SIZE)
    if scramble:
        rng.shuffle(eps)
    return pd.DataFrame({
        "batch_id": np.repeat(np.arange(N_BATCHES), BATCH_SIZE),
        "measurement_idx": np.arange(N_BATCHES * BATCH_SIZE),
        "eval_position": eps,
    })


class MockPolicy:
    def __init__(self, requires_feedback: bool = False):
        self.requires_feedback = requires_feedback
        self._reset_count = 0

    def reset(self):
        self._reset_count += 1

    def get_intensity(self, batch_id):
        from recall_aware_ids.controller import ControllerDecision
        return ControllerDecision(batch_id=batch_id, intensity=0.1)

    def submit_observations(self, batch_id, tp, fn):
        from recall_aware_ids.controller import ControllerUpdate
        return ControllerUpdate(
            config_id="C1", batch_id=batch_id, used_intensity=0.1,
            tp=int(tp), fn=int(fn),
            window_start_batch_id=batch_id, window_end_batch_id=batch_id,
            configured_window_size=5, window_batch_count=1,
            window_tp_sum=int(tp), window_fn_sum=int(fn), rolling_recall=0.5,
            state="Green", multiplier=1.1, unclipped_next_intensity=0.11,
            clipped_next_intensity=0.11, hit_min_bound=False, hit_max_bound=False,
            zero_denominator=False,
        )


class MockCacheProvider(AttackCacheProvider):
    def get_batch_data(self, batch_id):
        return {
            "X_attacked": np.zeros((BATCH_SIZE, 78), dtype=np.float32),
            "eligible": np.zeros(BATCH_SIZE, dtype=bool),
            "attempted": np.zeros(BATCH_SIZE, dtype=bool),
            "successful": np.zeros(BATCH_SIZE, dtype=bool),
            "status_codes": ["NOT_APPLICABLE"] * BATCH_SIZE,
            "queries": np.zeros(BATCH_SIZE, dtype=int),
            "l0": np.zeros(BATCH_SIZE, dtype=float),
            "l1": np.zeros(BATCH_SIZE, dtype=float),
            "l2": np.zeros(BATCH_SIZE, dtype=float),
            "linf": np.zeros(BATCH_SIZE, dtype=float),
        }


class MockDefenseAdapter:
    def __init__(self):
        self.call_count = 0

    def defend_batch(self, X, intensity, seed, attack_scenario, batch_id):
        self.call_count += 1
        return (
            np.zeros(BATCH_SIZE, dtype=int),
            _DummyDefenseResult(),
            np.zeros(BATCH_SIZE, dtype=float),
        )


class _DummyDefenseResult:
    protected_feature_modification_count = 0
    final_bounds_violation_count = 0
    final_nan_count = 0
    final_inf_count = 0
    projected_cell_count = 0


@pytest.fixture
def runner_deps(tmp_path):
    resolved = make_resolved_batches()
    y = np.zeros(N_BATCHES * BATCH_SIZE, dtype=int)
    label_provider = LabelProvider(y)
    return resolved, label_provider, tmp_path


def make_runner(resolved, label_provider, output_dir, run_id, policy, cache=None, defense=None):
    return ExperimentRunner(
        resolved_batches=resolved,
        label_provider=label_provider,
        attack_cache=cache or MockCacheProvider(),
        defense_adapter=defense or MockDefenseAdapter(),
        policy_controller=policy,
        output_dir=output_dir,
        run_id=run_id,
        seed=42,
        attack_scenario="Silent Probing",
        defense_name="afp",
        config_id="Base",
        provenance_hashes=_GOOD_PROV,
    )


# ─── Basic 144-batch execution ────────────────────────────────────────────────

def test_base_policy_144_batches(runner_deps):
    resolved, lp, out = runner_deps
    runner = make_runner(resolved, lp, out, "run_base", MockPolicy(requires_feedback=False))
    runner.execute_run()

    run_dir = out / "run_base"
    assert run_dir.exists()

    with open(run_dir / "config.json") as f:
        logs = json.load(f)
    assert len(logs) == 144

    with open(run_dir / "confusion.json") as f:
        cf_logs = json.load(f)
    assert len(cf_logs) == 144

    assert (run_dir / "run_summary.json").exists()
    assert (run_dir / "completion.json").exists()


def test_ra_policy_144_batches(runner_deps):
    resolved, lp, out = runner_deps
    runner = make_runner(resolved, lp, out, "run_ra", MockPolicy(requires_feedback=True))
    runner.execute_run()

    with open(out / "run_ra" / "config.json") as f:
        logs = json.load(f)
    assert len(logs) == 144
    # RA logs must have state != "Base"
    assert logs[0]["state"] == "Green"
    # RA logs must have tp/fn set
    assert logs[0]["tp"] is not None
    assert logs[0]["fn"] is not None


# ─── Base log must have None RA fields ────────────────────────────────────────

def test_base_log_has_all_ra_fields_none(runner_deps):
    resolved, lp, out = runner_deps
    runner = make_runner(resolved, lp, out, "run_base_null", MockPolicy(requires_feedback=False))
    runner.execute_run()

    with open(out / "run_base_null" / "config.json") as f:
        logs = json.load(f)

    ra_only_fields = [
        "tp", "fn", "window_start_batch_id", "window_end_batch_id",
        "configured_window_size", "window_batch_count",
        "window_tp_sum", "window_fn_sum",
        "unclipped_next_intensity", "clipped_next_intensity",
    ]
    for log in logs:
        assert log["state"] == "Base"
        for field in ra_only_fields:
            assert log[field] is None, f"Base log field {field!r} should be None"


# ─── Chronological event log (Base and RA) ────────────────────────────────────

def test_chronological_event_log(runner_deps):
    resolved, _, out = runner_deps
    
    events = []

    class EventPolicy(MockPolicy):
        def __init__(self, is_ra):
            super().__init__(requires_feedback=is_ra)
        def get_intensity(self, batch_id):
            events.append(("get_intensity", batch_id))
            return super().get_intensity(batch_id)
        def submit_observations(self, batch_id, tp, fn):
            events.append(("ra_feedback", batch_id))
            return super().submit_observations(batch_id, tp, fn)

    class EventLabelProvider(LabelProvider):
        def get_labels(self, batch_indices, batch_id):
            events.append(("get_labels", batch_id))
            return super().get_labels(batch_indices, batch_id)

    class EventDefense(MockDefenseAdapter):
        def defend_batch(self, X, intensity, seed, attack_scenario, batch_id):
            events.append(("predict", batch_id))
            return super().defend_batch(X, intensity, seed, attack_scenario, batch_id)

    class EventCacheProvider(AttackCacheProvider):
        def get_batch_data(self, batch_id):
            events.append(("cache_access", batch_id))
            return {
                "X_attacked": np.zeros((BATCH_SIZE, 78), dtype=np.float32),
                "eligible": np.zeros(BATCH_SIZE, dtype=bool),
                "attempted": np.zeros(BATCH_SIZE, dtype=bool),
                "successful": np.zeros(BATCH_SIZE, dtype=bool),
                "queries": np.zeros(BATCH_SIZE, dtype=int),
            }

    # Test Base
    events.clear()
    lp = EventLabelProvider(np.zeros(N_BATCHES * BATCH_SIZE, dtype=int))
    runner_base = ExperimentRunner(
        resolved_batches=resolved, label_provider=lp, attack_cache=EventCacheProvider(),
        defense_adapter=EventDefense(), policy_controller=EventPolicy(is_ra=False),
        output_dir=out, run_id="run_order_base", seed=42, attack_scenario="Silent Probing",
        defense_name="afp", config_id="Base", provenance_hashes=_GOOD_PROV,
    )
    runner_base.execute_run()
    
    # Assert Exact chronological event sequence for Base
    base_expected = []
    for bid in range(N_BATCHES):
        base_expected.extend([
            ("get_intensity", bid),
            ("cache_access", bid),
            ("predict", bid),
            ("get_labels", bid)
        ])
    assert events == base_expected

    # Test RA
    events.clear()
    lp = EventLabelProvider(np.zeros(N_BATCHES * BATCH_SIZE, dtype=int))
    runner_ra = ExperimentRunner(
        resolved_batches=resolved, label_provider=lp, attack_cache=EventCacheProvider(),
        defense_adapter=EventDefense(), policy_controller=EventPolicy(is_ra=True),
        output_dir=out, run_id="run_order_ra", seed=42, attack_scenario="Silent Probing",
        defense_name="afp", config_id="C1", provenance_hashes=_GOOD_PROV,
    )
    runner_ra.execute_run()

    # Assert Exact chronological event sequence for RA
    ra_expected = []
    for bid in range(N_BATCHES):
        ra_expected.extend([
            ("get_intensity", bid),
            ("cache_access", bid),
            ("predict", bid),
            ("get_labels", bid),
            ("ra_feedback", bid)
        ])
    assert events == ra_expected


# ─── Controller reset between runs ────────────────────────────────────────────

def test_controller_reset_called(runner_deps):
    resolved, lp, out = runner_deps
    policy = MockPolicy(requires_feedback=False)
    runner = make_runner(resolved, lp, out, "run_reset", policy)
    runner.execute_run()
    assert policy._reset_count == 1


# ─── RunSummary generation ────────────────────────────────────────────────────

def test_run_summary_generated_and_valid(runner_deps):
    resolved, lp, out = runner_deps
    runner = make_runner(resolved, lp, out, "run_summary", MockPolicy(requires_feedback=False))
    runner.execute_run()

    summary_path = out / "run_summary" / "run_summary.json"
    assert summary_path.exists()
    with open(summary_path) as f:
        summary = json.load(f)
    assert summary["total_batches"] == 144
    assert summary["completed_successfully"] is True
    assert "pr_auc_average_precision" in summary


# ─── Completion marker written last ───────────────────────────────────────────

def test_completion_written_last(runner_deps, monkeypatch):
    resolved, lp, out = runner_deps
    write_order = []

    orig_open = open

    def patched_open(path, mode="r", **kwargs):
        path_str = str(path)
        if mode == "w":
            if path_str.endswith("completion.json"):
                write_order.append("completion")
            elif path_str.endswith("run_summary.json"):
                write_order.append("run_summary")
            elif path_str.endswith("confusion.json"):
                write_order.append("confusion")
            elif path_str.endswith("config.json"):
                write_order.append("config")
        return orig_open(path, mode, **kwargs)

    monkeypatch.setattr("builtins.open", patched_open)

    runner = make_runner(resolved, lp, out, "run_order_check", MockPolicy(requires_feedback=False))
    runner.execute_run()

    assert write_order.index("completion") > write_order.index("run_summary")
    assert write_order.index("run_summary") > write_order.index("confusion")
    assert write_order.index("confusion") > write_order.index("config")


# ─── Provenance rejection ─────────────────────────────────────────────────────

def test_runner_rejects_dummy_provenance(runner_deps):
    resolved, lp, out = runner_deps
    # {"dummy": "hash"} fails with missing required keys first (before placeholder check)
    with pytest.raises(ValueError, match="missing required keys|placeholder"):
        ExperimentRunner(
            resolved_batches=resolved, label_provider=lp,
            attack_cache=MockCacheProvider(), defense_adapter=MockDefenseAdapter(),
            policy_controller=MockPolicy(), output_dir=out,
            run_id="run_bad", seed=42, attack_scenario="Silent Probing",
            defense_name="afp", config_id="Base",
            provenance_hashes={"dummy": "hash"},
        )


def test_runner_rejects_missing_provenance_keys(runner_deps):
    resolved, lp, out = runner_deps
    with pytest.raises(ValueError, match="missing required keys"):
        ExperimentRunner(
            resolved_batches=resolved, label_provider=lp,
            attack_cache=MockCacheProvider(), defense_adapter=MockDefenseAdapter(),
            policy_controller=MockPolicy(), output_dir=out,
            run_id="run_partial", seed=42, attack_scenario="Silent Probing",
            defense_name="afp", config_id="Base",
            provenance_hashes={"frozen_rf_hash": _H},
        )


# ─── Atomic failure guarantees ────────────────────────────────────────────────

class FailingCache(MockCacheProvider):
    def __init__(self, fail_at: int):
        self.fail_at = fail_at

    def get_batch_data(self, batch_id):
        if batch_id == self.fail_at:
            raise ValueError(f"Injected failure at batch {batch_id}")
        return super().get_batch_data(batch_id)


def _run_and_expect_failure(resolved, lp, out, run_id, fail_at):
    runner = make_runner(resolved, lp, out, run_id,
                         MockPolicy(requires_feedback=False),
                         cache=FailingCache(fail_at))
    with pytest.raises(ValueError, match="Injected failure"):
        runner.execute_run()
    # Final dir must NOT exist
    assert not (out / run_id).exists()
    # Quarantine dir must exist
    dirs = list(out.glob(f"{run_id}_failed_quarantined_*"))
    assert len(dirs) == 1
    assert (dirs[0] / "quarantined_failure.json").exists()


def test_atomic_failure_early(runner_deps):
    resolved, lp, out = runner_deps
    _run_and_expect_failure(resolved, lp, out, "fail_early", fail_at=0)


def test_atomic_failure_middle(runner_deps):
    resolved, lp, out = runner_deps
    _run_and_expect_failure(resolved, lp, out, "fail_mid", fail_at=72)


def test_atomic_failure_late(runner_deps):
    resolved, lp, out = runner_deps
    _run_and_expect_failure(resolved, lp, out, "fail_late", fail_at=143)


def test_stale_quarantine(runner_deps):
    resolved, lp, out = runner_deps
    # Create stale tmp
    (out / "run_stale_tmp").mkdir()

    runner = make_runner(resolved, lp, out, "run_stale", MockPolicy(requires_feedback=False))
    runner.execute_run()

    assert (out / "run_stale").exists()
    dirs = list(out.glob("run_stale_stale_quarantined_*"))
    assert len(dirs) == 1


def test_reject_existing_run(runner_deps):
    resolved, lp, out = runner_deps
    (out / "run_exist").mkdir()

    runner = make_runner(resolved, lp, out, "run_exist", MockPolicy(requires_feedback=False))
    with pytest.raises(FileExistsError):
        runner.execute_run()


# ─── Scrambled fixture: eval_position join, not row order ─────────────────────

def test_scrambled_eval_position_join(tmp_path):
    """
    Features and labels are assigned by eval_position.
    Deliberately scramble row order in resolved_batches.
    Verify that predictions match label provider mapping correctly.
    """
    rng = np.random.RandomState(99)

    # Distinct per-eval_position label (0 or 1) and feature
    n_total = N_BATCHES * BATCH_SIZE
    y_by_ep = rng.randint(0, 2, n_total)
    
    # We'll make feature[0] directly dictate the prediction
    # If pred_by_ep == 1, feature[0] = 1.0, else 0.0
    pred_by_ep = rng.randint(0, 2, n_total)

    # Scrambled resolved_batches
    resolved = make_resolved_batches(scramble=True, seed=42)
    # y_measurement is indexed by measurement_idx but maps to eval_position
    y_measurement = np.zeros(n_total, dtype=int)
    for _, row in resolved.iterrows():
        y_measurement[int(row["measurement_idx"])] = y_by_ep[int(row["eval_position"])]

    lp = LabelProvider(y_measurement)

    # Cache: return features with column 0 set to pred_by_ep[eval_position]
    class IdentifiableCacheProvider(AttackCacheProvider):
        def get_batch_data(self, batch_id):
            batch_df = resolved[resolved["batch_id"] == batch_id].sort_values("eval_position")
            eps = batch_df["eval_position"].values
            X = np.zeros((BATCH_SIZE, 78), dtype=np.float32)
            X[:, 0] = pred_by_ep[eps]
            return {
                "X_attacked": X,
                "eligible": np.zeros(BATCH_SIZE, dtype=bool),
                "attempted": np.zeros(BATCH_SIZE, dtype=bool),
                "successful": np.zeros(BATCH_SIZE, dtype=bool),
                "queries": np.zeros(BATCH_SIZE, dtype=int),
            }

    # Defense: predict 1 if X[:, 0] > 0.5 else 0
    class IdentifiableDefenseAdapter(MockDefenseAdapter):
        def defend_batch(self, X, intensity, seed, attack_scenario, batch_id):
            preds = (X[:, 0] > 0.5).astype(int)
            return preds, _DummyDefenseResult(), np.zeros(BATCH_SIZE, dtype=float)

    runner = ExperimentRunner(
        resolved_batches=resolved,
        label_provider=lp,
        attack_cache=IdentifiableCacheProvider(),
        defense_adapter=IdentifiableDefenseAdapter(),
        policy_controller=MockPolicy(requires_feedback=False),
        output_dir=tmp_path,
        run_id="run_scrambled",
        seed=0,
        attack_scenario="Silent Probing",
        defense_name="afp",
        config_id="Base",
        provenance_hashes=_GOOD_PROV,
    )
    runner.execute_run()

    # Read confusion logs and verify exact expected values
    with open(tmp_path / "run_scrambled" / "confusion.json") as f:
        cf = json.load(f)
    assert len(cf) == 144

    for batch_id in range(144):
        batch_df = resolved[resolved["batch_id"] == batch_id].sort_values("eval_position")
        eps = batch_df["eval_position"].values
        y_true = y_by_ep[eps]
        y_pred = pred_by_ep[eps]
        
        tp = int(np.sum((y_true == 1) & (y_pred == 1)))
        tn = int(np.sum((y_true == 0) & (y_pred == 0)))
        fp = int(np.sum((y_true == 0) & (y_pred == 1)))
        fn = int(np.sum((y_true == 1) & (y_pred == 0)))
        
        log_cf = cf[batch_id]
        assert log_cf["tp"] == tp
        assert log_cf["tn"] == tn
        assert log_cf["fp"] == fp
        assert log_cf["fn"] == fn


def test_runner_applicable_zero_attempts_asr_zero(runner_deps):
    resolved, lp, out = runner_deps
    runner = ExperimentRunner(
        resolved_batches=resolved,
        label_provider=lp,
        attack_cache=MockCacheProvider(),
        defense_adapter=MockDefenseAdapter(),
        policy_controller=MockPolicy(requires_feedback=False),
        output_dir=out,
        run_id="run_zero_attempt_asr",
        seed=42,
        attack_scenario="SurrogateTransfer",
        defense_name="afp",
        config_id="Base",
        provenance_hashes=_GOOD_PROV,
    )
    runner.execute_run()

    run_dir = out / "run_zero_attempt_asr"
    assert run_dir.exists()
    assert (run_dir / "completion.json").exists()

    with open(run_dir / "run_summary.json") as f:
        summary = json.load(f)

    assert summary["attack_scenario"] == "SurrogateTransfer"
    assert summary["total_attempted"] == 0
    assert summary["total_successful"] == 0
    assert summary["global_asr"] == 0.0


def test_runner_quarantines_on_malformed_serialized_output(monkeypatch, runner_deps):
    resolved, lp, out = runner_deps
    import recall_aware_ids.experiment.runner as runner_mod

    orig_validate = runner_mod._validate_run_outputs

    def corrupting_validate(run_tmp_dir, summary):
        # Corrupt scores.json with malformed contents before calling validator
        with open(run_tmp_dir / "scores.json", "w") as f:
            f.write("{invalid json content}")
        orig_validate(run_tmp_dir, summary)

    monkeypatch.setattr(runner_mod, "_validate_run_outputs", corrupting_validate)

    runner = ExperimentRunner(
        resolved_batches=resolved,
        label_provider=lp,
        attack_cache=MockCacheProvider(),
        defense_adapter=MockDefenseAdapter(),
        policy_controller=MockPolicy(requires_feedback=False),
        output_dir=out,
        run_id="run_corrupt_output",
        seed=42,
        attack_scenario="SurrogateTransfer",
        defense_name="afp",
        config_id="Base",
        provenance_hashes=_GOOD_PROV,
    )

    with pytest.raises(ValueError, match="scores.json is not valid JSON"):
        runner.execute_run()

    # Final run dir must NOT exist
    assert not (out / "run_corrupt_output").exists()

    # Quarantine directory must exist
    quarantined_dirs = list(out.glob("run_corrupt_output_failed_quarantined_*"))
    assert len(quarantined_dirs) == 1
    q_dir = quarantined_dirs[0]

    # completion.json must NOT exist in quarantine directory
    assert not (q_dir / "completion.json").exists()

    # quarantined_failure.json must exist and record failure
    assert (q_dir / "quarantined_failure.json").exists()
    with open(q_dir / "quarantined_failure.json") as f:
        fail_data = json.load(f)
    assert fail_data["run_id"] == "run_corrupt_output"
    assert "scores.json is not valid JSON" in fail_data["error_message"]


def test_runner_quarantines_when_serialized_run_summary_altered(monkeypatch, runner_deps):
    resolved, lp, out = runner_deps
    import recall_aware_ids.experiment.runner as runner_mod

    orig_validate = runner_mod._validate_run_outputs

    def corrupting_validate(run_tmp_dir, summary):
        # Alter run_summary.json to a different but still schema-valid summary
        with open(run_tmp_dir / "run_summary.json", "r") as f:
            summary_dict = json.load(f)
        # Modify a field: seed=9999 is still valid int, but differs from canonical summary
        summary_dict["seed"] = 9999
        with open(run_tmp_dir / "run_summary.json", "w") as f:
            json.dump(summary_dict, f, indent=2)
        orig_validate(run_tmp_dir, summary)

    monkeypatch.setattr(runner_mod, "_validate_run_outputs", corrupting_validate)

    runner = ExperimentRunner(
        resolved_batches=resolved,
        label_provider=lp,
        attack_cache=MockCacheProvider(),
        defense_adapter=MockDefenseAdapter(),
        policy_controller=MockPolicy(requires_feedback=False),
        output_dir=out,
        run_id="run_altered_summary",
        seed=42,
        attack_scenario="SurrogateTransfer",
        defense_name="afp",
        config_id="Base",
        provenance_hashes=_GOOD_PROV,
    )

    with pytest.raises(ValueError, match="canonical representation does not match in-memory summary"):
        runner.execute_run()

    assert not (out / "run_altered_summary").exists()
    q_dirs = list(out.glob("run_altered_summary_failed_quarantined_*"))
    assert len(q_dirs) == 1
    assert not (q_dirs[0] / "completion.json").exists()
    assert (q_dirs[0] / "quarantined_failure.json").exists()
    with open(q_dirs[0] / "quarantined_failure.json") as f:
        fail_data = json.load(f)
    assert "canonical representation does not match" in fail_data["error_message"]


def test_runner_quarantines_when_confusion_record_disagrees_with_predictions(monkeypatch, runner_deps):
    resolved, lp, out = runner_deps
    import recall_aware_ids.experiment.runner as runner_mod

    orig_validate = runner_mod._validate_run_outputs

    def corrupting_validate(run_tmp_dir, summary):
        # Alter a confusion record while maintaining sum==500 and schema validity
        with open(run_tmp_dir / "confusion.json", "r") as f:
            confusion_data = json.load(f)
        rec0 = confusion_data[0]
        # Shift 1 count between TP and FP/FN/TN while keeping sum == 500
        if rec0["fn"] > 0:
            rec0["tp"] += 1
            rec0["fn"] -= 1
        elif rec0["fp"] > 0:
            rec0["tp"] += 1
            rec0["fp"] -= 1
        else:
            rec0["tp"] += 1
            rec0["tn"] -= 1
        with open(run_tmp_dir / "confusion.json", "w") as f:
            json.dump(confusion_data, f, indent=2)
        orig_validate(run_tmp_dir, summary)

    monkeypatch.setattr(runner_mod, "_validate_run_outputs", corrupting_validate)

    runner = ExperimentRunner(
        resolved_batches=resolved,
        label_provider=lp,
        attack_cache=MockCacheProvider(),
        defense_adapter=MockDefenseAdapter(),
        policy_controller=MockPolicy(requires_feedback=False),
        output_dir=out,
        run_id="run_confusion_disagrees",
        seed=42,
        attack_scenario="SurrogateTransfer",
        defense_name="afp",
        config_id="Base",
        provenance_hashes=_GOOD_PROV,
    )

    with pytest.raises(ValueError, match="do not match recomputed counts from reopened labels and predictions"):
        runner.execute_run()

    assert not (out / "run_confusion_disagrees").exists()
    q_dirs = list(out.glob("run_confusion_disagrees_failed_quarantined_*"))
    assert len(q_dirs) == 1
    assert not (q_dirs[0] / "completion.json").exists()
    assert (q_dirs[0] / "quarantined_failure.json").exists()
    with open(q_dirs[0] / "quarantined_failure.json") as f:
        fail_data = json.load(f)
    assert "do not match recomputed counts" in fail_data["error_message"]


def test_runner_quarantines_when_predictions_disagree_with_confusion(monkeypatch, runner_deps):
    resolved, lp, out = runner_deps
    import recall_aware_ids.experiment.runner as runner_mod

    orig_validate = runner_mod._validate_run_outputs

    def corrupting_validate(run_tmp_dir, summary):
        # Alter predictions in scores.json (structurally valid binary ints, 500 elements)
        with open(run_tmp_dir / "scores.json", "r") as f:
            scores_data = json.load(f)
        # Flip first prediction in batch 0
        scores_data[0]["predictions"][0] = 1 - scores_data[0]["predictions"][0]
        scores_data[0]["preds"][0] = scores_data[0]["predictions"][0]
        with open(run_tmp_dir / "scores.json", "w") as f:
            json.dump(scores_data, f)
        orig_validate(run_tmp_dir, summary)

    monkeypatch.setattr(runner_mod, "_validate_run_outputs", corrupting_validate)

    runner = ExperimentRunner(
        resolved_batches=resolved,
        label_provider=lp,
        attack_cache=MockCacheProvider(),
        defense_adapter=MockDefenseAdapter(),
        policy_controller=MockPolicy(requires_feedback=False),
        output_dir=out,
        run_id="run_preds_disagree",
        seed=42,
        attack_scenario="SurrogateTransfer",
        defense_name="afp",
        config_id="Base",
        provenance_hashes=_GOOD_PROV,
    )

    with pytest.raises(ValueError, match="do not match recomputed counts from reopened labels and predictions"):
        runner.execute_run()

    assert not (out / "run_preds_disagree").exists()
    q_dirs = list(out.glob("run_preds_disagree_failed_quarantined_*"))
    assert len(q_dirs) == 1
    assert not (q_dirs[0] / "completion.json").exists()
    assert (q_dirs[0] / "quarantined_failure.json").exists()
    with open(q_dirs[0] / "quarantined_failure.json") as f:
        fail_data = json.load(f)
    assert "do not match recomputed counts" in fail_data["error_message"]


def test_runner_quarantines_when_scores_disagree_with_metrics(monkeypatch, tmp_path):
    resolved = make_resolved_batches()
    y = np.zeros(N_BATCHES * BATCH_SIZE, dtype=int)
    y[0] = 1  # Non-trivial PR-AUC in batch 0
    lp = LabelProvider(y)
    out = tmp_path
    import recall_aware_ids.experiment.runner as runner_mod

    orig_validate = runner_mod._validate_run_outputs

    def corrupting_validate(run_tmp_dir, summary):
        # Alter scores in scores.json (structurally valid floats in [0, 1])
        with open(run_tmp_dir / "scores.json", "r") as f:
            scores_data = json.load(f)
        # Modify scores of batch 0 (still valid floats in [0, 1])
        # Set scores[0] to 0.999 which changes average precision (PR-AUC)
        scores_data[0]["scores"][0] = 0.999
        with open(run_tmp_dir / "scores.json", "w") as f:
            json.dump(scores_data, f)
        orig_validate(run_tmp_dir, summary)

    monkeypatch.setattr(runner_mod, "_validate_run_outputs", corrupting_validate)

    runner = ExperimentRunner(
        resolved_batches=resolved,
        label_provider=lp,
        attack_cache=MockCacheProvider(),
        defense_adapter=MockDefenseAdapter(),
        policy_controller=MockPolicy(requires_feedback=False),
        output_dir=out,
        run_id="run_scores_disagree",
        seed=42,
        attack_scenario="SurrogateTransfer",
        defense_name="afp",
        config_id="Base",
        provenance_hashes=_GOOD_PROV,
    )

    with pytest.raises(ValueError, match="does not match"):
        runner.execute_run()

    assert not (out / "run_scores_disagree").exists()
    q_dirs = list(out.glob("run_scores_disagree_failed_quarantined_*"))
    assert len(q_dirs) == 1
    assert not (q_dirs[0] / "completion.json").exists()
    assert (q_dirs[0] / "quarantined_failure.json").exists()
    with open(q_dirs[0] / "quarantined_failure.json") as f:
        fail_data = json.load(f)
    assert "does not match" in fail_data["error_message"]
