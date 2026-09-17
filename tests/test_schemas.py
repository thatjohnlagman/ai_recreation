"""
Schema validation tests.
Proves malformed objects are REJECTED at construction time, not merely that valid ones pass.
"""
import json
import math
import dataclasses
import pytest

from recall_aware_ids.experiment.schemas import (
    AttackCacheManifest,
    AttackedSampleStatus,
    BatchConfigLog,
    BatchConfusionLog,
    RunSummary,
    CompletionMarker,
    FailureRecord,
    _REQUIRED_PROVENANCE_KEYS,
)

# Canonical 64-char hex value used throughout
_H = "a" * 64
_GOOD_PROV = {k: _H for k in _REQUIRED_PROVENANCE_KEYS}


# ---------------------------------------------------------------------------
# CompletionMarker
# ---------------------------------------------------------------------------

def test_completion_marker_rejects_empty():
    with pytest.raises(ValueError, match="nonempty"):
        CompletionMarker(run_id="r", timestamp="t", provenance_hashes={})


def test_completion_marker_rejects_dummy_key():
    hashes = {**_GOOD_PROV, "dummy": _H}
    with pytest.raises(ValueError, match="placeholder"):
        CompletionMarker(run_id="r", timestamp="t", provenance_hashes=hashes)


def test_completion_marker_rejects_dummy_value():
    hashes = {k: "dummy" for k in _REQUIRED_PROVENANCE_KEYS}
    with pytest.raises(ValueError, match="placeholder"):
        CompletionMarker(run_id="r", timestamp="t", provenance_hashes=hashes)


def test_completion_marker_rejects_non_hex_value():
    bad = {**_GOOD_PROV}
    bad["frozen_rf_hash"] = "not_hex_at_all"
    with pytest.raises(ValueError, match="64-char lowercase hex"):
        CompletionMarker(run_id="r", timestamp="t", provenance_hashes=bad)


def test_completion_marker_rejects_missing_required_keys():
    partial = {"frozen_rf_hash": _H}
    with pytest.raises(ValueError, match="missing required keys"):
        CompletionMarker(run_id="r", timestamp="t", provenance_hashes=partial)


def test_completion_marker_accepts_valid():
    cm = CompletionMarker(run_id="run1", timestamp="2026-01-01T00:00:00Z", provenance_hashes=_GOOD_PROV)
    assert cm.run_id == "run1"
    # JSON-serialisable
    json.dumps(dataclasses.asdict(cm))


# ---------------------------------------------------------------------------
# BatchConfigLog — rejection tests
# ---------------------------------------------------------------------------

def _valid_base_log(**overrides):
    defaults = dict(
        run_id="r", batch_id=0, config_id="Base", intensity=0.5,
        state="Base", multiplier=1.0, rolling_recall=0.0,
        hit_min_bound=False, hit_max_bound=False, zero_denominator=False,
    )
    defaults.update(overrides)
    return BatchConfigLog(**defaults)


def _valid_ra_log(**overrides):
    defaults = dict(
        run_id="r", batch_id=0, config_id="C1", intensity=0.5,
        state="Green", multiplier=1.1, rolling_recall=0.9,
        hit_min_bound=False, hit_max_bound=False, zero_denominator=False,
        tp=5, fn=1, window_start_batch_id=0, window_end_batch_id=0,
        configured_window_size=5, window_batch_count=1,
        window_tp_sum=5, window_fn_sum=1,
        unclipped_next_intensity=0.55, clipped_next_intensity=0.55,
    )
    defaults.update(overrides)
    return BatchConfigLog(**defaults)


def test_batch_config_log_rejects_negative_batch_id():
    with pytest.raises(ValueError, match="non-negative"):
        _valid_base_log(batch_id=-1)


def test_batch_config_log_rejects_bool_batch_id():
    with pytest.raises(TypeError, match="bool"):
        _valid_base_log(batch_id=True)


def test_batch_config_log_rejects_invalid_state():
    with pytest.raises(ValueError, match="state"):
        _valid_base_log(state="Unknown")


def test_batch_config_log_rejects_nonfinite_intensity():
    with pytest.raises(ValueError):
        _valid_base_log(intensity=float("inf"))


def test_batch_config_log_rejects_base_with_tp():
    with pytest.raises(ValueError, match="tp=None"):
        _valid_base_log(tp=5)


def test_batch_config_log_rejects_ra_missing_tp():
    with pytest.raises(ValueError, match="tp and fn"):
        BatchConfigLog(
            run_id="r", batch_id=0, config_id="C1", intensity=0.5,
            state="Green", multiplier=1.1, rolling_recall=0.9,
            hit_min_bound=False, hit_max_bound=False, zero_denominator=False,
            # tp and fn missing (None)
        )


def test_batch_config_log_rejects_negative_tp():
    with pytest.raises(ValueError, match="non-negative"):
        _valid_ra_log(tp=-1)


def test_batch_config_log_accepts_valid_base():
    log = _valid_base_log()
    assert log.state == "Base"
    assert log.tp is None


def test_batch_config_log_accepts_valid_ra():
    log = _valid_ra_log()
    assert log.state == "Green"
    assert log.tp == 5


# ---------------------------------------------------------------------------
# BatchConfusionLog — rejection tests
# ---------------------------------------------------------------------------

def _valid_confusion_log(**overrides):
    defaults = dict(
        run_id="r", batch_id=0, tp=100, fp=100, tn=150, fn=150,
        accuracy=0.5, recall=0.4, precision=0.5, f1=0.44,
        balanced_accuracy=0.5, pr_auc_average_precision=0.5,
        eligible_count=10, attempted_count=5, successful_count=1,
        asr_applicable=True, asr=0.2,
    )
    defaults.update(overrides)
    return BatchConfusionLog(**defaults)


def test_batch_confusion_log_rejects_negative_tp():
    with pytest.raises(ValueError, match="non-negative"):
        _valid_confusion_log(tp=-1)


def test_batch_confusion_log_rejects_bool_tp():
    with pytest.raises(TypeError, match="bool"):
        _valid_confusion_log(tp=True)


def test_batch_confusion_log_rejects_out_of_range_recall():
    with pytest.raises(ValueError, match="recall"):
        _valid_confusion_log(recall=1.5)


def test_batch_confusion_log_rejects_asr_when_not_applicable():
    with pytest.raises(ValueError, match="asr_applicable=False"):
        _valid_confusion_log(asr_applicable=False, asr=0.5)


def test_batch_confusion_log_rejects_null_asr_when_applicable():
    with pytest.raises(ValueError, match="asr_applicable=True"):
        _valid_confusion_log(asr_applicable=True, asr=None)


def test_batch_confusion_log_rejects_attempted_gt_eligible():
    with pytest.raises(ValueError, match="attempted_count"):
        _valid_confusion_log(eligible_count=3, attempted_count=10, successful_count=1)


def test_batch_confusion_log_rejects_successful_gt_attempted():
    with pytest.raises(ValueError, match="successful_count"):
        _valid_confusion_log(attempted_count=5, successful_count=10)

def test_batch_confusion_log_enforces_exact_500_sum():
    with pytest.raises(ValueError, match="must sum to exactly 500"):
        # Sum = 1 + 2 + 3 + 4 = 10 (not 500)
        _valid_confusion_log(tp=1, fp=2, tn=3, fn=4)

def test_batch_confusion_log_rejects_nonfinite_metric():
    with pytest.raises(ValueError):
        _valid_confusion_log(accuracy=float("nan"))


def test_batch_confusion_log_accepts_silent_probing():
    # ASR not applicable, none attempted/successful, asr=None
    log = BatchConfusionLog(
        run_id="r", batch_id=0, tp=100, fp=100, tn=150, fn=150,
        accuracy=0.5, recall=0.5, precision=0.5, f1=0.5,
        balanced_accuracy=0.5, pr_auc_average_precision=0.5,
        eligible_count=0, attempted_count=0, successful_count=0,
        asr_applicable=False, asr=None,
    )
    assert log.asr is None

def test_batch_confusion_log_enforces_zero_asr_when_zero_attempts():
    with pytest.raises(ValueError, match="asr must be 0.0 when attempted is 0"):
        BatchConfusionLog(
            run_id="r", batch_id=0, tp=100, fp=100, tn=150, fn=150,
            accuracy=0.5, recall=0.5, precision=0.5, f1=0.5,
            balanced_accuracy=0.5, pr_auc_average_precision=0.5,
            eligible_count=10, attempted_count=0, successful_count=0,
            asr_applicable=True, asr=0.5
        )


# ---------------------------------------------------------------------------
# AttackedSampleStatus — rejection tests
# ---------------------------------------------------------------------------

def _valid_status(**overrides):
    defaults = dict(
        eval_position=100, eligible=True, attempted=True, successful=False,
        status_code="TARGET_REJECTION", queries_used=5,
        l0=1.0, l1=2.0, l2=3.0, linf=4.0,
    )
    defaults.update(overrides)
    return AttackedSampleStatus(**defaults)


def test_attacked_sample_status_rejects_invalid_status_code():
    with pytest.raises(ValueError, match="status_code"):
        _valid_status(status_code="MADE_UP_CODE")


def test_attacked_sample_status_rejects_negative_l0():
    with pytest.raises(ValueError, match="l0"):
        _valid_status(l0=-1.0)


def test_attacked_sample_status_rejects_nonfinite_l1():
    with pytest.raises(ValueError, match="l1"):
        _valid_status(l1=float("inf"))


def test_attacked_sample_status_rejects_successful_without_attempted():
    with pytest.raises(ValueError, match="attempted"):
        _valid_status(successful=True, attempted=False)


def test_attacked_sample_status_rejects_attempted_without_eligible():
    with pytest.raises(ValueError, match="eligible"):
        _valid_status(attempted=True, eligible=False)


def test_attacked_sample_status_rejects_non_bool_eligible():
    with pytest.raises(TypeError, match="bool"):
        _valid_status(eligible=1)


def test_attacked_sample_status_rejects_negative_queries():
    with pytest.raises(ValueError, match="non-negative"):
        _valid_status(queries_used=-1)


def test_attacked_sample_status_accepts_not_applicable():
    s = AttackedSampleStatus(
        eval_position=0, eligible=False, attempted=False, successful=False,
        status_code="NOT_APPLICABLE", queries_used=0,
        l0=0.0, l1=0.0, l2=0.0, linf=0.0,
    )
    assert s.status_code == "NOT_APPLICABLE"


# ---------------------------------------------------------------------------
# RunSummary — rejection tests
# ---------------------------------------------------------------------------

def _valid_run_summary(**overrides):
    defaults = dict(
        run_id="r", seed=42, attack_scenario="SilentProbing", defense="afp",
        config_id="Base", total_batches=144,
        tp=18000, fp=18000, tn=18000, fn=18000,
        accuracy=0.5, recall=0.5, precision=0.5, f1=0.5,
        balanced_accuracy=0.5, pr_auc_average_precision=0.5,
        total_eligible=72000,
        total_attempted=72000,
        total_successful=72000,
        total_queries=1000,
        cache_identity={"some": "hash"},
        status_code_counts={"SUCCESS": 72000},
        l0_summary={"mean": 0.0},
        l1_summary={"mean": 0.0},
        l2_summary={"mean": 0.0},
        linf_summary={"mean": 0.0},
        global_asr=1.0, completed_successfully=True,
    )
    defaults.update(overrides)
    return RunSummary(**defaults)


def test_run_summary_rejects_wrong_batch_count():
    with pytest.raises(ValueError, match="total_batches"):
        _valid_run_summary(total_batches=100)


def test_run_summary_rejects_negative_tp():
    with pytest.raises(ValueError, match="non-negative"):
        _valid_run_summary(tp=-1)


def test_run_summary_rejects_out_of_range_asr():
    with pytest.raises(ValueError, match="global_asr"):
        _valid_run_summary(global_asr=1.5)


def test_run_summary_rejects_attempted_gt_eligible():
    with pytest.raises(ValueError, match="total_attempted"):
        _valid_run_summary(total_eligible=5, total_attempted=10)

def test_run_summary_enforces_exact_72000_sum():
    with pytest.raises(ValueError, match="must sum to exactly 72000"):
        # Sum = 10 + 0 + 90 + 0 = 100 (not 72000)
        _valid_run_summary(tp=10, fp=0, tn=90, fn=0)


# ---------------------------------------------------------------------------
# JSON round-trip for all valid schemas
# ---------------------------------------------------------------------------

def test_json_roundtrip_all_schemas():
    # AttackedSampleStatus
    ass = AttackedSampleStatus(
        eval_position=100, eligible=True, attempted=True, successful=False,
        status_code="TARGET_REJECTION", queries_used=5,
        l0=1.0, l1=2.0, l2=3.0, linf=4.0,
    )
    loaded = json.loads(json.dumps(dataclasses.asdict(ass)))
    assert loaded["eval_position"] == 100
    assert loaded["l0"] == 1.0

    # BatchConfigLog (Base)
    bcl_base = _valid_base_log()
    loaded_base = json.loads(json.dumps(dataclasses.asdict(bcl_base)))
    assert loaded_base["state"] == "Base"
    assert loaded_base["tp"] is None

    # BatchConfigLog (RA)
    bcl_ra = _valid_ra_log()
    loaded_ra = json.loads(json.dumps(dataclasses.asdict(bcl_ra)))
    assert loaded_ra["state"] == "Green"
    assert loaded_ra["tp"] == 5

    # BatchConfusionLog
    bcf = _valid_confusion_log()
    loaded_bcf = json.loads(json.dumps(dataclasses.asdict(bcf)))
    assert loaded_bcf["asr"] == pytest.approx(0.2)

    # RunSummary
    rs = _valid_run_summary()
    loaded_rs = json.loads(json.dumps(dataclasses.asdict(rs)))
    assert loaded_rs["completed_successfully"] is True

    # CompletionMarker
    cm = CompletionMarker(run_id="r", timestamp="2026-09-17T00:00:00Z", provenance_hashes=_GOOD_PROV)
    loaded_cm = json.loads(json.dumps(dataclasses.asdict(cm)))
    assert loaded_cm["run_id"] == "r"
