import json
import dataclasses
import pytest
from recall_aware_ids.experiment.schemas import (
    AttackCacheManifest,
    AttackedSampleStatus,
    BatchConfigLog,
    BatchConfusionLog,
    RunSummary,
    CompletionMarker,
    FailureRecord
)

def test_completion_marker_validation():
    with pytest.raises(ValueError):
        CompletionMarker(run_id="run1", timestamp="now", provenance_hashes={})
    
    cm = CompletionMarker(run_id="run1", timestamp="now", provenance_hashes={"a": "b"})
    assert cm.run_id == "run1"

def test_json_roundtrip():
    # Test AttackedSampleStatus
    ass = AttackedSampleStatus(
        eval_position=100, eligible=True, attempted=True, successful=False,
        status_code="failed", queries_used=5, l0=1.0, l1=2.0, l2=3.0, linf=4.0
    )
    ass_dict = dataclasses.asdict(ass)
    ass_json = json.dumps(ass_dict)
    ass_loaded = json.loads(ass_json)
    assert ass_loaded["eval_position"] == 100
    assert ass_loaded["l0"] == 1.0

    # Test BatchConfigLog
    bcl = BatchConfigLog(
        run_id="run1", batch_id=0, config_id="cfg1", intensity=0.5,
        state="Green", multiplier=1.5, rolling_recall=0.9,
        hit_min_bound=False, hit_max_bound=False, zero_denominator=False,
        tp=10, fn=2, window_start_batch_id=0, window_end_batch_id=0,
        configured_window_size=10, window_batch_count=1,
        window_tp_sum=10, window_fn_sum=2, unclipped_next_intensity=0.75,
        clipped_next_intensity=0.75, fs_effective_d=None
    )
    bcl_json = json.dumps(dataclasses.asdict(bcl))
    assert json.loads(bcl_json)["state"] == "Green"
    assert json.loads(bcl_json)["tp"] == 10

    # Test BatchConfusionLog
    bcf = BatchConfusionLog(
        run_id="run1", batch_id=0, tp=1, fp=2, tn=3, fn=4,
        accuracy=0.5, recall=0.2, precision=0.3, f1=0.24,
        balanced_accuracy=0.5, pr_auc_average_precision=0.4,
        eligible_count=10, attempted_count=5, successful_count=1,
        asr_applicable=True, asr=0.2
    )
    bcf_json = json.dumps(dataclasses.asdict(bcf))
    assert json.loads(bcf_json)["asr"] == 0.2
    
    # Test RunSummary
    rs = RunSummary(
        run_id="run1", seed=42, attack_scenario="Pilot", defense="afp",
        config_id="cfg", total_batches=144,
        tp=10, fp=20, tn=30, fn=40, accuracy=0.5, recall=0.2, precision=0.3, f1=0.4,
        balanced_accuracy=0.5, pr_auc_average_precision=0.6,
        total_eligible=100, total_attempted=50, total_successful=10, global_asr=0.2,
        completed_successfully=True
    )
    rs_json = json.dumps(dataclasses.asdict(rs))
    assert json.loads(rs_json)["global_asr"] == 0.2
