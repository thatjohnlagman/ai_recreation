import pytest
import numpy as np
import pandas as pd
from pathlib import Path
import json

from recall_aware_ids.experiment.runner import ExperimentRunner, AttackCacheProvider, LabelProvider

class MockPolicy:
    def __init__(self, requires_feedback=False):
        self.requires_feedback = requires_feedback
        from recall_aware_ids.controller import ControllerDecision, ControllerUpdate
        self.decision = ControllerDecision(batch_id=0, intensity=0.1)
        self.update = ControllerUpdate(
            config_id="cfg", batch_id=0, used_intensity=0.1, tp=1, fn=1,
            window_start_batch_id=0, window_end_batch_id=0,
            configured_window_size=5, window_batch_count=1,
            window_tp_sum=1, window_fn_sum=1, rolling_recall=0.5,
            state="Green", multiplier=1.1, unclipped_next_intensity=0.11,
            clipped_next_intensity=0.11, hit_min_bound=False, hit_max_bound=False, zero_denominator=False
        )

    def reset(self):
        pass

    def get_intensity(self, batch_id):
        from recall_aware_ids.controller import ControllerDecision
        return ControllerDecision(batch_id=batch_id, intensity=0.1)
        
    def submit_observations(self, batch_id, tp, fn):
        return self.update

class MockCacheProvider(AttackCacheProvider):
    def get_batch_data(self, batch_id):
        return {
            'X_attacked': np.zeros((500, 78)),
            'eligible': np.ones(500, dtype=bool),
            'attempted': np.zeros(500, dtype=bool),
            'successful': np.zeros(500, dtype=bool),
            'queries': np.zeros(500, dtype=int)
        }

class MockDefenseAdapter:
    def defend_batch(self, X, intensity, seed, attack_scenario, batch_id):
        return np.zeros(500, dtype=int), {}, np.zeros(500, dtype=float)

@pytest.fixture
def mock_runner_data(tmp_path):
    resolved_batches = pd.DataFrame({
        'batch_id': np.repeat(np.arange(144), 500),
        'measurement_idx': np.arange(72000),
        'eval_position': np.arange(72000)
    })
    y_measurement = np.zeros(72000, dtype=int)
    label_provider = LabelProvider(y_measurement)
    
    return resolved_batches, label_provider, tmp_path

def test_base_policy_execution(mock_runner_data):
    resolved_batches, label_provider, output_dir = mock_runner_data
    runner = ExperimentRunner(
        resolved_batches=resolved_batches,
        label_provider=label_provider,
        attack_cache=MockCacheProvider(),
        defense_adapter=MockDefenseAdapter(),
        policy_controller=MockPolicy(requires_feedback=False),
        output_dir=output_dir,
        run_id="run_base",
        seed=42,
        attack_scenario="Silent Probing",
        defense_name="afp",
        config_id="cfg"
    )
    runner.execute_run()
    assert (output_dir / "run_base").exists()
    assert (output_dir / "run_base" / "completion.json").exists()
    assert (output_dir / "run_base" / "config.json").exists()
    
    with open(output_dir / "run_base" / "config.json") as f:
        logs = json.load(f)
        assert len(logs) == 144
        assert logs[0]["state"] == "Base"
        assert logs[0]["multiplier"] == 1.0
        
    assert len(label_provider.access_log) == 144

def test_ra_policy_execution(mock_runner_data):
    resolved_batches, label_provider, output_dir = mock_runner_data
    runner = ExperimentRunner(
        resolved_batches=resolved_batches,
        label_provider=label_provider,
        attack_cache=MockCacheProvider(),
        defense_adapter=MockDefenseAdapter(),
        policy_controller=MockPolicy(requires_feedback=True),
        output_dir=output_dir,
        run_id="run_ra",
        seed=42,
        attack_scenario="Silent Probing",
        defense_name="afp",
        config_id="cfg"
    )
    runner.execute_run()
    
    with open(output_dir / "run_ra" / "config.json") as f:
        logs = json.load(f)
        assert len(logs) == 144
        assert logs[0]["state"] == "Green"

class FailingCacheProvider(MockCacheProvider):
    def __init__(self, fail_at_batch):
        self.fail_at_batch = fail_at_batch
    def get_batch_data(self, batch_id):
        if batch_id == self.fail_at_batch:
            raise ValueError("Injected failure")
        return super().get_batch_data(batch_id)

def test_atomic_failure_middle(mock_runner_data):
    resolved_batches, label_provider, output_dir = mock_runner_data
    runner = ExperimentRunner(
        resolved_batches=resolved_batches,
        label_provider=label_provider,
        attack_cache=FailingCacheProvider(fail_at_batch=72),
        defense_adapter=MockDefenseAdapter(),
        policy_controller=MockPolicy(requires_feedback=True),
        output_dir=output_dir,
        run_id="run_fail",
        seed=42,
        attack_scenario="Silent Probing",
        defense_name="afp",
        config_id="cfg"
    )
    with pytest.raises(ValueError, match="Injected failure"):
        runner.execute_run()
        
    assert not (output_dir / "run_fail").exists()
    # Find quarantined dir
    dirs = list(output_dir.glob("run_fail_failed_quarantined_*"))
    assert len(dirs) == 1
    assert (dirs[0] / "quarantined_failure.json").exists()

def test_stale_quarantine(mock_runner_data):
    resolved_batches, label_provider, output_dir = mock_runner_data
    # create a stale tmp dir
    (output_dir / "run_stale_tmp").mkdir()
    
    runner = ExperimentRunner(
        resolved_batches=resolved_batches,
        label_provider=label_provider,
        attack_cache=MockCacheProvider(),
        defense_adapter=MockDefenseAdapter(),
        policy_controller=MockPolicy(requires_feedback=False),
        output_dir=output_dir,
        run_id="run_stale",
        seed=42,
        attack_scenario="Silent Probing",
        defense_name="afp",
        config_id="cfg"
    )
    runner.execute_run()
    assert (output_dir / "run_stale").exists()
    # The stale tmp should be moved
    dirs = list(output_dir.glob("run_stale_stale_quarantined_*"))
    assert len(dirs) == 1

def test_reject_existing_run(mock_runner_data):
    resolved_batches, label_provider, output_dir = mock_runner_data
    (output_dir / "run_exist").mkdir()
    
    runner = ExperimentRunner(
        resolved_batches=resolved_batches,
        label_provider=label_provider,
        attack_cache=MockCacheProvider(),
        defense_adapter=MockDefenseAdapter(),
        policy_controller=MockPolicy(requires_feedback=False),
        output_dir=output_dir,
        run_id="run_exist",
        seed=42,
        attack_scenario="Silent Probing",
        defense_name="afp",
        config_id="cfg"
    )
    with pytest.raises(FileExistsError):
        runner.execute_run()

def test_label_access_timing_proved(mock_runner_data):
    resolved_batches, label_provider, output_dir = mock_runner_data
    class OrderCheckingPolicy(MockPolicy):
        def __init__(self, lp):
            super().__init__(requires_feedback=True)
            self.lp = lp
        def get_intensity(self, batch_id):
            # When getting intensity for batch_id, label provider should NOT have accessed it yet
            for access in self.lp.access_log:
                if access["batch_id"] == batch_id:
                    raise AssertionError("Label accessed BEFORE policy decision!")
            return super().get_intensity(batch_id)
            
    policy = OrderCheckingPolicy(label_provider)
    runner = ExperimentRunner(
        resolved_batches=resolved_batches,
        label_provider=label_provider,
        attack_cache=MockCacheProvider(),
        defense_adapter=MockDefenseAdapter(),
        policy_controller=policy,
        output_dir=output_dir,
        run_id="run_timing",
        seed=42,
        attack_scenario="Silent Probing",
        defense_name="afp",
        config_id="cfg"
    )
    runner.execute_run()
