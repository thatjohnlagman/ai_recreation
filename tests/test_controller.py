import pytest
import numpy as np
import yaml
from pathlib import Path
import dataclasses
import ast
import pandas as pd
from recall_aware_ids.controller import RecallAwareController, ControllerUpdate
from recall_aware_ids.defenses.afp import AdaptiveFeaturePoisoning

ROOT = Path(__file__).parents[1]

@pytest.fixture
def base_config():
    return {
        "id": "C1_TEST",
        "sensitivity_focus": "test",
        "window_size": 3,
        "Rcritical": 0.85,
        "Rmin": 0.95,
        "fast_decay": 0.5,
        "slow_decay": 0.9,
        "growth_factor": 1.1,
        "primary_config": True
    }

@pytest.fixture
def afp_defense_config():
    return {
        "epsilon_base": 0.0003,
        "intensity_min": 0.0,
        "intensity_max": 0.0003
    }

def test_batch_0_base_intensity(base_config, afp_defense_config):
    controller = RecallAwareController(base_config, afp_defense_config, "afp")
    dec = controller.get_intensity(0)
    assert dec.batch_id == 0
    assert dec.intensity == 0.0003

def test_timing_contract(base_config, afp_defense_config):
    controller = RecallAwareController(base_config, afp_defense_config, "afp")
    
    # 1. Submitting before get_intensity must fail
    with pytest.raises(RuntimeError, match="Cannot submit observations without first calling get_intensity"):
        controller.submit_observations(0, 100, 0)
        
    # 2. get_intensity twice must fail
    controller.get_intensity(0)
    with pytest.raises(RuntimeError, match="a decision is already pending"):
        controller.get_intensity(0)
        
    # 3. submit observations for wrong batch must fail
    with pytest.raises(ValueError, match="Pending decision exists for batch 0"):
        controller.submit_observations(1, 100, 0)

    # 4. Out of order get_intensity must fail
    with pytest.raises(ValueError, match="Expected batch_id 0"):
        controller.reset()
        controller.get_intensity(1)

def test_failure_atomicity(base_config, afp_defense_config):
    controller = RecallAwareController(base_config, afp_defense_config, "afp")
    controller.get_intensity(0)
    
    initial_deque = list(controller.window)
    initial_intensity = controller.current_intensity
    initial_expected = controller.expected_batch_id
    
    # Intentionally fail submission with bad type
    with pytest.raises(TypeError):
        controller.submit_observations(0, True, 0) # type: ignore
        
    # State should remain completely intact
    assert controller.pending_decision is not None
    assert controller.expected_batch_id == initial_expected
    assert controller.current_intensity == initial_intensity
    assert list(controller.window) == initial_deque

    # Successful submission clears state
    controller.submit_observations(0, 100, 0)
    assert controller.pending_decision is None
    assert controller.expected_batch_id == 1

def test_exact_branches_fresh_controllers(base_config, afp_defense_config):
    # Rcritical = 0.85, Rmin = 0.95
    # Test Rcritical EXACTLY -> Yellow
    c1 = RecallAwareController(base_config, afp_defense_config, "afp")
    c1.get_intensity(0)
    upd1 = c1.submit_observations(0, 85, 15) # 0.85 exactly
    assert upd1.state == "Yellow"

    # Test Rmin EXACTLY -> Green
    c2 = RecallAwareController(base_config, afp_defense_config, "afp")
    c2.get_intensity(0)
    upd2 = c2.submit_observations(0, 95, 5) # 0.95 exactly
    assert upd2.state == "Green"

    # Test immediately below Rcritical -> Red
    c3 = RecallAwareController(base_config, afp_defense_config, "afp")
    c3.get_intensity(0)
    upd3 = c3.submit_observations(0, 84, 16) # 0.84
    assert upd3.state == "Red"

    # Test immediately below Rmin -> Yellow
    c4 = RecallAwareController(base_config, afp_defense_config, "afp")
    c4.get_intensity(0)
    upd4 = c4.submit_observations(0, 94, 6) # 0.94
    assert upd4.state == "Yellow"

def test_rolling_window_eviction_and_warmup(base_config, afp_defense_config):
    controller = RecallAwareController(base_config, afp_defense_config, "afp")
    
    # Warm up 1 batch
    controller.get_intensity(0)
    u1 = controller.submit_observations(0, 10, 0)
    assert u1.configured_window_size == 3
    assert u1.window_batch_count == 1
    
    # Warm up 2 batches
    controller.get_intensity(1)
    u2 = controller.submit_observations(1, 20, 0)
    assert u2.window_batch_count == 2
    
    # 3 batches (full)
    controller.get_intensity(2)
    u3 = controller.submit_observations(2, 30, 0)
    assert u3.window_batch_count == 3
    
    # Eviction on 4th insert
    controller.get_intensity(3)
    u4 = controller.submit_observations(3, 40, 0)
    assert u4.window_batch_count == 3
    assert u4.window_start_batch_id == 1
    assert u4.window_end_batch_id == 3
    assert u4.window_tp_sum == 90 # 20+30+40

def test_aggregate_recall_calculation(base_config, afp_defense_config):
    controller = RecallAwareController(base_config, afp_defense_config, "afp")
    controller.get_intensity(0)
    controller.submit_observations(0, 100, 0) # batch recall 1.0 (N=100)
    controller.get_intensity(1)
    controller.submit_observations(1, 0, 10) # batch recall 0.0 (N=10)
    
    # Mean of batch recalls: (1.0 + 0.0) / 2 = 0.50
    # Aggregate recall: 100 / (100 + 10) = 100 / 110 ≈ 0.90909
    # Ensure it uses aggregate, not mean!
    controller.get_intensity(2)
    upd = controller.submit_observations(2, 0, 0) 
    
    assert np.isclose(upd.rolling_recall, 100/110)
    assert not np.isclose(upd.rolling_recall, 0.50)

def test_zero_denominator(base_config, afp_defense_config):
    controller = RecallAwareController(base_config, afp_defense_config, "afp", zero_division_value=0.0)
    controller.get_intensity(0)
    upd = controller.submit_observations(0, 0, 0)
    
    assert upd.zero_denominator is True
    assert upd.rolling_recall == 0.0
    assert upd.state == "Red"

def test_clipping_bounds(base_config):
    # Positive minimum bound to test floor logic
    pos_min_config = {
        "epsilon_base": 0.0004,
        "intensity_min": 0.0001,
        "intensity_max": 0.0004
    }
    controller = RecallAwareController(base_config, pos_min_config, "afp")
    # Base is 0.0004
    controller.get_intensity(0)
    upd1 = controller.submit_observations(0, 0, 100) # Red -> 0.0004 * 0.5 = 0.0002
    assert upd1.clipped_next_intensity == 0.0002
    assert upd1.hit_min_bound is False

    controller.get_intensity(1)
    upd2 = controller.submit_observations(1, 0, 100) # Red -> 0.0002 * 0.5 = 0.0001 exactly min
    assert np.isclose(upd2.unclipped_next_intensity, 0.0001)
    assert np.isclose(upd2.clipped_next_intensity, 0.0001)
    assert upd2.hit_min_bound is True

    controller.get_intensity(2)
    upd3 = controller.submit_observations(2, 0, 100) # Red -> 0.0001 * 0.5 = 0.00005 (below min)
    assert np.isclose(upd3.unclipped_next_intensity, 0.00005)
    assert np.isclose(upd3.clipped_next_intensity, 0.0001)
    assert upd3.hit_min_bound is True

    # Test Green recovery
    controller.get_intensity(3)
    upd4 = controller.submit_observations(3, 10000, 0) # Green -> 0.0001 * 1.1 = 0.00011
    assert upd4.state == "Green"
    assert np.isclose(upd4.clipped_next_intensity, 0.00011)
    assert upd4.hit_max_bound is False

def test_deterministic_isolation_and_reset(base_config, afp_defense_config):
    c1 = RecallAwareController(base_config, afp_defense_config, "afp")
    c1.get_intensity(0)
    u1 = c1.submit_observations(0, 100, 0)
    
    c2 = RecallAwareController(base_config, afp_defense_config, "afp")
    c2.get_intensity(0)
    u2 = c2.submit_observations(0, 100, 0)
    
    # Must be perfectly identical JSON serializable dictionaries
    import json
    j1 = json.dumps(dataclasses.asdict(u1))
    j2 = json.dumps(dataclasses.asdict(u2))
    assert j1 == j2

    # Reset while pending abandons
    c1.get_intensity(1)
    assert c1.pending_decision is not None
    c1.reset()
    assert c1.pending_decision is None
    assert c1.expected_batch_id == 0
    assert c1.current_intensity == 0.0003

def test_rejection_of_invalid_types(base_config, afp_defense_config):
    controller = RecallAwareController(base_config, afp_defense_config, "afp")
    
    # False cannot act as batch 0, True cannot act as batch 1
    with pytest.raises(TypeError, match="cannot be a boolean"):
        controller.get_intensity(False)
    with pytest.raises(TypeError, match="cannot be a boolean"):
        controller.get_intensity(True)
        
    controller.get_intensity(0)
    with pytest.raises(TypeError, match="cannot be a boolean"):
        controller.submit_observations(0, True, False) # type: ignore
        
    with pytest.raises(TypeError, match="integral type"):
        controller.submit_observations(0, 1.5, 0) # type: ignore
        
    with pytest.raises(ValueError, match="non-negative"):
        controller.submit_observations(0, -1, 0)
        
    # Numpy int should work and be normalized to Python int
    upd = controller.submit_observations(0, np.int64(10), np.int32(0))
    assert type(upd.tp) is int
    assert upd.tp == 10

def test_validation_of_malformed_config(base_config, afp_defense_config):
    bad_config = base_config.copy()
    bad_config['fast_decay'] = 1.2
    with pytest.raises(ValueError, match="Decay/growth factors must satisfy"):
        RecallAwareController(bad_config, afp_defense_config, "afp")
        
    # Infinite growth factor
    bad_config2 = base_config.copy()
    bad_config2['growth_factor'] = float('inf')
    with pytest.raises(ValueError, match="must be finite"):
        RecallAwareController(bad_config2, afp_defense_config, "afp")

    # Missing keys
    bad_config3 = base_config.copy()
    del bad_config3['Rcritical']
    with pytest.raises(ValueError, match="Missing required key"):
        RecallAwareController(bad_config3, afp_defense_config, "afp")

    # Not a mapping
    with pytest.raises(ValueError, match="config must be a mapping"):
        RecallAwareController([], afp_defense_config, "afp")

def test_frozen_dataclass_mutation(base_config, afp_defense_config):
    c1 = RecallAwareController(base_config, afp_defense_config, "afp")
    dec = c1.get_intensity(0)
    upd = c1.submit_observations(0, 10, 0)
    
    with pytest.raises(dataclasses.FrozenInstanceError):
        dec.intensity = 0.5
    with pytest.raises(dataclasses.FrozenInstanceError):
        upd.tp = 20

def test_noise_logic_absence():
    controller_path = ROOT / "src/recall_aware_ids/controller/recall_controller.py"
    with open(controller_path, "r") as f:
        tree = ast.parse(f.read())
        
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                assert alias.name != 'random', "random module imported"
        elif isinstance(node, ast.ImportFrom):
            assert node.module != 'random', "random module imported"
            if node.module == 'numpy':
                for alias in node.names:
                    assert alias.name != 'random', "numpy.random imported"
        elif isinstance(node, ast.Attribute):
            if isinstance(node.value, ast.Name) and node.value.id == 'np' and node.attr == 'random':
                pytest.fail("np.random used")

def test_common_randomness(base_config, afp_defense_config):
    # Dummy required objects for AFP
    feature_names = [f"feat_{i}" for i in range(78)]
    modifiable_mask = [True] * 63 + [False] * 15
    train_bounds = pd.DataFrame({"train_min": [-100.0] * 78, "train_max": [100.0] * 78}, index=feature_names)
    benign_profile = pd.DataFrame({"mu": [0.0] * 78, "sigma": [1.0] * 78}, index=feature_names)

    # Simulate a defense object for Base (where intensity_max is always the parameter)
    base_afp = AdaptiveFeaturePoisoning(feature_names, modifiable_mask, train_bounds, benign_profile)
    
    # Simulate RA passing the identical intensity
    ra_afp = AdaptiveFeaturePoisoning(feature_names, modifiable_mask, train_bounds, benign_profile)
    
    c = RecallAwareController(base_config, afp_defense_config, "afp")
    dec = c.get_intensity(0)
    
    X = np.ones((5, 78), dtype=np.float32)
    
    # Base uses intensity_max directly, RA uses controller's decision
    # Ensure they both provide identical outputs when parameters match exactly
    # We pass the same attack_scenario="calibration", defense="afp", batch_id=0
    
    base_out, _, _ = base_afp.defend(X, epsilon_base=afp_defense_config['epsilon_base'], alpha=0.5, seed=42, attack_scenario="calibration", batch_id=0)
    ra_out, _, _ = ra_afp.defend(X, epsilon_base=dec.intensity, alpha=0.5, seed=42, attack_scenario="calibration", batch_id=0)
    
    np.testing.assert_array_equal(base_out, ra_out)
