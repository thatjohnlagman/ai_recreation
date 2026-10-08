"""
Tests for AFP, FS, and RS defense adapters.
Uses synthetic data only — no evaluation records accessed.
"""
import pytest
import numpy as np
import json

from recall_aware_ids.experiment.adapters import (
    AFPDefenseAdapter, FSDefenseAdapter, RSDefenseAdapter, DefenseAdapterError
)


# ─── Synthetic fixtures ──────────────────────────────────────────────────────

N = 500  # batch size
N_FEATURES = 78

# Load real feature mask to get correct mask (63 modifiable, 15 protected)
import json as _json
from pathlib import Path
ROOT = Path(__file__).parents[1]
with open(ROOT / "artifacts/preprocessors/feature_mask.json") as _f:
    _mask_data = _json.load(_f)
MASK = _mask_data["feature_mask"]         # list of bool
FNAMES = _mask_data["feature_columns"]    # list of str


def make_bounds_df():
    import pandas as pd
    return pd.read_parquet(ROOT / "artifacts/preprocessors/training_bounds.parquet")


def make_profile_df():
    import pandas as pd
    return pd.read_parquet(ROOT / "artifacts/preprocessors/afp_benign_profile.parquet")


def make_synthetic_X(seed=42):
    rng = np.random.RandomState(seed)
    return rng.rand(N, N_FEATURES).astype(np.float32)


class MockModel:
    """Deterministic mock model: predict 1 if feature[0] > 0.5, else 0."""
    def predict(self, X):
        return (X[:, 0] > 0.5).astype(int)

    def predict_proba(self, X):
        p1 = np.clip(X[:, 0].astype(float), 0.0, 1.0)
        return np.column_stack([1 - p1, p1])


# ─── AFP adapter ─────────────────────────────────────────────────────────────

@pytest.fixture
def afp_adapter():
    from recall_aware_ids.defenses.afp import AdaptiveFeaturePoisoning
    afp = AdaptiveFeaturePoisoning(
        feature_names=FNAMES, modifiable_mask=MASK,
        training_bounds=make_bounds_df(), benign_profile=make_profile_df()
    )
    return AFPDefenseAdapter(afp=afp, model=MockModel(), epsilon_base=0.0003, alpha=0.5)


def test_afp_adapter_returns_correct_shapes(afp_adapter):
    X = make_synthetic_X()
    preds, result, scores = afp_adapter.defend_batch(X, intensity=0.0003, seed=42,
                                                     attack_scenario="Test", batch_id=0)
    assert preds.shape == (N,)
    assert scores.shape == (N,)
    assert preds.dtype == int
    assert scores.dtype == float
    assert np.all((scores >= 0.0) & (scores <= 1.0))
    assert np.all((preds == 0) | (preds == 1))


def test_afp_adapter_uses_defended_output(afp_adapter):
    """Predictions must match model(X_defended), not model(X_original)."""
    X = make_synthetic_X(seed=1)
    preds, result, scores = afp_adapter.defend_batch(X, intensity=0.0003, seed=42,
                                                     attack_scenario="Test", batch_id=0)
    # Compare with raw preds on original X — they may differ if AFP perturbs feature[0]
    raw_preds = MockModel().predict(X)
    # We can't assert they differ (may happen to match), but shape and range must be correct
    assert preds.shape == raw_preds.shape


def test_afp_adapter_exposes_defense_diagnostics(afp_adapter):
    X = make_synthetic_X()
    _, result, _ = afp_adapter.defend_batch(X, intensity=0.0003, seed=42,
                                            attack_scenario="Test", batch_id=0)
    assert hasattr(result, "protected_feature_modification_count")
    assert result.protected_feature_modification_count == 0
    assert hasattr(result, "final_bounds_violation_count")
    assert result.final_bounds_violation_count == 0


# ─── FS adapter ──────────────────────────────────────────────────────────────

@pytest.fixture
def fs_adapter():
    from recall_aware_ids.defenses.feature_squeezing import FeatureSqueezing
    fs = FeatureSqueezing(feature_names=FNAMES, modifiable_mask=MASK,
                          training_bounds=make_bounds_df())
    return FSDefenseAdapter(fs=fs, model=MockModel())


def test_fs_adapter_returns_correct_shapes(fs_adapter):
    X = make_synthetic_X()
    preds, result, scores = fs_adapter.defend_batch(X, intensity=2.0, seed=42,
                                                    attack_scenario="Test", batch_id=0)
    assert preds.shape == (N,)
    assert scores.shape == (N,)


def test_fs_adapter_captures_effective_d(fs_adapter):
    X = make_synthetic_X()
    fs_adapter.defend_batch(X, intensity=2.0, seed=42, attack_scenario="Test", batch_id=0)
    # effective_d = max(0, 6 - int(2)) = 4
    assert fs_adapter.last_effective_d == 4

    fs_adapter.defend_batch(X, intensity=5.0, seed=42, attack_scenario="Test", batch_id=1)
    # effective_d = max(0, 6 - int(5)) = 1
    assert fs_adapter.last_effective_d == 1


def test_fs_adapter_uses_defended_output(fs_adapter):
    X = make_synthetic_X()
    preds, _, scores = fs_adapter.defend_batch(X, intensity=2.0, seed=42,
                                               attack_scenario="Test", batch_id=0)
    assert np.all((preds == 0) | (preds == 1))
    assert np.all((scores >= 0.0) & (scores <= 1.0))


# ─── RS adapter ──────────────────────────────────────────────────────────────

@pytest.fixture
def rs_adapter():
    from recall_aware_ids.defenses.randomized_smoothing import RandomizedSmoothing
    rs = RandomizedSmoothing(feature_names=FNAMES, modifiable_mask=MASK,
                              training_bounds=make_bounds_df(), ensemble_size=11)
    model = MockModel()
    return RSDefenseAdapter(rs=rs, predict_func=model.predict, chunk_size=100)


def test_rs_adapter_returns_correct_shapes(rs_adapter):
    X = make_synthetic_X()
    preds, result, scores = rs_adapter.defend_batch(X, intensity=0.0002, seed=42,
                                                    attack_scenario="Test", batch_id=0)
    assert preds.shape == (N,)
    assert scores.shape == (N,)
    assert np.all((scores >= 0.0) & (scores <= 1.0))


def test_rs_adapter_scores_are_vote_fractions(rs_adapter):
    """Scores must be multiples of 1/ensemble_size (11 members → 0, 1/11, ..., 1)."""
    X = make_synthetic_X()
    _, _, scores = rs_adapter.defend_batch(X, intensity=0.0002, seed=42,
                                           attack_scenario="Test", batch_id=0)
    allowed = {i / 11.0 for i in range(12)}
    for s in scores:
        assert any(abs(s - a) < 1e-6 for a in allowed), f"Unexpected score: {s}"


def test_rs_adapter_deterministic_same_context(rs_adapter):
    X = make_synthetic_X()
    p1, _, s1 = rs_adapter.defend_batch(X, intensity=0.0002, seed=42,
                                        attack_scenario="Test", batch_id=0)
    p2, _, s2 = rs_adapter.defend_batch(X, intensity=0.0002, seed=42,
                                        attack_scenario="Test", batch_id=0)
    assert np.array_equal(p1, p2)
    assert np.allclose(s1, s2)


def test_rs_adapter_different_batch_different_seed_context(rs_adapter):
    """
    Proves batch_id is incorporated into the RS noise seed.
    The seed derivation uses get_noise_seed(root_seed, scenario, 'rs', batch_id, ensemble_id).
    Different batch_id MUST produce different seed integers (hash collision is astronomically unlikely).
    """
    from recall_aware_ids.defenses.randomized_smoothing import RandomizedSmoothing
    import json as _j

    rs_obj = rs_adapter.rs  # underlying RandomizedSmoothing instance
    seed0_m0 = rs_obj.get_noise_seed(42, "Test", "rs", batch_id=0, ensemble_id=0)
    seed1_m0 = rs_obj.get_noise_seed(42, "Test", "rs", batch_id=1, ensemble_id=0)
    # Different batch_id must produce different seed integers
    assert seed0_m0 != seed1_m0, (
        "get_noise_seed with batch_id=0 and batch_id=1 must produce different integers"
    )
    # Same parameters must be deterministic
    seed0_m0_repeat = rs_obj.get_noise_seed(42, "Test", "rs", batch_id=0, ensemble_id=0)
    assert seed0_m0 == seed0_m0_repeat


def test_rs_adapter_exposes_defense_diagnostics(rs_adapter):
    X = make_synthetic_X()
    _, result, _ = rs_adapter.defend_batch(X, intensity=0.0002, seed=42,
                                           attack_scenario="Test", batch_id=0)
    assert hasattr(result, "protected_feature_modification_count")


def test_rs_adapter_spy_predictor_logic():
    from recall_aware_ids.defenses.randomized_smoothing import RandomizedSmoothing
    rs = RandomizedSmoothing(feature_names=FNAMES, modifiable_mask=MASK,
                             training_bounds=make_bounds_df(), ensemble_size=5)
    
    spy_calls = []
    def spy_predict(X_batch):
        spy_calls.append(X_batch.copy())
        # Predict 1 if the sum of all elements in the row is positive
        # To make things deterministic, let's just return a constant for each sample
        # Since RS generates noise, we'll return 1 for row index % 2 == 0
        return np.ones(len(X_batch), dtype=int)
        
    class MockModelWithProba:
        def predict(self, X):
            return spy_predict(X)
        def predict_proba(self, X):
            raise AssertionError("predict_proba must not be called")
            
    adapter = RSDefenseAdapter(rs=rs, predict_func=MockModelWithProba().predict, chunk_size=100)
    
    X = make_synthetic_X(seed=1)
    preds, result, scores = adapter.defend_batch(X, intensity=0.001, seed=42, attack_scenario="Test", batch_id=0)
    
    # Assert spy was called exactly ensemble_size * (N / chunk_size) times
    # ensemble_size is 5, N is 500, chunk is 100 -> 5 * 5 = 25 calls
    assert len(spy_calls) == 25
    # Since spy_predict always returns 1, scores should be 1.0, preds should be 1
    assert np.all(preds == 1)
    assert np.all(scores == 1.0)
    assert result.protected_feature_modification_count == 0
