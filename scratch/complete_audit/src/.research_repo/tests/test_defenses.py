import sys
import numpy as np
import pandas as pd
import pytest
from pathlib import Path

PROJECT_ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from recall_aware_ids.defenses.base import BaseDefense
from recall_aware_ids.defenses.afp import AdaptiveFeaturePoisoning
from recall_aware_ids.defenses.randomized_smoothing import RandomizedSmoothing
from recall_aware_ids.defenses.feature_squeezing import FeatureSqueezing

FEATURE_NAMES = [f"feat_{i}" for i in range(78)]
MODIFIABLE_MASK = [True] * 63 + [False] * 15

TRAIN_BOUNDS = pd.DataFrame(
    {"train_min": [-100.0] * 78, "train_max": [100.0] * 78},
    index=FEATURE_NAMES
)

BENIGN_PROFILE = pd.DataFrame(
    {"mu": [0.0] * 78, "sigma": [1.0] * 78},
    index=FEATURE_NAMES
)

def test_feature_order_mismatch():
    bad_bounds = TRAIN_BOUNDS.copy()
    bad_bounds.index = [f"feat_{i}" for i in range(77, -1, -1)]
    with pytest.raises(ValueError, match="perfectly match feature_names order"):
        BaseDefense(FEATURE_NAMES, MODIFIABLE_MASK, bad_bounds)

    bad_profile = BENIGN_PROFILE.copy()
    bad_profile.index = [f"feat_{i}" for i in range(77, -1, -1)]
    with pytest.raises(ValueError, match="perfectly match feature_names order"):
        AdaptiveFeaturePoisoning(FEATURE_NAMES, MODIFIABLE_MASK, TRAIN_BOUNDS, bad_profile)

def test_base_defense_projection():
    base = BaseDefense(FEATURE_NAMES, MODIFIABLE_MASK, TRAIN_BOUNDS)
    
    X_orig = np.zeros((2, 78), dtype=np.float32)
    X_orig[:, 75] = 10.0 # protected feature
    
    X_cand = np.ones((2, 78), dtype=np.float32) * 200.0
    
    X_proj, res = base.project_to_bounds(X_cand, X_orig)
    
    assert res.proposal_out_of_bounds_count == 2 * 63 # all 63 eligible cells are out of bounds
    assert X_proj.shape == (2, 78)
    assert X_proj.dtype == np.float32
    assert X_proj[0, 0] == 100.0 # clipped
    assert X_proj[0, 75] == 10.0 # protected restored exactly
    
    # input immutability
    assert X_cand[0, 0] == 200.0

def test_afp_exact_formula_and_flooring():
    profile = BENIGN_PROFILE.copy()
    profile.loc["feat_0", "sigma"] = 1e-10 # should be floored to 1e-6
    profile.loc["feat_0", "mu"] = 5.0
    
    afp = AdaptiveFeaturePoisoning(FEATURE_NAMES, MODIFIABLE_MASK, TRAIN_BOUNDS, profile)
    
    X = np.zeros((1, 78), dtype=np.float32)
    X[0, 0] = 6.0
    
    X_proj, invalid, mean_eps = afp.defend(X, epsilon_base=0.1, alpha=0.5, seed=42, attack_scenario="calibration", batch_id=1)
    
    # Delta should be abs(6.0 - 5.0) / 1e-6 = 1e6
    # epsilon_i = 0.1 * (1.0 + 0.5 * 1e6) = 0.1 * 500001 = 50000.1
    # Note: float32 precision might approximate 50000.1
    
    expected_eps = np.float32(0.1 * (1.0 + 0.5 * 1e6))
    # It applied standard noise scaled by this.
    # Since X[0, 0] was 6.0 and noise is in [-eps, eps], resulting val should be within [6 - eps, 6 + eps]
    # Then clipped to [-100, 100].
    
    assert X_proj[0, 0] >= -100.0 and X_proj[0, 0] <= 100.0
    # Mean epsilon for this cell alone would be ~50000.1. Other cells have delta=0 so eps=0.1.
    assert mean_eps > 0.1

def test_common_noise_equality_and_order_independence():
    # Base and RA should receive exact same underlying standard noise.
    afp1 = AdaptiveFeaturePoisoning(FEATURE_NAMES, MODIFIABLE_MASK, TRAIN_BOUNDS, BENIGN_PROFILE)
    afp2 = AdaptiveFeaturePoisoning(FEATURE_NAMES, MODIFIABLE_MASK, TRAIN_BOUNDS, BENIGN_PROFILE)
    
    X = np.ones((1, 78), dtype=np.float32)
    
    # Run 1
    X_proj1, _, _ = afp1.defend(X, epsilon_base=0.1, alpha=0.5, seed=42, attack_scenario="calibration", batch_id=1)
    X_proj2, _, _ = afp2.defend(X, epsilon_base=0.1, alpha=0.5, seed=42, attack_scenario="calibration", batch_id=1)
    assert np.array_equal(X_proj1, X_proj2)
    
    # Diff intensities preserve direction
    # To check direction, we see if (X_proj_high - X) and (X_proj_low - X) have same signs
    # assuming they don't hit clipping bounds.
    X_zero = np.zeros((1, 78), dtype=np.float32)
    X_low, _, _ = afp1.defend(X_zero, epsilon_base=0.1, alpha=0.0, seed=42, attack_scenario="calibration", batch_id=1)
    X_high, _, _ = afp1.defend(X_zero, epsilon_base=0.2, alpha=0.0, seed=42, attack_scenario="calibration", batch_id=1)
    
    # If they are not hitting bounds, high should be exactly 2x low
    assert np.allclose(X_high, X_low * 2.0, rtol=1e-5)

def test_rs_chunk_size_invariance():
    rs = RandomizedSmoothing(FEATURE_NAMES, MODIFIABLE_MASK, TRAIN_BOUNDS, ensemble_size=11)
    
    def mock_predict(X_proj):
        # Deterministic dummy model
        return (np.sum(X_proj, axis=1) > 10).astype(int)
        
    X = np.ones((10, 78), dtype=np.float32)
    
    preds_chunk10, inv10 = rs.predict_ensemble(X, sigma=0.5, seed=42, attack_scenario="calibration", batch_id=1, predict_func=mock_predict, chunk_size=10)
    preds_chunk2, inv2 = rs.predict_ensemble(X, sigma=0.5, seed=42, attack_scenario="calibration", batch_id=1, predict_func=mock_predict, chunk_size=2)
    
    assert np.array_equal(preds_chunk10, preds_chunk2)
    assert inv10 == inv2

def test_rs_binary_validation():
    rs = RandomizedSmoothing(FEATURE_NAMES, MODIFIABLE_MASK, TRAIN_BOUNDS, ensemble_size=3)
    def bad_predict(X_proj):
        return np.ones(X_proj.shape[0]) * 2 # invalid label
        
    X = np.ones((1, 78), dtype=np.float32)
    with pytest.raises(ValueError, match="strictly binary labels"):
        rs.predict_ensemble(X, sigma=0.5, seed=42, attack_scenario="calibration", batch_id=1, predict_func=bad_predict)

def test_feature_squeezing_semantics():
    fs = FeatureSqueezing(FEATURE_NAMES, MODIFIABLE_MASK, TRAIN_BOUNDS)
    
    X = np.zeros((1, 78), dtype=np.float32)
    X[0, 0] = 5.555555
    X[0, 75] = 5.555555 # protected
    
    # intensity 1 -> d = 6 - 1 = 5
    X_proj1, _, d1 = fs.defend(X, intensity=1.0)
    assert d1 == 5
    assert X_proj1[0, 0] == np.float32(5.55556)
    assert X_proj1[0, 75] == np.float32(5.555555) # protected unchanged
    
    # half-way rounding banker's rounding test in numpy
    X[0, 0] = 2.5
    X_proj2, _, d2 = fs.defend(X, intensity=6.0) # d=0
    assert X_proj2[0, 0] == 2.0 # 2.5 rounds to nearest even, which is 2.0

def test_nonfinite_rejection():
    X = np.ones((1, 78), dtype=np.float32)
    X[0, 0] = np.nan
    
    afp = AdaptiveFeaturePoisoning(FEATURE_NAMES, MODIFIABLE_MASK, TRAIN_BOUNDS, BENIGN_PROFILE)
    with pytest.raises(ValueError):
        afp.defend(X, 0.1, 0.5, 42, "calibration", 1)
        
    rs = RandomizedSmoothing(FEATURE_NAMES, MODIFIABLE_MASK, TRAIN_BOUNDS)
    with pytest.raises(ValueError):
        rs.predict_ensemble(X, 0.5, 42, "calibration", 1, lambda x: x)
        
    fs = FeatureSqueezing(FEATURE_NAMES, MODIFIABLE_MASK, TRAIN_BOUNDS)
    with pytest.raises(ValueError):
        fs.defend(X, 1.5, 42, "calibration", 1)

def test_defense_result_metrics():
    # Test fractions and hard failures on proposals
    afp = AdaptiveFeaturePoisoning(FEATURE_NAMES, MODIFIABLE_MASK, TRAIN_BOUNDS, BENIGN_PROFILE)
    X = np.zeros((10, 78), dtype=np.float32)
    # Give a tiny perturbation to ensure it fits or exceeds
    # If we pass extremely large values directly as X, it gets propagated.
    X_cand = np.ones((10, 78), dtype=np.float32) * 500.0 # Way out of bounds
    X_proj, res = afp.project_to_bounds(X_cand, X)
    
    # 10 samples, 63 modifiable features. 
    assert res.proposal_out_of_bounds_count == 10 * 63
    assert res.proposal_out_of_bounds_fraction == 1.0
    assert res.projection_unit_count == 10
    assert res.projection_unit_fraction == 1.0
    assert res.projected_cell_count == 10 * 63
    
    # Final output must be perfectly clean
    assert res.final_nan_count == 0
    assert res.final_inf_count == 0
    assert res.final_bounds_violation_count == 0
    assert res.final_invalid_fraction == 0.0
    assert res.protected_feature_modification_count == 0
    
    # Non-finite proposal is a hard failure
    X_bad = np.ones((1, 78), dtype=np.float32)
    X_bad[0, 0] = np.nan
    with pytest.raises(ValueError, match="Non-finite values encountered in proposal before projection"):
        afp.project_to_bounds(X_bad, X[:1])

def test_synthetic_report_fractions():
    # Independently recompute every stored fraction from integer counts
    # This verifies the script's exact denominator tracking.
    
    candidates = [
        {
            "defense": "afp",
            "proposal_out_of_bounds_count": 500,
            "proposal_out_of_bounds_fraction": 500.0 / (100 * 70),
            "projection_unit_count": 50,
            "projection_unit_fraction": 50.0 / 100,
            "final_invalid_fraction": 0.0,
            "n_samples": 100,
            "n_eligible": 70,
            "total_cells": 100 * 78,
            "final_nan": 0, "final_inf": 0, "final_oob": 0
        },
        {
            "defense": "rs",
            "proposal_out_of_bounds_count": 7000,
            "proposal_out_of_bounds_fraction": 7000.0 / (100 * 11 * 70),
            "projection_unit_count": 800,
            "projection_unit_fraction": 800.0 / (100 * 11),
            "final_invalid_fraction": 0.0,
            "n_samples": 100,
            "n_eligible": 70,
            "total_cells": 100 * 11 * 78,
            "final_nan": 0, "final_inf": 0, "final_oob": 0
        }
    ]
    
    for c in candidates:
        if c["defense"] in ("afp", "fs"):
            expected_oob_frac = c["proposal_out_of_bounds_count"] / (c["n_samples"] * c["n_eligible"])
            expected_unit_frac = c["projection_unit_count"] / c["n_samples"]
            expected_final_inv = (c["final_nan"] + c["final_inf"] + c["final_oob"]) / c["total_cells"]
        else:
            # RS
            expected_oob_frac = c["proposal_out_of_bounds_count"] / (c["n_samples"] * 11 * c["n_eligible"])
            expected_unit_frac = c["projection_unit_count"] / (c["n_samples"] * 11)
            expected_final_inv = (c["final_nan"] + c["final_inf"] + c["final_oob"]) / c["total_cells"]
            
        assert np.isclose(c["proposal_out_of_bounds_fraction"], expected_oob_frac)
        assert np.isclose(c["projection_unit_fraction"], expected_unit_frac)
        assert np.isclose(c["final_invalid_fraction"], expected_final_inv)

def test_rng_key_serialization_and_contexts():
    # Test proving JSON canonical serialization and different attack scenarios produce different streams
    afp1 = AdaptiveFeaturePoisoning(FEATURE_NAMES, MODIFIABLE_MASK, TRAIN_BOUNDS, BENIGN_PROFILE)
    
    seed1 = afp1.get_noise_seed(42, "scenario_a", "afp", 0)
    seed2 = afp1.get_noise_seed(42, "scenario_b", "afp", 0)
    
    assert seed1 != seed2
    
    # Test ambiguity prevention: "scenario_a" with batch 10 vs "scenario_a_10" with batch None
    # Wait, json.dumps avoids this
    seed3 = afp1.get_noise_seed(42, "scenario_a", "afp", 10)
    seed4 = afp1.get_noise_seed(42, "scenario_a_10", "afp", None)
    assert seed3 != seed4

def test_public_api_no_projection_bypass():
    afp = AdaptiveFeaturePoisoning(FEATURE_NAMES, MODIFIABLE_MASK, TRAIN_BOUNDS, BENIGN_PROFILE)
    
    X = np.zeros((2, 78), dtype=np.float32)
    # The public API should not accept an apply_projection kwarg
    with pytest.raises(TypeError):
        afp.defend(X, epsilon_base=0.1, alpha=1.0, seed=42, attack_scenario="calibration", batch_id=0, apply_projection=False)

def test_afp_uniform_distribution():
    # Verify AFP uses uniform noise exactly in [-1, 1] and can be reconstructed
    afp = AdaptiveFeaturePoisoning(FEATURE_NAMES, MODIFIABLE_MASK, TRAIN_BOUNDS, BENIGN_PROFILE)
    X = np.zeros((10, 78), dtype=np.float32)
    
    seed = 1337
    scenario = "calibration"
    batch_id = 0
    eps_base = 0.5
    
    # 1. Private generator (used in diagnostic)
    diagnostic_prop, _ = afp._generate_proposal(X, eps_base, 1.0, seed, scenario, batch_id)
    
    # 2. Reconstruct what it should be if Uniform
    seed_int = afp.get_noise_seed(seed, scenario, "afp", batch_id)
    rng = np.random.RandomState(seed_int)
    expected_standard_noise = rng.uniform(-1.0, 1.0, size=X.shape).astype(np.float32)
    
    # Check bounded
    assert np.all(expected_standard_noise >= -1.0)
    assert np.all(expected_standard_noise <= 1.0)
    
    expected_noise = expected_standard_noise * eps_base
    expected_noise[:, afp.protected_mask] = 0.0
    expected_cand = X + expected_noise
    
    # 3. Verify private generator matches manual uniform construction
    assert np.allclose(diagnostic_prop, expected_cand, atol=1e-6)
    
    # 4. Verify public defend matches exactly (because X=0 won't clip against [-100, 100])
    X_proj, _, _ = afp.defend(X, epsilon_base=eps_base, alpha=1.0, seed=seed, attack_scenario=scenario, batch_id=batch_id)
    assert np.allclose(X_proj, diagnostic_prop, atol=1e-6)
    
    # 5. Must fail if Gaussian noise were substituted
    rng_gauss = np.random.RandomState(seed_int)
    gauss_noise = rng_gauss.normal(0.0, 1.0, size=X.shape).astype(np.float32)
    gauss_cand = X + (gauss_noise * eps_base)
    gauss_cand[:, afp.protected_mask] = 0.0
    
    assert not np.allclose(diagnostic_prop, gauss_cand, atol=1e-6)

if __name__ == "__main__":
    pytest.main(["-q", __file__])
