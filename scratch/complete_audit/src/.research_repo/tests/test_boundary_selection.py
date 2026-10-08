import pytest
import numpy as np
from recall_aware_ids.experiment.boundary_selection import select_boundary_targets

def test_boundary_selection():
    # 72000 positions
    eligible_mask = np.zeros(72000, dtype=bool)
    eligible_mask[1000:2000] = True  # 1000 eligible
    
    selected = select_boundary_targets(eligible_mask, seed=42, n_attack_samples=200, official_mode=True)
    
    # Meaningful assertions instead of tautological ones
    assert selected.dtype == bool
    assert selected.shape == (72000,)
    assert np.sum(selected) == 200
    assert np.all(eligible_mask[selected])  # Subset of eligible
    
    # Determinism
    selected2 = select_boundary_targets(eligible_mask, seed=42, n_attack_samples=200, official_mode=True)
    assert np.array_equal(selected, selected2)
    
    # Rejections
    with pytest.raises(ValueError, match="1D boolean numpy array"):
        select_boundary_targets([True, False], seed=42, official_mode=False)
        
    with pytest.raises(ValueError, match="72000 positions"):
        select_boundary_targets(np.ones(100, dtype=bool), seed=42, official_mode=True)
        
    with pytest.raises(TypeError, match="finite integer"):
        select_boundary_targets(eligible_mask, seed=True, official_mode=True)
        
    with pytest.raises(TypeError, match="positive integer"):
        select_boundary_targets(eligible_mask, seed=42, n_attack_samples=False, official_mode=True)

    with pytest.raises(ValueError, match="Insufficient"):
        select_boundary_targets(eligible_mask, seed=42, n_attack_samples=2000, official_mode=True)
