import numpy as np

def select_boundary_targets(eligible_mask: np.ndarray, seed: int, n_attack_samples: int = 200, official_mode: bool = True) -> np.ndarray:
    """
    Selects exactly `n_attack_samples` across the complete measurement pool deterministically.
    """
    if isinstance(eligible_mask, bool) or not isinstance(eligible_mask, np.ndarray) or eligible_mask.ndim != 1 or eligible_mask.dtype != bool:
        raise ValueError("eligible_mask must be a 1D boolean numpy array")
        
    if official_mode and len(eligible_mask) != 72000:
        raise ValueError(f"Official mode requires exactly 72000 positions, got {len(eligible_mask)}")
        
    if isinstance(seed, bool) or not isinstance(seed, (int, np.integer)):
        raise TypeError("seed must be a finite integer")
        
    if isinstance(n_attack_samples, bool) or not isinstance(n_attack_samples, (int, np.integer)) or n_attack_samples <= 0:
        raise TypeError("n_attack_samples must be a positive integer")

    eligible_indices = np.where(eligible_mask)[0]
    
    if len(eligible_indices) < n_attack_samples:
        raise ValueError(f"Insufficient eligible samples for Boundary Attack. Found {len(eligible_indices)}, require {n_attack_samples}.")

    if official_mode and n_attack_samples != 200:
        raise ValueError(f"Official mode requires exactly 200 attack targets, got {n_attack_samples}")
        
    rng = np.random.RandomState(int(seed))
    
    # Select deterministically without replacement
    selected_indices = rng.choice(eligible_indices, size=n_attack_samples, replace=False)
    
    attempted_mask = np.zeros_like(eligible_mask, dtype=bool)
    attempted_mask[selected_indices] = True
    
    return attempted_mask
