"""
utils/seeds.py — Global deterministic seed management.
Set once at process startup using set_global_seeds().
"""
import random
import numpy as np


def set_global_seeds(seed: int = 42) -> None:
    """Set Python, NumPy, and scikit-learn (via env) seeds deterministically."""
    import os
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    # scikit-learn uses numpy's global RNG for most operations.
    # Pass random_state=seed explicitly to all sklearn objects.
