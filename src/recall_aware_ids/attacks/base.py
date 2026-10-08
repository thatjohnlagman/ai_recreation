import numpy as np

class AttackResult:
    def __init__(self, sample_id, X_adv, eligible, attempted, success, status_code, message, query_count, magnitudes):
        self.sample_id = sample_id
        self.X_adv = X_adv
        self.eligible = eligible
        self.attempted = attempted
        self.success = success  # True, False, or None
        self.status_code = status_code # Standardized code
        self.message = message # Human readable message
        self.query_count = query_count
        self.magnitudes = magnitudes # dict of l0, l1, l2, linf

class BaseAttack:
    """
    Abstract base class for black-box adversarial attack scenarios.
    Enforces protected feature preservation and training bounds clipping.
    """
    def __init__(self, feature_names, modifiable_mask, training_bounds):
        if len(feature_names) != 78:
            raise ValueError(f"Expected 78 feature names, got {len(feature_names)}")
        if len(modifiable_mask) != 78:
            raise ValueError(f"Expected 78 mask elements, got {len(modifiable_mask)}")
            
        # Validate feature/bounds order explicitly using index matching
        if hasattr(training_bounds, "index") and list(training_bounds.index) != list(range(78)):
            # If it's a pandas DataFrame with feature names as index
            if list(training_bounds.index) != list(feature_names):
                raise ValueError("training_bounds index does not perfectly match feature_names order.")
                
        self.feature_names = feature_names
        self.modifiable_mask = np.array(modifiable_mask, dtype=bool)
        self.protected_mask = ~self.modifiable_mask
        if 'feature' in training_bounds.columns:
            training_bounds = training_bounds.set_index('feature')
            
        min_col = 'train_min' if 'train_min' in training_bounds.columns else 'min'
        max_col = 'train_max' if 'train_max' in training_bounds.columns else 'max'
        
        self.train_min = np.array(training_bounds[min_col].values, dtype=np.float32)
        self.train_max = np.array(training_bounds[max_col].values, dtype=np.float32)
        if len(self.train_min) != 78 or len(self.train_max) != 78:
            raise ValueError("Bounds must be length 78")
            
        if not np.isfinite(self.train_min).all() or not np.isfinite(self.train_max).all():
            raise ValueError("Training bounds must be finite")
            
        if not np.all(self.train_min <= self.train_max):
            raise ValueError("Minimum bounds must be <= maximum bounds")

    def project_and_clip(self, X_adv, X_orig):
        """
        Works on a copy.
        Restores protected features.
        Clips modifiable features to inclusive [train_min, train_max].
        Rejects nonfinite inputs/outputs.
        """
        if X_adv.shape != (78,) or X_orig.shape != (78,):
            raise ValueError("project_and_clip expects exactly 1D array of shape (78,)")
            
        if not np.isfinite(X_adv).all() or not np.isfinite(X_orig).all():
            raise ValueError("Non-finite values encountered in input.")
            
        X_proj = np.array(X_adv, dtype=np.float32, copy=True)
        X_orig_32 = np.array(X_orig, dtype=np.float32)
        
        # Restore protected features perfectly
        X_proj[self.protected_mask] = X_orig_32[self.protected_mask]
        
        # Clip only modifiable features
        X_proj[self.modifiable_mask] = np.clip(
            X_proj[self.modifiable_mask], 
            self.train_min[self.modifiable_mask], 
            self.train_max[self.modifiable_mask]
        )
        
        if not np.isfinite(X_proj).all():
            raise ValueError("Non-finite values produced after clipping.")
            
        # Verify protected equality after projection
        if not np.array_equal(X_proj[self.protected_mask], X_orig_32[self.protected_mask]):
            raise ValueError("Protected feature restoration failed.")
            
        return X_proj

    def calculate_magnitudes(self, X_adv, X_orig):
        """
        Computes L0 (exact float32 changed features), L1, L2, L_inf (float64 calculation).
        """
        x_a = np.array(X_adv, dtype=np.float32)
        x_o = np.array(X_orig, dtype=np.float32)
        
        # Exact float32 inequality for L0
        l0 = np.sum(x_a != x_o)
        
        # Convert to float64 BEFORE subtraction to avoid precision issues
        diff = x_a.astype(np.float64) - x_o.astype(np.float64)
        
        l1 = np.sum(np.abs(diff))
        l2 = np.sqrt(np.sum(diff ** 2))
        linf = np.max(np.abs(diff)) if len(diff) > 0 else 0.0
        
        return {"l0": int(l0), "l1": float(l1), "l2": float(l2), "linf": float(linf)}

    def generate(self, X, oracle, sample_id, **kwargs):
        raise NotImplementedError
