import numpy as np
from recall_aware_ids.defenses.base import BaseDefense

class FeatureSqueezing(BaseDefense):
    def defend(self, X, intensity, seed=None, attack_scenario="none", batch_id=None):
        """
        Applies FS to the batch X.
        d = max(0, 6 - int(intensity))
        x_prime = round(x, d)
        """
        if not np.isfinite(intensity) or intensity < 0.0 or intensity > 6.0:
            raise ValueError(f"Feature squeezing intensity must be in [0, 6], got {intensity}")

        if not np.isfinite(X).all():
            raise ValueError("Non-finite values encountered in input.")
            
        if not np.isfinite(intensity):
            raise ValueError("Intensity must be a finite number.")
            
        d = max(0, 6 - int(intensity))
        
        # Rounding applied to eligible features
        X_cand = np.array(X, dtype=np.float32, copy=True)
        # Using np.round which employs 'round half to even' (bankers rounding)
        X_cand[:, self.modifiable_mask] = np.round(X[:, self.modifiable_mask], decimals=d)
        
        X_proj, result = self.project_to_bounds(X_cand, X)
        
        return X_proj, result, d
