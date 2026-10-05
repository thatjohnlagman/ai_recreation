import numpy as np
from defenses.base import BaseDefense

class AdaptiveFeaturePoisoning(BaseDefense):
    def __init__(self, feature_names, modifiable_mask, training_bounds, benign_profile):
        super().__init__(feature_names, modifiable_mask, training_bounds)
        
        # Verify benign profile
        if 'feature' in benign_profile.columns:
            benign_profile = benign_profile.set_index('feature')
            
        if hasattr(benign_profile, "index") and list(benign_profile.index) != list(range(78)):
            if list(benign_profile.index) != list(self.feature_names):
                raise ValueError("benign_profile index does not perfectly match feature_names order.")
                
        mean_col = 'benign_mean' if 'benign_mean' in benign_profile.columns else 'mu'
        std_col = 'benign_std' if 'benign_std' in benign_profile.columns else 'sigma'
        
        self.mu = np.array(benign_profile[mean_col].values, dtype=np.float32)
        self.sigma = np.array(benign_profile[std_col].values, dtype=np.float32)
        
        if len(self.mu) != 78 or len(self.sigma) != 78:
            raise ValueError("benign_profile must have length 78")
            
        if not np.isfinite(self.mu).all() or not np.isfinite(self.sigma).all():
            raise ValueError("benign_profile must be finite")

    def _generate_proposal(self, X, epsilon_base, alpha, seed, attack_scenario, batch_id):
        if not np.isfinite(epsilon_base) or epsilon_base < 0.0:
            raise ValueError(f"epsilon_base must be finite and non-negative, got {epsilon_base}")
        if not np.isfinite(alpha) or alpha < 0.0:
            raise ValueError(f"alpha must be finite and non-negative, got {alpha}")
        
        if not np.isfinite(X).all():
            raise ValueError("Non-finite values encountered in input.")
            
        seed_int = self.get_noise_seed(seed, attack_scenario, "afp", batch_id)
        rng = np.random.RandomState(seed_int)
        
        standard_noise = rng.uniform(
            low=-1.0, 
            high=1.0, 
            size=X.shape
        ).astype(np.float32)
        
        sigma_safe = np.maximum(np.abs(self.sigma), 1.0e-6)
        delta = np.abs(X - self.mu) / sigma_safe
        epsilon_i = epsilon_base * (1.0 + alpha * delta)
        
        noise = standard_noise * epsilon_i
        noise[:, self.protected_mask] = 0.0
        
        X_cand = X + noise
        return X_cand, epsilon_i

    def defend(self, X, epsilon_base, alpha, seed, attack_scenario, batch_id):
        """
        Applies AFP to the batch X.
        sigma_safe = np.maximum(np.abs(sigma), 1.0e-6)
        delta = np.abs(x - mu) / sigma_safe
        epsilon_i = epsilon_base * (1.0 + alpha * delta)
        """
        X_cand, epsilon_i = self._generate_proposal(X, epsilon_base, alpha, seed, attack_scenario, batch_id)
        
        X_proj, result = self.project_to_bounds(X_cand, X)
        
        # For calibration reporting, we also want the mean epsilon_i across eligible cells
        mean_epsilon_i = float(np.mean(epsilon_i[:, self.modifiable_mask]))
        
        return X_proj, result, mean_epsilon_i
