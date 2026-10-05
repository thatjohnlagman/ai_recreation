import numpy as np
from defenses.base import BaseDefense

class RandomizedSmoothing(BaseDefense):
    def __init__(self, feature_names, modifiable_mask, training_bounds, ensemble_size=11):
        super().__init__(feature_names, modifiable_mask, training_bounds)
        self.ensemble_size = ensemble_size

    def predict_ensemble(self, X, sigma, seed, attack_scenario, batch_id, predict_func, chunk_size=100, return_scores=False):
        if not np.isfinite(sigma) or sigma < 0.0:
            raise ValueError(f"sigma must be finite and non-negative, got {sigma}")
        
        if not np.isfinite(X).all():
            raise ValueError("Non-finite values encountered in input.")
            
        n_samples = X.shape[0]
        final_preds = np.zeros(n_samples, dtype=int)
        
        total_original_out_of_bounds_count = 0
        total_original_eligible_out_of_bounds_count = 0
        total_oob_count = 0
        total_projection_unit_count = 0
        total_projected_cell_count = 0
        total_final_nan_count = 0
        total_final_inf_count = 0
        total_final_bounds_violation_count = 0
        total_protected_feature_modification_count = 0
        
        member_votes = np.zeros((n_samples, 2), dtype=int)
        
        for m in range(self.ensemble_size):
            seed_int = self.get_noise_seed(seed, attack_scenario, "rs", batch_id, ensemble_id=m)
            rng = np.random.RandomState(seed_int)
            
            # Generate the FULL standard noise matrix exactly once per ensemble member
            standard_noise_full = rng.normal(0.0, 1.0, size=X.shape).astype(np.float32)
            
            for start_idx in range(0, n_samples, chunk_size):
                end_idx = min(start_idx + chunk_size, n_samples)
                X_chunk = X[start_idx:end_idx]
                
                # Consume slices of that matrix
                noise = standard_noise_full[start_idx:end_idx] * sigma
                noise[:, self.protected_mask] = 0.0
                
                X_cand = X_chunk + noise
                X_proj, res = self.project_to_bounds(X_cand, X_chunk)
                
                if m == 0:
                    total_original_out_of_bounds_count += res.original_out_of_bounds_count
                    total_original_eligible_out_of_bounds_count += res.original_eligible_out_of_bounds_count
                    
                total_oob_count += res.proposal_out_of_bounds_count
                total_projection_unit_count += res.projection_unit_count
                total_projected_cell_count += res.projected_cell_count
                total_final_nan_count += res.final_nan_count
                total_final_inf_count += res.final_inf_count
                total_final_bounds_violation_count += res.final_bounds_violation_count
                total_protected_feature_modification_count += res.protected_feature_modification_count
                
                preds = predict_func(X_proj)
                
                if not np.all(np.isin(preds, [0, 1])):
                    raise ValueError("Prediction function must return strictly binary labels [0, 1].")
                    
                for i, p in enumerate(preds):
                    member_votes[start_idx + i, int(p)] += 1
                    
            # standard_noise_full is garbage collected here at the end of the outer loop iteration
            
        for i in range(n_samples):
            if member_votes[i, 1] > member_votes[i, 0]:
                final_preds[i] = 1
            else:
                final_preds[i] = 0
                
        total_eligible_cells = n_samples * self.ensemble_size * int(np.sum(self.modifiable_mask))
        total_cells = n_samples * self.ensemble_size * 78
        total_units = n_samples * self.ensemble_size
        
        proposal_out_of_bounds_fraction = total_oob_count / total_eligible_cells if total_eligible_cells > 0 else 0.0
        projection_unit_fraction = total_projection_unit_count / total_units if total_units > 0 else 0.0
        
        final_invalid_count = total_final_nan_count + total_final_inf_count + total_final_bounds_violation_count
        final_invalid_fraction = final_invalid_count / total_cells if total_cells > 0 else 0.0
        
        from defenses.base import DefenseResult
        result = DefenseResult(
            original_out_of_bounds_count=total_original_out_of_bounds_count,
            original_eligible_out_of_bounds_count=total_original_eligible_out_of_bounds_count,
            proposal_out_of_bounds_count=total_oob_count,
            proposal_out_of_bounds_fraction=proposal_out_of_bounds_fraction,
            projection_unit_count=total_projection_unit_count,
            projection_unit_fraction=projection_unit_fraction,
            projected_cell_count=total_projected_cell_count,
            final_nan_count=total_final_nan_count,
            final_inf_count=total_final_inf_count,
            final_bounds_violation_count=total_final_bounds_violation_count,
            final_invalid_fraction=final_invalid_fraction,
            protected_feature_modification_count=total_protected_feature_modification_count
        )
        if return_scores:
            positive_vote_fraction = member_votes[:, 1] / self.ensemble_size
            return final_preds, result, positive_vote_fraction
        return final_preds, result
