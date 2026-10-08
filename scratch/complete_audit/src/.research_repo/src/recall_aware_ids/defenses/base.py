import hashlib
import struct
import json
import numpy as np
from dataclasses import dataclass

@dataclass(frozen=True)
class DefenseResult:
    original_out_of_bounds_count: int
    original_eligible_out_of_bounds_count: int
    proposal_out_of_bounds_count: int
    proposal_out_of_bounds_fraction: float
    projection_unit_count: int
    projection_unit_fraction: float
    projected_cell_count: int
    final_nan_count: int
    final_inf_count: int
    final_bounds_violation_count: int
    final_invalid_fraction: float
    protected_feature_modification_count: int

class BaseDefense:
    def __init__(self, feature_names, modifiable_mask, training_bounds):
        if len(feature_names) != 78:
            raise ValueError(f"Expected 78 feature names, got {len(feature_names)}")
        if len(modifiable_mask) != 78:
            raise ValueError(f"Expected 78 mask elements, got {len(modifiable_mask)}")
            
        n_eligible = int(np.sum(modifiable_mask))
        n_protected = 78 - n_eligible
        if n_eligible != 63 or n_protected != 15:
            raise ValueError(f"Expected exactly 63 eligible and 15 protected features. Got {n_eligible} eligible, {n_protected} protected.")
            
        if 'feature' in training_bounds.columns:
            training_bounds = training_bounds.set_index('feature')
            
        if hasattr(training_bounds, "index") and list(training_bounds.index) != list(range(78)):
            if list(training_bounds.index) != list(feature_names):
                raise ValueError("training_bounds index does not perfectly match feature_names order.")
                
        self.feature_names = feature_names
        self.modifiable_mask = np.array(modifiable_mask, dtype=bool)
        self.protected_mask = ~self.modifiable_mask
        
        min_col = 'train_min' if 'train_min' in training_bounds.columns else 'min'
        max_col = 'train_max' if 'train_max' in training_bounds.columns else 'max'
        self.train_min = np.array(training_bounds[min_col].values, dtype=np.float32)
        self.train_max = np.array(training_bounds[max_col].values, dtype=np.float32)
        
        if not np.isfinite(self.train_min).all() or not np.isfinite(self.train_max).all():
            raise ValueError("Training bounds contain non-finite values.")
            
        if not np.all(self.train_min <= self.train_max):
            raise ValueError("Training bounds have minimum > maximum.")

        if len(self.modifiable_mask) != 78:
            raise ValueError("Modifiable mask must match 78 features.")
            
        self.train_min = training_bounds["train_min"].values.astype(np.float32)
        self.train_max = training_bounds["train_max"].values.astype(np.float32)
        
    def get_noise_seed(self, root_seed, attack_scenario, defense_mechanism, batch_id=None, ensemble_id=None):
        if batch_id is None:
            batch_id = "none"
        if ensemble_id is None:
            ensemble_id = "none"
            
        context = [root_seed, attack_scenario, defense_mechanism, batch_id, ensemble_id]
        context_str = json.dumps(context)
        
        return int(hashlib.sha256(context_str.encode("utf-8")).hexdigest(), 16) % (2**32)

    def project_to_bounds(self, X_cand, X_orig):
        """
        Works on a copy.
        Restores protected features exactly.
        Clips modifiable features to inclusive [train_min, train_max].
        Returns the projected array and the DefenseResult object.
        """
        if not np.isfinite(X_cand).all():
            raise ValueError("Non-finite values encountered in proposal before projection.")
            
        X_proj = np.array(X_cand, dtype=np.float32, copy=True)
        X_orig_32 = np.array(X_orig, dtype=np.float32)
        
        n_samples = X_proj.shape[0]
        n_eligible_features = int(np.sum(self.modifiable_mask))
        total_eligible_cells = n_samples * n_eligible_features
        total_cells = n_samples * 78
        
        # Original bounds violations (strict)
        orig_oob_mask = (X_orig_32 < self.train_min) | (X_orig_32 > self.train_max)
        original_out_of_bounds_count = int(np.sum(orig_oob_mask))
        
        orig_eligible = X_orig_32[:, self.modifiable_mask]
        eligible_mins = self.train_min[self.modifiable_mask]
        eligible_maxs = self.train_max[self.modifiable_mask]
        
        orig_eligible_oob_mask = (orig_eligible < eligible_mins) | (orig_eligible > eligible_maxs)
        original_eligible_out_of_bounds_count = int(np.sum(orig_eligible_oob_mask))
        
        effective_mins = np.minimum(orig_eligible, eligible_mins)
        effective_maxs = np.maximum(orig_eligible, eligible_maxs)
        
        eligible_cands = X_proj[:, self.modifiable_mask]
        
        # Diagnostics pre-projection (using effective bounds)
        oob_mask = (eligible_cands < effective_mins) | (eligible_cands > effective_maxs)
        proposal_out_of_bounds_count = int(np.sum(oob_mask))
        proposal_out_of_bounds_fraction = proposal_out_of_bounds_count / total_eligible_cells if total_eligible_cells > 0 else 0.0
        
        # Projection units (rows with at least one OOB eligible cell)
        rows_with_oob = np.any(oob_mask, axis=1)
        projection_unit_count = int(np.sum(rows_with_oob))
        projection_unit_fraction = projection_unit_count / n_samples if n_samples > 0 else 0.0
        
        # Apply Clipping
        # A proposal that crosses the effective bounds is clipped.
        X_proj[:, self.modifiable_mask] = np.clip(
            eligible_cands, 
            effective_mins, 
            effective_maxs
        )
        
        # Calculate exactly how many cells were changed by projection
        post_clip_eligible = X_proj[:, self.modifiable_mask]
        projected_cell_count = int(np.sum(post_clip_eligible != eligible_cands))
        
        # Restore Protected Features and count modifications if any occurred
        protected_orig = X_orig_32[:, self.protected_mask]
        X_proj[:, self.protected_mask] = protected_orig
        
        # Calculate final state metrics on ALL cells (since final state must be pristine relative to effective bounds)
        final_nan_count = int(np.sum(np.isnan(X_proj)))
        final_inf_count = int(np.sum(np.isinf(X_proj)))
        
        protected_proj = X_proj[:, self.protected_mask]
        protected_feature_modification_count = int(np.sum(protected_proj != protected_orig))
        
        # Final OOB check using effective bounds for all cells
        effective_mins_all = np.minimum(X_orig_32, self.train_min)
        effective_maxs_all = np.maximum(X_orig_32, self.train_max)
        
        final_oob_mask = (X_proj < effective_mins_all) | (X_proj > effective_maxs_all)
        final_bounds_violation_count = int(np.sum(final_oob_mask & np.isfinite(X_proj)))
        
        final_invalid_count = final_nan_count + final_inf_count + final_bounds_violation_count
        final_invalid_fraction = final_invalid_count / total_cells if total_cells > 0 else 0.0
        
        result = DefenseResult(
            original_out_of_bounds_count=original_out_of_bounds_count,
            original_eligible_out_of_bounds_count=original_eligible_out_of_bounds_count,
            proposal_out_of_bounds_count=proposal_out_of_bounds_count,
            proposal_out_of_bounds_fraction=proposal_out_of_bounds_fraction,
            projection_unit_count=projection_unit_count,
            projection_unit_fraction=projection_unit_fraction,
            projected_cell_count=projected_cell_count,
            final_nan_count=final_nan_count,
            final_inf_count=final_inf_count,
            final_bounds_violation_count=final_bounds_violation_count,
            final_invalid_fraction=final_invalid_fraction,
            protected_feature_modification_count=protected_feature_modification_count
        )
        
        return X_proj, result
