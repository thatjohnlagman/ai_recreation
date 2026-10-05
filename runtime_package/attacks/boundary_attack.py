import numpy as np
from attacks.base import BaseAttack, AttackResult
from attacks.oracle import BudgetExhaustedFailure, BatchBudgetFailure, DuplicateSampleIDsFailure

class DecisionBoundaryAttack(BaseAttack):
    def __init__(self, feature_names, modifiable_mask, training_bounds, max_queries=50, binary_search_steps=10):
        super().__init__(feature_names, modifiable_mask, training_bounds)
        self.max_queries = max_queries
        self.binary_search_steps = binary_search_steps

    def generate(self, X_orig, oracle, sample_id, true_label, reference_pool):
        """
        reference_pool is an array of known benign samples deterministically ordered.
        """
        # Validate Oracle config agreement early
        if oracle.max_queries_per_sample is None or oracle.max_queries_per_sample != self.max_queries:
            raise ValueError(f"Oracle budget ({oracle.max_queries_per_sample}) mismatches attack limit ({self.max_queries}). None is not accepted.")
            
        X_orig_32 = np.array(X_orig, dtype=np.float32)
        
        # 1. Eligibility Check
        orig_pred = oracle.predict(X_orig_32, sample_ids=[sample_id], stage="eligibility")
        if isinstance(orig_pred, (BudgetExhaustedFailure, BatchBudgetFailure, DuplicateSampleIDsFailure)):
            return self._fail_result(sample_id, X_orig_32, "BUDGET_EXHAUSTION", oracle)
        orig_pred = orig_pred[0]
        
        if orig_pred == 0:
            status = "INELIGIBLE_FALSE_NEGATIVE" if true_label == 1 else "INELIGIBLE_TRUE_BENIGN"
            return self._ineligible_result(sample_id, X_orig_32, oracle, status)
            
        if true_label != 1:
            return self._ineligible_result(sample_id, X_orig_32, oracle, "INELIGIBLE_TRUE_BENIGN")
            
        # 2. Endpoint Verification (Search for valid reference)
        valid_reference = None
        for ref in reference_pool:
            # Check budget to see if we can finish: 1 endpoint + exactly 10 steps + 1 verification = 12 queries.
            # E.g. limit is 50. Eligibility took 1. Current = 1.
            # Max endpoints we can screen = 38 (38 endpoints + 1 eligibility + 11 = 50 limit).
            current = oracle.get_query_count(sample_id)
            if self.max_queries - current < (1 + self.binary_search_steps + 1):
                return self._fail_result(sample_id, X_orig_32, "INSUFFICIENT_BUDGET_FOR_FULL_SEARCH", oracle)
                
            # Construct masked reference
            masked_ref = np.array(ref, dtype=np.float32, copy=True)
            masked_ref[self.protected_mask] = X_orig_32[self.protected_mask]
            
            try:
                masked_ref = self.project_and_clip(masked_ref, X_orig_32)
            except ValueError:
                continue
                
            ref_pred = oracle.predict(masked_ref, sample_ids=[sample_id], stage="endpoint_screening")
            if isinstance(ref_pred, (BudgetExhaustedFailure, BatchBudgetFailure, DuplicateSampleIDsFailure)):
                return self._fail_result(sample_id, X_orig_32, "BUDGET_EXHAUSTION", oracle)
                
            if ref_pred[0] == 0:
                valid_reference = masked_ref
                break
                
        if valid_reference is None:
            return self._fail_result(sample_id, X_orig_32, "NO_FEASIBLE_CANDIDATE", oracle)
            
        # 3. Binary Search Interpolation (EXACTLY 10 queries executed)
        x_attack = np.array(X_orig_32, dtype=np.float64)
        x_benign = np.array(valid_reference, dtype=np.float64)
        
        for step in range(self.binary_search_steps):
            # Safe float64 midpoint calculation
            x_mid = x_attack + (x_benign - x_attack) / 2.0
            
            # Cast back to float32 for model and bounds
            x_mid_32 = np.array(x_mid, dtype=np.float32)
            
            try:
                x_mid_32 = self.project_and_clip(x_mid_32, X_orig_32)
            except ValueError:
                return self._fail_result(sample_id, X_orig_32, "PROJECTION_FAILED_DURING_SEARCH", oracle)
                
            mid_pred = oracle.predict(x_mid_32, sample_ids=[sample_id], stage=f"binary_search_step_{step}")
            if isinstance(mid_pred, (BudgetExhaustedFailure, BatchBudgetFailure, DuplicateSampleIDsFailure)):
                return self._fail_result(sample_id, X_orig_32, "BUDGET_EXHAUSTION", oracle)
                
            if mid_pred[0] == 0:
                x_benign = np.array(x_mid_32, dtype=np.float64)
            else:
                x_attack = np.array(x_mid_32, dtype=np.float64)
                
        # 4. Final Verification
        try:
            x_benign_32 = self.project_and_clip(np.array(x_benign, dtype=np.float32), X_orig_32)
        except ValueError:
            return self._fail_result(sample_id, X_orig_32, "PROJECTION_FAILED_BEFORE_TRANSFER", oracle)
            
        final_pred = oracle.predict(x_benign_32, sample_ids=[sample_id], stage="verification")
        if isinstance(final_pred, (BudgetExhaustedFailure, BatchBudgetFailure, DuplicateSampleIDsFailure)):
            return self._fail_result(sample_id, X_orig_32, "BUDGET_EXHAUSTION", oracle)
            
        if final_pred[0] == 0:
            return AttackResult(
                sample_id=sample_id,
                X_adv=x_benign_32,
                eligible=True,
                attempted=True,
                success=True,
                status_code="SUCCESS",
                message="Final verification confirmed evasion",
                query_count=oracle.get_query_count(sample_id),
                magnitudes=self.calculate_magnitudes(x_benign_32, X_orig_32)
            )
        else:
            return self._fail_result(sample_id, X_orig_32, "TARGET_REJECTION", oracle)

    def _fail_result(self, sample_id, X_orig_32, code, oracle):
        return AttackResult(
            sample_id=sample_id,
            X_adv=X_orig_32,
            eligible=True,
            attempted=True,
            success=False,
            status_code=code,
            message=f"Attack failed: {code}",
            query_count=oracle.get_query_count(sample_id),
            magnitudes=self.calculate_magnitudes(X_orig_32, X_orig_32)
        )
        
    def _ineligible_result(self, sample_id, X_orig_32, oracle, status):
        return AttackResult(
            sample_id=sample_id,
            X_adv=X_orig_32,
            eligible=False,
            attempted=False,
            success=None,
            status_code=status,
            message=f"Input ineligible: {status}",
            query_count=oracle.get_query_count(sample_id),
            magnitudes=self.calculate_magnitudes(X_orig_32, X_orig_32)
        )
