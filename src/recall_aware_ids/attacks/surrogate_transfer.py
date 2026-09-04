import numpy as np
from sklearn.tree import DecisionTreeClassifier
from recall_aware_ids.attacks.base import BaseAttack, AttackResult
from recall_aware_ids.attacks.oracle import BudgetExhaustedFailure, BatchBudgetFailure, DuplicateSampleIDsFailure

class SurrogateTransferAttack(BaseAttack):
    def __init__(self, feature_names, modifiable_mask, training_bounds, effective_seed=42):
        super().__init__(feature_names, modifiable_mask, training_bounds)
        self.effective_seed = effective_seed
        self.surrogate = None
        self.benign_class_idx = None
        
    def fit_surrogate(self, X_pool, y_pool_oracle):
        """
        Fit surrogate strictly using oracle-derived labels.
        """
        self.surrogate = DecisionTreeClassifier(max_depth=None, random_state=self.effective_seed)
        self.surrogate.fit(X_pool, y_pool_oracle)
        
        classes = self.surrogate.classes_
        if 0 not in classes:
            raise ValueError("Surrogate did not learn the benign class!")
        self.benign_class_idx = np.where(classes == 0)[0][0]
        
    def _extract_benign_paths(self):
        tree = self.surrogate.tree_
        paths = []
        
        def traverse(node_id, current_path):
            if tree.children_left[node_id] == tree.children_right[node_id]:
                pred_counts = tree.value[node_id][0]
                pred_class_idx = np.argmax(pred_counts)
                if pred_class_idx == self.benign_class_idx:
                    paths.append((node_id, current_path))
                return
                
            feature = tree.feature[node_id]
            threshold = tree.threshold[node_id]
            
            traverse(tree.children_left[node_id], current_path + [(feature, "<=", threshold)])
            traverse(tree.children_right[node_id], current_path + [(feature, ">", threshold)])
            
        traverse(0, [])
        return paths

    def _resolve_intervals(self, path):
        intervals = {f: [-np.inf, np.inf] for f in range(78)}
        
        for feature, op, threshold in path:
            t32 = np.float32(threshold)
            
            if op == "<=":
                val = t32 if t32 <= threshold else np.nextafter(t32, -np.inf, dtype=np.float32)
                intervals[feature][1] = min(intervals[feature][1], val)
                
            elif op == ">":
                val = t32 if t32 > threshold else np.nextafter(t32, np.inf, dtype=np.float32)
                intervals[feature][0] = max(intervals[feature][0], val)
                
        return intervals

    def generate_candidate(self, X_orig):
        """
        Generates a candidate using ONLY the surrogate DT.
        """
        if self.surrogate is None:
            raise RuntimeError("Surrogate must be fitted before candidate generation.")
            
        X_orig_32 = np.array(X_orig, dtype=np.float32)
        benign_paths = self._extract_benign_paths()
        
        feasible_candidates = []
        
        for leaf_id, path in benign_paths:
            intervals = self._resolve_intervals(path)
            
            contradiction = False
            for f in range(78):
                if intervals[f][0] > intervals[f][1]:
                    contradiction = True
                    break
            if contradiction: continue
                
            protected_fails = False
            for f in range(78):
                if self.protected_mask[f]:
                    orig_val = X_orig_32[f]
                    if orig_val < intervals[f][0] or orig_val > intervals[f][1]:
                        protected_fails = True
                        break
            if protected_fails: continue
                
            X_cand = np.array(X_orig_32, copy=True)
            for f in range(78):
                if self.modifiable_mask[f]:
                    if X_cand[f] < intervals[f][0]:
                        X_cand[f] = intervals[f][0]
                    elif X_cand[f] > intervals[f][1]:
                        X_cand[f] = intervals[f][1]
                        
            try:
                X_cand = self.project_and_clip(X_cand, X_orig_32)
            except ValueError:
                continue
                
            surrogate_pred = self.surrogate.predict(X_cand.reshape(1, -1))[0]
            if surrogate_pred != 0:
                continue
                
            mags = self.calculate_magnitudes(X_cand, X_orig_32)
            feasible_candidates.append({
                "X_adv": X_cand,
                "l0": mags["l0"],
                "l2": mags["l2"],
                "linf": mags["linf"],
                "leaf_id": leaf_id,
                "mags": mags
            })
            
        if not feasible_candidates:
            return None, "NO_FEASIBLE_CANDIDATE"
            
        # Tie-breaker lexicographical sort: min L0, min L2, min Linf, lowest leaf_id
        feasible_candidates.sort(key=lambda c: (c["l0"], c["l2"], c["linf"], c["leaf_id"]))
        best = feasible_candidates[0]
        
        return best["X_adv"], best["mags"]
        
    def evaluate_transfer(self, X_cand, X_orig, oracle, sample_id, true_label):
        """
        Submits finalized candidate to target oracle.
        """
        orig_pred = oracle.predict(X_orig, sample_ids=[sample_id], stage="eligibility")
        if isinstance(orig_pred, (BudgetExhaustedFailure, BatchBudgetFailure, DuplicateSampleIDsFailure)):
            return AttackResult(sample_id, X_orig, False, True, False, "BUDGET_EXHAUSTION", orig_pred.message, oracle.get_query_count(sample_id), self.calculate_magnitudes(X_orig, X_orig))
            
        orig_pred = orig_pred[0]
        
        if orig_pred == 0:
            status = "INELIGIBLE_FALSE_NEGATIVE" if true_label == 1 else "INELIGIBLE_TRUE_BENIGN"
            return AttackResult(sample_id, X_orig, False, False, None, status, f"Input ineligible: {status}", oracle.get_query_count(sample_id), self.calculate_magnitudes(X_orig, X_orig))
            
        if true_label != 1:
            return AttackResult(sample_id, X_orig, False, False, None, "INELIGIBLE_TRUE_BENIGN", "Input ineligible: INELIGIBLE_TRUE_BENIGN", oracle.get_query_count(sample_id), self.calculate_magnitudes(X_orig, X_orig))
            
        if X_cand is None:
            return AttackResult(sample_id, X_orig, True, True, False, "NO_FEASIBLE_CANDIDATE", "Candidate generation yielded no valid candidate.", oracle.get_query_count(sample_id), self.calculate_magnitudes(X_orig, X_orig))
            
        # Re-apply projection immediately before final query
        try:
            X_cand = self.project_and_clip(X_cand, X_orig)
        except ValueError:
            return AttackResult(sample_id, X_orig, True, True, False, "PROJECTION_FAILED_BEFORE_TRANSFER", "Failed to project and clip candidate safely.", oracle.get_query_count(sample_id), self.calculate_magnitudes(X_orig, X_orig))
            
        final_pred = oracle.predict(X_cand, sample_ids=[sample_id], stage="verification")
        if isinstance(final_pred, (BudgetExhaustedFailure, BatchBudgetFailure, DuplicateSampleIDsFailure)):
            return AttackResult(sample_id, X_cand, True, True, False, "BUDGET_EXHAUSTION", final_pred.message, oracle.get_query_count(sample_id), self.calculate_magnitudes(X_cand, X_orig))
            
        final_pred = final_pred[0]
        success = (final_pred == 0)
        reason = "SUCCESS" if success else "TARGET_REJECTION"
        
        return AttackResult(
            sample_id=sample_id,
            X_adv=X_cand if success else X_orig,
            eligible=True,
            attempted=True,
            success=success,
            status_code=reason,
            message=f"Transfer result: {reason}",
            query_count=oracle.get_query_count(sample_id),
            magnitudes=self.calculate_magnitudes(X_cand if success else X_orig, X_orig)
        )
