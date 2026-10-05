import numpy as np

class BudgetExhaustedFailure:
    def __init__(self, sample_id, stage, query_limit):
        self.sample_id = sample_id
        self.stage = stage
        self.query_limit = query_limit
        self.message = f"Query budget exhausted ({query_limit}) for sample {sample_id} at stage: {stage}"

class BatchBudgetFailure:
    def __init__(self, exhausted_ids):
        self.exhausted_ids = exhausted_ids
        self.message = f"Batch query rejected due to budget exhaustion for IDs: {exhausted_ids}"

class DuplicateSampleIDsFailure:
    def __init__(self, duplicates):
        self.duplicates = duplicates
        self.message = f"Batch query rejected due to duplicate sample IDs: {duplicates}"


class BlackBoxOracle:
    """
    Restricted hard-label Oracle interface for the Random Forest.
    This provides a constrained supported API for attack implementations.
    It does not provide absolute cryptographic/sandbox security, but strictly 
    avoids retaining the full model object to prevent accidental or intentional 
    access to internals like `predict_proba` or `estimators_`.
    """
    def __init__(self, predict_func, max_queries_per_sample=None):
        self._predict_func = predict_func  # Only retain the hard-label callable
        self.max_queries_per_sample = max_queries_per_sample
        
        self.global_query_count = 0
        self.per_sample_query_count = {}
        self.per_sample_history = {}
        
    def get_query_count(self, sample_id):
        return self.per_sample_query_count.get(sample_id, 0)
        
    def predict(self, X, sample_ids, stage="unknown"):
        """
        Hard-label prediction.
        """
        if not isinstance(X, np.ndarray):
            X = np.array(X)
        if X.ndim == 1:
            X = X.reshape(1, -1)
            
        if X.shape[1] != 78:
            raise ValueError(f"Oracle expects 78 features, got {X.shape[1]}")
            
        if isinstance(sample_ids, str) or not isinstance(sample_ids, (list, tuple, np.ndarray)):
            raise TypeError("sample_ids must be a sequence matching the number of rows in X.")
            
        if len(sample_ids) != X.shape[0]:
            raise ValueError(f"Length of sample_ids ({len(sample_ids)}) must match rows in X ({X.shape[0]}).")
            
        # 1. Duplicate ID Check
        seen = set()
        duplicates = set()
        for sid in sample_ids:
            if sid in seen:
                duplicates.add(sid)
            seen.add(sid)
            
        if duplicates:
            return DuplicateSampleIDsFailure(list(duplicates))
            
        # 2. Atomic Preflight
        if self.max_queries_per_sample is not None:
            exhausted_ids = []
            for sid in sample_ids:
                current = self.per_sample_query_count.get(sid, 0)
                if current >= self.max_queries_per_sample:
                    exhausted_ids.append(sid)
            
            if exhausted_ids:
                if len(sample_ids) == 1:
                    return BudgetExhaustedFailure(sample_ids[0], stage, self.max_queries_per_sample)
                else:
                    return BatchBudgetFailure(exhausted_ids)
                    
        # 3. Charge budget
        for sid in sample_ids:
            self.per_sample_query_count[sid] = self.per_sample_query_count.get(sid, 0) + 1
            hist = self.per_sample_history.get(sid, [])
            hist.append(stage)
            self.per_sample_history[sid] = hist
            
        self.global_query_count += X.shape[0]
        
        # 4. Model evaluation
        return self._predict_func(X)
