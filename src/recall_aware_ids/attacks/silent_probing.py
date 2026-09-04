import numpy as np
from recall_aware_ids.attacks.base import BaseAttack, AttackResult

class SilentProbingAttack(BaseAttack):
    def generate(self, X_orig, oracle, sample_id, true_label=None):
        """
        Silent Probing makes zero target queries.
        It does not manufacture eligibility or success booleans.
        """
        X_copy = np.array(X_orig, dtype=np.float32, copy=True)
        
        # 0 queries made
        magnitudes = self.calculate_magnitudes(X_copy, X_orig)
        
        return AttackResult(
            sample_id=sample_id,
            X_adv=X_copy,
            eligible=None,
            attempted=False,
            success=None,
            status_code="STUDY_DEFINED_SILENT_PROBING",
            message="Silent Probing makes no changes.",
            query_count=0,
            magnitudes=magnitudes
        )
