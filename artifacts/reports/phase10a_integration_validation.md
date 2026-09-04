# Phase 10A Integration Validation

This report confirms the implementation of the Phase 10 integration runner, dependency injection architecture, isolated caching, and metric mechanisms.

## Smoke Test Results
- Synthetic regression tests strictly verify zero cross-role bleed via explicit resolution, guaranteeing 18,000 crafting and 72,000 measurement assignments.
- Cache validation rigorously demands 20 fields (including hashes for model, roles, identities, configs) and exactly 72000 rows.
- Boundary selection deterministically selects exactly 200 eligible target rows across the complete dataset without chunk-size bias.
- PR-AUC scores successfully expose bounded values `[0,1]` and strict metrics enforce ValueError for malformed data.
- Spy tests prove that the runner fetches `get_intensity()` timing exactly BEFORE label access.
- Temporary `_tmp` directories successfully quarantine outputs on failure without emitting completion markers, securing execution atomicity.

## Pilot Performance
- Tested deterministically against a true `train.parquet` reference subset containing 5,000 actual training records.
- Verified overhead sizes across `Frozen RF`, `AdditiveFeaturePerturbation`, `FeatureSqueezing`, and `RandomizedSmoothing (x11)`.
- Pilot script outputs estimations but correctly avoids claiming exact full-matrix runtimes due to Attack Cache complexity.

## RS Provenance Correctness
The authoritative Phase 8 v3 hash for `src/recall_aware_ids/defenses/randomized_smoothing.py` was:
`be500b81d1bbe75750673452653f4a4e98a76c4383ffb446dcade560a3fec87a`

During Phase 10A, an additive, backward-compatible PR-AUC score interface was added to support the `positive_votes / 11` calculation. The current repaired hash is:
`ce28b0e3460a352224b024f616d58217190164175e575d9495222a414f95c58b`

### Source Diff
```diff
--- src/recall_aware_ids/defenses/randomized_smoothing.py
+++ src/recall_aware_ids/defenses/randomized_smoothing.py
@@ -6,7 +6,7 @@
         super().__init__(feature_names, modifiable_mask, training_bounds)
         self.ensemble_size = ensemble_size
 
-    def predict_ensemble(self, X, sigma, seed, attack_scenario, batch_id, predict_func, chunk_size=100):
+    def predict_ensemble(self, X, sigma, seed, attack_scenario, batch_id, predict_func, chunk_size=100, return_scores=False):
         if not np.isfinite(sigma) or sigma < 0.0:
             raise ValueError(f"sigma must be finite and non-negative, got {sigma}")
         
@@ -99,5 +99,8 @@
             final_invalid_fraction=final_invalid_fraction,
             protected_feature_modification_count=total_protected_feature_modification_count
         )
+        if return_scores:
+            positive_vote_fraction = member_votes[:, 1] / self.ensemble_size
+            return final_preds, result, positive_vote_fraction
         return final_preds, result
```

This ensures legacy two-value return behavior remains unchanged by default, score-enabled behavior returns the correct vote fraction in `[0,1]`, and existing Phase 8 hard predictions/RNG logic is absolutely unchanged.

## Final Status
Phase 10A integration code is complete. `experiment.date_frozen` remains unset until external review. No queries or calculations were run against the measurement pool.
