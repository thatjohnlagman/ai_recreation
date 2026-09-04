# Phase 7: Attack Semantics and Definitions

## Attack Success & Eligibility Definitions
- **Eligible Population:** Samples whose true label is Attack AND the Target model initially predicts as Attack. (Original false negatives are ineligible).
- **Attempted Attacks:** Eligible samples submitted to the attack generator.
- **Successful Evasion:** A modified sample that is predicted as Benign by the target model, generated within the query budget, and strictly adhering to all bounds/masks.
- **Evasion Success Rate:** `(Successful Evasions) / (Eligible Attempted Samples)`
- **Post-Attack Target Recall:** Re-evaluated Recall score on the applicable evaluation population using the finalized adversarial/clean samples.
- **Silent Probing:** Does not modify samples and is assigned a success rate of `not_applicable`. Naturally misclassified unchanged records are model false negatives, not successful evasions.

## Black-Box Restrictions and Perturbations
- Target Random Forest is accessed *only* through a hard-label `predict()` Oracle callable.
- The Oracle interface acts as a restricted supported API for attack implementations. While it does not provide an absolute cryptographic security sandbox, it strictly avoids retaining the full model object to prevent accidental or intentional access to internals like `predict_proba` or `estimators_`.
- The Oracle strictly tracks per-sample query consumption, enforces a hard budget (e.g., 50 for Boundary Attack), handles duplicate IDs atomically, and blocks batch processing if any ID lacks budget.
- 15 protected features are completely immutable. The remaining 63 are clipped exactly to their representation inside the `training_bounds.parquet`.
- Magnitudes (L0, L1, L2, L∞) are computed after all bounds projection is applied. L0 defines exactly modified `float32` features; others use `float64` subtraction to guarantee arithmetic accuracy before measuring absolute distance.

## Seed and Cache Semantics
- **Effective Attack Seed:** `effective_attack_seed = run_seed`. (Defaults to 42 for smoke testing).
- **Seed-Invariant:** `silent_probing` uses no stochastic elements.
- **Surrogate Decision Tree:** Randomness during DT fitting and lexicographic deterministic candidate sorting (L0, L2, L∞, leaf ID) relies on `effective_attack_seed`.
- **Boundary Attack:** Random selection of benign endpoints and subsequent reference ordering relies on `effective_attack_seed`.
- **Cache Identity:** To guarantee exactly paired experimentation, the attack cache identity encapsulates:
  - Scenario Name
  - Effective Attack Seed
  - Attack Configuration Hash
  - Evaluation-Role Manifest Hash
  - Evaluation-Batch Manifest Hash
  - Target Model Hash
  - Scaler, Feature-Name, Feature-Mask, and Training-Bound hashes
- *Cache Reuse:* A single cache realization for a `(Scenario, Seed)` tuple will be deterministically applied across all three defenses and both Base and RA configurations during Phase 10. No caches are written to disk during Phase 7 smoke testing.

## Historical Provenance (attacks.yaml)
- **Phase 6 Creation Hash:** `0db73105875613c95a4e8a1bcb60aaf100d13c4f633791ebe1325de0ac9b2c16` (Prior to terminology fix)
- **Phase 7 Repaired Hash:** `fbd596219990125d8a75da0b1d0e35bdd82c1429052407115ab2ab4f4f91cd2b` (Current, with `preserve_protected_features`)
