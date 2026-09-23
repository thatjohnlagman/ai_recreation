# Phase 11 Statistical Method Source Map

This document traces every statistical choice to its authoritative frozen source in the experimental protocol prior to Phase 10 evaluation.

| Method Component | Authoritative Source | Location | Frozen Before Eval? | Note |
|---|---|---|---|---|
| **Two-tailed paired t-test** | `docs/EXPERIMENT_PROTOCOL.md` | Section "Frozen Parameters" (Line 45) | Yes | Specified as the primary test. |
| **Alpha level (0.05)** | `docs/EXPERIMENT_PROTOCOL.md` | Section "Frozen Parameters" (Line 46) | Yes | |
| **Batch-Level Pairing Key** | `docs/IMPLEMENTATION_DECISIONS.md` | Section 16 "Statistical Diagnostics" (Line 123) | Yes | `(seed, attack_scenario, defense_mechanism, batch_id)` explicitly mandated. |
| **Cohen's dz Effect Size** | `docs/IMPLEMENTATION_DECISIONS.md` | Section 16 "Statistical Diagnostics" (Line 124) | Yes | |
| **95% Confidence Interval** | `docs/IMPLEMENTATION_DECISIONS.md` | Section 16 "Statistical Diagnostics" (Line 125) | Yes | Paired difference CI. |
| **Shapiro-Wilk Diagnostic** | `docs/IMPLEMENTATION_DECISIONS.md` | Section 16 "Statistical Diagnostics" (Line 127) | Yes | Normality diagnostic on paired differences. |
| **Wilcoxon Signed-Rank** | `docs/IMPLEMENTATION_DECISIONS.md` | Section 16 "Statistical Diagnostics" (Line 126) | Yes | Supplementary test if normality is questionable. |
| **Holm Correction** | `docs/IMPLEMENTATION_DECISIONS.md` | Section 16 "Statistical Diagnostics" (Line 126) | Yes | Used for multiplicity adjustment. |
| **Undefined Metrics** | `docs/IMPLEMENTATION_DECISIONS.md` | Section 15 "Undefined Metric Handling" (Lines 116-118) | Yes | 0.0 value substitution (not NaN). |

## Structural Amendment: The Statistical Unit Problem
**Mandate Limitation:** `IMPLEMENTATION_DECISIONS.md` Section 16 mandates the batch-level pairing key `(seed, attack, defense, batch_id)`. However, batches are serially dependent repeated measures nested within execution pairs (the independent experimental unit). Treating all 6,480 batch pairs as independent is statistically invalid (pseudoreplication).

**Resolution:** 
1. The batch-level paired t-test (N=6480 per metric) will be documented and reported descriptively to satisfy the explicit thesis mandate.
2. A **run-level paired t-test** (N=45 pairs) will serve as the primary structurally valid inferential test. The independent replication occurs at the execution (seed) level, aggregating the nested batches safely.

## Hypothesis Formulation & Holm Families
- **Hypothesis Family:** The test is performed separately for each Attack × Defense cell (3 attacks × 3 defenses = 9 cells).
- **Null Hypothesis (H0):** The true mean paired difference (C1 - Base) in the metric is zero.
- **Alternative Hypothesis (H1):** The true mean paired difference (C1 - Base) in the metric is non-zero.
- **Holm Multiplicity Family:** Adjustments are applied across the 9 cells for a single metric. Tests for different metrics (e.g. Precision vs Recall) belong to separate families.
- **Mixed Directions:** A difference is only considered "effective" if it is statistically significant *and* the sign of the mean difference reflects improvement (per `EXPERIMENT_PROTOCOL.md` Scientific Integrity Commitments, Line 115).
- **Zero-variance differences:** If all paired differences are exactly zero (variance = 0), the p-value is 1.0.
