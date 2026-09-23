# Phase 11 Analysis Specification

**Protocol Version:** 2.0.0 (Corrective Analysis Lockdown)
**Lock Timestamp:** 2026-09-23T16:05:00+08:00
**Lock Commit:** c247a70eae2e9d866591b1e02ae6422154faced4
**Source Map:** See `docs/PHASE11_STATISTICAL_METHOD_SOURCE_MAP.md` for explicit mapping to frozen pre-evaluation protocol files.

## Experimental Units and Nesting Structure
- **Independent Experimental Unit:** The execution/seed level `(seed, attack_scenario, defense_mechanism, controller_config)`. The unique stochastic environment across the 5 primary seeds represents the independent trials.
- **Repeated-Measure Unit:** The 144 chronological 500-record measurement batches. These are nested within the independent executions.
- **Nesting Dependence:** Because the controller state at batch $t+1$ depends upon the outcomes of earlier batches, the batches within a given run are serially dependent and do not represent independent observations.

## Inferential Statistics Structure (The "Statistical Unit" Amendment)
The formal `IMPLEMENTATION_DECISIONS.md` protocol explicitly mandated the use of a batch-level pairing key `(seed, attack_scenario, defense_mechanism, batch_id)`. We document that interpreting the batch count as the independent sample size (N=6480 per cell) is pseudoreplication.

To resolve this limitation without replacing the mandated test:
1. **Primary Structural Inference (Amendment):** A **run-level paired t-test** ($N=5$ pairs per Attack×Defense cell, aggregating all 144 batches) will serve as the structurally sound inferential test that satisfies independence assumptions.
2. **Mandated Fallback:** The **batch-level paired t-test** ($N=720$ pairs per Attack×Defense cell) is performed and reported purely to satisfy the pre-registered protocol mandate, but its p-values must be explicitly flagged with the dependence limitation.

## Hypothesis Families and Adjustments
- **Hypothesis Formulation:** Tests are evaluated separately for each of the 9 *Attack × Defense* experimental cells.
- **Null Hypothesis ($H_0$):** The true mean paired difference ($C_1 - Base$) in the selected metric is exactly zero.
- **Alternative Hypothesis ($H_1$):** The true mean paired difference ($C_1 - Base$) in the selected metric is non-zero.
- **Multiplicity Families:** The Holm adjustment (step-down procedure) is applied strictly within a single metric across the 9 cells. Different metrics (e.g., F1 vs. Precision) form separate families. 

## Statistical Methodology
- **Test:** Two-tailed paired t-test (`scipy.stats.ttest_rel`).
- **Effect Size:** Cohen's $d_z$ formula: $d_z = \frac{\bar{X}_{diff}}{s_{diff}}$
- **Confidence Interval (CI):** 95% CI on the paired difference mean using the t-distribution.
- **Normality Check:** Shapiro-Wilk test on the paired differences. 
- **Wilcoxon Signed-Rank:** Computed identically on the paired differences as a deterministic sensitivity fallback if Shapiro-Wilk detects significant non-normality, but the primary results remain the t-test.
- **Alpha:** 0.05.
- **Zero-Variance Differences:** If all paired differences are identically zero (variance = 0), the resulting t-statistic is 0.0 and the p-value is 1.0.
- **Mixed Directions Rule:** A statistically significant difference is only interpreted as "effective" if the sign of the mean difference reflects a genuine improvement (e.g. positive for Recall and F1).

## Missing Data and Alias Policy
- **Undefined Metrics:** When $TP+FP=0$ or $TP+FN=0$, Precision/Recall substitute to $0.0$.
- **Aliases:** C1 aliases (Sensitivity matrix) resolve directly to the execution metrics of the targeted primary run. They are isolated into the `sensitivity_*` tables and strictly prohibited from duplicating independent trials in the `primary_*` tables.
- **Missing Data:** Missing batches or corrupted executions trigger catastrophic failure. Unweighted batch averaging is mathematically prohibited; run metrics must be cleanly regenerated from confusion totals.
