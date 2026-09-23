# Phase 11 Analysis Specification

**Protocol Version:** 1.0.0
**Date Frozen:** 2026-09-17T21:23:51+08:00
**Primary Aim:** Evaluate the performance of Recall-Aware Control for Perturbation Defenses in Intrusion Detection Systems Against Black-Box Probing Attacks.

## Experimental Units
- **Population:** Black-box probing attacks against binary Random Forest IDS.
- **Experimental Unit:** 500-record measurement batches (from a disjoint 72,000-record measurement pool).
- **Execution Level:** The independent execution is defined by the unique combination of `(seed, attack_scenario, defense_mechanism, controller_config)`.
- **Aliases:** C1 sensitivity aliases map directly to the primary C1 run and must NEVER be treated as independent duplicates.

## Research Questions & Metrics
- **RQ1:** Base results descriptive performance.
- **RQ2:** C1 Recall-Aware descriptive performance.
- **RQ3:** Comparative effectiveness (C1 minus Base).
- **RQ4:** Sensitivity analysis (C1–C7) descriptive performance.

- **Metrics Used:** Accuracy, Precision, Recall, F1-Score, Balanced Accuracy.
- **Handling of Undefined Metrics:**
  - If TP+FP=0: Precision = 0.0
  - If TP+FN=0: Recall = 0.0
  - If Precision+Recall=0: F1 = 0.0

## Inferential Statistics Specification
- **Statistical Test:** Two-tailed paired t-test (`scipy.stats.ttest_rel`).
- **Pairing Key:** `(seed, attack_scenario, defense_mechanism, batch_id)`.
- **Direction of Differences:** C1 minus Base (C1 - Base).
- **Effect Size:** Cohen's dz (mean difference divided by standard deviation of differences).
- **Confidence Interval:** 95% CI on the paired mean difference.
- **Assumption Checks:** Shapiro-Wilk test on paired differences to check normality.
- **Fallback Test:** Wilcoxon signed-rank test (computed as a supplementary check if normality is questionable, but primary results remain the paired t-test).
- **Significance Level (Alpha):** 0.05.
- **Multiple-Comparison Correction:** Holm-adjusted p-values across tests.
- **Interpretation Rule:** A "significant difference" is only considered "effective" if the sign of the mean difference is positive for performance metrics (Recall, F1). Mixed results are reported transparently.

## Execution Handling
- **Missing or Invalid Data:** Any execution missing expected batches (144) or containing malformed schema is rejected. No unweighted batch averaging is permitted.
- **Quarantine Exclusion:** Quarantined or active staging directories are explicitly excluded from analysis.
- **Parquet Exclusions:** All evaluation and inference parquet outputs are prohibited from being opened during the analysis execution phase; only JSON serialization artifacts are used.
