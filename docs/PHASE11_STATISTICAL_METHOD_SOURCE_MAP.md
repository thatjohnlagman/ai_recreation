# Phase 11 Statistical Method Source Map

This document traces every statistical choice to its authoritative frozen source to distinguish between thesis-mandated methods, frozen implementation operationalizations, and supplementary diagnostics.

## 1. Methods Explicitly Required by the Approved Thesis
The following methods are structurally mandated by the approved thesis and constitute the authoritative primary inference procedure.

| Method Component | Authoritative Source | Note |
|---|---|---|
| **Experimental/Pairing Unit** | Approved Thesis (p. 69) | Batch-level Base/C1 pairing. |
| **Analysis Grouping** | Approved Thesis (p. 70) | Analysis grouping and RQ method summary. |
| **Primary Inferential Test** | Approved Thesis (pp. 70–73) | Two-tailed paired t-test. |
| **Confirmatory Metrics** | Approved Thesis (pp. 70–73) | Precision, Recall, and F1-score. |
| **Significance Level (Alpha)** | Approved Thesis (pp. 70–73) | α = 0.05 |
| **Descriptive Treatment (RQ1/2/4)**| Approved Thesis (pp. 70–73) | Descriptive handling of RQ1/RQ2/RQ4, as applicable. |

**Methodological Limitation Disclosure:** The thesis requires a batch-level paired test. However, consecutive batches within an execution may be serially dependent because the Recall-Aware controller carries state forward. Treating all 6,480 batch pairs as independent is a methodological limitation affecting interpretation. Despite this limitation, the batch-level test remains the unalterable, confirmatory primary test.

## 2. Frozen Implementation Details
These operationalize the approved thesis and were frozen before evaluation.

| Method Component | Authoritative Source | Note |
|---|---|---|
| **Exact Pairing Key** | `IMPLEMENTATION_DECISIONS.md` (Line 123) | `(seed, attack_scenario, defense_mechanism, batch_id)` |

## 3. Supplementary Diagnostics and Robustness Analyses
The following methods were added during implementation to provide additional rigor, but they **are not replacements for the thesis-required primary test** and do not alter the thesis hypothesis decision rule.

| Method Component | Source | Purpose / Note |
|---|---|---|
| **Run-Level Paired Test** | Implementation | Supplementary robustness analysis treating the execution/seed as the independent unit to evaluate sensitivity to the batch-level serial dependence. |
| **Cohen's dz** | `IMPLEMENTATION_DECISIONS.md` | Supplementary effect size calculation. |
| **95% Confidence Interval** | `IMPLEMENTATION_DECISIONS.md` | Supplementary diagnostic for the paired difference mean. |
| **Shapiro-Wilk Diagnostic** | `IMPLEMENTATION_DECISIONS.md` | Supplementary check for normality of paired differences. |
| **Wilcoxon Signed-Rank** | `IMPLEMENTATION_DECISIONS.md` | Supplementary non-parametric fallback if normality is questionable. |
| **Holm Correction** | `IMPLEMENTATION_DECISIONS.md` | Supplementary multiplicity adjustment. |
