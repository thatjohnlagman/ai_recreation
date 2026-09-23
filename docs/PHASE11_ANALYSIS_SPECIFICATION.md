# Phase 11 Analysis Specification

**Protocol Version:** 2.1.0 (Corrective Analysis Lockdown)
**Lock Timestamp:** 2026-09-23T16:21:00+08:00
**Lock Commit:** [To Be Computed]
**Source Map:** See `docs/PHASE11_STATISTICAL_METHOD_SOURCE_MAP.md` for explicit mapping to frozen pre-evaluation protocol files.

## Experimental Units and Nesting Structure
- **Execution Unit:** The unique stochastic environment representing an independent execution is `(seed, attack_scenario, defense_mechanism, controller_config)`.
- **Primary Pairing Unit:** The approved thesis protocol explicitly requires pairing at the **batch level**. Each Base batch is paired with the corresponding controller batch for the same `(seed, attack_scenario, defense_mechanism, batch_id)`.
- **Nesting Dependence Limitation:** The protocol explicitly mandates this batch-level evaluation. However, because the controller state at batch $t+1$ depends upon the outcomes of earlier batches, consecutive batches within a run are serially dependent. Treating them as independent in the primary analysis is a recognized methodological limitation affecting interpretation. Despite this, the batch-level test must remain the thesis-required primary inference.

## Inferential Statistics Structure (RQ3)
- **Primary Test:** Two-tailed paired t-test at the batch level.
- **Confirmatory Metrics:** Precision, Recall, and F1-score. (Accuracy and Balanced Accuracy may be reported supplementally, but the thesis specifically designates Precision, Recall, and F1-score).
- **Direction of Differences:** C1 minus Base (C1 - Base).
- **Significance Level (Alpha):** 0.05.

### Supplementary Diagnostics
The following tests are defined as supplementary robustness and sensitivity analyses and are not replacements for the thesis-required primary test:
- **Run-Level Paired Test:** A supplementary paired t-test using the execution (seed) as the independent unit, to evaluate sensitivity against the batch-level dependence limitation.
- **Effect Size:** Cohen's dz.
- **Confidence Intervals:** 95% CI on the paired difference mean.
- **Shapiro-Wilk Diagnostic & Wilcoxon Signed-Rank:** Supplementary checks for normality assumptions.
- **Holm Correction:** Supplementary multiplicity adjustment.

## Unresolved Statistical Ambiguity
**Ambiguity regarding Multiplicity Families and Pooling Structure:**
The approved thesis states: *"The main analysis is summarized by defense mechanism and configuration, with attack scenario retained as an evaluation condition."*
This creates an ambiguity regarding the exact pooling structure and multiple-comparison family:
- Is the test performed individually for each of the 9 *Attack × Defense* cells?
- Or is the test pooled across all attack scenarios for each defense mechanism?
- Furthermore, the exact Holm multiplicity family is not unambiguously established by the approved documents.

**STOP:** This specific ambiguity is documented here and must be explicitly resolved and approved prior to conducting inference in Phase 11B. We will not invent a family or pooling structure after observing the data.

## Missing Data and Alias Policy
- **Undefined Metrics:** When $TP+FP=0$ or $TP+FN=0$, Precision/Recall substitute to $0.0$.
- **Aliases:** C1 aliases (Sensitivity matrix) resolve directly to the execution metrics of the targeted primary run. They are isolated into the `sensitivity_*` tables and strictly prohibited from duplicating independent trials in the `primary_*` tables.
- **Missing Data:** Missing batches or corrupted executions trigger catastrophic failure. Unweighted batch averaging is mathematically prohibited; run metrics must be cleanly regenerated from confusion totals.
