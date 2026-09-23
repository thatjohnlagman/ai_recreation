# Phase 11B Pooling and Multiplicity Decision Memo (Blinded)

**Status:** Phase 11B execution is blocked pending explicit approval of the statistical pooling structure and multiple-comparison family. No metric values have been inspected. This memo derives solely from matrix structure and the approved thesis text.

## 1. The Ambiguity in the Approved Thesis

The approved thesis states (p. 70):
> *"The main analysis is summarized by defense mechanism and configuration, with attack scenario retained as an evaluation condition."*

This creates structural ambiguity for Phase 11B Hypothesis Testing. How should the 144 batch pairs for a single configuration (Base vs. C1) be grouped for inferential tests?

## 2. Structural Options (Derived from Matrix Only)

There are 3 Defense Mechanisms × 3 Attack Scenarios = 9 Experimental Cells. Each cell contains 5 independent executions × 144 batches = 720 batch-level repeated measures.

### Option A: Cell-Wise Testing (No Pooling Across Attacks)
- **Grouping:** Test each of the 9 Attack × Defense cells independently.
- **Pairs per Test:** $N = 720$ batch pairs.
- **Total Primary Tests (per Metric):** 9 tests.
- **Thesis Support:** Retains attack scenario strictly as an isolated condition, but fails to summarize *across* the defense mechanism globally.

### Option B: Pooled Testing (Summarized by Defense)
- **Grouping:** Pool the 3 Attack Scenarios for a given Defense Mechanism.
- **Pairs per Test:** $N = 720 \times 3 = 2,160$ batch pairs.
- **Total Primary Tests (per Metric):** 3 tests.
- **Thesis Support:** Directly addresses "summarized by defense mechanism", while attack scenario remains a nested evaluation condition within the pool.

## 3. Recommended Thesis Interpretation

Based solely on the approved thesis instruction to summarize by defense mechanism *with attack scenario retained as an evaluation condition*, **Option B (Pooled Testing)** most closely aligns with the literal text.

## 4. Supplementary Holm Multiplicity Family

The thesis does not explicitly define a multiplicity correction family. The previously implemented Holm correction is supplementary. 
If Option B is selected, the supplementary Holm family should consist of the 3 tests within a single metric. If Option A is selected, the family should consist of the 9 tests within a single metric.

## Request for Explicit Approval
Please explicitly instruct whether to use **Option A** or **Option B** for the primary batch-level test grouping before Phase 11B can proceed. Do not conduct inference until this is resolved.
