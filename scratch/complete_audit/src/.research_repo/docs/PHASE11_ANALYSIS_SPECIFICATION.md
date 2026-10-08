# Phase 11 Analysis Specification

**Status:** Option B is prospectively approved and locked.
**Lock Timestamp:** 2026-09-23T09:44:33.571916+00:00

## Structure
- **Primary observation:** paired batch difference, C1 minus Base.
- **Pooling:** Pool the three attack scenarios within each defense.
- **Pairs:** 2,160 paired batches per defense/metric.
- **Tests:** Nine primary tests: 3 defenses × Precision, Recall, and F1.

## Decision Rule
- The thesis decision rule uses each raw two-tailed paired-t p-value at alpha 0.05.
- Holm adjustment across one family containing all nine primary tests is supplementary and must not replace the raw-p thesis decision.

## Supplementary Analysis
- Supplementary run-level analysis uses 15 paired executions per defense/metric and a separate nine-test Holm family.
- RQ4 remains descriptive only.

## Limitations
- Batch serial dependence is disclosed as a limitation.

## Degenerate Result Contract
- **all-zero differences:** t=0, p=1, CI=[0,0], dz=0
- **constant nonzero differences:** signed infinity t statistic, raw p=0, degenerate CI=[mean,mean], dz=null/NA, and an explicit warning.
