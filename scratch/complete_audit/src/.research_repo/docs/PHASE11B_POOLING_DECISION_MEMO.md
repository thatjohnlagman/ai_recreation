# Phase 11B Pooling Decision Memo

**Status:** Option B is prospectively approved and locked.
**Lock Timestamp:** 2026-09-23T09:44:33.571916+00:00

## Decision
- Option B is prospectively approved and locked.
- Primary observation: paired batch difference, C1 minus Base.
- Pool the three attack scenarios within each defense.
- 2,160 paired batches per defense/metric.
- Nine primary tests: 3 defenses × Precision, Recall, and F1.
- The thesis decision rule uses each raw two-tailed paired-t p-value at alpha 0.05.
- Holm adjustment across one family containing all nine primary tests is supplementary and must not replace the raw-p thesis decision.
- Supplementary run-level analysis uses 15 paired executions per defense/metric and a separate nine-test Holm family.
- Batch serial dependence is disclosed as a limitation.
- RQ4 remains descriptive only.

## Degenerate Result Contract
- all-zero differences: t=0, p=1, CI=[0,0], dz=0;
- constant nonzero differences: signed infinity t statistic, raw p=0, degenerate CI=[mean,mean], dz=null/NA, and an explicit warning.
