# Phase 6: Evaluation Batches Report

## Role Splitting
The strict 90,000-record Evaluation partition (`X_eval`) was cleanly partitioned into two entirely disjoint structural roles utilizing Seed 42:
* **Crafting/Query Pool (20%):** 18,000 records reserved exclusively as surrogate queries and benign boundary references. Prohibited from measurement and evaluation metrics.
  * Benign: 14,936
  * Attack: 3,064
* **Measurement Pool (80%):** 72,000 records reserved exclusively for statistical evaluation.
  * Benign: 59,743
  * Attack: 12,257

## Batch Construction
The 72,000 Measurement rows were sequentially batched into exactly 144 distinct batches of 500 records. Because the `12,257` measurement attacks do not divide evenly by 144:
* **17 batches** contain exactly 86 attacks (and 414 benign).
* **127 batches** contain exactly 85 attacks (and 415 benign).

These batches are immutable. The assignment of `_source_file` and `_raw_row_idx` to `batch_id` and `within_batch_position` will never change across any perturbation run. Future code must load `X_eval.parquet` directly via `eval_position` to retrieve features without duplicating data.
