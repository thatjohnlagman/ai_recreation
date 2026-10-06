# Phase 4 Preprocessing Report
    
**Input File:** working_sample.parquet (SHA-256: 1a612e440ea68d1526524d8788463b90bfe0a2e287c11e336136c236f12c73b9)
**Input Rows:** 300,000
**NaN/Inf Drops:** 0

## Split Strategy
* **Train Fraction:** 0.7 (210,000 total train/cal)
* **Eval Fraction:** 0.30000000000000004 (90,000)
* **Calibration:** 0.15 of Train
* **Split Seed:** 42
* **Disjointness:** Confirmed (Train, Eval, and Calibration share no indices)
* **Transformations:** `StandardScaler` fitted on Train only, applied to all.

## Partition Counts
* **Train:** 178,500
  * Classes: {0: 148114, 1: 30386}
* **Evaluation:** 90,000
  * Classes: {0: 74679, 1: 15321}
* **Calibration:** 31,500
  * Classes: {0: 26138, 1: 5362}

## Features
* **Total Included Features:** 78
* **Modifiable Features:** 63
* **Protected Features:** 15
* **Excluded Cols:** {'y_binary', 'timestamp', 'Flow ID', '_source_file', 'Label', 'Dst IP', '_orig_row_idx', 'label', 'attack_family', 'Timestamp', 'Src Port', '_raw_row_idx', 'Src IP'}
* **Metadata Saved:** ['_source_file', '_raw_row_idx', 'y_binary', 'attack_family']

## Performance
* **Runtime:** 1.8s
* **Peak RAM:** 1253.5 MB

_Generated automatically by scripts/preprocess_dataset.py_
