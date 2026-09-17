# Phase 10D Evaluation Runtime and Storage Estimates

## Executive Summary
This document provides empirical M4 runtime measurements and projected official evaluation matrix execution times and disk footprints for Phase 10.
All benchmarks were performed on this Apple Silicon M4 system using **training-derived data only** (5,000 samples from `X_train.parquet`, 10 batches of 500 samples).
Zero evaluation Parquets (`X_eval.parquet`, `metadata_eval.parquet`) or official caches were accessed or modified during this benchmark.

---

## 1. Measured Training-Derived Pilot Timings (500-Sample Batches)

| Defense Mechanism | Measured Mean per 500-Row Batch | Per-Sample Latency | 10-Batch Test Total |
| :--- | :--- | :--- | :--- |
| **Adaptive Feature Poisoning (AFP)** | 0.0517 s | 0.10 ms/sample | 0.517 s |
| **Feature Squeezing (FS)** | 0.0462 s | 0.09 ms/sample | 0.462 s |
| **Randomized Smoothing (RS)** (11-member) | 1.0297 s | 2.06 ms/sample | 10.297 s |

---

## 2. Projected Official Evaluation Run Times (144 Batches = 72,000 Samples per Run)

Attack caching is **100% precomputed** in the 15 official attack caches (`artifacts/caches/`). During evaluation, batch data is loaded directly from pre-validated Parquet files into memory, eliminating attack generation overhead.

| Scope | AFP Component | FS Component | RS Component | Defense-Inference Lower Bound | Full Wall-Clock Projection |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Single Run (1 execution)** | 7.45 s (0.12 min) | 6.65 s (0.11 min) | 148.28 s (2.47 min) | — | — |
| **Primary Comparison (90 references)** | 30 runs: 3.72 min | 30 runs: 3.33 min | 30 runs: 74.14 min | **81.19 min (1.35 h)** | **~1.3 – 1.4 h** |
| **Full Evaluation Matrix (252 unique runs)** | 84 runs: 10.43 min | 84 runs: 9.31 min | 84 runs: 207.59 min | **227.33 min (3.79 h)** | **~3.7 – 4.0 h** |
| **C1 Sensitivity Aliases (27 references)** | Instant | Instant | Instant | **0.00 min** (metadata pointer) | **< 1 s** |

> [!IMPORTANT]
> **Defense-Inference Lower Bound vs Full Wall-Clock Prediction**:
> The **3.79-hour figure** (reconciling the 3.21h–3.30h range observed under varying CPU thermal/load conditions) represents a **defense-inference lower bound only**.
> It strictly isolates CPU/RAM matrix defense inference. It excludes:
> - Parquet attack-cache loading and deserialization;
> - JSON serialization for `scores.json`, `confusion.json`, and `config.json`;
> - Global metric recalculation (PR-AUC, Balanced Accuracy, Macro F1 across 72,000 samples);
> - Output reopening, re-parsing, and cross-metric validation (`_validate_run_outputs()`);
> - Directory staging, atomic renames, and filesystem metadata operations.
> Accounting for an estimated 15% to 25% orchestration and I/O overhead, the true end-to-end wall-clock execution time is realistically estimated at **~3.7 to 4.0 hours**.

> [!NOTE]
> **Dominant Defense**: Randomized Smoothing (RS) performs 11 Random Forest inference passes per sample and accounts for approximately **91.3%** of total execution time.
> **Parallelization Potential**: Runs across different seeds and defenses are embarrassingly parallel.

---

## 3. Storage Projections (Production Schema Aligned)

For each completed run, 5 output files are published:
1. `scores.json`: 144 batches × 500 aligned rows containing dual fields (`y_true` + `labels`, `preds` + `predictions`, `scores`) and defense diagnostic counts ~ **2.23 MB**
2. `confusion.json`: 144 batch confusion and derivable metric records ~ **56.85 KB**
3. `config.json`: 144 batch controller state, intensity, and feedback records ~ **43.35 KB**
4. `run_summary.json`: global recomputed metrics, ASR, and provenance ~ **4.0 KB**
5. `completion.json`: atomic completion marker ~ **1.0 KB**

| Target Scope | Total Runs | Per-Run Footprint | Aggregate Storage Footprint |
| :--- | :--- | :--- | :--- |
| **Primary Comparison** | 90 runs | 2.33 MB | **0.20 GB** |
| **Full Unique Executions** | 252 unique runs | 2.33 MB | **0.57 GB** |
| **Available Disk Capacity** | — | — | **> 68 GB free** |

Storage requirement of ~**0.57 GB** is well within the 68+ GB available on this host.
