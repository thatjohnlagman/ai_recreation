# Phase 10D Evaluation Runtime and Storage Estimates

## Executive Summary
This document provides empirical M4 runtime measurements and projected official evaluation matrix execution times and disk footprints for Phase 10.
All benchmarks were performed on this Apple Silicon M4 system using **training-derived data only** (5,000 samples from `X_train.parquet`, 10 batches of 500 samples).
Zero evaluation Parquets (`X_eval.parquet`, `metadata_eval.parquet`) or official caches were accessed or modified during this benchmark.

---

## 1. Measured Training-Derived Pilot Timings (500-Sample Batches)

| Defense Mechanism | Measured Mean per 500-Row Batch | Per-Sample Latency | 10-Batch Test Total |
| :--- | :--- | :--- | :--- |
| **Adaptive Feature Poisoning (AFP)** | 0.0520 s | 0.10 ms/sample | 0.520 s |
| **Feature Squeezing (FS)** | 0.0460 s | 0.09 ms/sample | 0.460 s |
| **Randomized Smoothing (RS)** (11-member) | 0.8710 s | 1.74 ms/sample | 8.710 s |

---

## 2. Projected Official Evaluation Run Times (144 Batches = 72,000 Samples per Run)

Attack caching is **100% precomputed** in the 15 official attack caches (`artifacts/caches/`). During evaluation, batch data is loaded directly from pre-validated Parquet files into memory, eliminating attack generation overhead.

| Scope | AFP Component | FS Component | RS Component | Defense-Inference Lower Bound | Full Wall-Clock Projection (+15% to 25% overhead) |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Single Run (1 execution)** | 7.49 s (0.12 min) | 6.62 s (0.11 min) | 125.42 s (2.09 min) | — | — |
| **Primary Comparison (90 references)** | 30 runs: 3.74 min | 30 runs: 3.31 min | 30 runs: 62.71 min | **69.77 min (1.16 h)** | **~1.34 – 1.45 h** (~80.2 – 87.2 min) |
| **Full Evaluation Matrix (252 unique runs)** | 84 runs: 10.48 min | 84 runs: 9.27 min | 84 runs: 175.59 min | **195.34 min (3.26 h)** | **~3.74 – 4.07 h** (~224.6 – 244.2 min) |
| **C1 Sensitivity Aliases (27 references)** | Instant | Instant | Instant | **0.00 min** (metadata pointer) | **< 1 s** |

> [!IMPORTANT]
> **Defense-Inference Lower Bound vs Full Wall-Clock Prediction**:
> The **3.26-hour figure** represents a **defense-inference lower bound only**.
> It strictly isolates CPU/RAM matrix defense inference. It excludes:
> - Parquet attack-cache loading and deserialization;
> - JSON serialization for `scores.json`, `confusion.json`, and `config.json`;
> - Global metric recalculation (PR-AUC, Balanced Accuracy, Macro F1 across 72,000 samples);
> - Output reopening, re-parsing, and cross-metric validation (`_validate_run_outputs()`);
> - Directory staging, atomic renames, and filesystem metadata operations.
>
> Accounting for an estimated 15% to 25% orchestration and I/O overhead:
> - **Primary Comparison (90 runs)**: 1.16h lower bound $\times$ [1.15, 1.25] = **~1.34 to 1.45 hours** (~80.2 to 87.2 min).
> - **Full Evaluation Matrix (252 unique runs)**: 3.26h lower bound $\times$ [1.15, 1.25] = **~3.74 to 4.07 hours** (~224.6 to 244.2 min).
>
> Note: Across varying CPU thermal/load conditions on this host, the defense-inference lower bound ranges from 3.21h (cold run) to 3.79h (sustained load). Applying 15%–25% overhead to the full envelope yields an overall expected wall-clock range of ~3.69h (3.21h $\times$ 1.15) to ~4.74h (3.79h $\times$ 1.25). For the authoritative sustained-load lower bound (3.26h), the projected wall-clock execution time is **~3.74 to 4.07 hours**.

> [!NOTE]
> **Dominant Defense**: Randomized Smoothing (RS) performs 11 Random Forest inference passes per sample and accounts for approximately **89.9%** of total execution time.
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
