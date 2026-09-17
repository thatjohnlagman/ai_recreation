# Phase 10D Evaluation Runtime and Storage Estimates

## Executive Summary
This document provides empirical M4 runtime measurements and projected official evaluation matrix execution times and disk footprints for Phase 10.
All benchmarks were performed on this Apple Silicon M4 system using **training-derived data only** (5,000 samples from `X_train.parquet`, 10 batches of 500 samples).
Zero evaluation Parquets or official caches were accessed or modified during this benchmark.

---

## 1. Measured Training-Derived Pilot Timings (500-Sample Batches)

| Defense Mechanism | Measured Mean per 500-Row Batch | Per-Sample Latency | 10-Batch Test Total |
| :--- | :--- | :--- | :--- |
| **Adaptive Feature Poisoning (AFP)** | 0.0520 s | 0.10 ms/sample | 0.520 s |
| **Feature Squeezing (FS)** | 0.0454 s | 0.09 ms/sample | 0.454 s |
| **Randomized Smoothing (RS)** (11-member) | 0.8586 s | 1.72 ms/sample | 8.586 s |

---

## 2. Projected Official Evaluation Run Times (144 Batches = 72,000 Samples per Run)

Attack caching is **100% precomputed** in the 15 official attack caches (`artifacts/caches/`). During evaluation, batch data is loaded directly from pre-validated Parquet files into memory, eliminating attack generation overhead.

| Scope | AFP Component | FS Component | RS Component | Total Projected Time |
| :--- | :--- | :--- | :--- | :--- |
| **Single Run (1 execution)** | 7.49 s (0.12 min) | 6.54 s (0.11 min) | 123.64 s (2.06 min) | — |
| **Primary Comparison (90 references)** | 30 runs: 3.74 min | 30 runs: 3.27 min | 30 runs: 61.82 min | **68.83 min (1.15 h)** |
| **Full Evaluation Matrix (252 unique runs)** | 84 runs: 10.48 min | 84 runs: 9.16 min | 84 runs: 173.10 min | **192.74 min (3.21 h)** |
| **C1 Sensitivity Aliases (27 references)** | Instant | Instant | Instant | **0.00 min** (instant metadata link) |

> [!NOTE]
> **Dominant Defense**: Randomized Smoothing (RS) performs 11 Random Forest inference passes per sample and accounts for approximately **89.8%** of total execution time.
> **Parallelization Potential**: Runs across different seeds and defenses are embarrassingly parallel.

---

## 3. Storage Projections

For each completed run, 5 output files are published:
1. `scores.json`: 144 batches × 500 aligned rows (true labels, predictions, positive scores) ~ **0.83 MB**
2. `confusion.json`: 144 batch confusion and derivable metric records ~ **10.72 KB**
3. `config.json`: 144 batch controller state and intensity records ~ **8.05 KB**
4. `run_summary.json`: global recomputed metrics, ASR, and provenance ~ **2.0 KB**
5. `completion.json`: atomic completion marker ~ **1.0 KB**

| Target Scope | Total Runs | Per-Run Footprint | Aggregate Storage Footprint |
| :--- | :--- | :--- | :--- |
| **Primary Comparison** | 90 runs | 0.85 MB | **0.07 GB** |
| **Full Unique Executions** | 252 unique runs | 0.85 MB | **0.21 GB** |
| **Available Disk Capacity** | — | — | **1048575 / > 60 GB free** |

Storage requirement of ~**0.21 GB** is well within the 68+ GB available on this host.
