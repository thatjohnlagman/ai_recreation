# MERGED STATE HANDOFF

## Overview
This repository merges the standalone Recall-Aware IDS (server.py, attacker_sim.py, frontend dashboard) with the research dataset (.research_repo). 
The operator UI now supports toggling between **Live Computed Benchmarks** (derived directly from real-time attacker_sim.py streaming batches) and **Static Thesis Results** (pre-scraped directly from .research_repo/artifacts/evaluation_runs).

## Pipeline Flow
- `attacker_sim.py` -> Streams simulated batched adversarial/benign flows to `server.py` `/stream_batch`.
- `server.py` -> Processes flows through Base and Recall-Aware arms. Computes `rolling_recall` organically over batches.
- `controller` -> Adjusts intensity bounds based on feedback.
- `frontend/results.js` -> Receives updates. When "Live Benchmark" is ON, displays dynamic state and matrices. When OFF, falls back to `STATIC_C1_C7_MATRIX` and `STATIC_BENCHMARK_SUMMARY` (which were populated via `scratch/scrape_research_repo.py` averaging the true thesis runs).

## Configurations
- Demo uses `server.py` hardcoded defaults for AFP limits and C1-C7 thresholds, which might slightly differ from `.research_repo` YAML configs. 
- The research results displayed when the toggle is OFF accurately represent the fully completed thesis simulations.
