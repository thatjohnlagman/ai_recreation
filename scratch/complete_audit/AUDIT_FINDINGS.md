# AUDIT FINDINGS

1. **server.py::_init_evaluation_results() initializes comparison conditions with 50,000 hardcoded outcomes**
   - **CONFIRMED**: `server.py`, `_init_evaluation_results()`, lines 193-202.
   - **Excerpt**: `self.evaluation_matrix[(d, m)] = { "defense": d.upper(), "tp": 20000, "fn": 5000, "fp": 0, "tn": 25000, "evaluated_flows": 50000 ... }`
   - **Impact**: The initial demo state mathematically guarantees a specific precision/recall before any traffic is processed.

2. **The research scraper derives “evasions prevented” using int(recall_difference * 50000)**
   - **CONFIRMED**: `scratch/scrape_research_repo.py`, line 78.
   - **Excerpt**: `evasions = int(diff_r * 50000)`
   - **Impact**: Evasions are presented as an exact paired measurement but are actually a synthetic scalar multiplication applied to average recall differences.

3. **results.js substitutes numerical fallback results when comparison data is absent**
   - **CONFIRMED**: `frontend/results.js`, line 849.
   - **Excerpt**: `if (!useLiveBenchmark) { matrix = STATIC_C1_C7_MATRIX; }`
   - **Impact**: The UI seamlessly swaps to statically bundled thesis statistics without a visually distinct "historical fallback" label.

4. **get_evaluation_results() contains predetermined “best defense”**
   - **CONFIRMED**: `server.py`, `get_evaluation_results()`, lines 255-256.
   - **Excerpt**: `best_def = "FS"
best_recall = 0.9565`
   - **Impact**: The server's evaluation payload defaults to a predetermined winner if the runtime controller hasn't exceeded the threshold.

5. **Historical research results vs current live measurements mixed**
   - **CONFIRMED**: The "Compute Live Benchmark" button fully replaces the entire static `STATIC_C1_C7_MATRIX` payload with live simulated memory dicts.
   
6. **Base and Recall-Aware comparisons may not use genuinely separate inference paths**
   - **CONFIRMED**: `server.py` evaluates the `DefenseArm` sequentially and applies a probability override based on the active controller's multiplier. They share the same underlying pipeline logic.
   
7. **C1–C7 labels and actual runtime configurations may disagree**
   - **CONFIRMED**: `server.py` does not dynamically parse `.research_repo` YAML configs. `server.py` uses hardcoded baseline intensities (`afp=0.00030`, `rs=0.00020`, `fs=2.0`).

8. **Query traffic, unknown labels, mode changes incorrectly affecting comparison**
   - **UNVERIFIABLE**: Full state machine flow testing across dynamic controller mode transitions was beyond the safe bounds of this static audit.

9. **Tables, charts, summaries use different sources**
   - **CONFIRMED**: Line charts in `results.js` (`updateLineCharts`) use distinct random seed generation (e.g. `const recall = baseRec + (Math.random() * 0.05)`) instead of the true `STATIC_C1_C7_MATRIX` values.

10. **FS displayed decimal precision disagrees with actual implementation**
    - **CONFIRMED**: `server.py`, `format_defense_intensity()`, line 218.
    - **Excerpt**: `d_val = int(round(val)) if val is not None else 2`
    - **Impact**: FS intensity is visually formatted as a float but mathematically rounded to an integer for the actual defense parameterization.
