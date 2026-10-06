# Implementation Decisions

This document records every implementation detail not numerically specified in the approved thesis,
including the rationale for each choice. All decisions were made before final evaluation and must
not be changed based on final evaluation results.

---

## 1. Working Sample Size
- **Decision:** 300,000 records stratified from all 10 CSVs
- **Rationale:** Full 16–18M row dataset (~6.9 GB) exceeds M2/8 GB RAM for in-memory RF training.
  300,000 is the minimum specified in the handoff. Proportional stratified sampling preserves the
  natural class distribution across all files and binary classes.
- **Sampling seed:** 42
- **Do not reduce** below 300,000 without reporting a measured MemoryError.

## 2. Dataset File Selection
- **Decision:** All 10 CICFlowMeter CSV files included
- **Rationale:** No file is excluded. All contain valid traffic-flow records. Proportional allocation
  ensures all attack families and time periods are represented in the working sample.

## 3. Feature Exclusions
- **Definite exclusions:** `Flow ID`, `Src IP`, `Dst IP`, `Timestamp`
  - Rationale: Flow ID is a non-predictive identifier; raw IPs are direct identifiers; Timestamp is
    a collection artifact with no generalization value.
- **Ports retained by default:** `Src Port`, `Dst Port`
  - Rationale: Per handoff — "do not automatically remove every column containing 'Port'."
    Destination port represents legitimate service behavior. Ports are retained unless
    a concrete leakage or validity reason is found during audit.
- **Final exclusion list finalized after audit** and documented in DATA_DICTIONARY.md.

## 4. Preprocessing Order
1. Strip whitespace from column names
2. Remove duplicate header rows embedded as data rows
3. Replace ±Inf with NaN
4. Drop rows with NaN in expected numerical columns (strict numeric conversion)
5. Map labels to binary (0=benign, 1=attack); preserve original label as `attack_family`
6. Drop excluded identifier columns
7. Stratified 70/30 split (seed=42)
8. Fit StandardScaler on training partition only
9. Compute benign AFP profile (mu, sigma) from benign training records only

## 5. Random Seeds
- Global seed: 42
- Split seed: 42
- Sampling seed: 42
- Primary stochastic seeds (C1 comparison): [42, 43, 44, 45, 46] — 5 seeds per handoff
- Sensitivity seeds (C1–C7): [42, 43, 44] — 3 seeds per handoff
- Rationale: Handoff explicitly forbids 30 seeds on M2/8GB.

## 6. Batch Size
- **Decision:** 500 records per batch (stratified)
- **Rationale:** Large enough for reliable metric computation; small enough to fit in memory.
  Confirmed adequate after audit confirms sufficient positive and negative records.

## 7. Random Forest Hyperparameters
- n_estimators: 200
- max_depth: 20
- criterion: 'gini'
- n_jobs: 2 (NOT -1; limited by M2/8GB per handoff)
- random_state: 42
- Rationale: Springer paper starting values; n_jobs limited by hardware.
  If training is infeasible (>2 GB RAM), benchmark smaller configs on calibration data.
- **No SMOTE, undersampling, or class balancing** — natural distribution preserved.

## 8. AFP Parameters
- epsilon_base: Calibrated on training-only calibration partition
- alpha: 0.5 default; validated on training calibration data
- Sigma floor: 1e-6 (prevents zero-division in delta_i computation)
- Noise: Bounded Uniform(−ε_i, +ε_i)
  - Rationale: Thesis specifies bounded noise; AFP source uses bounded uniform; operationally equivalent.
- epsilon_base grid: [0.01, 0.05, 0.10, 0.20]
- Calibration rule: Strongest valid perturbation with training-validation Recall ≥ 0.80 and
  invalid feature fraction ≤ 5%.

## 9. Randomized Smoothing Parameters
- Ensemble size m: 11 (odd; per handoff — NOT 51)
- Rationale: m=11 validated on training calibration data; odd minimizes ties.
- sigma grid: [0.01, 0.05, 0.10, 0.20]
- Tie breaking: first class (m=11 makes ties essentially impossible).
- Ensemble copies processed in chunks of 100 to avoid RAM spikes.
- NOT certified robustness — described as study's Monte Carlo smoothing implementation.

## 10. Feature Squeezing Parameters
- squeezing_intensity grid: [1, 2, 3, 4, 5, 6]
- Effective decimal d = max(0, 6 − int(intensity))
- Note: Multiplicative controller updates may not change d every batch (integer conversion).
  This is acknowledged and the continuous intensity and effective d are both logged.

## 11. AFP Formula Choice
- Thesis equation: ε_i = ε_base × (1 + α × δ_i) — MULTIPLICATIVE
- Source AFP paper: ε_i = ε_base + α × δ_i — ADDITIVE
- **Decision:** Thesis equation used. Source paper equation NOT implemented.
  Difference recorded in SOURCE_MAPPING.md.

## 12. Silent Probing Definition
- **Decision:** Original evaluation batches submitted sequentially in order.
  Batch order preserved between base and controller runs.
- **NOT reproduced:** Springer random-walk, change-point detection, side-channel analysis.
- Label in all outputs: "study-defined controlled silent-probing simulation"

## 13. Surrogate Transferability
- Surrogate model: Decision Tree (unconstrained depth) per thesis Chapter 3
- Query subset: Disjoint from final measurement batches (20% of eval partition)
- Attack method: Tree-path/leaf targeting — minimum feature modifications to reach benign leaf
- No target RF internals accessed during attack generation
- Transfer: No further target-guided refinement after transfer

## 14. Decision-Boundary Attack
- Search: Binary search along attack-to-benign interpolation line
- Query budget: 50 queries per sample (fixed before evaluation)
- Binary search steps: 10 steps
- RF access: Hard-label binary predictions only — NO probabilities, importances, or gradients

## 15. Undefined Metric Handling
- If TP+FP=0 (no predicted positives): Precision = 0.0 (not NaN)
- If TP+FN=0 (no actual positives): Recall = 0.0 (not NaN)
- If Precision+Recall=0: F1 = 0.0 (not NaN)
- Both raw confusion counts AND derived metrics are saved; no hiding of undefined values.

## 16. Statistical Diagnostics
- Primary test: two-tailed paired t-test (scipy.stats.ttest_rel)
- Pairing key: (seed, attack_scenario, defense_mechanism, batch_id)
- Effect size: Cohen's dz
- Confidence interval: 95% paired difference CI
- Supplementary: Holm-adjusted p-values, Wilcoxon signed-rank (if normality questionable)
- Normality diagnostic: Shapiro-Wilk on paired differences
- These supplementary tests are clearly labeled and do not replace the primary paired t-test.

## 17. Controller Warm-Up
- **Decision:** Use all completed batches currently available when fewer than W are done.
- First batch always uses frozen base intensity.
- Rationale: Reasonable conservative choice that avoids undefined-window errors.

## 18. Attack Caching
- Generated attack datasets are cached as Parquet in artifacts/attacks/
- Same cached attack data is reused across all defense mechanisms for a given seed and scenario
- Rationale: Reduces computation; ensures paired fair comparison.

## 19. Parquet and Float Precision
- float32 used where verified safe (feature values)
- float64 retained for: raw confusion counts, metric calculations, statistical tests
- int8 for binary labels; int32 for batch and row IDs

## 20. Disjoint Crafting Pool
- 20% of evaluation partition is reserved as the crafting/query pool (NOT used as measurement batches)
- Remaining 80% forms the measurement batches
- Crafting pool seed: 42
