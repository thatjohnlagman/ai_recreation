# Experiment Protocol

> **FROZEN DOCUMENT** — Do not modify after final evaluation begins.
> All decisions below are finalized before running the final experiment matrix.
> SHA-256 hashes of config files are recorded in every run's metadata.

---

## Study Identity

- **Thesis title:** Performance of Recall-Aware Control for Perturbation Defenses in Intrusion Detection Systems Against Black-Box Probing Attacks
- **Classification:** Binary (0=benign, 1=attack)
- **Dataset:** CSE-CIC-IDS2018 — 10 CICFlowMeter CSV files
- **Working sample:** 300,000 records (stratified proportional from all files)
- **Split:** Stratified 70/30 (seed=42)
- **Target classifier:** Random Forest (frozen after training)

---

## Frozen Parameters

| Parameter | Value |
|---|---|
| Working sample size | 300,000 |
| Sampling seed | 42 |
| Split seed | 42 |
| Train fraction | 0.70 |
| Eval fraction | 0.30 |
| Batch size | 500 |
| Primary seeds | [42, 43, 44, 45, 46] |
| Sensitivity seeds | [42, 43, 44] |
| RF n_estimators | 200 |
| RF max_depth | 20 |
| RF criterion | gini |
| RF n_jobs | 2 |
| RS ensemble size m | 11 |
| AFP sigma floor | 1e-6 |
| AFP noise distribution | Bounded Uniform |
| Controller C1 W | 5 |
| Controller C1 Rcritical | 0.85 |
| Controller C1 Rmin | 0.95 |
| Controller C1 fast_decay | 0.40 |
| Controller C1 slow_decay | 0.90 |
| Controller C1 growth_factor | 1.05 |
| Statistical test | Two-tailed paired t-test |
| Significance level α | 0.05 |
| Pairing key | (seed, attack, defense, batch_id) |

Defense-specific intensity calibration values are filled here after Phase 8 calibration:

| Defense | epsilon_base / sigma / squeezing_intensity | intensity_min | intensity_max |
|---|---|---|---|
| AFP | 0.0003 | 0.0 | 0.0003 |
| RS | 0.0002 | 0.0 | 0.0002 |
| FS | 2 | 0.0 | 2.0 |

**Protocol Amendment (Phase 8 Calibration):** The controller bounds for AFP and RS have been updated to `intensity_min = 0.0` (down from 0.001) to support valid identity controls, and `intensity_max` is strictly bounded to the selected strongest valid calibration intensity for all defenses. This prevents upward clipping to invalid regimes. This amendment was made prior to Phase 9 and explicitly before any final evaluation data were accessed.

---

## Experiment Matrix

### Primary runs (RQ1–RQ3, C1)

3 attack scenarios × 3 defenses × 2 configurations (base, controller) × 5 seeds = **90 runs**

| Attack | Defense | Config |
|---|---|---|
| Silent Probing | AFP | Base |
| Silent Probing | AFP | C1 Controller |
| Silent Probing | RS | Base |
| Silent Probing | RS | C1 Controller |
| Silent Probing | FS | Base |
| Silent Probing | FS | C1 Controller |
| Surrogate Transfer | AFP | Base |
| Surrogate Transfer | AFP | C1 Controller |
| ... | ... | ... |
| Boundary Attack | FS | C1 Controller |

### Sensitivity runs (RQ4, C1–C7)

3 attacks × 3 defenses × 7 configs × 3 seeds = **189 runs**

---

## Leakage Prevention Rules

1. StandardScaler fitted on training partition only.
2. AFP benign profile (mu, sigma) computed from benign training records only.
3. Defense calibration uses training-derived calibration partition only.
4. Surrogate training uses a disjoint query pool — NOT final measurement batches.
5. Boundary attack reference pool is also disjoint from final measurement batches.
6. Controller warm-up uses completed batch TP/FN only — never current batch labels.
7. No hyperparameter or intensity selection based on final evaluation outcomes.

---

## Output Tables (Appendix 1 of Thesis)

- **B1:** Experiment Summary (base mean vs controller mean, t-value, p-value, decision)
- **B2.1:** Batch-Level Configuration/Controller Log
- **B2.2:** Batch-Level Confusion and Metric Log
- **B3:** Paired t-Test Summary (all defenses × all metrics)
- **B4:** Controller Parameter Sensitivity Summary (C1–C7)

All tables generated from raw machine-readable logs. No manual entry of results.

---

## Scientific Integrity Commitments

- If RA is ineffective or harmful, result is reported honestly.
- All seeds, attack scenarios, and metrics are reported — none discarded.
- Statistical significance, direction, effect size, and practical importance are clearly separated.
- "Significant difference" ≠ "effective" without checking the sign of the mean difference.
- Controller pinned at minimum intensity is reported as such, not as "balanced regulation."

---

## Pre-Evaluation Clarification (Sept 2026)

*Previous Protocol SHA-256:* `c10055f2759dce8ac7b65218902f44e334b35167b8d530be73831244de342ec4`

Before final evaluation commenced, the following structural clarification to the predefined `attacks.yaml` leakage rules was explicitly recorded. This does not alter the underlying methodology or results, but concretizes the evaluation split mathematics:
1. The 90,000-record `X_eval` allocation is strictly partitioned into an **18,000-record crafting pool** (20%) and a disjoint **72,000-record measurement pool** (80%).
2. Final statistical analysis and evaluation runs exclusively use **144 measurement batches of 500 records** derived solely from the measurement pool.
3. The crafting pool is reserved for surrogate queries and boundary benign references. It is absolutely prohibited from being included in measurement metrics or paired tests.
