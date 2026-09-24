# Chapter 4: Results and Interpretation

## 4.1 Introduction
This chapter presents the statistical analysis of the Recall-Aware (RA) feedback controller across three existing defense mechanisms: Adaptive Feature Poisoning (AFP), Randomized Smoothing (RS), and Feature Squeezing (FS). The proposed contribution of this thesis is the RA controller, which dynamically adjusts perturbation intensity using Rolling Recall. The valid causal comparison within this study is each implemented fixed-intensity Base defense versus the same defense augmented with the C1 Recall-Aware controller (denoted as AFP + RA, RS + RA, FS + RA).

## 4.2 RQ1: Base Defense Performance
Table 4.1 summarizes the Precision, Recall, and F1-Score of the fixed-intensity Base defenses across attack scenarios.

| Defense | Scenario | Precision | Recall | F1-Score |
|---|---|---|---|---|
| AFP | SilentProbing | 99.92% | 80.66% | 89.21% |
| AFP | SurrogateTransfer | 99.93% | 80.62% | 89.19% |
| AFP | DecisionBoundary | 99.93% | 79.40% | 88.43% |
| FEATURE_SQUEEZING | SilentProbing | 99.86% | 93.19% | 96.39% |
| FEATURE_SQUEEZING | SurrogateTransfer | 99.86% | 93.19% | 96.39% |
| FEATURE_SQUEEZING | DecisionBoundary | 99.86% | 91.66% | 95.56% |
| RANDOMIZED_SMOOTHING | SilentProbing | 99.95% | 80.95% | 89.39% |
| RANDOMIZED_SMOOTHING | SurrogateTransfer | 99.95% | 81.06% | 89.47% |
| RANDOMIZED_SMOOTHING | DecisionBoundary | 99.95% | 79.70% | 88.63% |

## 4.3 RQ2: Controller-Augmented Defense Performance
Table 4.2 summarizes the performance of the defenses augmented with the C1 RA controller.

| Defense | Scenario | Precision | Recall | F1-Score |
|---|---|---|---|---|
| AFP + RA | SilentProbing | 99.59% | 93.70% | 96.53% |
| AFP + RA | SurrogateTransfer | 99.61% | 93.64% | 96.51% |
| AFP + RA | DecisionBoundary | 99.52% | 92.25% | 95.72% |
| FEATURE_SQUEEZING + RA | SilentProbing | 99.74% | 93.94% | 96.74% |
| FEATURE_SQUEEZING + RA | SurrogateTransfer | 99.74% | 93.93% | 96.73% |
| FEATURE_SQUEEZING + RA | DecisionBoundary | 99.70% | 92.49% | 95.94% |
| RANDOMIZED_SMOOTHING + RA | SilentProbing | 99.64% | 93.68% | 96.55% |
| RANDOMIZED_SMOOTHING + RA | SurrogateTransfer | 99.61% | 93.70% | 96.54% |
| RANDOMIZED_SMOOTHING + RA | DecisionBoundary | 99.52% | 92.25% | 95.72% |

## 4.4 RQ3: Statistical Differences (Base vs. C1)
The locked primary analysis consists of 2,160 C1 minus Base batch pairs per defense and metric. Raw two-tailed paired t-test p-values at α = .05 determine the primary decision. Holm adjustment is provided as supplementary robustness evidence. Note that serial dependence at the batch level is a known limitation.

| Defense | Metric | Base Mean | C1 Mean | C1-Base Diff | 95% CI | t-stat (df=2159) | p_raw | Decision | Holm | dz | Direction |
|---|---|---|---|---|---|---|---|---|---|---|---|
| AFP | precision | 0.9993 | 0.9957 | -0.0036 | [-0.0038, -0.0033] | -24.90 | <.001 | significant | <.001 | -0.54 | decreased |
| AFP | recall | 0.8023 | 0.9319 | 0.1297 | [0.1281, 0.1313] | 161.50 | <.001 | significant | <.001 | 3.47 | improved |
| AFP | f1_score | 0.8894 | 0.9625 | 0.0731 | [0.0721, 0.0740] | 149.81 | <.001 | significant | <.001 | 3.22 | improved |
| RANDOMIZED_SMOOTHING | precision | 0.9995 | 0.9959 | -0.0036 | [-0.0039, -0.0033] | -25.85 | <.001 | significant | <.001 | -0.56 | decreased |
| RANDOMIZED_SMOOTHING | recall | 0.8057 | 0.9321 | 0.1264 | [0.1249, 0.1280] | 159.38 | <.001 | significant | <.001 | 3.43 | improved |
| RANDOMIZED_SMOOTHING | f1_score | 0.8916 | 0.9627 | 0.0711 | [0.0701, 0.0720] | 147.58 | <.001 | significant | <.001 | 3.18 | improved |
| FEATURE_SQUEEZING | precision | 0.9986 | 0.9973 | -0.0013 | [-0.0015, -0.0011] | -13.00 | <.001 | significant | <.001 | -0.28 | decreased |
| FEATURE_SQUEEZING | recall | 0.9268 | 0.9345 | 0.0077 | [0.0073, 0.0081] | 37.78 | <.001 | significant | <.001 | 0.81 | improved |
| FEATURE_SQUEEZING | f1_score | 0.9611 | 0.9647 | 0.0036 | [0.0033, 0.0038] | 29.48 | <.001 | significant | <.001 | 0.63 | improved |

## 4.5 RQ4: Sensitivity Analysis (C1–C7)
Descriptive performance across configurations C1 through C7 demonstrates that alternative parameter selections yield distinct operating points, with no single configuration universally optimal across all metrics.

## 4.6 Integrated Discussion and AFP Interpretation
The original AFP paper reported high accuracy in some scenarios while reporting very low attack recall (e.g., 0.01). Accuracy can conceal missed attacks, particularly under class imbalance. While noting these internal reporting inconsistencies in the original paper, the valid internal comparison in this study is Base AFP versus AFP + RA under matched experimental conditions.

In our controlled evaluation, RA increased AFP Recall and F1 across all scenarios. Specifically, Recall increased for Silent Probing (approx. 80.66% Base to 93.70% C1), Surrogate Transfer (approx. 80.62% Base to 93.64% C1), and Decision Boundary (approx. 79.40% Base to 92.25% C1). This improvement came with a small, statistically significant decrease in Precision. The controller effectively mitigated recall degradation associated with fixed perturbation intensity under this study’s conditions.

## 4.7 Summary of Findings
The proposed Recall-Aware controller improved Recall and F1-Score when added to AFP, Randomized Smoothing, and Feature Squeezing under the frozen controlled evaluation, with a small reduction in Precision.

## 4.8 Methodological Limitations
- Serial dependence in batch-level observations is present.
- Results are constrained to the specific dataset, attacks, classifier, and conditions tested.
- The study does not claim universal real-world superiority or perfect attack prevention.
