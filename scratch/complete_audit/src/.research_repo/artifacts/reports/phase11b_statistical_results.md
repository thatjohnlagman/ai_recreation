# Phase 11B Statistical Results

This report summarizes the locked Phase 11B statistical analysis.

## Important Disclosures
- Serial dependence is acknowledged at the batch level. The primary batch-level results are interpreted alongside the supplementary run-level results.
- No causal claims are made.
- No claims of universal IDS superiority are made.
- The methodology was fully prespecified and locked prior to this Phase 11B execution.

## Primary Batch-Level Results (n=2,160 pairs)
| Defense | Metric | Mean Diff | Raw p-value | Raw Decision | Holm p-value | Holm Decision |
|---|---|---|---|---|---|---|
| afp | precision | -0.0036 | 1.4911e-120 | significant | 2.9821e-120 | significant |
| afp | recall | 0.1297 | 0.0000e+00 | significant | 0.0000e+00 | significant |
| afp | f1_score | 0.0731 | 0.0000e+00 | significant | 0.0000e+00 | significant |
| randomized_smoothing | precision | -0.0036 | 1.4757e-128 | significant | 4.4271e-128 | significant |
| randomized_smoothing | recall | 0.1264 | 0.0000e+00 | significant | 0.0000e+00 | significant |
| randomized_smoothing | f1_score | 0.0711 | 0.0000e+00 | significant | 0.0000e+00 | significant |
| feature_squeezing | precision | -0.0013 | 2.9367e-37 | significant | 2.9367e-37 | significant |
| feature_squeezing | recall | 0.0077 | 3.6528e-240 | significant | 1.8264e-239 | significant |
| feature_squeezing | f1_score | 0.0036 | 7.5060e-161 | significant | 3.0024e-160 | significant |

## Supplementary Run-Level Results (n=15 pairs)
| Defense | Metric | Mean Diff | Raw p-value | Raw Decision | Holm p-value | Holm Decision |
|---|---|---|---|---|---|---|
| afp | precision | -0.0036 | 9.2734e-14 | significant | 2.7820e-13 | significant |
| afp | recall | 0.1297 | 1.3146e-27 | significant | 7.8874e-27 | significant |
| afp | f1_score | 0.0727 | 3.7580e-28 | significant | 2.6306e-27 | significant |
| randomized_smoothing | precision | -0.0036 | 1.2401e-12 | significant | 1.2401e-12 | significant |
| randomized_smoothing | recall | 0.1264 | 5.6061e-29 | significant | 4.4849e-28 | significant |
| randomized_smoothing | f1_score | 0.0707 | 2.3796e-29 | significant | 2.1417e-28 | significant |
| feature_squeezing | precision | -0.0013 | 5.7903e-13 | significant | 1.1581e-12 | significant |
| feature_squeezing | recall | 0.0077 | 6.8616e-19 | significant | 2.7446e-18 | significant |
| feature_squeezing | f1_score | 0.0035 | 3.0542e-19 | significant | 1.5271e-18 | significant |
