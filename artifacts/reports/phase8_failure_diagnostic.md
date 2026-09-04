# Phase 8 Failure Diagnostic Report

Identity Controls: PASSED

## Isolation of Noise vs Clipping
When passing the unclipped proposal to the model:
- AFP unclipped recall: 0.1568
- AFP clipped recall: 0.1568
- RS unclipped recall: 0.0819
- RS clipped recall: 0.0819

*(If both collapse similarly, noise sensitivity is the primary cause. If only clipped mode collapses, clipping is the primary cause.)*

## Logarithmic Diagnostics (Summary)
- AFP val=1e-06 alpha=0.25: Recall=0.9465, OOB=0.2891
- AFP val=1e-06 alpha=0.5: Recall=0.9465, OOB=0.2894
- AFP val=1e-06 alpha=1.0: Recall=0.9463, OOB=0.2900
- RS val=1e-06: Recall=0.9459, OOB=0.2887
- AFP val=3e-06 alpha=0.25: Recall=0.9452, OOB=0.2943
- AFP val=3e-06 alpha=0.5: Recall=0.9455, OOB=0.2946
- AFP val=3e-06 alpha=1.0: Recall=0.9454, OOB=0.2952
- RS val=3e-06: Recall=0.9455, OOB=0.2940
- AFP val=1e-05 alpha=0.25: Recall=0.9347, OOB=0.3012
- AFP val=1e-05 alpha=0.5: Recall=0.9338, OOB=0.3016
- AFP val=1e-05 alpha=1.0: Recall=0.9308, OOB=0.3023
- RS val=1e-05: Recall=0.9413, OOB=0.3007
- AFP val=3e-05 alpha=0.25: Recall=0.9008, OOB=0.3095
- AFP val=3e-05 alpha=0.5: Recall=0.8995, OOB=0.3100
- AFP val=3e-05 alpha=1.0: Recall=0.8974, OOB=0.3108
- RS val=3e-05: Recall=0.8971, OOB=0.3090
- AFP val=0.0001 alpha=0.25: Recall=0.8646, OOB=0.3185
- AFP val=0.0001 alpha=0.5: Recall=0.8603, OOB=0.3189
- AFP val=0.0001 alpha=1.0: Recall=0.8542, OOB=0.3195
- RS val=0.0001: Recall=0.8704, OOB=0.3182
- AFP val=0.0003 alpha=0.25: Recall=0.7546, OOB=0.3267
- AFP val=0.0003 alpha=0.5: Recall=0.7454, OOB=0.3270
- AFP val=0.0003 alpha=1.0: Recall=0.7258, OOB=0.3278
- RS val=0.0003: Recall=0.7837, OOB=0.3265
- AFP val=0.001 alpha=0.25: Recall=0.5265, OOB=0.3428
- AFP val=0.001 alpha=0.5: Recall=0.5168, OOB=0.3434
- AFP val=0.001 alpha=1.0: Recall=0.5007, OOB=0.3444
- RS val=0.001: Recall=0.5561, OOB=0.3423
- AFP val=0.003 alpha=0.25: Recall=0.3355, OOB=0.3609
- AFP val=0.003 alpha=0.5: Recall=0.3236, OOB=0.3616
- AFP val=0.003 alpha=1.0: Recall=0.3077, OOB=0.3628
- RS val=0.003: Recall=0.3665, OOB=0.3602
- AFP val=0.01 alpha=0.25: Recall=0.1568, OOB=0.3848
- AFP val=0.01 alpha=0.5: Recall=0.1499, OOB=0.3856
- AFP val=0.01 alpha=1.0: Recall=0.1367, OOB=0.3870
- RS val=0.01: Recall=0.0819, OOB=0.3839
