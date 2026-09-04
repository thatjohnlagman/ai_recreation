# Phase 9: Controller Semantics

This document details the mathematical correctness, timing contracts, rolling window logic, state transitions, bounding, and validation guarantees of the Recall-Aware Controller implemented in Phase 9.

## Timing Contract and State Machine
The controller rigorously separates decision-making from observation.
1. **Decision (Before Batch $t$)**: `get_intensity(batch_id)` must be called before the controller sees the labels for batch $t$. It creates a "pending decision". Calling it twice before submitting observations fails safely.
2. **Observation (After Batch $t$)**: `submit_observations(batch_id, TP, FN)` provides the labels. It strictly requires a pending decision for the exact `batch_id`. Submitting without getting an intensity fails.
3. **Atomicity**: Invalid submissions (bad types, mismatched IDs, negative counts) fail gracefully, leaving the controller state completely unmodified. Upon successful validation, the pending decision is cleared, the state is updated, and the expected batch ID is incremented.

## Rolling Recall and Window Calculation
The controller calculates recall directly from aggregated non-negative integer counts rather than averaging batch-wise recalls.
$$ Rolling Recall = \frac{\sum_{i=1}^{W} TP_i}{\sum_{i=1}^{W} (TP_i + FN_i)} $$
- The window holds a maximum of $W$ completed batches.
- **Warm-up**: If fewer than $W$ batches exist, the sum is taken over all available completed batches.
- **Zero Denominator**: If total $TP + FN = 0$ inside the window, the recall safely defaults to the configuration's explicit `zero_division_value` (default `0.0`), and a flag is recorded in the logs.

## State Transitions
Transitions operate on the aggregated rolling recall using strict `<` boundaries. 
- **Red** (`rolling_recall < Rcritical`): Intensity is scaled down aggressively using the `fast_decay` multiplier.
- **Yellow** (`rolling_recall < Rmin`): Intensity is scaled down slowly using the `slow_decay` multiplier.
- **Green** (Otherwise): Intensity is scaled up using the `growth_factor`.

*Note*: Exact equality with thresholds falls into the higher state (e.g. `rolling_recall == Rcritical` maps to Yellow, not Red). Multiplications are performed in `float64` precision.

## Clipping and Bounding
The resulting next-batch intensity is inclusively clipped to the defense-specific bounds `[intensity_min, intensity_max]`.
The state update logging accurately records both the unclipped theoretical intensity and the applied clipped intensity.

## Deterministic Isolation
A `reset()` function is exposed to completely sever state between runs. Resetting drops all history, expected batch IDs, and pending decisions. If `reset()` is called while a decision is pending, it intentionally abandons that unfinished run and returns the controller to a clean batch-0 state. It explicitly restores the `base_intensity` originally configured for the active defense, regardless of what `intensity_max` may be.

## Empirical Effectiveness Scope
Phase 9 guarantees ONLY mathematical correctness according to the thesis specification. Evaluation using real network data to determine empirical performance (effectiveness) belongs strictly to Phase 10.
