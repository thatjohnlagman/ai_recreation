"""
Analysis Script 2: Paired T-Test for RQ3
==========================================
Thesis: Performance Analysis of Recall-Aware Control for
        Intrusion Detection Perturbation Defenses Against Black-Box Probing

RQ3 (inferential): Is there a significant difference in performance
before and after applying the recall-aware controller?

Null Hypothesis (H0):
  There is no significant difference in the performance of each
  perturbation-based defense mechanism before and after applying
  the recall-aware controller in terms of Precision, Attack Recall,
  and F1-Score.

Statistical Test: Two-tailed paired t-test (scipy.stats.ttest_rel)
Alpha: 0.05
Decision: p <= 0.05 -> Reject H0 / p > 0.05 -> Fail to reject H0

Loads: results/batch_level_experiment_results.csv
Saves:
  results/rq3_paired_ttest_results.csv
  results/rq3_paired_ttest_interpretation.md
"""

import os
import sys
import pandas as pd
import numpy as np
from scipy import stats

# --- Path setup ---
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

INPUT_PATH   = os.path.join(ROOT, 'results', 'batch_level_experiment_results.csv')
OUTPUT_DIR   = os.path.join(ROOT, 'results')
RESULTS_PATH = os.path.join(OUTPUT_DIR, 'rq3_paired_ttest_results.csv')
INTERP_PATH  = os.path.join(OUTPUT_DIR, 'rq3_paired_ttest_interpretation.md')

ALPHA   = 0.05
METRICS = ['precision', 'attack_recall', 'f1_score']
GROUP_COLS = ['dataset', 'attack_scenario', 'defense_mechanism']
PAIR_KEY = ['dataset', 'attack_scenario', 'defense_mechanism', 'configuration', 'batch_id', 'run_mode']


def validate_no_duplicates(df):
    """
    Pre-flight check: abort if any group contains duplicate batch IDs.

    Duplicate batch IDs make the paired t-test invalid because one batch_id
    would correspond to multiple (different) metric values, breaking the 1:1 pairing.

    Exits with an error if any duplicates are found.
    """
    dups = df[df.duplicated(subset=PAIR_KEY, keep=False)]
    if len(dups) == 0:
        print("  [OK] No duplicate batch IDs found — data is clean.")
        return

    dup_summary = dups.groupby(
        ['dataset', 'attack_scenario', 'defense_mechanism', 'configuration']
    ).size().reset_index(name='duplicate_row_count')

    print("\n" + "=" * 60)
    print("[VALIDATION FAILED] Duplicate batch IDs detected in results.")
    print("The paired t-test CANNOT run until duplicates are resolved.")
    print("=" * 60)
    print(dup_summary.to_string(index=False))
    print("\nFix: Delete results/batch_level_experiment_results.csv and re-run:")
    print("  python experiments/run_revised_experiments.py --mode sample")
    print("=" * 60)
    sys.exit(1)


def validate_pair_alignment(base_df, ctrl_df, group_label, expected_n=None):
    """
    Check that base and controller DataFrames have the same batch IDs.
    Returns the common batch IDs (sorted), or exits if mismatch is critical.
    """
    base_ids = set(base_df['batch_id'].tolist())
    ctrl_ids = set(ctrl_df['batch_id'].tolist())

    if base_ids == ctrl_ids:
        if expected_n is not None and len(base_ids) != expected_n:
            print(f"  [WARNING] {group_label}: expected {expected_n} pairs, got {len(base_ids)}")
        return sorted(base_ids)

    missing_in_ctrl = base_ids - ctrl_ids
    missing_in_base = ctrl_ids - base_ids
    print(f"\n  [WARNING] {group_label}: batch ID mismatch.")
    if missing_in_ctrl:
        print(f"    base has {len(missing_in_ctrl)} batch IDs not in controller — using intersection")
    if missing_in_base:
        print(f"    controller has {len(missing_in_base)} batch IDs not in base — using intersection")

    common = sorted(base_ids & ctrl_ids)
    if len(common) < 2:
        print(f"  [ERROR] Only {len(common)} common batch IDs — cannot run t-test for {group_label}")
        return None
    return common



def run_paired_ttest(base_values, ctrl_values, alpha=ALPHA):
    """
    Run a two-tailed paired t-test between base and controller metric arrays.

    Parameters
    ----------
    base_values : array-like — metric values per batch under base config
    ctrl_values : array-like — metric values per batch under controller config
    alpha       : float — significance level

    Returns
    -------
    n_pairs      : int
    base_mean    : float
    ctrl_mean    : float
    mean_diff    : float (controller - base)
    t_stat       : float
    p_value      : float
    decision     : str — 'Reject H0' or 'Fail to reject H0'
    """
    base_arr = np.array(base_values, dtype=float)
    ctrl_arr = np.array(ctrl_values, dtype=float)

    n_pairs   = len(base_arr)
    base_mean = float(np.mean(base_arr))
    ctrl_mean = float(np.mean(ctrl_arr))
    mean_diff = ctrl_mean - base_mean

    if n_pairs < 2:
        return n_pairs, base_mean, ctrl_mean, mean_diff, np.nan, np.nan, 'Insufficient data'

    t_stat, p_value = stats.ttest_rel(base_arr, ctrl_arr, alternative='two-sided')

    decision = 'Reject H0' if p_value <= alpha else 'Fail to reject H0'

    return n_pairs, base_mean, ctrl_mean, mean_diff, float(t_stat), float(p_value), decision


def main():
    if not os.path.exists(INPUT_PATH):
        print(f"[ERROR] Results file not found: {INPUT_PATH}")
        print("  -> Run: python experiments/run_revised_experiments.py --mode sample")
        sys.exit(1)

    print("=" * 60)
    print("RQ3: PAIRED T-TEST ANALYSIS")
    print(f"Alpha = {ALPHA} | Two-tailed | scipy.stats.ttest_rel")
    print("=" * 60)

    df = pd.read_csv(INPUT_PATH)
    print(f"Loaded {len(df)} rows from {INPUT_PATH}")

    # --- PRE-FLIGHT VALIDATION ---
    # This will exit(1) immediately if any group has duplicate batch IDs.
    # Do NOT run the t-test on contaminated data.
    print("\nValidating data integrity...")
    validate_no_duplicates(df)

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    result_rows = []
    problems    = []

    # --- For each group and metric, run the paired t-test ---
    groups = df.groupby(GROUP_COLS)

    for group_key, group_df in groups:
        dataset, attack_scenario, defense_mechanism = group_key
        group_label = f"{dataset}/{attack_scenario}/{defense_mechanism}"

        base_df = group_df[group_df['configuration'] == 'base'].sort_values('batch_id')
        ctrl_df = group_df[group_df['configuration'] == 'controller_augmented'].sort_values('batch_id')

        # Validate pair alignment — uses validate_pair_alignment() which warns on mismatch
        common_ids = validate_pair_alignment(base_df, ctrl_df, group_label, expected_n=200)
        if common_ids is None:
            problems.append(f"SKIP: {group_label} — insufficient common batch IDs.")
            continue

        base_df = base_df[base_df['batch_id'].isin(common_ids)].sort_values('batch_id')
        ctrl_df = ctrl_df[ctrl_df['batch_id'].isin(common_ids)].sort_values('batch_id')

        if len(base_df) == 0 or len(ctrl_df) == 0:
            problems.append(f"SKIP: {group_label} — empty after alignment.")
            continue

        for metric in METRICS:
            base_values = base_df[metric].values
            ctrl_values = ctrl_df[metric].values

            n_pairs, base_mean, ctrl_mean, mean_diff, t_stat, p_value, decision = \
                run_paired_ttest(base_values, ctrl_values)

            result_rows.append({
                'dataset':           dataset,
                'attack_scenario':   attack_scenario,
                'defense_mechanism': defense_mechanism,
                'metric':            metric,
                'n_pairs':           n_pairs,
                'base_mean':         round(base_mean, 4),
                'controller_mean':   round(ctrl_mean, 4),
                'mean_difference':   round(mean_diff, 4),
                't_statistic':       round(t_stat, 4) if not np.isnan(t_stat) else 'N/A',
                'p_value':           round(p_value, 4) if not np.isnan(p_value) else 'N/A',
                'alpha':             ALPHA,
                'decision':          decision,
            })

    # --- Save results CSV ---
    results_df = pd.DataFrame(result_rows)
    results_df.to_csv(RESULTS_PATH, index=False)
    print(f"\nResults saved to: {RESULTS_PATH}")
    print(results_df.to_string(index=False))

    # --- Report any problems ---
    if problems:
        print("\n[WARNINGS]")
        for p in problems:
            print(f"  -> {p}")

    # --- Generate interpretation markdown ---
    _write_interpretation(results_df, problems)
    print(f"\nInterpretation saved to: {INTERP_PATH}")

    print("\n" + "=" * 60)
    print("RQ3 PAIRED T-TEST COMPLETE.")
    print("=" * 60)


def _write_interpretation(results_df, problems):
    """Write a plain-language interpretation of the t-test results."""

    reject = results_df[results_df['decision'] == 'Reject H0']
    fail   = results_df[results_df['decision'] == 'Fail to reject H0']

    lines = []
    lines.append("# RQ3: Paired T-Test Interpretation\n")
    lines.append("## Overview\n")
    lines.append(
        "This document explains the results of the paired t-test conducted for "
        "Research Question 3 (RQ3) of the thesis.\n"
    )

    lines.append("## What Was Compared\n")
    lines.append(
        "For each combination of dataset, attack scenario, and defense mechanism, "
        "a two-tailed paired t-test was conducted. The test compared:\n\n"
        "- **Base configuration**: the defense mechanism operating at a fixed high perturbation intensity, "
        "with no controller active.\n"
        "- **Controller-augmented configuration**: the same defense mechanism operating with the "
        "recall-aware controller adjusting perturbation intensity dynamically.\n\n"
        "The paired unit is the **evaluation batch**. Each batch in the base configuration "
        "is paired with the same batch in the controller-augmented configuration, as both "
        "configurations processed identical batches containing the same attack and benign samples "
        "in the same order.\n"
    )

    lines.append("## Null Hypothesis\n")
    lines.append(
        "> **H0:** There is no significant difference in the performance of each "
        "perturbation-based defense mechanism before and after applying the recall-aware "
        "controller in terms of Precision, Attack Recall, and F1-Score.\n"
    )

    lines.append("## Statistical Test\n")
    lines.append(
        f"- Test: Two-tailed paired t-test (`scipy.stats.ttest_rel`)\n"
        f"- Alpha: {ALPHA}\n"
        f"- Decision rule:\n"
        f"  - p-value ≤ {ALPHA} → **Reject H0** (significant difference)\n"
        f"  - p-value > {ALPHA} → **Fail to reject H0** (no significant difference)\n"
    )

    lines.append("## What p-value Means\n")
    lines.append(
        "The p-value is the probability of observing a difference as large as the one "
        "measured (or larger) if the null hypothesis were true — that is, if the controller "
        "had no real effect. A small p-value (≤ 0.05) indicates that the observed difference "
        "is unlikely to have occurred by chance alone.\n\n"
        "**Reject H0** does not mean the controller is universally better. "
        "It indicates that the observed difference is statistically significant — "
        "the controller provides evidence of a performance change.\n\n"
        "**Fail to reject H0** means the data does not provide sufficient evidence "
        "of a significant difference at the 0.05 level. This does not prove the controller "
        "has no effect; it may indicate limited power or a small effect size.\n"
    )

    lines.append("## Results Summary\n")
    lines.append(f"Total comparisons: {len(results_df)}\n")
    lines.append(f"Reject H0: {len(reject)}\n")
    lines.append(f"Fail to reject H0: {len(fail)}\n\n")

    lines.append("### Reject H0 (significant difference detected)\n")
    if len(reject) > 0:
        lines.append("| Dataset | Attack | Defense | Metric | Base Mean | Controller Mean | Diff | t | p |\n")
        lines.append("|---|---|---|---|---|---|---|---|---|\n")
        for _, row in reject.iterrows():
            lines.append(
                f"| {row['dataset']} | {row['attack_scenario']} | {row['defense_mechanism']} "
                f"| {row['metric']} | {row['base_mean']} | {row['controller_mean']} "
                f"| {row['mean_difference']} | {row['t_statistic']} | {row['p_value']} |\n"
            )
    else:
        lines.append("No comparisons reached significance.\n")

    lines.append("\n### Fail to Reject H0\n")
    if len(fail) > 0:
        lines.append("| Dataset | Attack | Defense | Metric | Base Mean | Controller Mean | p |\n")
        lines.append("|---|---|---|---|---|---|---|\n")
        for _, row in fail.iterrows():
            lines.append(
                f"| {row['dataset']} | {row['attack_scenario']} | {row['defense_mechanism']} "
                f"| {row['metric']} | {row['base_mean']} | {row['controller_mean']} "
                f"| {row['p_value']} |\n"
            )
    else:
        lines.append("All comparisons reached significance.\n")

    if problems:
        lines.append("\n## Data Quality Notes\n")
        for p in problems:
            lines.append(f"- {p}\n")

    lines.append("\n## Type I and Type II Error\n")
    lines.append(
        "- **Type I Error (False Positive, α = 0.05):** The probability of rejecting H0 when "
        "it is actually true — concluding the controller has an effect when it does not. "
        "By setting α = 0.05, this risk is controlled at 5%.\n\n"
        "- **Type II Error (False Negative, β):** The probability of failing to reject H0 "
        "when it is actually false — missing a real effect of the controller. "
        "Type II error is reduced by having a sufficient number of paired batches (n ≥ 30).\n"
    )

    lines.append("\n## Limitations\n")
    lines.append(
        "- Results are based on controlled offline simulation using benchmark datasets, "
        "not live network traffic.\n"
        "- The attack scenarios are simplified implementations of black-box probing strategies. "
        "Real-world attackers may use more sophisticated methods.\n"
        "- The controller's performance may vary with different initial intensity values "
        "or different threshold parameters.\n"
        "- The study does not claim that the controller universally improves all metrics "
        "across all conditions. Results are reported per condition and interpreted independently.\n"
    )

    with open(INTERP_PATH, 'w') as f:
        f.writelines(lines)


if __name__ == '__main__':
    main()
