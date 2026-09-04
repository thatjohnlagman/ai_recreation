"""
src/recall_aware_ids/statistics/paired_ttest.py
Statistical analysis for the thesis experiment.

Primary test (Thesis Chapter 3):
    Two-tailed paired t-test (scipy.stats.ttest_rel)
    Significance level: α = 0.05
    Pairing key: (seed, attack_scenario, defense_mechanism, batch_id)

Supplementary (labeled clearly, do not replace primary test):
    - Shapiro-Wilk normality test on paired differences
    - Wilcoxon signed-rank test (if normality questionable)
    - Cohen's dz (effect size)
    - 95% paired difference confidence interval
    - Holm-corrected p-values

Critical rule: significant p-value alone does NOT establish effectiveness.
The direction (sign) and practical magnitude of the mean difference must also
be reported and interpreted.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
from scipy import stats as scipy_stats


@dataclass
class PairedTestResult:
    """Result of a paired t-test for one (attack, defense, metric) triple."""
    attack_scenario: str
    defense_mechanism: str
    metric: str                  # "precision", "recall", or "f1_score"
    n_pairs: int
    mean_base: float
    mean_controller: float
    mean_difference: float       # controller - base
    std_difference: float
    t_statistic: float
    p_value: float
    significant: bool            # p_value < alpha
    direction: str               # "controller_better", "controller_worse", "no_change"
    cohens_dz: float
    ci_lower: float              # 95% CI on mean difference
    ci_upper: float
    # Supplementary
    shapiro_stat: Optional[float] = None
    shapiro_p: Optional[float] = None
    wilcoxon_stat: Optional[float] = None
    wilcoxon_p: Optional[float] = None
    holm_adjusted_p: Optional[float] = None
    alpha: float = 0.05
    null_hypothesis: str = "H0: no difference in mean metric between base and controller"
    decision: str = ""           # "Reject H0" or "Fail to reject H0"


def run_paired_ttest(
    base_values: np.ndarray,
    controller_values: np.ndarray,
    attack_scenario: str,
    defense_mechanism: str,
    metric: str,
    alpha: float = 0.05,
    run_supplementary: bool = True,
) -> PairedTestResult:
    """
    Run two-tailed paired t-test on paired batch-level metric observations.

    Parameters
    ----------
    base_values : array of floats, shape (n,)
        Metric values for base defense configuration.
    controller_values : array of floats, shape (n,)
        Metric values for controller-augmented configuration.
        Must be paired: base_values[i] and controller_values[i] have same
        (seed, attack, defense, batch_id).
    """
    base = np.asarray(base_values, dtype=np.float64)
    ctrl = np.asarray(controller_values, dtype=np.float64)

    if len(base) != len(ctrl):
        raise ValueError(f"Paired arrays must have equal length: {len(base)} vs {len(ctrl)}")
    if len(base) < 2:
        raise ValueError("Need at least 2 pairs for t-test.")

    differences = ctrl - base   # controller - base
    n = len(differences)
    mean_diff = float(differences.mean())
    std_diff = float(differences.std(ddof=1)) if n > 1 else 0.0

    # Primary: two-tailed paired t-test
    t_stat, p_val = scipy_stats.ttest_rel(ctrl, base)
    significant = bool(p_val < alpha)
    decision = "Reject H0" if significant else "Fail to reject H0"

    # Direction
    if significant:
        direction = "controller_better" if mean_diff > 0 else "controller_worse"
    else:
        direction = "no_significant_difference"

    # Cohen's dz
    cohens_dz = (mean_diff / std_diff) if std_diff > 0 else 0.0

    # 95% CI on mean difference
    se = std_diff / np.sqrt(n) if n > 0 else 0.0
    t_crit = scipy_stats.t.ppf(0.975, df=n - 1)
    ci_lower = mean_diff - t_crit * se
    ci_upper = mean_diff + t_crit * se

    # Supplementary diagnostics
    shapiro_stat = shapiro_p = wilcoxon_stat = wilcoxon_p = None
    if run_supplementary and n >= 3:
        try:
            shapiro_stat, shapiro_p = scipy_stats.shapiro(differences)
        except Exception:
            pass
        try:
            if not np.all(differences == 0):
                wilcoxon_stat, wilcoxon_p = scipy_stats.wilcoxon(differences)
        except Exception:
            pass

    return PairedTestResult(
        attack_scenario=attack_scenario,
        defense_mechanism=defense_mechanism,
        metric=metric,
        n_pairs=n,
        mean_base=float(base.mean()),
        mean_controller=float(ctrl.mean()),
        mean_difference=mean_diff,
        std_difference=std_diff,
        t_statistic=float(t_stat),
        p_value=float(p_val),
        significant=significant,
        direction=direction,
        cohens_dz=cohens_dz,
        ci_lower=float(ci_lower),
        ci_upper=float(ci_upper),
        shapiro_stat=float(shapiro_stat) if shapiro_stat is not None else None,
        shapiro_p=float(shapiro_p) if shapiro_p is not None else None,
        wilcoxon_stat=float(wilcoxon_stat) if wilcoxon_stat is not None else None,
        wilcoxon_p=float(wilcoxon_p) if wilcoxon_p is not None else None,
        alpha=alpha,
        null_hypothesis="H0: no difference in mean metric between base and controller configurations",
        decision=decision,
    )


def apply_holm_correction(results: list[PairedTestResult]) -> list[PairedTestResult]:
    """Apply Holm-Bonferroni correction to a set of test results (in-place)."""
    n = len(results)
    sorted_idx = sorted(range(n), key=lambda i: results[i].p_value)
    for rank, idx in enumerate(sorted_idx):
        adjusted = min(1.0, results[idx].p_value * (n - rank))
        results[idx].holm_adjusted_p = adjusted
    return results
