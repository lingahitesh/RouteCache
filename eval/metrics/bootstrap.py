"""
Bootstrap confidence interval computation.

SRS FR-8.7: "The harness shall report bootstrap confidence intervals
for all headline numbers."

Lesson 10, Section 20: Use the percentile method with B >= 1000
bootstrap resamples. Report the 95% CI [2.5th, 97.5th percentile].

Build Plan: Step 13
"""

import numpy as np
from typing import Callable


def bootstrap_ci(
    data: np.ndarray,
    statistic: Callable = np.mean,
    n_bootstrap: int = 1000,
    confidence: float = 0.95,
    seed: int = 42,
) -> tuple[float, float, float]:
    """
    Compute a bootstrap confidence interval for a statistic.

    Args:
        data: 1-D array of observations.
        statistic: Function to compute the statistic (default: mean).
        n_bootstrap: Number of bootstrap resamples (>= 1000).
        confidence: Confidence level (default: 0.95 for 95% CI).
        seed: Random seed for reproducibility.

    Returns:
        (point_estimate, ci_lower, ci_upper)

    Example:
        costs = np.array([0.001, 0.002, 0.0, 0.0, 0.003, ...])
        mean_cost, lo, hi = bootstrap_ci(costs)
        print(f"Mean cost: ${mean_cost:.4f} (95% CI: ${lo:.4f}–${hi:.4f})")
    """
    rng = np.random.RandomState(seed)
    n = len(data)
    point_estimate = float(statistic(data))

    # Generate bootstrap resamples
    bootstrap_stats = np.array([
        statistic(data[rng.randint(0, n, size=n)])
        for _ in range(n_bootstrap)
    ])

    # Percentile method
    alpha = 1 - confidence
    ci_lower = float(np.percentile(bootstrap_stats, 100 * alpha / 2))
    ci_upper = float(np.percentile(bootstrap_stats, 100 * (1 - alpha / 2)))

    return point_estimate, ci_lower, ci_upper


def bootstrap_cost_saved(
    system_costs: np.ndarray,
    baseline_costs: np.ndarray,
    n_bootstrap: int = 1000,
    seed: int = 42,
) -> tuple[float, float, float]:
    """
    Bootstrap CI for cost-saved percentage.

    Returns (point_estimate, ci_lower, ci_upper) as percentages.
    """
    def savings_stat(indices):
        sys = system_costs[indices].sum()
        base = baseline_costs[indices].sum()
        if base == 0:
            return 0.0
        return (1 - sys / base) * 100

    rng = np.random.RandomState(seed)
    n = len(system_costs)
    point = savings_stat(np.arange(n))

    bootstraps = []
    for _ in range(n_bootstrap):
        idx = rng.randint(0, n, size=n)
        bootstraps.append(savings_stat(idx))

    bootstraps = np.array(bootstraps)
    return point, float(np.percentile(bootstraps, 2.5)), float(np.percentile(bootstraps, 97.5))


def bootstrap_wrong_hit_rate(
    is_wrong: np.ndarray,    # boolean array: True = wrong hit
    n_bootstrap: int = 1000,
    seed: int = 42,
) -> tuple[float, float, float]:
    """
    Bootstrap CI for wrong-hit rate.

    Returns (point_estimate, ci_lower, ci_upper) as fractions.
    """
    return bootstrap_ci(is_wrong.astype(float), np.mean, n_bootstrap, seed=seed)