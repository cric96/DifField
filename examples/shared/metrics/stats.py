"""Shared helpers for numeric metrics."""

from __future__ import annotations

import math

import numpy as np


def mean(values: list[float]) -> float:
    """Compute the arithmetic mean of a list of numbers. Ignores NaNs."""
    valid_values = [v for v in values if not math.isnan(v)]
    if not valid_values:
        return float('nan')
    return sum(valid_values) / max(1, len(valid_values))


def std(values: list[float]) -> float:
    """Compute the sample standard deviation of a list of numbers. Ignores NaNs."""
    valid_values = [v for v in values if not math.isnan(v)]
    if len(valid_values) < 2:
        return 0.0
    mean_value = mean(valid_values)
    return (
        sum((value - mean_value) ** 2 for value in valid_values) / (len(valid_values) - 1)
    ) ** 0.5


# Two-sided Student-t critical values at the 0.975 quantile, indexed by degrees of freedom.
_T_CRIT_975 = {
    1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447, 7: 2.365,
    8: 2.306, 9: 2.262, 10: 2.228, 11: 2.201, 12: 2.179, 13: 2.160, 14: 2.145,
    15: 2.131, 16: 2.120, 17: 2.110, 18: 2.101, 19: 2.093, 20: 2.086,
    21: 2.080, 22: 2.074, 23: 2.069, 24: 2.064, 25: 2.060, 26: 2.056,
    27: 2.052, 28: 2.048, 29: 2.045, 30: 2.042,
}


def ci95(values: list[float]) -> float:
    """Half-width of the 95% confidence interval of the mean (Student-t). Ignores NaNs."""
    finite = [v for v in values if not math.isnan(v)]
    n = len(finite)
    if n < 2:
        return 0.0
    t_crit = _T_CRIT_975.get(n - 1, 1.96)
    return t_crit * std(finite) / math.sqrt(n)


def aggregate(values: list[float]) -> dict[str, float]:
    """Aggregate a list of scalars into a mean / ci / std / n record."""
    return {"mean": mean(values), "ci": ci95(values), "std": std(values), "n": len(values)}


def aggregate_curve(curves: list[list[float]]) -> tuple[list[float], list[float]]:
    """Aggregate equal-length curves into per-step (mean, 95% CI) lists.

    Curves of differing length are truncated to the shortest.
    """
    if not curves:
        return [], []
    length = min(len(c) for c in curves)
    means, cis = [], []
    for i in range(length):
        col = [c[i] for c in curves]
        means.append(mean(col))
        cis.append(ci95(col))
    return means, cis


def bootstrap_band(curves, samples=2000, seed=0):
    """Mean curve and its 95% percentile bootstrap band over the runs (rows of ``curves``)."""
    curves = np.asarray(curves, dtype=float)
    generator = np.random.default_rng(seed)
    draws = curves[generator.integers(len(curves), size=(samples, len(curves)))].mean(1)
    low, high = np.quantile(draws, [0.025, 0.975], axis=0)
    return curves.mean(0), low, high


def interval(rows, metric, *, reference=None, samples=2000, seed=0):
    """Two-way clustered bootstrap for the crossed seed x shared-episode design.

    Differences are formed before resampling; the same episode draw is used
    for every sampled seed. A single seed or an incomplete/nonfinite panel
    cannot support a seed-level confidence interval.
    """
    table = {(row["seed"], row["episode"]): row["metrics"].get(metric) for row in rows}
    if len(table) != len(rows):
        raise ValueError("Duplicate seed/episode in interval panel")
    if reference is not None:
        baseline = {(row["seed"], row["episode"]): row["metrics"].get(metric) for row in reference}
        if len(baseline) != len(reference):
            raise ValueError("Duplicate seed/episode in reference panel")
        table = {
            key: table[key] - baseline[key]
            if table.get(key) is not None and baseline.get(key) is not None
            else None
            for key in table.keys() | baseline.keys()
        }
    seeds, episodes = sorted({k[0] for k in table}), sorted({k[1] for k in table})
    finite = [value for value in table.values() if value is not None and math.isfinite(value)]
    invalid = len(table) - len(finite)
    result = {
        "mean": float(np.mean(finite)) if finite and not invalid else None,
        "finite_mean": float(np.mean(finite)) if finite else None,
        "ci95": None,
        "seeds": len(seeds),
        "episodes": len(episodes),
        "invalid": invalid,
        "n": len(table),
    }
    if invalid or len(table) != len(seeds) * len(episodes):
        result["ci_reason"] = "nonfinite_or_incomplete_panel"
    elif len(seeds) < 2 or len(episodes) < 2:
        result["ci_reason"] = "insufficient_independent_seeds_or_episodes"
    else:
        matrix = np.array([[table[s, e] for e in episodes] for s in seeds])
        generator = np.random.default_rng(seed)
        draws = []
        for _ in range(samples):
            si = generator.integers(len(seeds), size=len(seeds))
            ei = generator.integers(len(episodes), size=len(episodes))
            draws.append(matrix[np.ix_(si, ei)].mean())
        result["ci95"] = np.quantile(draws, [0.025, 0.975]).tolist()
        result["ci_reason"] = "two_way_cluster_bootstrap"
    return result
