"""Shared helpers for numeric metrics."""

from __future__ import annotations

import math


def is_finite_number(value: object) -> bool:
    """Check if a value is a finite float or integer."""
    try:
        fval = float(value)  # type: ignore[arg-type]
        return math.isfinite(fval)
    except (TypeError, ValueError):
        return False


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
