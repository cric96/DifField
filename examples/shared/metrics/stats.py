"""Shared helpers for numeric metrics."""

from __future__ import annotations

import math


def is_finite_number(value: object) -> bool:
    """Check if a value is a finite float or integer."""
    try:
        fval = float(value)
        return math.isfinite(fval)
    except (TypeError, ValueError):
        return False


def mean(values: list[float]) -> float:
    """Compute the arithmetic mean of a list of numbers."""
    return sum(values) / max(1, len(values))


def std(values: list[float]) -> float:
    """Compute the sample standard deviation of a list of numbers."""
    if len(values) < 2:
        return 0.0
    mean_value = mean(values)
    return (
        sum((value - mean_value) ** 2 for value in values) / (len(values) - 1)
    ) ** 0.5
