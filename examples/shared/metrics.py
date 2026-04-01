"""Shared helpers for numeric metrics and nested summary access."""

from __future__ import annotations

import math
from typing import Any


def is_finite_number(value: object) -> bool:
    return isinstance(value, (int, float)) and math.isfinite(float(value))


def mean(values: list[float]) -> float:
    return sum(values) / max(1, len(values))


def std(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    mean_value = mean(values)
    return (sum((value - mean_value) ** 2 for value in values) / (len(values) - 1)) ** 0.5


def nested_get(payload: dict[str, Any], path: str, default: Any = None) -> Any:
    """Read nested values using dot notation, e.g. training.final_traj_loss."""
    current: Any = payload
    for part in path.split("."):
        if not isinstance(current, dict) or part not in current:
            return default
        current = current[part]
    return current
