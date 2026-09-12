"""Shared metrics layer: statistics and nested access."""

from .nested import nested_get
from .stats import aggregate, aggregate_curve, ci95, is_finite_number, mean, std

__all__ = [
    "aggregate",
    "aggregate_curve",
    "ci95",
    "is_finite_number",
    "mean",
    "nested_get",
    "std",
]
