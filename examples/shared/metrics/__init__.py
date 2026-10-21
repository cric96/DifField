"""Shared metrics layer: statistics and recovery after events."""

from .recovery import event_windows, recovery, recovery_summary, relative_deviation
from .stats import (
    aggregate,
    aggregate_curve,
    bootstrap_band,
    ci95,
    interval,
    interval_cell,
    mean,
    std,
)

__all__ = [
    "aggregate",
    "aggregate_curve",
    "bootstrap_band",
    "ci95",
    "event_windows",
    "interval",
    "interval_cell",
    "mean",
    "recovery",
    "recovery_summary",
    "relative_deviation",
    "std",
]
