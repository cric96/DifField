"""Shared metrics layer: statistics and nested access."""

from .nested import nested_get
from .stats import is_finite_number, mean, std

__all__ = ["is_finite_number", "mean", "nested_get", "std"]
