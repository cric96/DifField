"""Shared metrics layer: statistics and nested access."""

from .stats import is_finite_number, mean, std
from .nested import nested_get

__all__ = ["is_finite_number", "mean", "nested_get", "std"]
