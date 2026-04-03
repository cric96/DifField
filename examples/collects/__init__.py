"""Collect example family."""

from .large import main as large_main
from .small import main as small_main

__all__ = ["small_main", "large_main"]