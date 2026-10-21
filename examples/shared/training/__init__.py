"""Shared training layer: checkpoints and utilities."""

from .loop import frozen
from .utils import parse_int_csv

__all__ = ["frozen", "parse_int_csv"]
