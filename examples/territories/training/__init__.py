"""Territories training layer: optimization loop and metrics."""

from .loop import TerritoriesTrainingLoop
from .metrics import summary_metrics

__all__ = ["TerritoriesTrainingLoop", "summary_metrics"]
