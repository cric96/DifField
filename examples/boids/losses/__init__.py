"""Boids loss functions: trajectory and separation metrics."""

from .trajectory import trajectory_loss_components
from .separation import close_pair_distance_loss

__all__ = ["trajectory_loss_components", "close_pair_distance_loss"]
