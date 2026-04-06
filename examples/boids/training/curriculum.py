"""Curriculum learning schedules for boids training."""

from __future__ import annotations

import math


def _curriculum_ramp_epochs(total_epochs: int, ramp_fraction: float) -> int:
    """Compute how many epochs the curriculum ramp should last."""
    if total_epochs <= 1:
        return 1
    return max(1, int(round(total_epochs * ramp_fraction)))


def curriculum_horizon(
    epoch: int,
    total_epochs: int,
    min_horizon: int,
    max_horizon: int,
    ramp_fraction: float = 0.85,
) -> int:
    """Compute the current simulation horizon based on training progress."""
    if total_epochs <= 1:
        return max_horizon
    ramp_epochs = _curriculum_ramp_epochs(total_epochs, ramp_fraction)
    progress = min(1.0, epoch / ramp_epochs)
    return int(round(min_horizon + (max_horizon - min_horizon) * progress))


def scheduled_learning_rate(
    base_lr: float,
    epoch: int,
    total_epochs: int,
    *,
    ramp_fraction: float,
    final_lr_ratio: float,
) -> float:
    """Compute the learning rate using a constant phase followed by a cosine decay."""
    if total_epochs <= 1:
        return float(base_lr)
    ramp_epochs = _curriculum_ramp_epochs(total_epochs, ramp_fraction)
    if epoch <= ramp_epochs:
        return float(base_lr)
    tail_epochs = max(1, total_epochs - ramp_epochs - 1)
    progress = min(1.0, (epoch - ramp_epochs) / tail_epochs)
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return float(base_lr * (final_lr_ratio + (1.0 - final_lr_ratio) * cosine))
