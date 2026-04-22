"""Reusable movement physics helpers for spatial simulations."""

from __future__ import annotations

import torch


def normalize_vectors(x: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """L2-normalize vectors along the last dimension."""
    return x / (x.norm(dim=-1, keepdim=True) + eps)


def limit_speed(velocities: torch.Tensor, max_speed: float | torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Clamp per-node speed without changing direction."""
    speed = velocities.norm(dim=-1, keepdim=True).clamp_min(eps)
    scale = torch.clamp(max_speed / speed, max=1.0)
    return velocities * scale


def bounce_in_box(
    positions: torch.Tensor,
    velocities: torch.Tensor,
    low: float = 0.0,
    high: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Reflect velocity on box boundaries and clamp positions inside the box."""
    for dim in (0, 1):
        lo = positions[:, dim] < low
        hi = positions[:, dim] > high
        velocities[lo | hi, dim] *= -1.0
    return positions.clamp(low, high), velocities


def boids_acceleration_dense(
    positions: torch.Tensor,
    velocities: torch.Tensor,
    *,
    radius: float,
    sep: float,
    w_sep: float | torch.Tensor,
    w_align: float | torch.Tensor,
    w_cohesion: float | torch.Tensor,
) -> torch.Tensor:
    """Compute classic boids acceleration from dense pairwise geometry."""
    distances = torch.cdist(positions, positions)
    neighbor_mask = (distances <= radius) & (distances > 0)
    sep_mask = (distances <= sep) & (distances > 0)

    delta = positions.unsqueeze(1) - positions.unsqueeze(0)
    sep_force = (delta * sep_mask.unsqueeze(-1)).sum(dim=1)

    n_count = neighbor_mask.sum(dim=1, keepdim=True).clamp_min(1)
    neighbor_vel_sum = (velocities.unsqueeze(0) * neighbor_mask.unsqueeze(-1)).sum(dim=1)
    neighbor_pos_sum = (positions.unsqueeze(0) * neighbor_mask.unsqueeze(-1)).sum(dim=1)

    align_force = neighbor_vel_sum / n_count - velocities
    cohesion_force = neighbor_pos_sum / n_count - positions

    return (
        w_sep * normalize_vectors(sep_force)
        + w_align * align_force
        + w_cohesion * cohesion_force
    )
