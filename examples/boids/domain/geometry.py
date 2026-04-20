"""Geometry primitives for boids simulation: separation forces and initial state sampling."""

from __future__ import annotations

import torch

from autofield import normalize_vectors


def hard_separation_force(positions: torch.Tensor, sep: float) -> torch.Tensor:
    from autofield import gather_sum, normalize_vectors, scatter

    delta = positions - scatter(positions)
    dist = delta.norm(dim=-1)
    sep_mask = ((dist <= sep) & (dist > 0)).pointwise()
    sep_force = gather_sum(delta * sep_mask)
    return normalize_vectors(sep_force)


def sample_initial_state(
    num_nodes: int,
    *,
    seed: int,
    velocity_scale: float,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    generator = torch.Generator()
    generator.manual_seed(seed)
    positions = torch.rand(num_nodes, 2, generator=generator, dtype=torch.float32).to(
        device
    )
    if velocity_scale <= 0.0:
        velocities = torch.zeros_like(positions)
    else:
        directions = (
            torch.rand(num_nodes, 2, generator=generator, dtype=torch.float32) * 2.0
            - 1.0
        ).to(device)
        velocities = normalize_vectors(directions) * velocity_scale
    return positions, velocities
