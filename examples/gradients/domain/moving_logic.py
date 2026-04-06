"""Domain logic for moving nodes gradients."""

from __future__ import annotations

import torch

MAX_DIST = 100.0


def pairwise_shortest_hop(
    positions: torch.Tensor, source: int, radius: float
) -> torch.Tensor:
    """Compute ground truth hop distance in a moving network."""
    num_nodes = positions.shape[0]
    dist = torch.full(
        (num_nodes,), float("inf"), dtype=torch.float32, device=positions.device
    )
    dist[source] = 0.0

    pairwise_dist = torch.cdist(positions, positions)
    adj = (pairwise_dist <= radius) & (pairwise_dist > 0)

    frontier = torch.zeros(num_nodes, dtype=torch.bool, device=positions.device)
    frontier[source] = True
    for step in range(1, num_nodes + 1):
        neigh = (adj[frontier]).any(dim=0)
        newly = neigh & torch.isinf(dist)
        dist[newly] = float(step)
        if not newly.any():
            break
        frontier = newly

    return torch.nan_to_num(dist, nan=MAX_DIST, posinf=MAX_DIST, neginf=0.0)


def step_teacher_positions(
    pos: torch.Tensor, vel: torch.Tensor, dt: float = 0.1
) -> tuple[torch.Tensor, torch.Tensor]:
    """Update positions and handle boundary bounces for the teacher simulation."""
    new_pos = pos + dt * vel
    new_vel = vel.clone()
    for dim in (0, 1):
        low = new_pos[:, dim] < 0.0
        high = new_pos[:, dim] > 1.0
        new_vel[low | high, dim] *= -1.0
    return new_pos.clamp(0.0, 1.0), new_vel
