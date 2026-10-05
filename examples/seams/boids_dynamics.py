"""Shared teacher dynamics and three-weight DSL model for Boids."""

from typing import NamedTuple

import torch
from torch import nn

from diffield import AggregateContext, gather_avg, gather_sum, scatter
from diffield.sim import bounce_in_box, limit_speed, normalize_vectors

TEACHER = (0.05, 0.9, 0.35)
FIXED = (0.02, 0.08, 1.1)
RADIUS, SEPARATION, SPEED, DAMPING = 0.3, 0.08, 0.02, 0.94


class BoidsStep(NamedTuple):
    positions: torch.Tensor
    velocities: torch.Tensor
    preclip: torch.Tensor


def radius_edges(positions):
    """Directed radius graph, without self loops (including coincident neighbours)."""
    adjacency = torch.cdist(positions.detach(), positions.detach()) < RADIUS
    adjacency.fill_diagonal_(False)
    return adjacency.nonzero().T.contiguous()


def integrate(positions, velocities, acceleration):
    """Shared dt=1 physics; labels and teacher weights are never inputs here."""
    preclip = DAMPING * velocities + acceleration
    clipped = limit_speed(preclip, SPEED)
    positions, clipped = bounce_in_box(positions + clipped, clipped)
    return BoidsStep(positions, clipped, preclip)


def dense_step(positions, velocities, weights=TEACHER):
    """Independent oracle, also usable in float64 for numerical diagnostics."""
    adjacency = torch.cdist(positions.detach(), positions.detach()) < RADIUS
    adjacency.fill_diagonal_(False)
    degree = adjacency.sum(-1).clamp_min(1).unsqueeze(-1)
    matrix = adjacency.to(positions.dtype)
    alignment = matrix @ velocities / degree - velocities
    cohesion = matrix @ positions / degree - positions
    delta = positions[:, None, :] - positions[None, :, :]
    distance = delta.norm(dim=-1)
    close = adjacency & (distance > 0) & (distance <= SEPARATION)
    separation = normalize_vectors((delta * close.unsqueeze(-1)).sum(1))
    acceleration = weights[0] * separation + weights[1] * alignment + weights[2] * cohesion
    return integrate(positions, velocities, acceleration)


class Boids(nn.Module):
    def __init__(self):
        super().__init__()
        self.log_weights = nn.Parameter(torch.tensor(FIXED).log())

    def acceleration(self, pos, velocity):
        """The original three-weight field program, within an aggregate round."""
        alignment = gather_avg(scatter(velocity)) - velocity
        cohesion = gather_avg(scatter(pos)) - pos
        delta = pos - scatter(pos)
        distance = delta.norm(dim=-1)
        mask = ((distance > 0) & (distance <= SEPARATION)).pointwise()
        separation = normalize_vectors(gather_sum(delta * mask))
        weights = self.log_weights.exp()
        return weights[0] * separation + weights[1] * alignment + weights[2] * cohesion

    def step(self, positions, velocities, edge_index):
        context = AggregateContext(edge_index, len(positions))
        with context.round():
            acceleration = self.acceleration(positions, velocities)
        return integrate(positions, velocities, acceleration)
