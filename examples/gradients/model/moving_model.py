"""Learnable model for gradients in networks with moving nodes."""

from __future__ import annotations

import torch
from torch import nn

from diffield.dsl import AggregateContext, field, gather_min, iterate, mux, scatter
from diffield.sim import SpatialScenario

from ..domain.moving_logic import MAX_DIST


class MotionPolicy(nn.Module):
    """Neural network predicting acceleration for moving nodes."""

    def __init__(self, hidden: int = 32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(4, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
            nn.Linear(hidden, 2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Predict target velocities based on node features."""
        return self.net(x)


class LearnableMovingGradient(nn.Module):
    """Integrated model for learning both communication weights and motion policies."""

    def __init__(
        self, positions: torch.Tensor, radius: float, source_idx: int, learn_mode: str
    ):
        super().__init__()
        self.positions0 = positions
        self.radius = radius
        self.source_idx = source_idx
        self.learn_mode = learn_mode
        self.motion_policy = MotionPolicy()
        self.w_raw = nn.Parameter(torch.tensor(0.54))
        self.dt = 0.1
        self.max_speed = 0.04

    @property
    def w(self) -> torch.Tensor:
        """Learnable hop weight."""
        return torch.nn.functional.softplus(self.w_raw) + 1e-3

    def _clip_box(
        self, positions: torch.Tensor, velocities: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Handle bouncing off the unit box boundaries."""
        for dim in (0, 1):
            low = positions[:, dim] < 0.0
            high = positions[:, dim] > 1.0
            velocities[low | high, dim] *= -1.0
        return positions.clamp(0.0, 1.0), velocities

    def forward(self, rounds: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Execute a multi-round simulation of both communication and motion."""
        positions = self.positions0.clone()
        velocities = torch.zeros_like(positions)
        scenario = SpatialScenario(
            positions=positions, edge_radius=self.radius, device=positions.device
        )
        ctx = AggregateContext(
            scenario.edge_index, scenario.num_nodes, edge_weight=scenario.edge_weight
        )

        source = scenario.marker(self.source_idx)

        pred_seq = []
        for _ in range(rounds):
            scenario.sync_context(ctx._ctx)
            with ctx.round():
                dist = iterate(
                    field.of(MAX_DIST),
                    lambda dist_old: mux(
                        source, field.of(0.0), gather_min(scatter(dist_old + self.w))
                    ),
                    name="dist",
                )

            dist_feat = torch.nan_to_num(dist, nan=0.0, posinf=10.0, neginf=0.0)
            features = torch.cat(
                [
                    positions,
                    dist_feat.unsqueeze(-1),
                    velocities.norm(dim=1, keepdim=True),
                ],
                dim=1,
            )
            dv = self.motion_policy(features)
            new_vel = velocities + self.dt * dv
            speed = new_vel.norm(dim=1, keepdim=True).clamp_min(1e-8)
            new_vel = new_vel * torch.clamp(self.max_speed / speed, max=1.0)

            # Gradient blocking for specific learning modes
            if self.learn_mode == "ac":
                new_vel = new_vel.detach()
            elif self.learn_mode == "motion":
                dist = dist.detach()

            positions = positions + self.dt * new_vel
            positions, velocities = self._clip_box(positions, new_vel)
            scenario.update_positions(positions, refresh_topology=True)
            pred_seq.append(dist)

        return torch.stack(pred_seq, dim=0), positions
