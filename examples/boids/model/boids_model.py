"""PyTorch module for learnable aggregate boids."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
import torch.nn as nn

from ..domain.dynamics import (
    reference_boids_velocity_step,
    rollout_with_dynamic_topology,
    step_boids_dynamics,
)
from .parameterization import (
    _softplus_param,
    _inverse_softplus_target,
)

if TYPE_CHECKING:
    from ..domain.specs import ModelSpec, SimulationSpec


class LearnableAggregateBoids(nn.Module):
    """Learnable boids model parameterizing separation, alignment, and cohesion."""

    def __init__(
        self,
        positions0: torch.Tensor,
        radius: float,
        sep: float,
        dt: float,
        damping: float,
        max_speed: float,
        init_connectivity: str = "hybrid",
        init_k_neighbors: int = 8,
        init_min_degree: int = 2,
        init_w_sep_target: float = 0.10,
        init_w_align_target: float = 2.40,
        init_w_cohesion_target: float = 0.08,
    ):
        super().__init__()
        self.positions0 = positions0
        self.radius = radius
        self.sep = sep
        self.dt = dt
        self.damping = damping
        self.max_speed = max_speed
        self.init_connectivity = init_connectivity
        self.init_k_neighbors = init_k_neighbors
        self.init_min_degree = init_min_degree

        self.w_sep_raw = nn.Parameter(
            torch.tensor(_inverse_softplus_target(init_w_sep_target))
        )
        self.w_align_raw = nn.Parameter(
            torch.tensor(_inverse_softplus_target(init_w_align_target))
        )
        self.w_cohesion_raw = nn.Parameter(
            torch.tensor(_inverse_softplus_target(init_w_cohesion_target))
        )

        self.last_init_graph_stats = {}
        self.last_rollout_graph_health = {}
        self.last_rollout_speed_health = {}

    @classmethod
    def from_specs(
        cls,
        *,
        positions0: torch.Tensor,
        simulation: "SimulationSpec",
        model: "ModelSpec",
    ) -> "LearnableAggregateBoids":
        """Factory creating a model from high-level specifications."""
        return cls(
            positions0=positions0,
            radius=simulation.radius,
            sep=simulation.sep,
            dt=simulation.dt,
            damping=simulation.damping,
            max_speed=simulation.max_speed,
            init_connectivity=model.init_connectivity,
            init_k_neighbors=model.init_k_neighbors,
            init_min_degree=model.init_min_degree,
            init_w_sep_target=model.init_w_sep_target,
            init_w_align_target=model.init_w_align_target,
            init_w_cohesion_target=model.init_w_cohesion_target,
        )

    @property
    def w_sep(self) -> torch.Tensor:
        return _softplus_param(self.w_sep_raw)

    @property
    def w_align(self) -> torch.Tensor:
        return _softplus_param(self.w_align_raw)

    @property
    def w_cohesion(self) -> torch.Tensor:
        return _softplus_param(self.w_cohesion_raw)

    def trainable_parameters(self) -> list[nn.Parameter]:
        """Return parameters that should be optimized."""
        return [self.w_sep_raw, self.w_align_raw, self.w_cohesion_raw]

    def step(
        self,
        *,
        positions: torch.Tensor,
        velocities: torch.Tensor,
        use_init_connectivity: bool,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Single step of learnable boids dynamics."""
        return step_boids_dynamics(
            positions=positions,
            velocities=velocities,
            radius=self.radius,
            init_connectivity=self.init_connectivity,
            init_k_neighbors=self.init_k_neighbors,
            init_min_degree=self.init_min_degree,
            dt=self.dt,
            use_init_connectivity=use_init_connectivity,
            velocity_step=self._velocity_step,
        )

    def rollout(
        self,
        rounds: int,
        *,
        positions0: torch.Tensor | None = None,
        velocities0: torch.Tensor | None = None,
        trunc_window: int | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Full simulation rollout with learnable parameters."""
        base_positions = self.positions0 if positions0 is None else positions0
        init_vel = (
            torch.zeros_like(base_positions) if velocities0 is None else velocities0
        )

        pos_seq, vel_seq, final_pos, init_graph_stats, graph_health, speed_health = (
            rollout_with_dynamic_topology(
                rounds,
                positions0=base_positions,
                velocities0=init_vel,
                radius=self.radius,
                init_connectivity=self.init_connectivity,
                init_k_neighbors=self.init_k_neighbors,
                init_min_degree=self.init_min_degree,
                dt=self.dt,
                max_speed=self.max_speed,
                trunc_window=trunc_window,
                velocity_step=self._velocity_step,
            )
        )

        self.last_init_graph_stats = init_graph_stats
        self.last_rollout_graph_health = graph_health
        self.last_rollout_speed_health = speed_health

        return pos_seq, vel_seq, final_pos

    def _velocity_step(
        self, vel: torch.Tensor, pos: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Internal velocity step using current learnable parameters."""
        return reference_boids_velocity_step(
            vel,
            pos,
            dt=self.dt,
            sep=self.sep,
            w_sep=self.w_sep,
            w_align=self.w_align,
            w_cohesion=self.w_cohesion,
            damping=self.damping,
            max_speed=self.max_speed,
        )
