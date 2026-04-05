"""Model and simulation primitives for the active learnable boids workflow."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
import torch.nn as nn

from .logics import (
    reference_boids_velocity_step,
    rollout_with_dynamic_topology,
    step_boids_dynamics,
    teacher_rollout,
)

if TYPE_CHECKING:
    from .config import ModelSpec, SimulationSpec


__all__ = ["LearnableAggregateBoids", "teacher_rollout"]


def _softplus_param(raw: torch.Tensor, minimum: float = 1e-4) -> torch.Tensor:
    return torch.nn.functional.softplus(raw) + minimum


def _bounded_sigmoid(raw: torch.Tensor, low: float, high: float) -> torch.Tensor:
    return low + (high - low) * torch.sigmoid(raw)


def _inverse_sigmoid_target(value: float) -> float:
    clipped = min(max(float(value), 1e-4), 1.0 - 1e-4)
    return float(torch.logit(torch.tensor(clipped, dtype=torch.float32)).item())


def _inverse_softplus_target(value: float, minimum: float = 1e-4) -> float:
    adjusted = max(float(value) - minimum, 1e-6)
    return float(torch.log(torch.expm1(torch.tensor(adjusted, dtype=torch.float32))).item())


def _inverse_bounded_sigmoid_target(value: float, low: float, high: float) -> float:
    if high <= low:
        raise ValueError("max speed bounds must satisfy high > low")
    clipped = min(max(float(value), low + 1e-6), high - 1e-6)
    scaled = (clipped - low) / (high - low)
    return _inverse_sigmoid_target(scaled)


class LearnableAggregateBoids(nn.Module):
    def __init__(
        self,
        positions0: torch.Tensor,
        radius: float,
        sep: float,
        dt: float,
        init_connectivity: str = "hybrid",
        init_k_neighbors: int = 8,
        init_min_degree: int = 2,
        init_w_sep_target: float = 0.10,
        init_w_align_target: float = 2.40,
        init_w_cohesion_target: float = 0.08,
        init_damping_target: float = 0.55,
        init_max_speed_target: float = 0.05,
        max_speed_min: float = 0.002,
        max_speed_max: float = 0.06,
    ):
        super().__init__()
        self.positions0 = positions0
        self.radius = radius
        self.sep = sep
        self.dt = dt
        self.init_connectivity = init_connectivity
        self.init_k_neighbors = init_k_neighbors
        self.init_min_degree = init_min_degree
        self.last_init_graph_stats = {"num_edges": float("nan"), "min_degree": float("nan"), "num_components": float("nan")}
        self.last_rollout_graph_health = {"mean_num_edges": float("nan"), "mean_min_degree": float("nan"), "max_num_components": float("nan")}
        self.last_rollout_speed_health = {"mean_pre_clip_speed": float("nan"), "mean_cap_fraction": float("nan")}
        self.max_speed_min = float(max_speed_min)
        self.max_speed_max = float(max_speed_max)

        self.w_sep_raw = nn.Parameter(torch.tensor(_inverse_softplus_target(init_w_sep_target)))
        self.w_align_raw = nn.Parameter(torch.tensor(_inverse_softplus_target(init_w_align_target)))
        self.w_cohesion_raw = nn.Parameter(torch.tensor(_inverse_softplus_target(init_w_cohesion_target)))
        self.damping_raw = nn.Parameter(torch.tensor(_inverse_sigmoid_target(init_damping_target)))
        self.register_buffer(
            "max_speed_raw",
            torch.tensor(_inverse_bounded_sigmoid_target(init_max_speed_target, self.max_speed_min, self.max_speed_max)),
        )

    @classmethod
    def from_specs(
        cls,
        *,
        positions0: torch.Tensor,
        simulation: "SimulationSpec",
        model: "ModelSpec",
    ) -> "LearnableAggregateBoids":
        return cls(
            positions0=positions0,
            radius=simulation.radius,
            sep=simulation.sep,
            dt=simulation.dt,
            init_connectivity=model.init_connectivity,
            init_k_neighbors=model.init_k_neighbors,
            init_min_degree=model.init_min_degree,
            init_w_sep_target=model.init_w_sep_target,
            init_w_align_target=model.init_w_align_target,
            init_w_cohesion_target=model.init_w_cohesion_target,
            init_damping_target=model.init_damping_target,
            init_max_speed_target=model.init_max_speed_target,
            max_speed_min=model.max_speed_min,
            max_speed_max=model.max_speed_max,
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

    @property
    def damping(self) -> torch.Tensor:
        return torch.sigmoid(self.damping_raw)

    @property
    def max_speed(self) -> torch.Tensor:
        return _bounded_sigmoid(self.max_speed_raw, self.max_speed_min, self.max_speed_max)

    def trainable_parameters(self) -> list[nn.Parameter]:
        return [self.w_sep_raw, self.w_align_raw, self.w_cohesion_raw, self.damping_raw]

    def step(
        self,
        *,
        positions: torch.Tensor,
        velocities: torch.Tensor,
        use_init_connectivity: bool,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
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

    def _rollout_with_dynamic_topology(
        self,
        rounds: int,
        *,
        positions0: torch.Tensor,
        velocities0: torch.Tensor,
        trunc_window: int | None,
        velocity_step,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        pos_seq, vel_seq, final_pos, init_graph_stats, graph_health, speed_health = rollout_with_dynamic_topology(
            rounds,
            positions0=positions0,
            velocities0=velocities0,
            radius=self.radius,
            init_connectivity=self.init_connectivity,
            init_k_neighbors=self.init_k_neighbors,
            init_min_degree=self.init_min_degree,
            dt=self.dt,
            max_speed=self.max_speed,
            trunc_window=trunc_window,
            velocity_step=velocity_step,
        )
        self.last_init_graph_stats = init_graph_stats
        self.last_rollout_graph_health = graph_health
        self.last_rollout_speed_health = speed_health
        return pos_seq, vel_seq, final_pos

    def rollout(
        self,
        rounds: int,
        *,
        positions0: torch.Tensor | None = None,
        velocities0: torch.Tensor | None = None,
        trunc_window: int | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        base_positions = self.positions0 if positions0 is None else positions0
        init_vel = torch.zeros_like(base_positions) if velocities0 is None else velocities0
        return self._rollout_with_dynamic_topology(
            rounds,
            positions0=base_positions,
            velocities0=init_vel,
            trunc_window=trunc_window,
            velocity_step=self._velocity_step,
        )

    def _velocity_step(self, vel: torch.Tensor, pos: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        pre_clip_vel, clipped_vel = reference_boids_velocity_step(
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
        resolved_pre_clip = torch.nan_to_num(pre_clip_vel, nan=0.0, posinf=0.0, neginf=0.0)
        resolved_clipped = torch.nan_to_num(clipped_vel, nan=0.0, posinf=0.0, neginf=0.0)
        pre_clip_speed = resolved_pre_clip.norm(dim=-1)
        if hasattr(self, "_rollout_pre_clip_speeds"):
            self._rollout_pre_clip_speeds.append(float(pre_clip_speed.mean().detach().item()))
            self._rollout_cap_fractions.append(float((pre_clip_speed > self.max_speed).float().mean().detach().item()))
        return resolved_pre_clip, resolved_clipped
