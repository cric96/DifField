"""Teacher rollout generation for boids imitation learning."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from .dynamics import reference_boids_velocity_step, rollout_with_dynamic_topology

if TYPE_CHECKING:
    from ..domain.specs import ModelSpec, SimulationSpec, TeacherDynamics


@torch.no_grad()
def teacher_rollout(
    positions0: torch.Tensor,
    velocities0: torch.Tensor,
    rounds: int,
    radius: float,
    init_connectivity: str,
    init_k_neighbors: int,
    init_min_degree: int,
    sep: float,
    dt: float,
    damping: float,
    max_speed: float,
    w_sep: float = 1.4,
    w_align: float = 0.8,
    w_cohesion: float = 0.6,
    return_preclip: bool = False,
) -> (
    tuple[torch.Tensor, torch.Tensor] | tuple[torch.Tensor, torch.Tensor, torch.Tensor]
):
    """Execute a boids simulation using teacher (ground truth) parameters."""
    pre_clip_seq: list[torch.Tensor] = []

    def teacher_velocity(
        prev: torch.Tensor, pos: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        pre_clip_vel, clipped_vel = reference_boids_velocity_step(
            prev,
            pos,
            dt=dt,
            sep=sep,
            w_sep=w_sep,
            w_align=w_align,
            w_cohesion=w_cohesion,
            damping=damping,
            max_speed=max_speed,
        )
        pre_clip_seq.append(
            torch.nan_to_num(pre_clip_vel, nan=0.0, posinf=0.0, neginf=0.0)
        )
        return pre_clip_vel, clipped_vel

    pos_seq, vel_seq, _, _, _, _ = rollout_with_dynamic_topology(
        rounds,
        positions0=positions0,
        velocities0=velocities0,
        radius=radius,
        init_connectivity=init_connectivity,
        init_k_neighbors=init_k_neighbors,
        init_min_degree=init_min_degree,
        dt=dt,
        max_speed=max_speed,
        trunc_window=None,
        velocity_step=teacher_velocity,
    )
    if return_preclip:
        return pos_seq, vel_seq, torch.stack(pre_clip_seq, dim=0)
    return pos_seq, vel_seq


@torch.no_grad()
def teacher_rollout_from_specs(
    *,
    positions0: torch.Tensor,
    velocities0: torch.Tensor | None = None,
    rounds: int | None = None,
    simulation: "SimulationSpec",
    teacher: "TeacherDynamics",
    model: "ModelSpec",
    return_preclip: bool = False,
) -> (
    tuple[torch.Tensor, torch.Tensor] | tuple[torch.Tensor, torch.Tensor, torch.Tensor]
):
    """Execute a teacher rollout based on high-level experiment specifications."""
    return teacher_rollout(
        positions0=positions0,
        velocities0=(
            torch.zeros_like(positions0) if velocities0 is None else velocities0
        ),
        rounds=simulation.rounds if rounds is None else rounds,
        radius=simulation.radius,
        init_connectivity=model.init_connectivity,
        init_k_neighbors=model.init_k_neighbors,
        init_min_degree=model.init_min_degree,
        sep=simulation.sep,
        dt=simulation.dt,
        damping=simulation.damping,
        max_speed=simulation.max_speed,
        w_sep=teacher.w_sep,
        w_align=teacher.w_align,
        w_cohesion=teacher.w_cohesion,
        return_preclip=return_preclip,
    )
