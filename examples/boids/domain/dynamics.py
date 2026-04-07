"""Boids simulation dynamics: velocity updates and rollouts."""

from __future__ import annotations

from typing import Callable

import torch
from autofield import bounce_in_box, limit_speed, nbr, rep
from autofield.dsl import AggregateContext

from .geometry import hard_separation_force
from .graph import edge_connectivity_stats, graph_health_summary
from .scenario import build_scenario


VelocityStep = Callable[[torch.Tensor, torch.Tensor], tuple[torch.Tensor, torch.Tensor]]


def reference_boids_velocity_step(
    vel: torch.Tensor,
    pos: torch.Tensor,
    *,
    dt: float,
    sep: float,
    w_sep: float | torch.Tensor,
    w_align: float | torch.Tensor,
    w_cohesion: float | torch.Tensor,
    damping: float | torch.Tensor,
    max_speed: float | torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute pre-clipped and clipped velocity for the standard boids model."""
    neigh_vel = nbr(vel, aggr="mean")
    neigh_pos = nbr(pos, aggr="mean")
    align_force = neigh_vel - vel
    cohesion_force = neigh_pos - pos
    sep_force = hard_separation_force(pos, sep)
    acc = w_sep * sep_force + w_align * align_force + w_cohesion * cohesion_force
    pre_clip_vel = damping * vel + dt * acc
    return pre_clip_vel, limit_speed(pre_clip_vel, max_speed)


def reference_boids_velocity_update(
    vel: torch.Tensor,
    pos: torch.Tensor,
    *,
    dt: float,
    sep: float,
    w_sep: float | torch.Tensor,
    w_align: float | torch.Tensor,
    w_cohesion: float | torch.Tensor,
    damping: float | torch.Tensor,
    max_speed: float | torch.Tensor,
) -> torch.Tensor:
    """Wrapper returning only the clipped velocity."""
    pre_clip_vel, _ = reference_boids_velocity_step(
        vel,
        pos,
        dt=dt,
        sep=sep,
        w_sep=w_sep,
        w_align=w_align,
        w_cohesion=w_cohesion,
        damping=damping,
        max_speed=max_speed,
    )
    return limit_speed(pre_clip_vel, max_speed)


def step_boids_dynamics(
    *,
    positions: torch.Tensor,
    velocities: torch.Tensor,
    radius: float,
    init_connectivity: str,
    init_k_neighbors: int,
    init_min_degree: int,
    dt: float,
    use_init_connectivity: bool,
    velocity_step: VelocityStep,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Single step of boids dynamics with dynamic topology."""
    scenario = build_scenario(
        positions,
        radius=radius,
        init_connectivity=init_connectivity,
        init_k_neighbors=init_k_neighbors,
        init_min_degree=init_min_degree,
        ensure_init_connected=use_init_connectivity,
    )
    ctx = AggregateContext(
        scenario.edge_index, scenario.num_nodes, edge_weight=scenario.edge_weight
    )
    scenario.sync_context(ctx._ctx)
    pos_t = scenario.positions
    pre_clip_holder: list[torch.Tensor] = []

    def step_velocity_update(prev: torch.Tensor) -> torch.Tensor:
        pre_clip_vel, clipped_vel = velocity_step(prev, pos_t)
        pre_clip_holder.append(
            torch.nan_to_num(pre_clip_vel, nan=0.0, posinf=0.0, neginf=0.0)
        )
        return torch.nan_to_num(clipped_vel, nan=0.0, posinf=0.0, neginf=0.0)

    with ctx.round():
        vel = rep(velocities, step_velocity_update, name="vel")

    new_pos = pos_t + dt * vel
    new_pos, vel_bounced = bounce_in_box(new_pos, vel)
    return new_pos, vel_bounced, pre_clip_holder[0]


def rollout_with_dynamic_topology(
    rounds: int,
    *,
    positions0: torch.Tensor,
    velocities0: torch.Tensor,
    radius: float,
    init_connectivity: str,
    init_k_neighbors: int,
    init_min_degree: int,
    dt: float,
    max_speed: float | torch.Tensor,
    trunc_window: int | None,
    velocity_step: VelocityStep,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    dict[str, float],
    dict[str, float],
    dict[str, float],
]:
    """Multi-round boids simulation with topology refresh at every step."""
    positions = positions0.clone()
    init_vel = velocities0.clone()
    scenario = build_scenario(
        positions,
        radius=radius,
        init_connectivity=init_connectivity,
        init_k_neighbors=init_k_neighbors,
        init_min_degree=init_min_degree,
        ensure_init_connected=True,
    )
    num_edges, min_degree, num_components = edge_connectivity_stats(
        scenario.edge_index, scenario.num_nodes
    )
    init_graph_stats = {
        "num_edges": num_edges,
        "min_degree": min_degree,
        "num_components": num_components,
    }
    ctx = AggregateContext(
        scenario.edge_index, scenario.num_nodes, edge_weight=scenario.edge_weight
    )
    scenario.sync_context(ctx._ctx)
    positions_seq = []
    velocities_seq = []
    graph_samples = []
    pre_clip_speeds: list[float] = []
    cap_fractions: list[float] = []

    for round_idx in range(rounds):
        if (
            trunc_window is not None
            and trunc_window > 0
            and round_idx > 0
            and round_idx % trunc_window == 0
        ):
            scenario.update_positions(
                scenario.positions.detach(), refresh_topology=False
            )
            detached_vel = ctx._ctx.state.get_or_init(init_vel, name="vel").detach()
            ctx._ctx.state.update(detached_vel, name="vel")

        scenario.sync_context(ctx._ctx)
        pos_t = scenario.positions

        def rollout_velocity_update(prev: torch.Tensor) -> torch.Tensor:
            pre_clip_vel, clipped_vel = velocity_step(prev, pos_t)
            resolved_pre_clip = torch.nan_to_num(
                pre_clip_vel, nan=0.0, posinf=0.0, neginf=0.0
            )
            resolved_clipped = torch.nan_to_num(
                clipped_vel, nan=0.0, posinf=0.0, neginf=0.0
            )
            pre_clip_speed = resolved_pre_clip.norm(dim=-1)
            cap_value = (
                max_speed.to(pre_clip_speed)
                if isinstance(max_speed, torch.Tensor)
                else pre_clip_speed.new_tensor(max_speed)
            )
            pre_clip_speeds.append(float(pre_clip_speed.mean().detach().item()))
            cap_fractions.append(
                float((pre_clip_speed > cap_value).float().mean().detach().item())
            )
            return resolved_clipped

        with ctx.round():
            vel = rep(init_vel, rollout_velocity_update, name="vel")

        new_pos = pos_t + dt * vel
        new_pos, vel_bounced = bounce_in_box(new_pos, vel)
        scenario.update_positions(new_pos, refresh_topology=True)
        ctx._ctx.state.update(vel_bounced, name="vel")

        positions_seq.append(scenario.positions)
        velocities_seq.append(vel_bounced)
        graph_samples.append(
            edge_connectivity_stats(scenario.edge_index, scenario.num_nodes)
        )

    graph_health = graph_health_summary(graph_samples)
    speed_health = {
        "mean_pre_clip_speed": float(
            sum(pre_clip_speeds) / max(1, len(pre_clip_speeds))
        ),
        "mean_cap_fraction": float(sum(cap_fractions) / max(1, len(cap_fractions))),
    }
    return (
        torch.stack(positions_seq, dim=0),
        torch.stack(velocities_seq, dim=0),
        scenario.positions,
        init_graph_stats,
        graph_health,
        speed_health,
    )
