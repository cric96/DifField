"""Visualization utilities for gradients in moving networks."""

from __future__ import annotations

from typing import TYPE_CHECKING
import torch
from autofield import SpatialScenario, mux, nbr, rep
from autofield.dsl import AggregateContext, field
from ..domain.moving_logic import MAX_DIST

try:
    from shared.plotting import export_moving_gif
except ImportError:
    from ...shared.plotting import export_moving_gif

if TYPE_CHECKING:
    from ..model.moving_model import LearnableMovingGradient


@torch.no_grad()
def render_moving_gradient_evolution(
    model: "LearnableMovingGradient",
    rounds: int,
    output_path: str,
    title: str = "Moving Nodes Gradient Evolution",
    fps: int = 10,
) -> None:
    """Run a rollout and save a GIF of the moving network evolution."""
    positions = model.positions0.clone()
    velocities = torch.zeros_like(positions)
    scenario = SpatialScenario(
        positions=positions, edge_radius=model.radius, device=positions.device
    )
    ctx = AggregateContext(
        scenario.edge_index, scenario.num_nodes, edge_weight=scenario.edge_weight
    )
    source = torch.zeros(
        positions.shape[0], dtype=torch.float32, device=positions.device
    )
    source[model.source_idx] = 1.0

    pos_by_round = {}
    val_by_round = {}
    edge_index_by_round = {}

    for r in range(rounds):
        scenario.sync_context(ctx._ctx)
        with ctx.round():
            dist = rep(
                MAX_DIST,
                lambda dist_old: mux(
                    source, field.of(0.0), nbr(dist_old + model.w, aggr="min")
                ),
                name="dist",
            )

        pos_by_round[r] = positions.clone()
        val_by_round[r] = dist.clone()
        edge_index_by_round[r] = scenario.edge_index.clone()

        dist_feat = torch.nan_to_num(dist, nan=0.0, posinf=10.0, neginf=0.0)
        features = torch.cat(
            [positions, dist_feat.unsqueeze(-1), velocities.norm(dim=1, keepdim=True)],
            dim=1,
        )
        dv = model.motion_policy(features)
        new_vel = velocities + model.dt * dv
        speed = new_vel.norm(dim=1, keepdim=True).clamp_min(1e-8)
        new_vel = new_vel * torch.clamp(model.max_speed / speed, max=1.0)

        positions = positions + model.dt * new_vel
        positions, velocities = model._clip_box(positions, new_vel)
        scenario.update_positions(positions, refresh_topology=True)

    export_moving_gif(
        positions_by_round=pos_by_round,
        values_by_round=val_by_round,
        source_idx=model.source_idx,
        output_path=output_path,
        title=title,
        edge_index_by_round=edge_index_by_round,
        show_links=True,
        fps=fps,
    )
