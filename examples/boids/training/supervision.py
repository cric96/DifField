"""Supervision logic for boids imitation learning."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
import torch.nn.functional as F

from ..losses.separation import close_pair_distance_loss

if TYPE_CHECKING:
    from ..model.boids_model import LearnableAggregateBoids
    from .trace import BoidsTrace


def teacher_forced_step_losses(
    model: "LearnableAggregateBoids",
    *,
    trace: "BoidsTrace",
    velocity_loss_weight: float,
    sep: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute teacher-forced losses by stepping the model from teacher states."""
    total_loss = torch.zeros((), device=trace.positions0.device)
    total_pos_loss = torch.zeros((), device=trace.positions0.device)
    total_vel_loss = torch.zeros((), device=trace.positions0.device)
    total_sep_focus_loss = torch.zeros((), device=trace.positions0.device)

    state_pos = trace.positions0
    state_vel = trace.velocities0
    num_steps = int(trace.pos_seq.shape[0])

    for step_idx, (target_pos, target_vel, target_preclip_vel) in enumerate(
        zip(trace.pos_seq, trace.vel_seq, trace.preclip_vel_seq)
    ):
        pred_pos, pred_vel, pred_preclip_vel = model.step(
            positions=state_pos,
            velocities=state_vel,
            use_init_connectivity=step_idx == 0,
        )
        pos_loss = F.mse_loss(pred_pos, target_pos)
        vel_loss = F.mse_loss(pred_preclip_vel, target_preclip_vel)
        loss = pos_loss + velocity_loss_weight * vel_loss

        # Separation focus loss
        sep_focus_loss = close_pair_distance_loss(
            pred_pos.unsqueeze(0),
            target_pos.unsqueeze(0),
            sep=sep,
        )

        total_loss = total_loss + loss
        total_pos_loss = total_pos_loss + pos_loss
        total_vel_loss = total_vel_loss + vel_loss
        total_sep_focus_loss = total_sep_focus_loss + sep_focus_loss

        # Teacher forcing: next step starts from teacher state
        state_pos = target_pos
        state_vel = target_vel

    scale = 1.0 / max(1, num_steps)
    return (
        total_loss * scale,
        total_pos_loss * scale,
        total_vel_loss * scale,
        total_sep_focus_loss * scale,
    )
