"""Trajectory-based loss functions for boids."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def trajectory_loss_components(
    pred_pos_seq: torch.Tensor,
    pred_vel_seq: torch.Tensor,
    teacher_pos_seq: torch.Tensor,
    teacher_vel_seq: torch.Tensor,
    *,
    velocity_weight: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute weighted sum of position and velocity MSE losses."""
    pos_loss = torch.stack(
        [
            F.mse_loss(pred_pos, teacher_pos)
            for pred_pos, teacher_pos in zip(pred_pos_seq, teacher_pos_seq)
        ]
    ).mean()
    vel_loss = torch.stack(
        [
            F.mse_loss(pred_vel, teacher_vel)
            for pred_vel, teacher_vel in zip(pred_vel_seq, teacher_vel_seq)
        ]
    ).mean()
    total_loss = pos_loss + velocity_weight * vel_loss
    return total_loss, pos_loss, vel_loss
