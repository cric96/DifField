"""Loss functions for boids imitation and diagnostics."""

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
    pos_loss = torch.stack([
        F.mse_loss(pred_pos, teacher_pos)
        for pred_pos, teacher_pos in zip(pred_pos_seq, teacher_pos_seq)
    ]).mean()
    vel_loss = torch.stack([
        F.mse_loss(pred_vel, teacher_vel)
        for pred_vel, teacher_vel in zip(pred_vel_seq, teacher_vel_seq)
    ]).mean()
    total_loss = pos_loss + velocity_weight * vel_loss
    return total_loss, pos_loss, vel_loss


def close_pair_distance_loss(
    pred_pos_seq: torch.Tensor,
    teacher_pos_seq: torch.Tensor,
    *,
    sep: float,
    focus_scale: float = 1.5,
    sharpness: float = 24.0,
) -> torch.Tensor:
    if pred_pos_seq.numel() == 0:
        return pred_pos_seq.new_zeros(())

    focus_radius = focus_scale * sep
    losses = []
    for pred_pos, teacher_pos in zip(pred_pos_seq, teacher_pos_seq):
        pred_dist = torch.cdist(pred_pos, pred_pos)
        teacher_dist = torch.cdist(teacher_pos, teacher_pos)
        near_weight = torch.maximum(
            torch.sigmoid(sharpness * (focus_radius - pred_dist)),
            torch.sigmoid(sharpness * (focus_radius - teacher_dist)),
        )
        mask = 1.0 - torch.eye(pred_pos.shape[0], device=pred_pos.device, dtype=pred_pos.dtype)
        weight = near_weight * mask
        losses.append((((pred_dist - teacher_dist).pow(2)) * weight).sum() / weight.sum().clamp_min(1e-6))
    return torch.stack(losses).mean()