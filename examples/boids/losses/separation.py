"""Distance-based losses focusing on close boid pairs."""

from __future__ import annotations

import torch


def close_pair_distance_loss(
    pred_pos_seq: torch.Tensor,
    teacher_pos_seq: torch.Tensor,
    *,
    sep: float,
    focus_scale: float = 1.5,
    sharpness: float = 24.0,
) -> torch.Tensor:
    """Compute MSE loss on pairwise distances, weighted toward close boids."""
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
        mask = 1.0 - torch.eye(
            pred_pos.shape[0], device=pred_pos.device, dtype=pred_pos.dtype
        )
        weight = near_weight * mask
        losses.append(
            (((pred_dist - teacher_dist).pow(2)) * weight).sum()
            / weight.sum().clamp_min(1e-6)
        )
    return torch.stack(losses).mean()
