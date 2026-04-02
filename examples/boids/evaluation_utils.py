"""Evaluation utilities for learnable boids experiments."""

from __future__ import annotations

import torch
import torch.nn as nn

from .config import SimulationSpec, TeacherDynamics
from .model import LearnableAggregateBoids, teacher_rollout_from_specs


@torch.no_grad()
def evaluate_seed(
    model: LearnableAggregateBoids,
    *,
    seed: int,
    simulation: SimulationSpec,
    teacher: TeacherDynamics,
) -> tuple[float, float]:
    old_pos0 = model.positions0
    try:
        torch.manual_seed(seed)
        eval_positions0 = torch.rand(simulation.num_nodes, 2, device=model.w_sep_raw.device)
        teacher_pos_seq, _ = teacher_rollout_from_specs(
            positions0=eval_positions0,
            simulation=simulation,
            teacher=teacher,
        )
        model.positions0 = eval_positions0
        pred_pos_seq, _, final_pos = model.rollout(simulation.rounds)
        traj_loss = nn.functional.mse_loss(pred_pos_seq, teacher_pos_seq).item()
        center_error = (teacher_pos_seq[-1].mean(dim=0) - final_pos.mean(dim=0)).norm().item()
        return float(traj_loss), float(center_error)
    finally:
        model.positions0 = old_pos0
