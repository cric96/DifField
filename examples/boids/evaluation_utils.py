"""Evaluation utilities for learnable boids experiments."""

from __future__ import annotations

import torch

from .config import ModelSpec, SimulationSpec, TeacherDynamics
from .model import LearnableAggregateBoids, sample_initial_boids_state, teacher_rollout_from_specs, trajectory_loss_components


@torch.no_grad()
def evaluate_seed(
    model: LearnableAggregateBoids,
    *,
    seed: int,
    simulation: SimulationSpec,
    teacher: TeacherDynamics,
    model_spec: ModelSpec,
    velocity_loss_weight: float,
) -> tuple[float, float, float, float]:
    eval_positions0, eval_velocities0 = sample_initial_boids_state(
        simulation.num_nodes,
        seed=seed,
        velocity_scale=simulation.init_velocity_scale,
        device=model.w_sep_raw.device,
    )
    teacher_pos_seq, teacher_vel_seq = teacher_rollout_from_specs(
        positions0=eval_positions0,
        velocities0=eval_velocities0,
        rounds=simulation.rounds,
        simulation=simulation,
        teacher=teacher,
        model=model_spec,
    )
    pred_pos_seq, pred_vel_seq, final_pos = model.rollout(
        simulation.rounds,
        positions0=eval_positions0,
        velocities0=eval_velocities0,
    )
    total_loss, pos_loss, vel_loss = trajectory_loss_components(
        pred_pos_seq,
        pred_vel_seq,
        teacher_pos_seq,
        teacher_vel_seq,
        velocity_weight=velocity_loss_weight,
    )
    center_error = (teacher_pos_seq[-1].mean(dim=0) - final_pos.mean(dim=0)).norm().item()
    return float(total_loss.item()), float(pos_loss.item()), float(vel_loss.item()), float(center_error)
