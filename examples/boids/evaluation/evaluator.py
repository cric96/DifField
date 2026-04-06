"""Evaluation engine for learnable boids models."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from ..domain.geometry import sample_initial_state
from ..domain.teacher import teacher_rollout_from_specs
from ..losses.trajectory import trajectory_loss_components

if TYPE_CHECKING:
    from ..domain.specs import ModelSpec, SimulationSpec, TeacherDynamics
    from ..model.boids_model import LearnableAggregateBoids


@torch.no_grad()
def evaluate_seed(
    model: LearnableAggregateBoids,
    *,
    seed: int,
    simulation: SimulationSpec,
    teacher: TeacherDynamics,
    model_spec: ModelSpec,
    velocity_loss_weight: float,
    rounds: int | None = None,
) -> dict[str, float]:
    """Evaluate the model on a specific random seed against teacher dynamics."""
    eval_rounds = simulation.rounds if rounds is None else rounds
    eval_positions0, eval_velocities0 = sample_initial_state(
        simulation.num_nodes,
        seed=seed,
        velocity_scale=simulation.init_velocity_scale,
        device=model.w_sep_raw.device,
    )
    teacher_pos_seq, teacher_vel_seq = teacher_rollout_from_specs(
        positions0=eval_positions0,
        velocities0=eval_velocities0,
        rounds=eval_rounds,
        simulation=simulation,
        teacher=teacher,
        model=model_spec,
    )
    pred_pos_seq, pred_vel_seq, final_pos = model.rollout(
        eval_rounds,
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
    center_error = (
        (teacher_pos_seq[-1].mean(dim=0) - final_pos.mean(dim=0)).norm().item()
    )
    return {
        "horizon": float(eval_rounds),
        "total_loss": float(total_loss.item()),
        "pos_loss": float(pos_loss.item()),
        "vel_loss": float(vel_loss.item()),
        "per_step_loss": float(total_loss.item() / max(1, eval_rounds)),
        "center_error": float(center_error),
    }


def mean_eval_metrics(items: list[dict[str, float]]) -> dict[str, float]:
    """Compute the mean of metrics across multiple evaluation seeds."""
    if not items:
        return {
            "horizon": float("nan"),
            "total_loss": float("nan"),
            "pos_loss": float("nan"),
            "vel_loss": float("nan"),
            "per_step_loss": float("nan"),
            "center_error": float("nan"),
        }
    return {
        key: float(sum(item[key] for item in items) / len(items)) for key in items[0]
    }
