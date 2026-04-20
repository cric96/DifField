"""Comprehensive summary generation for boids training runs."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, TYPE_CHECKING

from .recovery import extract_learned_parameters, extract_teacher_parameters_from_spec

if TYPE_CHECKING:
    from ..domain.specs import LearnableBoidsSpec


@dataclass
class BoidsSummaryBuilder:
    """Builder pattern for nested summary payload generation."""

    run_name: str
    spec: "LearnableBoidsSpec"
    history: dict[str, list[float]]
    model: Any
    effective_curriculum_max_horizon: int
    recovery: dict[str, dict[str, float]]

    def build(self) -> dict[str, Any]:
        """Construct the full nested summary dictionary."""
        return {
            "run": self._run_payload(),
            "configuration": self._configuration_payload(),
            "training": self._training_payload(),
            "validation": self._validation_payload(),
            "parameters": {
                "teacher": extract_teacher_parameters_from_spec(self.spec),
                "learned": extract_learned_parameters(self.history),
                "recovery": self.recovery,
            },
            "diagnostics": self._diagnostics_payload(),
        }

    def _run_payload(self) -> dict[str, Any]:
        return {
            "run_name": self.run_name,
            "seed": self.spec.seed,
            "epochs": self.spec.training.epochs,
            "rounds": self.spec.simulation.rounds,
            "num_nodes": self.spec.simulation.num_nodes,
        }

    def _configuration_payload(self) -> dict[str, Any]:
        replay_trace_dir = self.spec.training.replay_trace_dir
        return {
            "training_objective": "trace_teacher_forced_preclip_stepwise",
            "supervision_mode": self.spec.training.supervision_mode,
            "replay_trace_dir": (
                None if replay_trace_dir is None else str(replay_trace_dir)
            ),
            "save_replay_traces": self.spec.training.save_replay_traces,
            "init_w_sep_target": self.spec.model.init_w_sep_target,
            "init_w_align_target": self.spec.model.init_w_align_target,
            "init_w_cohesion_target": self.spec.model.init_w_cohesion_target,
            "damping": self.spec.simulation.damping,
            "max_speed": self.spec.simulation.max_speed,
            "init_connectivity": self.spec.model.init_connectivity,
            "init_k_neighbors": self.spec.model.init_k_neighbors,
            "init_min_degree": self.spec.model.init_min_degree,
            "init_velocity_scale": self.spec.simulation.init_velocity_scale,
            "curriculum_min_horizon": self.spec.training.min_horizon,
            "curriculum_max_horizon": self.effective_curriculum_max_horizon,
            "curriculum_ramp_fraction": self.spec.training.curriculum_ramp_fraction,
            "final_lr_ratio": self.spec.training.final_lr_ratio,
            "trunc_window": self.spec.training.trunc_window,
            "num_initial_conditions": self.spec.training.num_initial_conditions,
            "velocity_loss_weight": self.spec.training.velocity_loss_weight,
            "separation_loss_weight": self.spec.training.separation_loss_weight,
        }

    def _training_payload(self) -> dict[str, float | bool]:
        return {
            "final_total_loss": self.history["total"][-1],
            "final_pos_loss": self.history["pos_loss"][-1],
            "final_vel_loss": self.history["vel_loss"][-1],
            "final_per_step_loss": self.history["per_step_loss"][-1],
            "final_objective_loss": self.history["objective_loss"][-1],
            "final_sep_focus_loss": self.history["sep_focus_loss"][-1],
            "final_center_error": self.history["center_error"][-1],
            "best_pos_loss": min(self.history["pos_loss"]),
            "best_vel_loss": min(self.history["vel_loss"]),
            "best_total_loss": min(self.history["total"]),
            "best_per_step_loss": min(self.history["per_step_loss"]),
            "best_objective_loss": min(self.history["objective_loss"]),
            "has_nan": bool(any(value != value for value in self.history["total"])),
        }

    def _validation_payload(self) -> dict[str, Any]:
        return {
            "curriculum_horizon": {
                "rounds": self.history["val_curriculum_horizon"][-1],
                "final_total_loss": self.history["val_curriculum_total_loss"][-1],
                "final_pos_loss": self.history["val_curriculum_pos_loss"][-1],
                "final_vel_loss": self.history["val_curriculum_vel_loss"][-1],
                "final_per_step_loss": self.history["val_curriculum_per_step_loss"][-1],
                "final_center_error": self.history["val_curriculum_center_error"][-1],
            },
            "full_horizon": {
                "rounds": self.history["val_full_horizon"][-1],
                "final_total_loss": self.history["val_full_total_loss"][-1],
                "final_pos_loss": self.history["val_full_pos_loss"][-1],
                "final_vel_loss": self.history["val_full_vel_loss"][-1],
                "final_per_step_loss": self.history["val_full_per_step_loss"][-1],
                "final_center_error": self.history["val_full_center_error"][-1],
            },
            "final_val_total_loss": self.history["val_full_total_loss"][-1],
            "final_val_pos_loss": self.history["val_full_pos_loss"][-1],
            "final_val_vel_loss": self.history["val_full_vel_loss"][-1],
            "final_val_center_error": self.history["val_full_center_error"][-1],
        }

    def _diagnostics_payload(self) -> dict[str, Any]:
        return {
            "initial_graph": {
                "num_edges": self.model.last_init_graph_stats.get(
                    "num_edges", float("nan")
                ),
                "min_degree": self.model.last_init_graph_stats.get(
                    "min_degree", float("nan")
                ),
                "num_components": self.model.last_init_graph_stats.get(
                    "num_components", float("nan")
                ),
            },
            "rollout_graph": {
                "mean_num_edges": self.model.last_rollout_graph_health.get(
                    "mean_num_edges", float("nan")
                ),
                "mean_min_degree": self.model.last_rollout_graph_health.get(
                    "mean_min_degree", float("nan")
                ),
                "max_num_components": self.model.last_rollout_graph_health.get(
                    "max_num_components", float("nan")
                ),
            },
            "rollout_speed": {
                "mean_pre_clip_speed": self.model.last_rollout_speed_health.get(
                    "mean_pre_clip_speed", float("nan")
                ),
                "mean_cap_fraction": self.model.last_rollout_speed_health.get(
                    "mean_cap_fraction", float("nan")
                ),
            },
        }
