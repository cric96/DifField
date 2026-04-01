"""Reporting helpers for learnable boids experiments."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


def extract_teacher_parameters(args: Any) -> dict[str, float]:
    return {
        "w_sep": float(args.teacher_w_sep),
        "w_align": float(args.teacher_w_align),
        "w_cohesion": float(args.teacher_w_cohesion),
        "damping": float(args.teacher_damping),
        "max_speed": float(args.teacher_max_speed),
    }


def extract_learned_parameters(history: dict[str, list[float]]) -> dict[str, float]:
    return {
        "w_sep": history["w_sep"][-1],
        "w_align": history["w_align"][-1],
        "w_cohesion": history["w_cohesion"][-1],
        "damping": history["damping"][-1],
        "max_speed": history["max_speed"][-1],
        "tau_align": history["tau_align"][-1],
        "tau_cohesion": history["tau_cohesion"][-1],
    }


def compute_parameter_recovery_metrics(
    teacher_params: dict[str, float],
    learned_params: dict[str, float],
) -> dict[str, dict[str, float]]:
    recovery: dict[str, dict[str, float]] = {}
    for name, teacher_value in teacher_params.items():
        learned_value = learned_params[name]
        abs_err = abs(learned_value - teacher_value)
        recovery[name] = {
            "teacher": teacher_value,
            "learned": learned_value,
            "abs_error": abs_err,
            "rel_error": abs_err / max(abs(teacher_value), 1e-9),
        }
    return recovery


@dataclass
class BoidsSummaryBuilder:
    """Builder pattern for nested summary payload generation."""

    run_name: str
    args: Any
    history: dict[str, list[float]]
    model: Any
    train_max_speed: bool
    init_max_speed_target: float
    recovery: dict[str, dict[str, float]]

    def build(self) -> dict[str, Any]:
        return {
            "run": self._run_payload(),
            "configuration": self._configuration_payload(),
            "training": self._training_payload(),
            "validation": self._validation_payload(),
            "parameters": {
                "teacher": extract_teacher_parameters(self.args),
                "learned": extract_learned_parameters(self.history),
                "recovery": self.recovery,
            },
            "diagnostics": self._diagnostics_payload(),
        }

    def _run_payload(self) -> dict[str, Any]:
        return {
            "run_name": self.run_name,
            "mode": self.args.mode,
            "seed": self.args.seed,
            "epochs": self.args.epochs,
            "rounds": self.args.rounds,
            "num_nodes": self.args.num_nodes,
        }

    def _configuration_payload(self) -> dict[str, Any]:
        return {
            "train_max_speed": self.train_max_speed,
            "init_damping_target": self.args.init_damping_target,
            "init_max_speed_target": self.init_max_speed_target,
            "init_connectivity": self.args.init_connectivity,
            "init_k_neighbors": self.args.init_k_neighbors,
            "init_min_degree": self.args.init_min_degree,
        }

    def _training_payload(self) -> dict[str, float | bool]:
        return {
            "final_total": self.history["total"][-1],
            "final_traj_loss": self.history["traj_loss"][-1],
            "final_reg_loss": self.history["reg_loss"][-1],
            "final_center_error": self.history["center_error"][-1],
            "best_traj_loss": min(self.history["traj_loss"]),
            "best_total": min(self.history["total"]),
            "has_nan": bool(any(value != value for value in self.history["total"])),
        }

    def _validation_payload(self) -> dict[str, float]:
        return {
            "final_val_traj_loss": self.history["val_traj_loss"][-1],
            "final_val_center_error": self.history["val_center_error"][-1],
        }

    def _diagnostics_payload(self) -> dict[str, Any]:
        return {
            "initial_graph": {
                "num_edges": self.model.last_init_graph_stats.get("num_edges", float("nan")),
                "min_degree": self.model.last_init_graph_stats.get("min_degree", float("nan")),
                "num_components": self.model.last_init_graph_stats.get("num_components", float("nan")),
            },
            "rollout_graph": {
                "mean_num_edges": self.model.last_rollout_graph_health.get("mean_num_edges", float("nan")),
                "mean_min_degree": self.model.last_rollout_graph_health.get("mean_min_degree", float("nan")),
                "max_num_components": self.model.last_rollout_graph_health.get("max_num_components", float("nan")),
            },
            "rollout_speed": {
                "mean_pre_clip_speed": self.model.last_rollout_speed_health.get("mean_pre_clip_speed", float("nan")),
                "mean_cap_fraction": self.model.last_rollout_speed_health.get("mean_cap_fraction", float("nan")),
            },
        }