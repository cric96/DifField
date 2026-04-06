"""Reporting helpers for learnable territories experiments."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

try:
    from .specs import LearnableTerritoriesSpec
except ImportError:
    from specs import LearnableTerritoriesSpec


def extract_teacher_parameters_from_spec(spec: LearnableTerritoriesSpec) -> dict[str, float]:
    return {
        "range_weight": float(spec.teacher.range_weight),
        "risk_weight": float(spec.teacher.risk_weight),
        "assignment_tau": float(spec.teacher.assignment_tau),
    }


def extract_learned_parameters(history: dict[str, list[float]]) -> dict[str, float]:
    return {
        "range_weight": float(history["range_weight"][-1]),
        "risk_weight": float(history["risk_weight"][-1]),
        "assignment_tau": float(history["assignment_tau"][-1]),
        "surcharge_weight": float(history["surcharge_weight"][-1]),
    }


def compute_parameter_recovery_metrics(
    teacher_params: dict[str, float],
    learned_params: dict[str, float],
) -> dict[str, dict[str, float]]:
    recovery: dict[str, dict[str, float]] = {}
    for name, teacher_value in teacher_params.items():
        learned_value = float(learned_params[name])
        abs_err = abs(learned_value - teacher_value)
        recovery[name] = {
            "teacher": float(teacher_value),
            "learned": learned_value,
            "abs_error": abs_err,
            "rel_error": abs_err / max(abs(teacher_value), 1e-9),
        }
    return recovery


def _last_finite(values: list[float]) -> float:
    for value in reversed(values):
        if math.isfinite(value):
            return float(value)
    return float("nan")


@dataclass
class TerritoriesSummaryBuilder:
    run_name: str
    spec: LearnableTerritoriesSpec
    history: dict[str, list[float]]
    flat_summary: dict[str, float | int | str]
    evaluation: dict[str, Any]
    recovery: dict[str, dict[str, float]]

    def build(self) -> dict[str, Any]:
        return {
            "run": self._run_payload(),
            "configuration": self._configuration_payload(),
            "training": self._training_payload(),
            "validation": self.evaluation,
            "parameters": {
                "teacher": extract_teacher_parameters_from_spec(self.spec),
                "learned": extract_learned_parameters(self.history),
                "recovery": self.recovery,
            },
        }

    def _run_payload(self) -> dict[str, Any]:
        effective_rounds = self.spec.program.rounds
        if effective_rounds <= 0:
            effective_rounds = 2 * (self.spec.grid.rows + self.spec.grid.cols)
        return {
            "run_name": self.run_name,
            "mode": self.spec.model.mode,
            "train_seed": self.spec.training.train_seed,
            "eval_seeds": list(self.spec.evaluation.eval_seeds),
            "rows": self.spec.grid.rows,
            "cols": self.spec.grid.cols,
            "connectivity": self.spec.grid.connectivity,
            "num_sinks": self.spec.program.num_sinks,
            "scenario_preset": self.spec.program.scenario_preset,
            "sink_positions": list(self.spec.program.sink_positions or ()),
            "rounds": effective_rounds,
        }

    def _configuration_payload(self) -> dict[str, Any]:
        return {
            "epochs": self.spec.training.epochs,
            "learning_rate": self.spec.training.lr,
            "risk_loss_weight": self.spec.training.risk_loss_weight,
            "clip_grad_norm": self.spec.training.clip_grad_norm,
            "hidden_dim": self.spec.model.hidden_dim,
            "init_range_weight_target": self.spec.model.init_range_weight_target,
            "init_risk_weight_target": self.spec.model.init_risk_weight_target,
            "init_assignment_tau_target": self.spec.model.init_assignment_tau_target,
            "init_surcharge_weight_target": self.spec.model.init_surcharge_weight_target,
        }

    def _training_payload(self) -> dict[str, Any]:
        return {
            "best_epoch": int(self.flat_summary["best_epoch"]),
            "best_score": float(self.flat_summary["best_score"]),
            "final_total_loss": float(self.history["total"][-1]),
            "final_load_loss": float(self.history["load_loss"][-1]),
            "final_risk_loss": float(self.history["risk_loss"][-1]),
            "final_owner_agreement": float(self.history["owner_agreement"][-1]),
            "final_conservation_error": float(self.history["conservation_error"][-1]),
            "final_grad_norm": float(self.history["grad_norm"][-1]),
            "final_val_total_loss": _last_finite(self.history["val_total"]),
            "final_val_load_loss": _last_finite(self.history["val_load_loss"]),
            "final_val_risk_loss": _last_finite(self.history["val_risk_loss"]),
            "final_val_owner_agreement": _last_finite(self.history["val_owner_agreement"]),
            "final_val_conservation_error": _last_finite(self.history["val_conservation_error"]),
        }