"""Parameter recovery analysis for territory imitation learning."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..domain.specs import LearnableTerritoriesSpec


def extract_teacher_parameters_from_spec(
    spec: "LearnableTerritoriesSpec",
) -> dict[str, float]:
    """Extract teacher parameter values from experiment specifications."""
    return {
        "range_weight": float(spec.teacher.range_weight),
        "risk_weight": float(spec.teacher.risk_weight),
        "assignment_tau": float(spec.teacher.assignment_tau),
    }


def extract_learned_parameters(history: dict[str, list[float]]) -> dict[str, float]:
    """Extract the final learned parameter values from training history."""
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
    """Compute absolute and relative error between teacher and learned parameters."""
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
