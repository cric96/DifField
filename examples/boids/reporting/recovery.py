"""Parameter recovery analysis for boids imitation learning."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..domain.specs import LearnableBoidsSpec


def extract_teacher_parameters_from_spec(spec: LearnableBoidsSpec) -> dict[str, float]:
    """Extract teacher parameter values from experiment specifications."""
    return {
        "w_sep": float(spec.teacher.w_sep),
        "w_align": float(spec.teacher.w_align),
        "w_cohesion": float(spec.teacher.w_cohesion),
        "damping": float(spec.teacher.damping),
        "max_speed": float(spec.teacher.max_speed),
    }


def extract_learned_parameters(history: dict[str, list[float]]) -> dict[str, float]:
    """Extract the final learned parameter values from training history."""
    return {
        "w_sep": history["w_sep"][-1],
        "w_align": history["w_align"][-1],
        "w_cohesion": history["w_cohesion"][-1],
        "damping": history["damping"][-1],
        "max_speed": history["max_speed"][-1],
    }


def compute_parameter_recovery_metrics(
    teacher_params: dict[str, float],
    learned_params: dict[str, float],
) -> dict[str, dict[str, float]]:
    """Compute absolute and relative error between teacher and learned parameters."""
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
