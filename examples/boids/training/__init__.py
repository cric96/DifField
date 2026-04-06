"""Boids training logic: curriculum, supervision, traces, and loop."""

from .curriculum import curriculum_horizon, scheduled_learning_rate
from .loop import BoidsTrainingLoop
from .supervision import teacher_forced_step_losses
from .trace import (
    BoidsTrace,
    build_supervision_traces,
    load_boids_trace,
    save_boids_trace,
    teacher_trace_from_specs,
)

__all__ = [
    "BoidsTrace",
    "BoidsTrainingLoop",
    "build_supervision_traces",
    "curriculum_horizon",
    "load_boids_trace",
    "save_boids_trace",
    "scheduled_learning_rate",
    "teacher_forced_step_losses",
    "teacher_trace_from_specs",
]
