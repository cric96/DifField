"""Boids domain layer: geometry, initialization, and graph analysis."""

from .geometry import hard_separation_force, sample_initial_state
from .graph import edge_connectivity_stats, graph_health_summary
from .scenario import build_scenario
from .dynamics import (
    VelocityStep,
    reference_boids_velocity_step,
    reference_boids_velocity_update,
    step_boids_dynamics,
    rollout_with_dynamic_topology,
)
from .teacher import teacher_rollout, teacher_rollout_from_specs

__all__ = [
    "VelocityStep",
    "build_scenario",
    "edge_connectivity_stats",
    "graph_health_summary",
    "hard_separation_force",
    "reference_boids_velocity_step",
    "reference_boids_velocity_update",
    "rollout_with_dynamic_topology",
    "sample_initial_state",
    "step_boids_dynamics",
    "teacher_rollout",
    "teacher_rollout_from_specs",
]
