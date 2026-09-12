"""Reusable simulation building blocks for diffield examples."""

from .engine import ProgramStep, SimulationEngine
from .events import EventSchedule, ScheduledEvent, SimulationRuntime
from .physics import boids_acceleration_dense, bounce_in_box, limit_speed, normalize_vectors
from .recording import SnapshotRecorder
from .scenario import (
    FullyConnectedScenario,
    GridScenario,
    RelaxedRadiusScenario,
    SpatialScenario,
    build_relaxed_radius_graph,
    build_spatial_graph,
)

__all__ = [
    "EventSchedule",
    "FullyConnectedScenario",
    "GridScenario",
    "ProgramStep",
    "RelaxedRadiusScenario",
    "ScheduledEvent",
    "SimulationEngine",
    "SimulationRuntime",
    "SnapshotRecorder",
    "SpatialScenario",
    "boids_acceleration_dense",
    "bounce_in_box",
    "build_relaxed_radius_graph",
    "build_spatial_graph",
    "limit_speed",
    "normalize_vectors",
]
