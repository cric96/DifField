"""Reusable simulation building blocks for autofield examples."""

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
    "ProgramStep",
    "SimulationEngine",
    "SimulationRuntime",
    "EventSchedule",
    "ScheduledEvent",
    "normalize_vectors",
    "limit_speed",
    "bounce_in_box",
    "boids_acceleration_dense",
    "SnapshotRecorder",
    "FullyConnectedScenario",
    "GridScenario",
    "RelaxedRadiusScenario",
    "SpatialScenario",
    "build_relaxed_radius_graph",
    "build_spatial_graph",
]
