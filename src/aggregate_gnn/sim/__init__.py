"""Reusable simulation building blocks for Aggregate GNN examples."""

from .engine import ProgramStep, SimulationEngine
from .events import EventSchedule, ScheduledEvent, SimulationRuntime
from .physics import boids_acceleration_dense, bounce_in_box, limit_speed, normalize_vectors
from .recording import SnapshotRecorder
from .scenario import FullyConnectedScenario, GridScenario, SpatialScenario, build_spatial_graph

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
    "SpatialScenario",
    "build_spatial_graph",
]
