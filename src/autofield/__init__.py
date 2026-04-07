"""autofield — aggregate computing and differentiable field calculus."""

from .core import AggregateContext, DeviceContext, RoundContext, StateManager, get_default_mode, set_default_mode, with_mode
from .dsl import branch, broadcast, collect_cast, const, field, gradient, gradient_cast, mid, mux, nbr, nbr_range, nbrRange, rep
from .functional import field_where, mask_edges, mask_edges_for_partition, scatter_aggr
from .layers import BranchLayer, MuxLayer, NbrLayer, RepLayer
from .pyg_backend import HAS_PYG
from .sim import (
    EventSchedule,
    FullyConnectedScenario,
    GridScenario,
    RelaxedRadiusScenario,
    ScheduledEvent,
    SimulationEngine,
    SimulationRuntime,
    SnapshotRecorder,
    SpatialScenario,
    boids_acceleration_dense,
    bounce_in_box,
    build_relaxed_radius_graph,
    build_spatial_graph,
    limit_speed,
    normalize_vectors,
)
from .utils import make_grid_graph

__all__ = [
    # Core
    "RoundContext",
    "StateManager",
    # DSL
    "AggregateContext",
    "DeviceContext",
    "rep",
    "nbr",
    "nbr_range",
    "nbrRange",
    "branch",
    "broadcast",
    "gradient_cast",
    "collect_cast",
    "mux",
    "const",
    "field",
    "gradient",
    "mid",
    # Layers (nn.Module)
    "RepLayer",
    "NbrLayer",
    "BranchLayer",
    "MuxLayer",
    # Functional
    "scatter_aggr",
    "mask_edges",
    "mask_edges_for_partition",
    "field_where",
    "HAS_PYG",
    # Mode configuration
    "get_default_mode",
    "set_default_mode",
    "with_mode",
    # Simulation
    "GridScenario",
    "SimulationEngine",
    "SimulationRuntime",
    "EventSchedule",
    "ScheduledEvent",
    "SnapshotRecorder",
    "SpatialScenario",
    "RelaxedRadiusScenario",
    "build_relaxed_radius_graph",
    "FullyConnectedScenario",
    "build_spatial_graph",
    "normalize_vectors",
    "limit_speed",
    "bounce_in_box",
    "boids_acceleration_dense",
    # Utilities
    "make_grid_graph",
]
