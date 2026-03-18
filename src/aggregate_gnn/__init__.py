"""aggregate_gnn — Aggregate Computing ↔ GNN isomorphism library."""

from .core import RoundContext, StateManager
from .dsl import AggregateContext, DeviceContext, branch, broadcast, const, field, mid, mux, nbr, rep
from .functional import mask_edges, mask_edges_for_partition, scatter_aggr, soft_where
from .layers import BranchLayer, MuxLayer, NbrLayer, RepLayer
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
    "branch",
    "broadcast",
    "mux",
    "const",
    "field",
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
    "soft_where",
    # Utilities
    "make_grid_graph",
]
