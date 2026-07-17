"""Functional DSL for aggregate computing on graphs."""

from torch import Tensor

from ..constants import ELECTION_NONE
from ..core import AggregateContext, DeviceContext
from .building_blocks import (
    broadcast,
    collect_cast,
    descend,
    elect,
    gradient,
    gradient_cast,
)
from .gathering import (
    gather_avg,
    gather_max,
    gather_min,
    gather_sum,
    has_neighbors,
    nbr_count,
)
from .primitives import branch, const, field, gather, iterate, mid, mux
from .scattering import LinkField, as_scatter_expr, scatter, scatter_range

Field = Tensor

__all__ = [
    "ELECTION_NONE",
    "AggregateContext",
    "DeviceContext",
    "Field",
    "LinkField",
    "as_scatter_expr",
    "branch",
    "broadcast",
    "collect_cast",
    "const",
    "descend",
    "elect",
    "field",
    "gather",
    "gather_avg",
    "gather_max",
    "gather_min",
    "gather_sum",
    "gradient",
    "gradient_cast",
    "has_neighbors",
    "iterate",
    "mid",
    "mux",
    "nbr_count",
    "scatter",
    "scatter_range",
]
