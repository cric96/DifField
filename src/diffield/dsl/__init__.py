"""Functional DSL for aggregate computing on graphs."""

from torch import Tensor

from ..constants import ELECTION_NONE
from ..core import AggregateContext, DeviceContext
from .building_blocks import (
    Election,
    bounded_election,
    broadcast,
    collect_cast,
    converge_cast,
    descend,
    distance_to,
    elect,
    follow,
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
from .primitives import aligned_on, branch, const, field, gather, iterate, mid, mux, nbr
from .scattering import (
    LinkField,
    as_scatter_expr,
    link_cat,
    link_map,
    membership,
    scatter,
    scatter_range,
)

Field = Tensor

__all__ = [
    "ELECTION_NONE",
    "AggregateContext",
    "DeviceContext",
    "Election",
    "Field",
    "LinkField",
    "aligned_on",
    "as_scatter_expr",
    "bounded_election",
    "branch",
    "broadcast",
    "collect_cast",
    "const",
    "converge_cast",
    "descend",
    "distance_to",
    "elect",
    "follow",
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
    "link_cat",
    "link_map",
    "membership",
    "mid",
    "mux",
    "nbr",
    "nbr_count",
    "scatter",
    "scatter_range",
]
