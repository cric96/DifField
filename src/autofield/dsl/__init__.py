"""Functional DSL for aggregate computing on graphs."""

from torch import Tensor

from ..core import AggregateContext, DeviceContext
from .building_blocks import broadcast, collect_cast, gradient, gradient_cast
from .scattering import LinkField, as_scatter_expr, scatter, scatter_range
from .primitives import branch, const, field, gather, mid, mux, iterate
from .gathering import gather_min, gather_max, gather_sum, gather_avg

Field = Tensor

__all__ = [
    "AggregateContext",
    "DeviceContext",
    "LinkField",
    "Field",
    "iterate",
    "scatter",
    "gather",
    "gather_min",
    "gather_max",
    "gather_sum",
    "gather_avg",
    "as_scatter_expr",
    "scatter_range",
    "branch",
    "broadcast",
    "gradient_cast",
    "collect_cast",
    "mux",
    "const",
    "field",
    "gradient",
    "mid",
]
