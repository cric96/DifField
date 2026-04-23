"""Functional DSL for aggregate computing on graphs."""

from torch import Tensor

from ..core import AggregateContext, DeviceContext
from .building_blocks import broadcast, collect_cast, gradient, gradient_cast
from .gathering import gather_avg, gather_max, gather_min, gather_sum
from .primitives import branch, const, field, gather, iterate, mid, mux
from .scattering import LinkField, as_scatter_expr, scatter, scatter_range

Field = Tensor

__all__ = [
    "AggregateContext",
    "DeviceContext",
    "Field",
    "LinkField",
    "as_scatter_expr",
    "branch",
    "broadcast",
    "collect_cast",
    "const",
    "field",
    "gather",
    "gather_avg",
    "gather_max",
    "gather_min",
    "gather_sum",
    "gradient",
    "gradient_cast",
    "iterate",
    "mid",
    "mux",
    "scatter",
    "scatter_range",
]
