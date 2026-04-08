"""Functional DSL for aggregate computing on graphs."""

from torch import Tensor

from ..core import AggregateContext, DeviceContext
from .building_blocks import broadcast, collect_cast, gradient, gradient_cast
from .neighbor import NeighborExpr, as_nbr_expr, nbr, nbr_range
from .primitives import branch, const, field, foldhood, mid, mux, rep
from .hoods import minhood, maxhood, sumhood, avghood

Field = Tensor

__all__ = [
    "AggregateContext",
    "DeviceContext",
    "NeighborExpr",
    "Field",
    "rep",
    "nbr",
    "foldhood",
    "minhood",
    "maxhood",
    "sumhood",
    "avghood",
    "as_nbr_expr",
    "nbr_range",
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
