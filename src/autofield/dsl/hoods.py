"""Helper functions for common neighborhood operations."""

from __future__ import annotations

import torch.nn as nn
from typing import Callable
from torch import Tensor

from .primitives import hood
from .neighbor import NeighborExpr

__all__ = [
    "minhood",
    "maxhood",
    "sumhood",
    "avghood",
]


def minhood(
    expr: NeighborExpr,
    include_self: bool | None = None,
    mode: str | None = None,
    tau: float | None = None,
    fill_value: float | None = None,
    tag: str | None = None,
) -> Tensor:
    """Aggregate a ``NeighborExpr`` with minimum reduction."""
    return hood(
        expr,
        aggr="min",
        include_self=include_self,
        mode=mode,
        tau=tau,
        fill_value=fill_value,
        tag=tag,
    )


def maxhood(
    expr: NeighborExpr,
    include_self: bool | None = None,
    mode: str | None = None,
    tau: float | None = None,
    fill_value: float | None = None,
    tag: str | None = None,
) -> Tensor:
    """Aggregate a ``NeighborExpr`` with maximum reduction."""
    return hood(
        expr,
        aggr="max",
        include_self=include_self,
        mode=mode,
        tau=tau,
        fill_value=fill_value,
        tag=tag,
    )


def sumhood(
    expr: NeighborExpr,
    include_self: bool | None = None,
    mode: str | None = None,
    tau: float | None = None,
    fill_value: float | None = None,
    tag: str | None = None,
) -> Tensor:
    """Aggregate a ``NeighborExpr`` with sum reduction."""
    return hood(
        expr,
        aggr="sum",
        include_self=include_self,
        mode=mode,
        tau=tau,
        fill_value=fill_value,
        tag=tag,
    )


def avghood(
    expr: NeighborExpr,
    include_self: bool | None = None,
    mode: str | None = None,
    tau: float | None = None,
    fill_value: float | None = None,
    tag: str | None = None,
) -> Tensor:
    """Aggregate a ``NeighborExpr`` with mean reduction."""
    return hood(
        expr,
        aggr="mean",
        include_self=include_self,
        mode=mode,
        tau=tau,
        fill_value=fill_value,
        tag=tag,
    )
