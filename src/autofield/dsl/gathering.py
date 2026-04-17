"""Helper functions for common neighborhood operations."""

from __future__ import annotations

import torch.nn as nn
from typing import Callable
from torch import Tensor

from .primitives import gather
from .scattering import LinkField

__all__ = [
    "gather_min",
    "gather_max",
    "gather_sum",
    "gather_avg",
]


def gather_min(
    expr: LinkField,
    include_self: bool | None = None,
    mode: str | None = None,
    tau: float | None = None,
    fill_value: float | None = None,
    tag: str | None = None,
) -> Tensor:
    """Aggregate a ``LinkField`` with minimum reduction."""
    return gather(
        expr,
        aggr="min",
        include_self=include_self,
        mode=mode,
        tau=tau,
        fill_value=fill_value,
        tag=tag,
    )


def gather_max(
    expr: LinkField,
    include_self: bool | None = None,
    mode: str | None = None,
    tau: float | None = None,
    fill_value: float | None = None,
    tag: str | None = None,
) -> Tensor:
    """Aggregate a ``LinkField`` with maximum reduction."""
    return gather(
        expr,
        aggr="max",
        include_self=include_self,
        mode=mode,
        tau=tau,
        fill_value=fill_value,
        tag=tag,
    )


def gather_sum(
    expr: LinkField,
    include_self: bool | None = None,
    mode: str | None = None,
    tau: float | None = None,
    fill_value: float | None = None,
    tag: str | None = None,
) -> Tensor:
    """Aggregate a ``LinkField`` with sum reduction."""
    return gather(
        expr,
        aggr="sum",
        include_self=include_self,
        mode=mode,
        tau=tau,
        fill_value=fill_value,
        tag=tag,
    )


def gather_avg(
    expr: LinkField,
    include_self: bool | None = None,
    mode: str | None = None,
    tau: float | None = None,
    fill_value: float | None = None,
    tag: str | None = None,
) -> Tensor:
    """Aggregate a ``LinkField`` with mean reduction."""
    return gather(
        expr,
        aggr="mean",
        include_self=include_self,
        mode=mode,
        tau=tau,
        fill_value=fill_value,
        tag=tag,
    )
