"""Helper functions for common neighborhood operations."""

from __future__ import annotations

from torch import Tensor

from .primitives import field, gather
from .scattering import LinkField

__all__ = [
    "gather_avg",
    "gather_max",
    "gather_min",
    "gather_sum",
    "has_neighbors",
    "nbr_count",
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


def nbr_count() -> Tensor:
    """Count each node's neighbours (the in-degree, as a float field)."""
    from .scattering import scatter  # noqa: PLC0415  (circular import)

    return gather_sum(scatter(field.ones()), fill_value=0.0)


def has_neighbors() -> Tensor:
    """Boolean field: ``True`` where a node has at least one neighbour."""
    return nbr_count() > 0
