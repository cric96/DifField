"""Edge-wise neighbor expressions and range primitives."""

from __future__ import annotations

import warnings
from typing import Callable

import torch
from torch import Tensor

from ..core import RoundContext
from .helpers import edge_sources_targets, ensure_field


class NeighborExpr:
    """Edge-wise neighborhood expression evaluated before aggregation."""

    __array_priority__ = 1000

    def __init__(self, evaluator: Callable[[RoundContext, Tensor, Tensor | None], Tensor]) -> None:
        self._evaluator = evaluator

    def evaluate(
        self,
        *,
        ctx: RoundContext,
        edge_index: Tensor,
        edge_weight: Tensor | None,
    ) -> Tensor:
        return self._evaluator(ctx, edge_index, edge_weight)

    def _binary(self, other: float | Tensor | "NeighborExpr", op: Callable[[Tensor, Tensor], Tensor]) -> "NeighborExpr":
        other_expr = _as_neighbor_expr(other)
        return NeighborExpr(
            lambda ctx, edge_index, edge_weight: op(
                self.evaluate(ctx=ctx, edge_index=edge_index, edge_weight=edge_weight),
                other_expr.evaluate(ctx=ctx, edge_index=edge_index, edge_weight=edge_weight),
            ),
        )

    def _rbinary(self, other: float | Tensor | "NeighborExpr", op: Callable[[Tensor, Tensor], Tensor]) -> "NeighborExpr":
        other_expr = _as_neighbor_expr(other)
        return NeighborExpr(
            lambda ctx, edge_index, edge_weight: op(
                other_expr.evaluate(ctx=ctx, edge_index=edge_index, edge_weight=edge_weight),
                self.evaluate(ctx=ctx, edge_index=edge_index, edge_weight=edge_weight),
            ),
        )

    def __add__(self, other: float | Tensor | "NeighborExpr") -> "NeighborExpr":
        return self._binary(other, torch.add)

    def __radd__(self, other: float | Tensor | "NeighborExpr") -> "NeighborExpr":
        return self._rbinary(other, torch.add)

    def __sub__(self, other: float | Tensor | "NeighborExpr") -> "NeighborExpr":
        return self._binary(other, torch.sub)

    def __rsub__(self, other: float | Tensor | "NeighborExpr") -> "NeighborExpr":
        return self._rbinary(other, torch.sub)

    def __mul__(self, other: float | Tensor | "NeighborExpr") -> "NeighborExpr":
        return self._binary(other, torch.mul)

    def __rmul__(self, other: float | Tensor | "NeighborExpr") -> "NeighborExpr":
        return self._rbinary(other, torch.mul)

    def __truediv__(self, other: float | Tensor | "NeighborExpr") -> "NeighborExpr":
        return self._binary(other, torch.div)

    def __rtruediv__(self, other: float | Tensor | "NeighborExpr") -> "NeighborExpr":
        return self._rbinary(other, torch.div)

    def __neg__(self) -> "NeighborExpr":
        return NeighborExpr(
            lambda ctx, edge_index, edge_weight: -self.evaluate(
                ctx=ctx,
                edge_index=edge_index,
                edge_weight=edge_weight,
            ),
        )


def _as_neighbor_expr(value: float | Tensor | NeighborExpr) -> NeighborExpr:
    if isinstance(value, NeighborExpr):
        return value

    def evaluate(ctx: RoundContext, edge_index: Tensor, _edge_weight: Tensor | None) -> Tensor:
        source_nodes, _target_nodes = edge_sources_targets(edge_index)
        field_value = ensure_field(value, ctx)
        return field_value[source_nodes]

    return NeighborExpr(evaluate)


def nbr_range() -> NeighborExpr:
    """Return the current edge metric / range as an edge-wise expression."""

    def evaluate(ctx: RoundContext, edge_index: Tensor, edge_weight: Tensor | None) -> Tensor:
        if edge_weight is not None:
            return edge_weight
        return torch.ones(edge_index.shape[1], device=ctx.edge_index.device, dtype=torch.float32)

    return NeighborExpr(evaluate)


def nbrRange() -> NeighborExpr:
    """Deprecated field-calculus-style alias for :func:`nbr_range`."""
    warnings.warn("nbrRange() is deprecated; use nbr_range() instead.", DeprecationWarning, stacklevel=2)
    return nbr_range()