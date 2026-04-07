"""Public DSL primitives built on top of the layer modules."""

from __future__ import annotations

from typing import Callable

import torch
import torch.nn as nn
from torch import Tensor

from ..constants import DEFAULT_TAU_BRANCH, DEFAULT_TAU_SOFT_AGGR
from ..core import RoundContext, current_context, with_context
from ..functional import soft_where
from .neighbor import NeighborExpr


class _LambdaModule(nn.Module):
    """Wrap a zero-arg callable for use as a branch sub-program."""

    def __init__(self, fn: Callable[[], Tensor]) -> None:
        super().__init__()
        self._fn = fn

    def forward(self, x: Tensor, ctx: RoundContext | None = None) -> Tensor:
        if ctx is not None:
            with with_context(ctx):
                return self._fn()
        return self._fn()


def rep(name: str, init: float | Tensor, fn: Callable[[Tensor], Tensor]) -> Tensor:
    r"""Temporal evolution (per-node recurrent state)."""
    from ..layers import RepLayer

    return RepLayer(name, init, fn)(torch.empty(0))


def nbr(
    expr: Tensor | NeighborExpr,
    aggr: str | Callable = "sum",
    mode: str = "hard",
    tau: float = DEFAULT_TAU_SOFT_AGGR,
    fill_value: float | None = None,
    edge_index: Tensor | None = None,
    edge_weight: Tensor | None = None,
    tag: str | None = None,
    include_self: bool | None = None,
) -> Tensor:
    r"""Neighborhood message passing."""
    from ..layers import NbrLayer

    return NbrLayer(
        aggr=aggr, mode=mode, tau=tau, fill_value=fill_value, include_self=include_self
    )(
        expr,
        edge_index=edge_index,
        edge_weight=edge_weight,
        tag=tag,
    )


def branch(
    cond: Tensor,
    if_true: Callable[[], Tensor],
    if_false: Callable[[], Tensor],
    branch_name: str = "branch",
    reset_states: dict[str, float | Tensor] | None = None,
    mode: str = "hard",
    tau: float = DEFAULT_TAU_BRANCH,
) -> Tensor:
    r"""Domain restriction with communication isolation and state reset."""
    from ..layers import BranchLayer

    true_mod = _LambdaModule(if_true) if not isinstance(if_true, nn.Module) else if_true
    false_mod = (
        _LambdaModule(if_false) if not isinstance(if_false, nn.Module) else if_false
    )
    return BranchLayer(
        true_mod,
        false_mod,
        branch_name=branch_name,
        reset_states=reset_states,
        mode=mode,
        tau=tau,
    )(torch.empty(0), cond)


def mux(
    cond: Tensor,
    if_true: Tensor | Callable[[], Tensor],
    if_false: Tensor | Callable[[], Tensor],
) -> Tensor:
    r"""Pointwise conditional selection (no topology change)."""
    val_true = if_true() if callable(if_true) else if_true
    val_false = if_false() if callable(if_false) else if_false
    return soft_where(cond, val_true, val_false)


def const(value: float) -> Tensor:
    """Broadcast a scalar to all nodes in the current context."""
    ctx = current_context()
    return torch.full(
        (ctx.num_nodes,), value, dtype=torch.float32, device=ctx.edge_index.device
    )


def mid() -> Tensor:
    """Return tensor of node IDs [0, 1, ..., N-1] as float."""
    ctx = current_context()
    return torch.arange(
        ctx.num_nodes, dtype=torch.float32, device=ctx.edge_index.device
    )


class field:
    """Context-aware field constructors."""

    @staticmethod
    def of(value: float) -> Tensor:
        return const(value)

    @staticmethod
    def zeros() -> Tensor:
        return const(0.0)

    @staticmethod
    def ones() -> Tensor:
        return const(1.0)

    @staticmethod
    def inf() -> Tensor:
        return const(float("inf"))
