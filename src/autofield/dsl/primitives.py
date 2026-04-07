"""Public DSL primitives built on top of the layer modules."""

from __future__ import annotations

import hashlib
import inspect
from typing import Callable

import torch
import torch.nn as nn
from torch import Tensor

from ..constants import DEFAULT_TAU_BRANCH, DEFAULT_TAU_SOFT_AGGR
from ..core import RoundContext, current_context, with_context
from ..core.mode import get_default_mode
from ..functional import field_where
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


def _auto_name(kind: str, **kwargs: object) -> str:
    """Generate a deterministic name from the caller's source location.

    Walks up the stack skipping frames that belong to autofield internals
    (e.g. ``_LambdaModule.forward`` when ``rep`` is called inside a branch
    lambda) so the name always reflects the user's source position.

    Additional keyword arguments (e.g. ``aggr``) are folded into the hash so
    that two calls on the same line with different parameters get distinct
    names.

    The identity of the calling code object is also included so that two
    different lambdas on the same source line (as is common inside
    ``branch``) produce distinct names.
    """
    frame = inspect.currentframe()
    while frame is not None:
        code = frame.f_code
        if not code.co_filename.endswith("autofield/dsl/primitives.py"):
            param_key = tuple(sorted(kwargs.items()))
            code_id = id(code)
            key = (
                f"{code.co_filename}:{frame.f_lineno}:"
                f"{code.co_name}:{code_id}:{param_key}"
            )
            short_hash = hashlib.md5(key.encode()).hexdigest()[:8]
            return f"{kind}_{short_hash}"
        frame = frame.f_back
    return f"{kind}_fallback"


def rep(
    init: float | Tensor,
    fn: Callable[[Tensor], Tensor],
    *,
    name: str | None = None,
) -> Tensor:
    r"""Temporal evolution (per-node recurrent state).

    Can be called as ``rep(init, fn)`` for auto-naming or
    ``rep(init, fn, name="my_name")`` for explicit naming.
    """
    from ..layers import RepLayer

    resolved_name = name if name is not None else _auto_name("r")

    return RepLayer(init, fn, name=resolved_name)(torch.empty(0))


def nbr(
    expr: Tensor | NeighborExpr,
    aggr: str | Callable = "sum",
    mode: str | None = None,
    tau: float | None = None,
    fill_value: float | None = None,
    edge_index: Tensor | None = None,
    edge_weight: Tensor | None = None,
    tag: str | None = None,
    include_self: bool | None = None,
) -> Tensor:
    r"""Neighborhood message passing.

    When *tag* is omitted, a deterministic identifier is derived from the
    caller's source location so that the same ``nbr()`` call always maps to
    the same export/override slot.
    """
    from ..layers import NbrLayer

    effective_mode = mode if mode is not None else get_default_mode()
    effective_tau = tau if tau is not None else DEFAULT_TAU_SOFT_AGGR

    if tag is None and not isinstance(expr, NeighborExpr):
        tag = _auto_name("n", aggr=aggr)

    return NbrLayer(
        aggr=aggr, mode=effective_mode, tau=effective_tau, fill_value=fill_value, include_self=include_self
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
    mode: str | None = None,
    tau: float | None = None,
) -> Tensor:
    r"""Domain restriction with communication isolation and state reset."""
    from ..layers import BranchLayer

    effective_mode = mode if mode is not None else get_default_mode()
    effective_tau = tau if tau is not None else DEFAULT_TAU_BRANCH

    true_mod = _LambdaModule(if_true) if not isinstance(if_true, nn.Module) else if_true
    false_mod = (
        _LambdaModule(if_false) if not isinstance(if_false, nn.Module) else if_false
    )
    return BranchLayer(
        true_mod,
        false_mod,
        branch_name=branch_name,
        reset_states=reset_states,
        mode=effective_mode,
        tau=effective_tau,
    )(torch.empty(0), cond)


def mux(
    cond: Tensor,
    if_true: Tensor | Callable[[], Tensor],
    if_false: Tensor | Callable[[], Tensor],
    mode: str | None = None,
    tau: float | None = None,
) -> Tensor:
    r"""Pointwise conditional selection (no topology change)."""
    val_true = if_true() if callable(if_true) else if_true
    val_false = if_false() if callable(if_false) else if_false
    return field_where(cond, val_true, val_false, mode=mode, tau=tau)


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
