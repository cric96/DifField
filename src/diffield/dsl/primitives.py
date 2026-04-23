"""Public DSL primitives built on top of the layer modules."""

from __future__ import annotations

import hashlib
import inspect
from typing import TYPE_CHECKING

import torch
from torch import Tensor, nn

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable

    from .scattering import LinkField

from ..constants import DEFAULT_TAU_BRANCH, DEFAULT_TAU_SOFT_AGGR
from ..core import RoundContext, current_context, with_context
from ..core.mode import get_default_mode
from ..functional import field_where


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

    Walks up the stack skipping frames that belong to diffield internals
    (e.g. ``_LambdaModule.forward`` when ``iterate`` is called inside a branch
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
        if not code.co_filename.endswith("diffield/dsl/primitives.py"):
            param_key = tuple(sorted(kwargs.items()))
            code_id = id(code)
            key = (
                f"{code.co_filename}:{frame.f_lineno}:"
                f"{code.co_name}:{code_id}:{param_key}"
            )
            short_hash = hashlib.sha256(key.encode()).hexdigest()[:8]
            return f"{kind}_{short_hash}"
        frame = frame.f_back
    return f"{kind}_fallback"


def iterate(
    init: Tensor,
    fn: Callable[[Tensor, Tensor, RoundContext], Tensor] | Callable[[Tensor], Tensor],
    *,
    name: str | None = None,
) -> Tensor:
    r"""Temporal evolution (per-node recurrent state).

    ``init`` is a node field tensor. Use ``field.of(...)``, ``field.zeros()``,
    ``field.inf()``, scenario helpers, or any tensor with first dimension equal
    to the number of nodes.

    Can be called as ``iterate(init, fn)`` for auto-naming or
    ``iterate(init, fn, name="my_name")`` for explicit naming.
    """
    from ..layers import IterateLayer  # noqa: PLC0415

    resolved_name = name if name is not None else _auto_name("r")

    return IterateLayer(init, fn, name=resolved_name)(torch.empty(0))


def gather(
    expr: LinkField,
    aggr: str | Callable | nn.Module = "sum",
    include_self: bool | None = None,
    mode: str | None = None,
    tau: float | None = None,
    fill_value: float | None = None,
    tag: str | None = None,
) -> Tensor:
    r"""Gather neighbours and fold edge-wise messages into a field.

    The input must be a :class:`LinkField`, typically built with :func:`scatter`
    and optionally combined with arithmetic or :func:`scatter_range`.

    When a tag is present, the source field is exported in the context and can
    be overridden by runtime message overrides.
    """
    from ..core import current_context  # noqa: PLC0415
    from ..layers import GatherLayer  # noqa: PLC0415

    ctx = current_context()
    effective_mode = mode if mode is not None else get_default_mode()
    effective_tau = tau if tau is not None else DEFAULT_TAU_SOFT_AGGR

    # Use provided tag or auto-generate one
    if tag is not None:
        expr.tag = tag
    elif expr.tag is None:
        expr.tag = _auto_name("h", aggr=aggr)

    return GatherLayer(
        aggr=aggr,
        mode=effective_mode,
        tau=effective_tau,
        fill_value=fill_value,
        include_self=include_self,
    )(expr, ctx=ctx, tag=expr.tag)


def branch(
    cond: Tensor,
    if_true: Callable[[], Tensor],
    if_false: Callable[[], Tensor],
    branch_name: str = "branch",
    reset_states: dict[str, Tensor] | None = None,
    mode: str | None = None,
    tau: float | None = None,
) -> Tensor:
    r"""Domain restriction with edge partitioning and optional state reset.

    Communication is restricted by masking edges per branch. Branch evaluations
    share the same underlying state manager and are coordinated via
    snapshot/restore semantics before merging. ``reset_states`` values must be
    node field tensors.
    """
    from ..layers import BranchLayer  # noqa: PLC0415

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
    """Build a constant node field by broadcasting a scalar in the current context."""
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


class field:  # noqa: N801
    """Context-aware constructors for node field tensors."""

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

    # TODO add description here

    @staticmethod
    def with_overrides(
        field: Tensor, overrides: Iterable[tuple[int, float]]
    ) -> Tensor:
        for node_id, value in overrides:
            field[node_id] = value
        return field

    @staticmethod
    def from_values(
        values: list[float] | Tensor,
    ) -> Tensor:
        ctx = current_context()
        tensor = torch.as_tensor(values, dtype=torch.float32, device=ctx.edge_index.device)
        if tensor.shape[0] != ctx.num_nodes:
            raise ValueError(
                f"Length of values ({tensor.shape[0]}) does not match "
                f"number of nodes ({ctx.num_nodes})"
            )
        return tensor
