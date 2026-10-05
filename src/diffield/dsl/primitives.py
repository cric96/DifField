"""Public DSL primitives built on top of the layer modules."""

from __future__ import annotations

from collections.abc import Callable, Generator
from contextlib import contextmanager

import torch
from torch import Tensor, nn

from ..constants import DEFAULT_TAU_BRANCH, DEFAULT_TAU_SOFT_AGGR
from ..core import RoundContext, current_context, sub_context, with_context
from ..core.alignment import SEP
from ..core.mode import get_default_mode, get_default_tau
from ..functional import field_where
from .helpers import ensure_field
from .scattering import LinkField, scatter


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


def iterate(
    init: Tensor,
    fn: Callable[[Tensor], Tensor],
    *,
    name: str | None = None,
) -> Tensor:
    r"""Temporal evolution (per-node recurrent state).

    ``init`` is a node field tensor. Use ``field.of(...)``, ``field.zeros()``,
    ``field.inf()``, scenario helpers, or any tensor with first dimension equal
    to the number of nodes.

    Each occurrence is identified by its position in the evaluation tree (see
    :mod:`diffield.core.alignment`), so two calls never share state.  ``name``
    is optional and labels the occurrence, which keeps its store key stable and
    readable — ``iterate(..., name="dist")`` becomes ``/it:dist``.
    """
    from ..layers import IterateLayer

    return IterateLayer(init, fn, name=name)(torch.empty(0))


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

    The gathered source field is exported in the context under this
    occurrence's alignment path, where it can be overridden by runtime message
    overrides.  ``tag`` is optional and labels the occurrence.
    """
    from ..core import current_context
    from ..layers import GatherLayer

    ctx = current_context()
    effective_mode = mode if mode is not None else get_default_mode()
    effective_tau = tau if tau is not None else get_default_tau(DEFAULT_TAU_SOFT_AGGR)

    label = tag if tag is not None else expr.tag
    expr.tag = ctx.align.key("gt", label)

    return GatherLayer(
        aggr=aggr,
        mode=effective_mode,
        tau=effective_tau,
        fill_value=fill_value,
        include_self=include_self,
    )(expr, ctx=ctx, tag=expr.tag)


def nbr(value: Tensor, *, default: float = 0.0, name: str | None = None) -> LinkField:
    r"""Neighbours' value of *value* from their previous round (field-calculus ``nbr``).

    The value travels as a state slot, so it is exchanged between independent
    devices like any ``iterate`` state.  A neighbour that has not published yet
    reads ``default``; a self link (``include_self``) reads the current value.
    Each call is one exchanged field: bind it once and reuse it.
    """
    ctx = current_context()
    value = ensure_field(value, ctx)
    key = ctx.align.key("nbr", name)
    published = ctx.state.get_or_init(torch.full_like(value, default), name=key)
    ctx.state.update(value, name=key)

    def evaluate(_ctx: RoundContext, edge_index: Tensor, _weight: Tensor | None) -> Tensor:
        source, target = edge_index
        own = (source == target).view(-1, *[1] * (value.dim() - 1))
        return torch.where(own, value[source], published[source])

    return LinkField(evaluate, _repr=f"nbr({name or 'field'})")


@contextmanager
def aligned_on(
    key: Tensor,
    *,
    weight: Tensor | None = None,
    name: str | None = None,
    mode: str | None = None,
) -> Generator[None, None, None]:
    r"""Partition the enclosed computation by *key* (ScaFi ``align``, FCPP ``split``,
    Collektive ``alignedOn``).

    Devices publish their key; inside the block a link ``j -> i`` exists only
    when ``j``'s published key equals ``i``'s current key, so each partition
    computes independently, centrally or on devices.  A device whose key
    changed restarts the enclosed blocks.  In soft mode a ``weight`` (e.g.
    membership confidence) makes :func:`~diffield.dsl.scattering.membership`
    equal ``w_j * w_i`` for additive blocks; the partition itself stays hard.
    """
    ctx = current_context()
    key = ensure_field(key, ctx).detach().float()
    soft = weight is not None and (mode or get_default_mode()) == "soft"
    own = ensure_field(weight, ctx) if soft else torch.ones_like(key)
    with ctx.align.scope("aligned_on", name) as slot:
        published = ctx.state.get_or_init(key.new_full((ctx.num_nodes, 2), float("nan")), name=slot)
        ctx.state.update(torch.stack((key, own), -1), name=slot)
        source, target = ctx.edge_index
        same = published[source, 0] == key[target]
        region = sub_context(
            ctx,
            ctx.edge_index[:, same],
            ctx.edge_weight[same],
            None if ctx.message_weight is None else ctx.message_weight[same],
        )
        if soft:
            link = scatter(published[:, 1]) * own
            region.membership = link if ctx.membership is None else ctx.membership * link
        with with_context(region):
            yield
    changed = (key != published[:, 0]) & ~published[:, 0].isnan()
    inner = [k for k in ctx.state.keys() if k.startswith(slot + SEP)]  # noqa: SIM118
    ctx.state.restrict(inner, keep=~changed, mode="hard")


def branch(
    cond: Tensor,
    if_true: Callable[[], Tensor],
    if_false: Callable[[], Tensor],
    branch_name: str | None = None,
    reset_states: dict[str, Tensor] | None = None,
    mode: str | None = None,
    tau: float | None = None,
) -> Tensor:
    r"""Domain restriction with edge partitioning.

    Communication is restricted by masking edges per branch, and each partition
    gets its own alignment frame so their persistent state is disjoint.  A node
    that leaves a partition reads that partition's initializers when it returns,
    matching field-calculus alignment, so ``reset_states`` is no longer needed
    and is ignored.  ``branch_name`` optionally labels the occurrence.
    """
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
    """Build a constant node field by broadcasting a scalar in the current context."""
    ctx = current_context()
    return torch.full(
        (ctx.num_nodes,), value, dtype=torch.float32, device=ctx.edge_index.device
    )


def mid() -> Tensor:
    """Return the device-identity field.

    Normally ``[0, 1, ..., N-1]``.  A :class:`~diffield.core.device.DeviceContext`
    holds only one device and its neighbours, so it reports the real network
    ids it was given rather than its local slot numbering.
    """
    ctx = current_context()
    return ctx.node_ids


class field:
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
    def with_overrides(field: Tensor, overrides: tuple[int, float]) -> Tensor:
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
                f"Length of values ({tensor.shape[0]}) does not match number of nodes ({ctx.num_nodes})"
            )
        return tensor
