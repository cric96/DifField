"""Functional DSL for Aggregate Computing on graphs.

Each DSL primitive delegates to its corresponding ``nn.Module`` layer,
ensuring a **single implementation** of the computation logic.

Usage
-----
    ctx = AggregateContext(edge_index, num_nodes)

    for t in range(T):
        with ctx.round():
            d = rep("dist", float('inf'), lambda d:
                mux(source,
                    const(0.0),
                    nbr(d + 1, aggr="min"),
                )
            )
"""

from __future__ import annotations

import threading
from contextlib import contextmanager
from typing import Callable

import torch
import torch.nn as nn
from torch import Tensor

from .core import RoundContext
from .constants import (
    BROADCAST_NEAR_ZERO,
    DEFAULT_TAU_BRANCH,
    DEFAULT_TAU_SOFT_AGGR,
    LOG_EPSILON,
)
from .functional import scatter_aggr, scatter_binary_fold, scatter_min_by_first, soft_where
from .pyg_backend import Data, maybe_make_data

# ---------------------------------------------------------------------------
# Thread-local context stack (allows nesting)
# ---------------------------------------------------------------------------

_thread_local = threading.local()


def _ctx_stack() -> list[RoundContext]:
    if not hasattr(_thread_local, "ctx_stack"):
        _thread_local.ctx_stack = []
    return _thread_local.ctx_stack


def _current_ctx() -> RoundContext:
    stack = _ctx_stack()
    if not stack:
        raise RuntimeError("No active AggregateContext. Use `with ctx.round(): ...`")
    return stack[-1]


def _ensure_field(value: float | Tensor) -> Tensor:
    ctx = _current_ctx()
    return _ensure_field_in_ctx(value, ctx)


def _ensure_field_in_ctx(value: float | Tensor, ctx: RoundContext) -> Tensor:
    if isinstance(value, Tensor):
        tensor = value.to(ctx.edge_index.device)
        if tensor.dim() == 0:
            return tensor.expand(ctx.num_nodes)
        if tensor.shape[0] != ctx.num_nodes:
            raise ValueError(
                f"Expected first dimension {ctx.num_nodes}, got {tuple(tensor.shape)}",
            )
        return tensor
    return torch.full((ctx.num_nodes,), float(value), dtype=torch.float32, device=ctx.edge_index.device)


def _broadcast_like(value: float | Tensor, template: Tensor) -> Tensor:
    if isinstance(value, Tensor):
        tensor = value.to(device=template.device, dtype=template.dtype)
        if tensor.dim() == 0:
            return tensor.expand_as(template)
        if tensor.shape == template.shape:
            return tensor
        if tensor.shape == template.shape[1:]:
            return tensor.unsqueeze(0).expand_as(template)
        raise ValueError(
            f"Cannot broadcast shape {tuple(tensor.shape)} to {tuple(template.shape)}",
        )
    return torch.full_like(template, float(value))


def _pack_cast_state(distance: Tensor, payload: Tensor) -> Tensor:
    payload_flat = payload.unsqueeze(-1) if payload.dim() == 1 else payload.reshape(payload.shape[0], -1)
    return torch.cat((distance.unsqueeze(-1), payload_flat), dim=-1)


def _unpack_cast_state(packed: Tensor, payload_shape: tuple[int, ...]) -> tuple[Tensor, Tensor]:
    distance = packed[:, 0]
    payload_flat = packed[:, 1:]
    if not payload_shape:
        return distance, payload_flat[:, 0]
    return distance, payload_flat.reshape((packed.shape[0],) + payload_shape)


def _require_scalar_field(name: str, value: float | Tensor) -> Tensor:
    field_value = _ensure_field(value)
    if field_value.dim() != 1:
        raise ValueError(f"{name} must be a scalar field shaped [num_nodes]")
    return field_value


def _validate_cast_mode(mode: str) -> None:
    if mode not in {"hard", "soft"}:
        raise ValueError(f"Unknown mode: {mode}")


def _admissible_parent_edges(potential: Tensor) -> tuple[Tensor, Tensor, Tensor]:
    ctx = _current_ctx()
    src, tgt = ctx.edge_index
    parent_potential = potential[src]
    child_potential = potential[tgt]
    path_cost = parent_potential + ctx.edge_weight
    tolerance = 1e-6 + 1e-5 * torch.maximum(path_cost.abs(), child_potential.abs())
    admissible = (parent_potential < child_potential) & (path_cost <= child_potential + tolerance)
    return admissible, parent_potential, path_cost


def _find_parent_ids(potential: Tensor) -> Tensor:
    ctx = _current_ctx()
    src, tgt = ctx.edge_index
    admissible, parent_potential, _path_cost = _admissible_parent_edges(potential)
    parent_candidates = torch.stack(
        (
            torch.where(
                admissible,
                parent_potential,
                torch.full_like(parent_potential, float("inf")),
            ),
            src.float(),
        ),
        dim=-1,
    )
    fill_row = torch.stack((field.inf(), const(-1.0)), dim=-1)
    best_parent = scatter_min_by_first(
        parent_candidates,
        tgt,
        ctx.num_nodes,
        mode="hard",
        tau=DEFAULT_TAU_SOFT_AGGR,
        fill_row=fill_row,
    )
    candidate_potential = best_parent[:, 0]
    candidate_parent = best_parent[:, 1].round().long()
    sentinel = torch.full_like(candidate_parent, -1)
    return torch.where(torch.isfinite(candidate_potential), candidate_parent, sentinel)


def _soft_parent_weights(potential: Tensor, tau: float) -> Tensor:
    ctx = _current_ctx()
    src, tgt = ctx.edge_index
    effective_tau = max(float(tau), LOG_EPSILON)
    parent_potential = potential[tgt]
    child_potential = potential[src]
    path_cost = parent_potential + ctx.edge_weight
    lower_gate = torch.sigmoid((child_potential - parent_potential) / effective_tau)
    path_gate = torch.sigmoid((child_potential - path_cost) / effective_tau)
    logits = -parent_potential / effective_tau
    bucket_max = scatter_aggr(logits, src, ctx.num_nodes, aggr="max", fill_value=float("-inf"))
    stabilized = (logits - bucket_max[src]).exp() * lower_gate * path_gate
    denom = scatter_aggr(stabilized, src, ctx.num_nodes, aggr="sum")
    return torch.where(
        denom[src] > 0,
        stabilized / denom[src].clamp(min=LOG_EPSILON),
        torch.zeros_like(stabilized),
    )


def _scale_messages(messages: Tensor, weights: Tensor) -> Tensor:
    scaled_weights = weights
    while scaled_weights.dim() < messages.dim():
        scaled_weights = scaled_weights.unsqueeze(-1)
    return messages * scaled_weights


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
            )
        )

    def _rbinary(self, other: float | Tensor | "NeighborExpr", op: Callable[[Tensor, Tensor], Tensor]) -> "NeighborExpr":
        other_expr = _as_neighbor_expr(other)
        return NeighborExpr(
            lambda ctx, edge_index, edge_weight: op(
                other_expr.evaluate(ctx=ctx, edge_index=edge_index, edge_weight=edge_weight),
                self.evaluate(ctx=ctx, edge_index=edge_index, edge_weight=edge_weight),
            )
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
            )
        )


def _as_neighbor_expr(value: float | Tensor | NeighborExpr) -> NeighborExpr:
    if isinstance(value, NeighborExpr):
        return value

    def evaluate(ctx: RoundContext, edge_index: Tensor, _edge_weight: Tensor | None) -> Tensor:
        source_nodes = edge_index[0]
        field_value = _ensure_field_in_ctx(value, ctx)
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
    """Field-calculus-style alias for :func:`nbr_range`."""

    return nbr_range()


# ---------------------------------------------------------------------------
# _LambdaModule – adapter from callable → nn.Module for BranchLayer
# ---------------------------------------------------------------------------

class _LambdaModule(nn.Module):
    """Wraps a zero-arg callable for use as a branch sub-program.

    When called with ``(x, ctx)``, pushes *ctx* onto the DSL context stack
    so that inner DSL calls (rep, nbr, …) see the correct sub-context
    (e.g. masked edges inside a branch partition).
    """

    def __init__(self, fn: Callable[[], Tensor]) -> None:
        super().__init__()
        self._fn = fn

    def forward(self, x: Tensor, ctx: RoundContext | None = None) -> Tensor:
        if ctx is not None:
            _ctx_stack().append(ctx)
        try:
            return self._fn()
        finally:
            if ctx is not None:
                _ctx_stack().pop()


# ---------------------------------------------------------------------------
# AggregateContext – entry point
# ---------------------------------------------------------------------------

class AggregateContext:
    """Top-level context wrapping a graph and execution state.

    Parameters
    ----------
    edge_index : Tensor [2, E]
    num_nodes : int
    edge_weight : Tensor [E], optional
    """

    def __init__(
        self,
        edge_index: Tensor | Data,
        num_nodes: int | None = None,
        edge_weight: Tensor | None = None,
    ) -> None:
        if isinstance(edge_index, Data):
            data = edge_index
            if data.num_nodes is None:
                raise ValueError("PyG Data must define num_nodes for AggregateContext")
            data_edge_weight = edge_weight
            if data_edge_weight is None and hasattr(data, "edge_attr"):
                data_edge_weight = data.edge_attr
            self._ctx = RoundContext(data.edge_index, int(data.num_nodes), edge_weight=data_edge_weight)
            self._ctx.data = data
            return

        if num_nodes is None:
            raise ValueError("num_nodes is required when constructing AggregateContext from edge_index")
        self._ctx = RoundContext(edge_index, num_nodes, edge_weight=edge_weight)

    @contextmanager
    def round(self):
        """Execute one round of the aggregate program."""
        _ctx_stack().append(self._ctx)
        try:
            with self._ctx.round():
                yield self._ctx
        finally:
            _ctx_stack().pop()

    def reset(self) -> None:
        self._ctx.reset()

    @property
    def round_num(self) -> int:
        return self._ctx.round_num


# ---------------------------------------------------------------------------
# DSL primitives – thin wrappers that delegate to nn.Module layers
# ---------------------------------------------------------------------------

def rep(
    name: str,
    init: float | Tensor,
    fn: Callable[[Tensor], Tensor],
) -> Tensor:
    r"""Temporal evolution (per-node recurrent state).

    rep(name, init, f) ≡ s^{(t)} = f(s^{(t-1)}),  s^{(0)} = init

    Delegates to :class:`~aggregate_gnn.layers.RepLayer`.
    """
    from .layers import RepLayer
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
) -> Tensor:
    r"""Neighborhood message passing.

    nbr(expr, aggr) ≡ m_i = ⊕_{j∈N(i)} expr_{j→i}

    Parameters
    ----------
    tag : str, optional
        Named message channel.  When given, *expr* is stored in
        ``ctx.exports[tag]`` and, in :class:`DeviceContext`, neighbor
        values injected via ``neighbor_messages={tag: [...]}`` replace
        the expression for neighbor nodes.

    Delegates to :class:`~aggregate_gnn.layers.NbrLayer`.
    """
    from .layers import NbrLayer
    return NbrLayer(aggr=aggr, mode=mode, tau=tau, fill_value=fill_value)(
        expr, edge_index=edge_index, edge_weight=edge_weight, tag=tag,
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
    r"""Domain restriction with communication isolation and state reset.

    branch(cond, p1, p2) – nodes with cond=True run p1, cond=False run p2.
    Messages do NOT cross partitions.

    Delegates to :class:`~aggregate_gnn.layers.BranchLayer`.
    """
    from .layers import BranchLayer
    true_mod = _LambdaModule(if_true) if not isinstance(if_true, nn.Module) else if_true
    false_mod = _LambdaModule(if_false) if not isinstance(if_false, nn.Module) else if_false
    return BranchLayer(
        true_mod, false_mod,
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
    r"""Pointwise conditional selection (no topology change).

    mux(cond, e1, e2) – BOTH e1 and e2 evaluated on full graph.
    Output: cond * e1 + (1-cond) * e2.

    Equivalent GNN: element-wise gating.
    """
    val_true = if_true() if callable(if_true) else if_true
    val_false = if_false() if callable(if_false) else if_false
    return soft_where(cond, val_true, val_false)


def broadcast(mask: Tensor, value: Tensor, name: str = "bc") -> Tensor:
    r"""Propagate a value from root nodes (where mask is True) to the rest of the network.

    broadcast(m, v) ≡ rep("bc", inf)(bc => mux(m, v, nbr(bc, aggr="min")))

    Parameters
    ----------
    mask : Tensor [N] (bool or float)
        The roots of the broadcast. If float, it's treated as True where value <= 0.05.
    value : Tensor [N]
        The values to broadcast from the roots.
    name : str, optional
        Unique name for the broadcast state.
    """
    # Handle casting: if float (e.g. a gradient), treat near-zero as True
    cond = mask if mask.dtype == torch.bool else (mask <= BROADCAST_NEAR_ZERO)
    
    return rep(f"_bc_{name}", float("inf"), lambda bc:
        mux(cond,
            value,
            nbr(bc, aggr="min")
        )
    )


def gradient(
    source: float | Tensor,
    weight: float | Tensor | NeighborExpr | None = None,
    *,
    name: str = "gradient",
    mode: str = "hard",
    tau: float = DEFAULT_TAU_SOFT_AGGR,
    fill_value: float = float("inf"),
) -> Tensor:
    r"""Compute a minimum-cost gradient / distance field from source nodes.

    When ``weight`` is omitted, the operator uses ``nbrRange()`` so the
    current edge metric drives propagation directly:

    - hop graphs with unit edge weights -> hop distance
    - spatial graphs with geometric edge weights -> geometric shortest path

    Passing ``weight`` explicitly preserves the classic convenience form
    ``gradient(source, w)``.
    """
    _validate_cast_mode(mode)
    source_field = _require_scalar_field("source", source)
    step = nbrRange() if weight is None else weight

    return rep(
        f"_grad_{name}",
        fill_value,
        lambda dist: mux(
            source_field,
            field.of(0.0),
            nbr(dist + step, aggr="min", mode=mode, tau=tau, fill_value=fill_value),
        ),
    )


def gradient_cast(
    source: float | Tensor,
    center: float | Tensor,
    accumulation: Callable[[Tensor], Tensor],
    *,
    name: str = "gradient_cast",
    mode: str = "hard",
    tau: float = DEFAULT_TAU_SOFT_AGGR,
) -> Tensor:
    r"""Propagate payloads outward along a minimum-potential gradient.

    Mirrors the classic field-calculus ``gradientCast`` building block while
    staying inside the differentiable tensor DSL. The propagated state packs
    the path cost in the first column and the payload in the remaining ones.
    """
    _validate_cast_mode(mode)
    source_field = _require_scalar_field("source", source)
    center_field = _ensure_field(center)
    payload_shape = center_field.shape[1:]
    init_state = _pack_cast_state(field.inf(), center_field)
    source_state = _pack_cast_state(field.zeros(), center_field)

    def update(state: Tensor, _x: Tensor, ctx: RoundContext) -> Tensor:
        old_distance, old_payload = _unpack_cast_state(state, payload_shape)
        src, tgt = ctx.edge_index
        messages = _pack_cast_state(
            old_distance[src] + ctx.edge_weight,
            accumulation(old_payload[src]),
        )
        propagated = scatter_min_by_first(
            messages,
            tgt,
            ctx.num_nodes,
            mode=mode,
            tau=tau,
            fill_row=init_state,
        )
        return soft_where(source_field, source_state, propagated)

    state = rep(f"_gc_{name}", init_state, update)
    return _unpack_cast_state(state, payload_shape)[1]


def collect_cast(
    potential: float | Tensor,
    local: float | Tensor,
    null: float | Tensor,
    accumulation: Callable[[Tensor, Tensor], Tensor],
    *,
    name: str = "collect_cast",
    mode: str = "hard",
    tau: float = DEFAULT_TAU_SOFT_AGGR,
) -> Tensor:
    r"""Collect payloads from children toward local minima of a potential field.

    Hard mode matches the standard collect-by-parent-tree semantics. Soft mode
    relaxes the parent choice into differentiable edge weights, which works
    best with pointwise reducers such as sums and weighted sums.
    """
    _validate_cast_mode(mode)
    potential_field = _require_scalar_field("potential", potential)
    local_field = _ensure_field(local)
    null_field = _broadcast_like(null, local_field)

    def update(collected: Tensor, _x: Tensor, ctx: RoundContext) -> Tensor:
        src, tgt = ctx.edge_index
        if mode == "hard":
            parent_ids = _find_parent_ids(potential_field)
            keep = parent_ids[src] == tgt
            child_values = scatter_binary_fold(
                collected[src[keep]],
                tgt[keep],
                ctx.num_nodes,
                accumulation,
                null_field,
            )
        else:
            parent_weight = _soft_parent_weights(potential_field, tau)
            child_values = scatter_binary_fold(
                _scale_messages(collected[src], parent_weight),
                tgt,
                ctx.num_nodes,
                accumulation,
                null_field,
            )
        return accumulation(local_field, child_values)

    return rep(f"_cc_{name}", local_field, update)


def const(value: float) -> Tensor:
    """Broadcast a scalar to all nodes in the current context."""
    ctx = _current_ctx()
    return torch.full((ctx.num_nodes,), value, dtype=torch.float32, device=ctx.edge_index.device)


def mid() -> Tensor:
    """Return tensor of node IDs [0, 1, ..., N-1] as float."""
    ctx = _current_ctx()
    return torch.arange(ctx.num_nodes, dtype=torch.float32, device=ctx.edge_index.device)


# ---------------------------------------------------------------------------
# field – context-aware field constructors for device-agnostic programs
# ---------------------------------------------------------------------------

class field:
    """Context-aware field constructors.

    Usage inside a ``with ctx.round():`` block::

        field.of(0.0)       # → Tensor [N] filled with 0.0
        field.zeros()       # → same as field.of(0.0)
        field.ones()        # → Tensor [N] filled with 1.0
        field.inf()         # → Tensor [N] filled with +∞

    These replace explicit ``torch.zeros(n)`` / ``torch.full((n,), v)``
    calls, making programs agnostic to the number of devices.
    """

    @staticmethod
    def of(value: float) -> Tensor:
        """Uniform field: every device holds *value*."""
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


# ---------------------------------------------------------------------------
# DeviceContext – local (single-device) execution
# ---------------------------------------------------------------------------

class DeviceContext:
    """Run an AC program from the perspective of a single device.

    Internally builds a **star graph**:

    * **Node 0** — *this* device (the one we care about).
    * **Nodes 1 … K** — its *K* neighbours, whose ``rep`` states are
      injected from outside before each round.

    The AC program itself is **unchanged** — it uses the same DSL
    primitives (``rep``, ``nbr``, ``mux``, ``const``, ``field.of``, …)
    and remains completely agnostic to the number of devices.

    Parameters
    ----------
    num_neighbors : int
        Degree of this device (*K*).

    Example
    -------
    ::

        device = DeviceContext(num_neighbors=2)
        for t in range(T):
            with device.round(neighbor_exports={"dist": nbr_dists}):
                d = gradient(source_field, w)
            my_d = device.result(d)
    """

    def __init__(
        self,
        num_neighbors: int,
        *,
        self_loop: bool = True,
        neighbor_ranges: float | list[float] | Tensor = 1.0,
    ) -> None:
        self._num_neighbors = num_neighbors
        self._self_loop = self_loop
        self._default_neighbor_ranges = neighbor_ranges
        N = num_neighbors + 1
        # Star graph: neighbours 1..K → node 0, optionally with self-loop 0→0
        if self_loop:
            src = list(range(N))
            tgt = [0] * N
        else:
            src = list(range(1, N))
            tgt = [0] * num_neighbors
        if len(src) == 0:
            edge_index = torch.zeros((2, 0), dtype=torch.long)
        else:
            edge_index = torch.tensor([src, tgt], dtype=torch.long)
        self._agg_ctx = AggregateContext(edge_index, N, edge_weight=self._build_edge_weight(neighbor_ranges))

    def _neighbor_range_tensor(self, neighbor_ranges: float | list[float] | Tensor) -> Tensor:
        if self._num_neighbors == 0:
            return torch.zeros((0,), dtype=torch.float32)

        raw = torch.as_tensor(neighbor_ranges, dtype=torch.float32)
        if raw.dim() == 0:
            ranges = raw.expand(self._num_neighbors).clone()
        else:
            ranges = raw.flatten()
            if ranges.shape[0] != self._num_neighbors:
                raise ValueError(
                    f"Expected {self._num_neighbors} neighbour ranges, got {ranges.shape[0]}",
                )
            ranges = ranges.clone()

        if (ranges < 0).any():
            raise ValueError("neighbor_ranges must be non-negative")
        return ranges

    def _build_edge_weight(self, neighbor_ranges: float | list[float] | Tensor) -> Tensor:
        neighbor_tensor = self._neighbor_range_tensor(neighbor_ranges)
        if self._self_loop:
            return torch.cat((torch.zeros(1, dtype=torch.float32), neighbor_tensor), dim=0)
        return neighbor_tensor

    # --- helpers ---

    def local_field(self, own: float, *, nbr: float | list[float] | Tensor = 0.0) -> Tensor:
        """Build a field with *own* for this device, *nbr* for neighbours.

        Parameters
        ----------
        own : float
            Value for node 0 (this device).
        nbr : float or list[float] or Tensor
            If a scalar, all K neighbours get this value.
            If a list/Tensor of length K, each neighbour gets its own value.
        """
        N = self._num_neighbors + 1
        if isinstance(nbr, (list, Tensor)):
            t = torch.as_tensor(nbr, dtype=torch.float32)
            f = torch.zeros(N, dtype=torch.float32)
            f[1 : 1 + t.shape[0]] = t
        else:
            f = torch.full((N,), nbr, dtype=torch.float32)
        f[0] = own
        return f

    @staticmethod
    def result(tensor: Tensor) -> Tensor:
        """Extract this device's value (node 0) from a full field."""
        return tensor[0]

    def get_state(self, name: str) -> float:
        """Return this device's ``rep`` state for *name*.

        Use this (not ``result()``) when building ``neighbor_exports``
        for another device — ``rep`` state may differ from the
        observable output when ``mux`` or ``branch`` wraps ``rep``
        externally.
        """
        return self._agg_ctx._ctx.state._states[name][0].item()

    def reset(self) -> None:
        """Clear all state (start from scratch)."""
        self._agg_ctx.reset()

    @property
    def round_num(self) -> int:
        return self._agg_ctx.round_num

    # --- round context manager ---

    @contextmanager
    def round(
        self,
        neighbor_exports: dict[str, list[float] | Tensor] | None = None,
        neighbor_messages: dict[str, list[float] | Tensor] | None = None,
        neighbor_ranges: float | list[float] | Tensor | None = None,
    ):
        """Execute one local round.

        Parameters
        ----------
        neighbor_exports : dict mapping rep-state names to neighbour values
            E.g. ``{"dist": [3.0, 5.0]}`` injects distances received from
            the two neighbours.  Only meaningful from round 2 onwards
            (round 1 uses the ``init`` value of ``rep``).
        neighbor_messages : dict mapping nbr *tags* to neighbour values
            E.g. ``{"my_tag": [1.0, 2.0]}`` — when ``nbr(expr, tag="my_tag")``
            is called, neighbor nodes use the injected values instead of
            evaluating *expr*.
        neighbor_ranges : float or list[float] or Tensor, optional
            Per-neighbour edge metrics for this round. When omitted, the
            default ranges passed to the constructor are used.
        """
        with self._agg_ctx.round() as ctx:
            effective_ranges = self._default_neighbor_ranges if neighbor_ranges is None else neighbor_ranges
            ctx.edge_weight = self._build_edge_weight(effective_ranges).to(ctx.edge_index.device)
            ctx.data = maybe_make_data(ctx.edge_index, ctx.num_nodes, ctx.edge_weight)
            # Overwrite neighbour rep-states (1..K) with externally-received values
            if neighbor_exports:
                for name, vals in neighbor_exports.items():
                    state = ctx.state._states.get(name)
                    if state is not None:
                        t = torch.as_tensor(vals, dtype=state.dtype)
                        if t.dim() == 0:
                            t = t.unsqueeze(0)
                        state[1 : 1 + t.shape[0]] = t
            # Inject tagged nbr messages for neighbor nodes
            if neighbor_messages:
                N = self._num_neighbors + 1
                for tag, vals in neighbor_messages.items():
                    t = torch.as_tensor(vals, dtype=torch.float32)
                    if t.dim() == 0:
                        t = t.unsqueeze(0)
                    override = torch.zeros(N, dtype=torch.float32)
                    override[1 : 1 + t.shape[0]] = t
                    ctx._neighbor_message_overrides[tag] = override
            yield ctx
