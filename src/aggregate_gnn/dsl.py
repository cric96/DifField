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
)
from .functional import soft_where
from .pyg_backend import Data

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
    expr: Tensor,
    aggr: str | Callable = "sum",
    mode: str = "hard",
    tau: float = DEFAULT_TAU_SOFT_AGGR,
    fill_value: float | None = None,
    edge_index: Tensor | None = None,
    edge_weight: Tensor | None = None,
    tag: str | None = None,
) -> Tensor:
    r"""Neighborhood message passing.

    nbr(expr, aggr) ≡ m_i = ⊕_{j∈N(i)} expr_j

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

    def __init__(self, num_neighbors: int, *, self_loop: bool = True) -> None:
        self._num_neighbors = num_neighbors
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
        self._agg_ctx = AggregateContext(edge_index, N)

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
        """
        with self._agg_ctx.round() as ctx:
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
