"""Aggregate Computing primitives as composable nn.Module layers.

Each layer follows the signature:
    forward(x: Tensor, ctx: RoundContext) -> Tensor
so they can be freely nested and composed.
"""

from __future__ import annotations

import inspect
from typing import Callable, Optional

import torch
import torch.nn as nn
from torch import Tensor

from .constants import (
    CONDITION_THRESHOLD,
    DEFAULT_TAU_BRANCH,
    DEFAULT_TAU_SOFT_AGGR,
    FILL_VALUE_DEFAULT,
    FILL_VALUE_MAX,
    FILL_VALUE_MIN,
)
from .core import RoundContext
from .functional import mask_edges_for_partition, scatter_aggr, soft_where


def _get_ctx(ctx: RoundContext | None) -> RoundContext:
    """Return *ctx* if given, else look up the DSL thread-local context.

    This lets every layer work in two modes:
    1. Explicit:  layer.forward(x, ctx=ctx)   — standalone / test usage
    2. Implicit:  layer(x)                     — inside `with ctx.round():`
    """
    if ctx is not None:
        return ctx
    # Import here to avoid circular import at module level
    from .dsl import _current_ctx
    return _current_ctx()


def _register_callable(
    module: nn.Module,
    attr_name: str,
    fn: Callable | nn.Module,
    child_name: str,
) -> None:
    """Assign *fn* as ``module.<attr_name>`` and register it as a sub-module
    when *fn* is an ``nn.Module``.
    """
    setattr(module, attr_name, fn)
    if isinstance(fn, nn.Module):
        module.add_module(child_name, fn)


# ---------------------------------------------------------------------------
# rep – temporal evolution (per-node recurrent cell)
# ---------------------------------------------------------------------------

class RepLayer(nn.Module):
    r"""Temporal recurrence:  s_i^{(t)} = update_fn(s_i^{(t-1)}, x_i, ctx).

    GNN Equivalence
    ---------------
    ``rep`` is isomorphic to a **per-node Recurrent Neural Network cell**
    (Elman / GRU / LSTM style).  The event-structure causality relation
    e_{i,t} → e_{i,t+1} maps directly to the hidden-state recurrence
    h_i^{(t)} = f_θ(h_i^{(t-1)}).  When the update_fn contains a ``NbrLayer``
    (i.e. ``rep`` wraps ``nbr``), the composition becomes a full **Recurrent
    MPNN** — equivalent to a weight-tied GNN applied T times::

        h_i^(t) = γ_θ(h_i^(t-1), ⊕_{j∈N(i)} φ_θ(h_j^(t-1)))

    Weight-tying across rounds mirrors the AC property that every device
    runs the *same* program at every round.

    Parameters
    ----------
    name : str
        Unique state identifier (for the StateManager).
    init_value : float or Tensor
        Initial state s^{(0)}.
    update_fn : callable(state, x, ctx) -> new_state
        Or a nn.Module with the same signature.
    """

    def __init__(
        self,
        name: str,
        init_value: float | Tensor,
        update_fn: Callable[[Tensor, Tensor, RoundContext], Tensor] | nn.Module,
    ) -> None:
        super().__init__()
        self.name = name
        self.init_value = init_value
        _register_callable(self, "update_fn", update_fn, "_update_fn")

    def forward(self, x: Tensor, ctx: RoundContext | None = None) -> Tensor:
        """Run one step of the recurrence.

        *ctx* is optional — when omitted the active DSL context is used,
        allowing ``RepLayer`` to work inside ``with ctx.round():`` blocks.

        ``update_fn`` may have 1, 2, or 3 positional parameters:
          - ``fn(state)``            — DSL style
          - ``fn(state, x)``         — input-aware
          - ``fn(state, x, ctx)``    — full access
        """
        ctx = _get_ctx(ctx)
        state = ctx.state.get_or_init(self.name, self.init_value)
        # Introspect update_fn arity to support 1, 2, or 3 positional args.
        # Fallback to 3 (full access) when signature cannot be determined
        # (e.g. built-in or C-extension callables).
        try:
            target = self.update_fn.forward if isinstance(self.update_fn, nn.Module) else self.update_fn
            sig = inspect.signature(target)
            num_positional_params = len(
                [p for p in sig.parameters.values()
                 if p.default is inspect.Parameter.empty]
            )
        except (ValueError, TypeError):
            num_positional_params = 3

        if num_positional_params <= 1:
            new_state = self.update_fn(state)
        elif num_positional_params == 2:
            new_state = self.update_fn(state, x)
        else:
            new_state = self.update_fn(state, x, ctx)
        ctx.state.update(self.name, new_state)
        return new_state


# ---------------------------------------------------------------------------
# nbr – message passing (MPNN aggregate step)
# ---------------------------------------------------------------------------

class NbrLayer(nn.Module):
    r"""Neighborhood aggregation:  m_i = ⊕_{j∈N(i)} φ(x_j).

    GNN Equivalence
    ---------------
    ``nbr`` is isomorphic to the **aggregate step** of a Message-Passing
    Neural Network (MPNN).  The event-structure communication relation
    e_{i,t} ↝ e_{j,t+1} maps to edges in the message-passing graph.
    The aggregation ⊕ ∈ {sum, mean, min, max} is shared between AC and GNN.

    Specific instances:
    - ``nbr(x, sum)``  ≡  GCN-style aggregation  (Â · X)
    - ``nbr(x, mean)`` ≡  mean-aggregation GNN   (D⁻¹ · A · X)
    - ``nbr(x, min)``  ≡  min-aggregation MPNN   (Bellman-Ford style)
    - ``nbr(φ(x), sum)`` with learnable φ ≡ general MPNN message step

    Parameters
    ----------
    aggr : str or callable
        Built-in: "sum", "mean", "min", "max".
        Custom: ``aggr(msg, index, num_nodes) -> Tensor [N, *F]``.
        When callable, *mode*/*tau*/*fill_value* are ignored.
    transform_fn : optional callable or nn.Module
        Applied to source features before aggregation (the φ function).
    mode : str  "hard" | "soft"
    tau : float
        Temperature for soft min/max.
    fill_value : float
        Fill for nodes with no incoming messages (matters for min/max).
    """

    def __init__(
        self,
        aggr: str | Callable = "sum",
        transform_fn: Optional[Callable[[Tensor], Tensor] | nn.Module] = None,
        mode: str = "hard",
        tau: float = DEFAULT_TAU_SOFT_AGGR,
        fill_value: float | None = None,
        tag: str | None = None,
    ) -> None:
        super().__init__()
        self.aggr = aggr
        self.mode = mode
        self.tau = tau
        self.tag = tag
        if fill_value is None:
            if isinstance(aggr, str):
                self.fill_value = (
                    FILL_VALUE_MIN if aggr == "min"
                    else FILL_VALUE_MAX if aggr == "max"
                    else FILL_VALUE_DEFAULT
                )
            else:
                self.fill_value = FILL_VALUE_DEFAULT
        else:
            self.fill_value = fill_value
        if transform_fn is not None:
            _register_callable(self, "transform_fn", transform_fn, "_transform_fn")
        else:
            self.transform_fn = None

    def forward(
        self,
        x: Tensor,
        ctx: RoundContext | None = None,
        edge_index: Tensor | None = None,
        edge_weight: Tensor | None = None,
        tag: str | None = None,
    ) -> Tensor:
        """Aggregate neighbor features.

        *ctx* is optional — when omitted the active DSL context is used.
        *tag* overrides ``self.tag`` when given (DSL pass-through).
        """
        ctx = _get_ctx(ctx)
        effective_tag = tag if tag is not None else self.tag

        # Store export under tag (original expr values)
        if effective_tag is not None:
            ctx.exports[effective_tag] = x

        # Use injected overrides when running in DeviceContext
        src = x
        if effective_tag is not None and effective_tag in ctx._neighbor_message_overrides:
            src = ctx._neighbor_message_overrides[effective_tag]

        edge_idx = edge_index if edge_index is not None else ctx.edge_index
        edge_wt = edge_weight if edge_weight is not None else ctx.edge_weight

        src_nodes, tgt_nodes = edge_idx[0], edge_idx[1]

        msg = src[src_nodes]  # [E, *F]
        if self.transform_fn is not None:
            msg = self.transform_fn(msg)

        if edge_wt is not None:
            w = edge_wt
            if msg.dim() > 1:
                w = w.unsqueeze(-1)
            msg = msg * w

        return scatter_aggr(
            msg, tgt_nodes, ctx.num_nodes,
            aggr=self.aggr, mode=self.mode, tau=self.tau, fill_value=self.fill_value,
        )


# ---------------------------------------------------------------------------
# branch – domain restriction with state reset
# ---------------------------------------------------------------------------

class BranchLayer(nn.Module):
    """Dynamic subgraph partitioning with cross-partition isolation.

    branch(cond)(true_branch)(false_branch)

    GNN Equivalence
    ---------------
    ``branch`` is isomorphic to a **dynamic subgraph GNN with state reset**.
    The event-structure conflict relation e_{i,t} # e_{j,t} (when i and j
    are in different branches) maps to **edge removal** in the GNN:

        E'_k = {(i,j) ∈ E | c_i = k ∧ c_j = k},   k ∈ {0, 1}

    Each partition runs an independent MPNN on its induced subgraph.
    When a node switches branch (c_i^(t) ≠ c_i^(t-1)), its recurrent
    state resets to init — this has no direct GNN analogue but matches
    the AC semantic of "entering a fresh computation domain".

    Soft mode uses sigmoid edge weights for differentiable masking:
        α_ij = σ(τ · (c_i · c_j + (1-c_i)(1-c_j) - 0.5))

    Parameters
    ----------
    true_branch, false_branch : nn.Module
        Sub-programs for each partition.  Signature: forward(x, ctx) -> Tensor.
    branch_name : str
        Unique identifier for branch-switch tracking.
    reset_states : dict[str, float|Tensor] | None
        State names → init values to reset on branch switch.
    mode, tau : hard/soft edge masking parameters.
    """

    def __init__(
        self,
        true_branch: nn.Module,
        false_branch: nn.Module,
        branch_name: str = "branch",
        reset_states: dict[str, float | Tensor] | None = None,
        mode: str = "hard",
        tau: float = DEFAULT_TAU_BRANCH,
    ) -> None:
        super().__init__()
        self.true_branch = true_branch
        self.false_branch = false_branch
        self.branch_name = branch_name
        self.reset_states = reset_states or {}
        self.mode = mode
        self.tau = tau

    def forward(self, x: Tensor, cond: Tensor, ctx: RoundContext | None = None) -> Tensor:
        """Run branch.

        *ctx* is optional — when omitted the active DSL context is used.
        """
        ctx = _get_ctx(ctx)
        # Track branch switches and reset states
        switched = ctx.state.track_branch(self.branch_name, cond)
        if self.reset_states:
            ctx.state.reset_states_for_nodes(switched, self.reset_states)

        # Mask edges and weights for each partition
        ei_true, ew_true = mask_edges_for_partition(
            ctx.edge_index, cond.bool(), partition=True, edge_weight=ctx.edge_weight,
            mode=self.mode, tau=self.tau,
        )
        ei_false, ew_false = mask_edges_for_partition(
            ctx.edge_index, cond.bool(), partition=False, edge_weight=ctx.edge_weight,
            mode=self.mode, tau=self.tau,
        )

        # Snapshot state before running branches
        saved_states = {k: v.clone() for k, v in ctx.state._states.items()}

        # Run true branch
        ctx_true = _sub_context(ctx, ei_true, ew_true)
        out_true = self.true_branch(x, ctx_true)
        states_after_true = {k: v.clone() for k, v in ctx.state._states.items()}

        # Restore state fully (remove keys created by the true branch)
        ctx.state._states = {k: v.clone() for k, v in saved_states.items()}
        ctx_false = _sub_context(ctx, ei_false, ew_false)
        out_false = self.false_branch(x, ctx_false)
        states_after_false = {k: v.clone() for k, v in ctx.state._states.items()}

        # Merge per-key: where cond >= CONDITION_THRESHOLD pick the state
        # produced by the true branch, otherwise the false branch.
        all_keys = set(states_after_true) | set(states_after_false)
        for key in all_keys:
            state_after_true = states_after_true.get(key)
            state_after_false = states_after_false.get(key)
            if state_after_true is not None and state_after_false is not None:
                cond_float = cond.float()
                if cond_float.dim() < state_after_true.dim():
                    cond_float = cond_float.unsqueeze(-1)
                ctx.state._states[key] = torch.where(
                    cond_float >= CONDITION_THRESHOLD,
                    state_after_true, state_after_false,
                )
            elif state_after_true is not None:
                ctx.state._states[key] = state_after_true
            else:
                ctx.state._states[key] = state_after_false

        return soft_where(cond, out_true, out_false)


# ---------------------------------------------------------------------------
# mux – pointwise conditional selection (no topology change)
# ---------------------------------------------------------------------------

class MuxLayer(nn.Module):
    """Pointwise conditional: compute BOTH branches on full topology, select per node.

    mux(cond)(true_branch)(false_branch)

    GNN Equivalence
    ---------------
    ``mux`` is isomorphic to **element-wise gating** — the same mechanism
    used in GRU gates, attention masks, and mixture-of-experts routing:

        y_i = c_i · e1_i + (1 - c_i) · e2_i

    Unlike ``branch``, the event structure has **no conflict**: all nodes
    communicate over the full topology regardless of condition c.  Both
    sub-expressions are evaluated on the complete graph.
    """

    def __init__(
        self,
        true_branch: nn.Module,
        false_branch: nn.Module,
    ) -> None:
        super().__init__()
        self.true_branch = true_branch
        self.false_branch = false_branch

    def forward(self, x: Tensor, cond: Tensor, ctx: RoundContext | None = None) -> Tensor:
        """Run mux.

        *ctx* is optional — when omitted the active DSL context is used.
        """
        ctx = _get_ctx(ctx)
        out_true = self.true_branch(x, ctx)
        out_false = self.false_branch(x, ctx)
        return soft_where(cond, out_true, out_false)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _sub_context(ctx: RoundContext, edge_index: Tensor, edge_weight: Tensor) -> RoundContext:
    """Create a shallow copy of *ctx* with a different topology.

    The ``state`` is **shared** so that branch sub-programs can read / write
    the same recurrent states.  ``exports`` and message overrides are
    forwarded so that tagged ``nbr`` calls keep working.
    """
    sub = RoundContext.__new__(RoundContext)
    sub.edge_index = edge_index
    sub.edge_weight = edge_weight
    sub.num_nodes = ctx.num_nodes
    sub.round_num = ctx.round_num
    sub.state = ctx.state  # shared state
    sub.exports = ctx.exports
    sub._neighbor_message_overrides = ctx._neighbor_message_overrides
    return sub
