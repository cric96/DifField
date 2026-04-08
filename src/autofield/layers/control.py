"""Control-flow layers such as branch and mux."""

from __future__ import annotations

import torch.nn as nn
from torch import Tensor

from ..constants import DEFAULT_TAU_BRANCH
from ..core import RoundContext, resolve_context, sub_context
from ..core.mode import get_default_mode
from ..functional import field_where, mask_edges_for_partition


class BranchLayer(nn.Module):
    """Dynamic subgraph partitioning with cross-partition isolation."""

    def __init__(
        self,
        true_branch: nn.Module,
        false_branch: nn.Module,
        branch_name: str = "branch",
        reset_states: dict[str, Tensor] | None = None,
        mode: str | None = None,
        tau: float | None = None,
    ) -> None:
        super().__init__()
        self.true_branch = true_branch
        self.false_branch = false_branch
        self.branch_name = branch_name
        self.reset_states = reset_states or {}
        self.mode = mode
        self.tau = tau

    def forward(
        self, x: Tensor, cond: Tensor, ctx: RoundContext | None = None
    ) -> Tensor:
        """Run branch."""
        ctx = resolve_context(ctx)
        switched = ctx.state.track_branch(self.branch_name, cond)
        if self.reset_states:
            ctx.state.reset_states_for_nodes(switched, self.reset_states)

        effective_mode = self.mode if self.mode is not None else get_default_mode()
        effective_tau = self.tau if self.tau is not None else DEFAULT_TAU_BRANCH

        edge_index_true, edge_weight_true, message_weight_true = (
            mask_edges_for_partition(
                ctx.edge_index,
                cond,
                partition=True,
                edge_weight=ctx.edge_weight,
                message_weight=ctx.message_weight,
                mode=effective_mode,
                tau=effective_tau,
            )
        )
        edge_index_false, edge_weight_false, message_weight_false = (
            mask_edges_for_partition(
                ctx.edge_index,
                cond,
                partition=False,
                edge_weight=ctx.edge_weight,
                message_weight=ctx.message_weight,
                mode=effective_mode,
                tau=effective_tau,
            )
        )

        saved_states = ctx.state.snapshot()

        ctx_true = sub_context(
            ctx, edge_index_true, edge_weight_true, message_weight_true
        )
        out_true = self.true_branch(x, ctx_true)
        states_after_true = ctx.state.snapshot()

        ctx.state.restore(saved_states)
        ctx_false = sub_context(
            ctx, edge_index_false, edge_weight_false, message_weight_false
        )
        out_false = self.false_branch(x, ctx_false)
        states_after_false = ctx.state.snapshot()

        all_keys = set(states_after_true) | set(states_after_false)
        for key in all_keys:
            state_after_true = states_after_true.get(key)
            state_after_false = states_after_false.get(key)
            if state_after_true is not None and state_after_false is not None:
                ctx.state.update(
                    field_where(
                        cond,
                        state_after_true,
                        state_after_false,
                        mode=effective_mode,
                        tau=effective_tau,
                    ),
                    name=key,
                )
            elif state_after_true is not None:
                ctx.state.update(state_after_true, name=key)
            else:
                ctx.state.update(state_after_false, name=key)

        return field_where(
            cond, out_true, out_false, mode=effective_mode, tau=effective_tau
        )


class MuxLayer(nn.Module):
    """Pointwise conditional: evaluate both branches, then select per node."""

    def __init__(
        self,
        true_branch: nn.Module,
        false_branch: nn.Module,
        mode: str | None = None,
        tau: float | None = None,
    ) -> None:
        super().__init__()
        self.true_branch = true_branch
        self.false_branch = false_branch
        self.mode = mode
        self.tau = tau

    def forward(
        self, x: Tensor, cond: Tensor, ctx: RoundContext | None = None
    ) -> Tensor:
        """Run mux."""
        ctx = resolve_context(ctx)
        out_true = self.true_branch(x, ctx)
        out_false = self.false_branch(x, ctx)
        effective_mode = self.mode if self.mode is not None else get_default_mode()
        effective_tau = self.tau if self.tau is not None else DEFAULT_TAU_BRANCH
        return field_where(
            cond, out_true, out_false, mode=effective_mode, tau=effective_tau
        )
