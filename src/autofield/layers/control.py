"""Control-flow layers such as branch and mux."""

from __future__ import annotations

import torch.nn as nn
from torch import Tensor

from ..constants import CONDITION_THRESHOLD, DEFAULT_TAU_BRANCH
from ..core import RoundContext, resolve_context, sub_context
from ..functional import mask_edges_for_partition, soft_where


class BranchLayer(nn.Module):
    """Dynamic subgraph partitioning with cross-partition isolation."""

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

    def forward(
        self, x: Tensor, cond: Tensor, ctx: RoundContext | None = None
    ) -> Tensor:
        """Run branch."""
        ctx = resolve_context(ctx)
        switched = ctx.state.track_branch(self.branch_name, cond)
        if self.reset_states:
            ctx.state.reset_states_for_nodes(switched, self.reset_states)

        edge_index_true, edge_weight_true, message_weight_true = (
            mask_edges_for_partition(
                ctx.edge_index,
                cond,
                partition=True,
                edge_weight=ctx.edge_weight,
                message_weight=ctx.message_weight,
                mode=self.mode,
                tau=self.tau,
            )
        )
        edge_index_false, edge_weight_false, message_weight_false = (
            mask_edges_for_partition(
                ctx.edge_index,
                cond,
                partition=False,
                edge_weight=ctx.edge_weight,
                message_weight=ctx.message_weight,
                mode=self.mode,
                tau=self.tau,
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
                cond_float = cond.float()
                if cond_float.dim() < state_after_true.dim():
                    cond_float = cond_float.unsqueeze(-1)
                ctx.state.update(
                    key,
                    soft_where(
                        cond_float >= CONDITION_THRESHOLD,
                        state_after_true,
                        state_after_false,
                    ),
                )
            elif state_after_true is not None:
                ctx.state.update(key, state_after_true)
            else:
                ctx.state.update(key, state_after_false)

        return soft_where(cond, out_true, out_false)


class MuxLayer(nn.Module):
    """Pointwise conditional: evaluate both branches, then select per node."""

    def __init__(self, true_branch: nn.Module, false_branch: nn.Module) -> None:
        super().__init__()
        self.true_branch = true_branch
        self.false_branch = false_branch

    def forward(
        self, x: Tensor, cond: Tensor, ctx: RoundContext | None = None
    ) -> Tensor:
        """Run mux."""
        ctx = resolve_context(ctx)
        out_true = self.true_branch(x, ctx)
        out_false = self.false_branch(x, ctx)
        return soft_where(cond, out_true, out_false)
