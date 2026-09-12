"""Control-flow layers such as branch and mux."""

from __future__ import annotations

import warnings

from torch import Tensor, nn

from ..constants import DEFAULT_TAU_BRANCH
from ..core import RoundContext, resolve_context, sub_context
from ..core.mode import get_default_mode
from ..functional import field_where, mask_edges_for_partition


class BranchLayer(nn.Module):
    """Dynamic subgraph partitioning with cross-partition isolation.

    The two partitions occupy distinct alignment frames (``.../br/T/`` and
    ``.../br/F/``), so their persistent state can never alias.  After both have
    run, each partition's slots are restricted to its own nodes, which realises
    the availability mask of the field-calculus store: a node that is not in a
    partition reads that partition's initializers rather than whatever the
    vectorised evaluation happened to compute for it.
    """

    def __init__(
        self,
        true_branch: nn.Module,
        false_branch: nn.Module,
        branch_name: str | None = None,
        reset_states: dict[str, Tensor] | None = None,
        mode: str | None = None,
        tau: float | None = None,
    ) -> None:
        super().__init__()
        self.true_branch = true_branch
        self.false_branch = false_branch
        self.branch_name = branch_name
        self.mode = mode
        self.tau = tau
        if reset_states:
            warnings.warn(
                "branch(reset_states=...) is ignored and will be removed: "
                "alignment now resets a partition's state for nodes outside it "
                "automatically.",
                DeprecationWarning,
                stacklevel=3,
            )

    def forward(
        self, x: Tensor, cond: Tensor, ctx: RoundContext | None = None
    ) -> Tensor:
        """Run branch."""
        ctx = resolve_context(ctx)

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

        with ctx.align.frame("br", self.branch_name):
            with ctx.align.frame("T") as frame_true:
                ctx_true = sub_context(
                    ctx, edge_index_true, edge_weight_true, message_weight_true
                )
                out_true = self.true_branch(x, ctx_true)
            with ctx.align.frame("F") as frame_false:
                ctx_false = sub_context(
                    ctx, edge_index_false, edge_weight_false, message_weight_false
                )
                out_false = self.false_branch(x, ctx_false)

        keep_true = cond.float()
        ctx.state.restrict(
            frame_true.minted, keep=keep_true, mode=effective_mode, tau=effective_tau
        )
        ctx.state.restrict(
            frame_false.minted,
            keep=1.0 - keep_true,
            mode=effective_mode,
            tau=effective_tau,
        )

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
