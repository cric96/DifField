"""Graph execution contexts for aggregate programs."""

from __future__ import annotations

from contextlib import contextmanager

import torch
from torch import Tensor

from ..pyg_backend import maybe_make_data
from .alignment import AlignedDict, Aligner, path_has_label
from .state import StateManager


class RoundContext:
    """Holds the graph topology and per-round execution state."""

    def __init__(
        self, edge_index: Tensor, num_nodes: int, edge_weight: Tensor | None = None
    ) -> None:
        self.edge_index = edge_index # shape [2, num_edges]
        self.edge_weight = (
            edge_weight
            if edge_weight is not None
            else torch.ones(
                edge_index.shape[1],
                device=edge_index.device,
            )
        )
        self.message_weight: Tensor | None = None
        self.num_nodes = num_nodes
        # Network-wide identity of each node, which a partial view (a single
        # device and its neighbours) can override with the real ids.
        self.node_ids = torch.arange(
            num_nodes, dtype=torch.float32, device=edge_index.device
        )
        self.data = maybe_make_data(self.edge_index, self.num_nodes, self.edge_weight)
        self.round_num = 0
        self.state = StateManager(num_nodes, device=edge_index.device)
        self.align = Aligner()
        self.exports: AlignedDict = AlignedDict()
        self._neighbor_message_overrides: AlignedDict = AlignedDict()

    @contextmanager
    def round(self):
        """Context manager for a single round of aggregate execution."""
        self.align.begin_round()
        self.exports.clear()
        self._neighbor_message_overrides.clear()
        yield self
        self.align.end_round()
        self.round_num += 1

    def reset(self) -> None:
        """Reset everything for a fresh execution."""
        self.round_num = 0
        self.state.reset()
        self.align = Aligner()
        self.exports.clear()
        self._neighbor_message_overrides.clear()
        self.message_weight = None

    def get_message_override(self, tag: str) -> Tensor | None:
        """Get the message override for a specific tag.

        Overrides are registered *before* the round runs, so they are usually
        keyed by the label the user wrote while *tag* is a full alignment path.
        Resolution therefore also runs in reverse: a stored label matches when
        the queried path carries it.
        """
        overrides = self._neighbor_message_overrides
        if not overrides:
            return None
        if tag in overrides:
            return overrides[tag]
        for stored, value in overrides.items():
            if path_has_label(tag, stored):
                return value
        return None

    def set_message_override(self, tag: str, value: Tensor) -> None:
        """Set a message override for a specific tag."""
        self._neighbor_message_overrides[tag] = value


def sub_context(
    ctx: RoundContext,
    edge_index: Tensor,
    edge_weight: Tensor | None,
    message_weight: Tensor | None,
) -> RoundContext:
    """Create a shallow copy of *ctx* with a different topology."""
    sub = RoundContext.__new__(RoundContext)
    sub.edge_index = edge_index
    sub.edge_weight = edge_weight if edge_weight is not None else ctx.edge_weight
    sub.message_weight = message_weight
    sub.num_nodes = ctx.num_nodes
    sub.node_ids = ctx.node_ids
    sub.round_num = ctx.round_num
    sub.state = ctx.state
    sub.align = ctx.align
    sub.exports = ctx.exports
    sub._neighbor_message_overrides = ctx._neighbor_message_overrides
    sub.data = maybe_make_data(sub.edge_index, sub.num_nodes, sub.edge_weight)
    return sub
