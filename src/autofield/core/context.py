"""Graph execution contexts for aggregate programs."""

from __future__ import annotations

from contextlib import contextmanager

import torch
from torch import Tensor

from ..pyg_backend import maybe_make_data
from .state import StateManager


class RoundContext:
    """Holds the graph topology and per-round execution state."""

    def __init__(
        self, edge_index: Tensor, num_nodes: int, edge_weight: Tensor | None = None
    ) -> None:
        self.edge_index = edge_index
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
        self.data = maybe_make_data(self.edge_index, self.num_nodes, self.edge_weight)
        self.round_num = 0
        self.state = StateManager(num_nodes, device=edge_index.device)
        self.exports: dict[str, Tensor] = {}
        self._neighbor_message_overrides: dict[str, Tensor] = {}

    @contextmanager
    def round(self):
        """Context manager for a single round of aggregate execution."""
        self.exports.clear()
        self._neighbor_message_overrides.clear()
        yield self
        self.round_num += 1

    def reset(self) -> None:
        """Reset everything for a fresh execution."""
        self.round_num = 0
        self.state.reset()
        self.exports.clear()
        self._neighbor_message_overrides.clear()
        self.message_weight = None

    def get_message_override(self, tag: str) -> Tensor | None:
        """Get the message override for a specific tag."""
        return self._neighbor_message_overrides.get(tag)

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
    sub.round_num = ctx.round_num
    sub.state = ctx.state
    sub.exports = ctx.exports
    sub._neighbor_message_overrides = ctx._neighbor_message_overrides
    sub.data = maybe_make_data(sub.edge_index, sub.num_nodes, sub.edge_weight)
    return sub
