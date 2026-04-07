"""Single-device execution helpers for local aggregate execution."""

from __future__ import annotations

from typing import TYPE_CHECKING
from contextlib import contextmanager

import torch
from torch import Tensor

from ..pyg_backend import maybe_make_data
from .execution import AggregateContext

if TYPE_CHECKING:
    from .context import RoundContext


class DeviceContext:
    """Run an aggregate program from the perspective of a single device."""

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
        num_nodes = num_neighbors + 1
        if self_loop:
            src = list(range(num_nodes))
            tgt = [0] * num_nodes
        else:
            src = list(range(1, num_nodes))
            tgt = [0] * num_neighbors

        if not src:
            edge_index = torch.zeros((2, 0), dtype=torch.long)
        else:
            edge_index = torch.tensor([src, tgt], dtype=torch.long)

        self._agg_ctx = AggregateContext(
            edge_index, num_nodes, edge_weight=self._build_edge_weight(neighbor_ranges)
        )

    def _neighbor_range_tensor(
        self, neighbor_ranges: float | list[float] | Tensor
    ) -> Tensor:
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

    def _build_edge_weight(
        self, neighbor_ranges: float | list[float] | Tensor
    ) -> Tensor:
        neighbor_tensor = self._neighbor_range_tensor(neighbor_ranges)
        if self._self_loop:
            return torch.cat(
                (torch.zeros(1, dtype=torch.float32), neighbor_tensor), dim=0
            )
        return neighbor_tensor

    def local_field(
        self, own: float, *, nbr: float | list[float] | Tensor = 0.0
    ) -> Tensor:
        """Build a field with *own* for this device and *nbr* for neighbours."""
        num_nodes = self._num_neighbors + 1
        if isinstance(nbr, (list, Tensor)):
            tensor = torch.as_tensor(nbr, dtype=torch.float32)
            field = torch.zeros(num_nodes, dtype=torch.float32)
            field[1 : 1 + tensor.shape[0]] = tensor
        else:
            field = torch.full((num_nodes,), nbr, dtype=torch.float32)
        field[0] = own
        return field

    @staticmethod
    def result(tensor: Tensor) -> Tensor:
        """Extract this device's value (node 0) from a full field."""
        return tensor[0]

    def get_state(self, name: str) -> float:
        """Return this device's ``rep`` state for *name*."""
        state_tensor = self._agg_ctx._ctx.state.get_state(name)
        if state_tensor is None:
            raise KeyError(f"State '{name}' not found")
        return state_tensor[0].item()

    def reset(self) -> None:
        self._agg_ctx.reset()

    @property
    def round_num(self) -> int:
        return self._agg_ctx.round_num

    def _inject_neighbor_exports(
        self,
        ctx: RoundContext,
        neighbor_exports: dict[str, list[float] | Tensor],
    ) -> None:
        for name, values in neighbor_exports.items():
            state = ctx.state.get_state(name)
            if state is None:
                continue
            tensor = torch.as_tensor(values, dtype=state.dtype)
            if tensor.dim() == 0:
                tensor = tensor.unsqueeze(0)
            state[1 : 1 + tensor.shape[0]] = tensor

    def _inject_neighbor_messages(
        self,
        ctx: RoundContext,
        neighbor_messages: dict[str, list[float] | Tensor],
    ) -> None:
        num_nodes = self._num_neighbors + 1
        for tag, values in neighbor_messages.items():
            tensor = torch.as_tensor(values, dtype=torch.float32)
            if tensor.dim() == 0:
                tensor = tensor.unsqueeze(0)
            override = torch.zeros(num_nodes, dtype=torch.float32)
            override[1 : 1 + tensor.shape[0]] = tensor
            ctx.set_message_override(tag, override)

    @contextmanager
    def round(
        self,
        neighbor_exports: dict[str, list[float] | Tensor] | None = None,
        neighbor_messages: dict[str, list[float] | Tensor] | None = None,
        neighbor_ranges: float | list[float] | Tensor | None = None,
    ):
        """Execute one local round."""
        with self._agg_ctx.round() as ctx:
            effective_ranges = (
                self._default_neighbor_ranges
                if neighbor_ranges is None
                else neighbor_ranges
            )
            ctx.edge_weight = self._build_edge_weight(effective_ranges).to(
                ctx.edge_index.device
            )
            ctx.data = maybe_make_data(ctx.edge_index, ctx.num_nodes, ctx.edge_weight)

            if neighbor_exports:
                self._inject_neighbor_exports(ctx, neighbor_exports)

            if neighbor_messages:
                self._inject_neighbor_messages(ctx, neighbor_messages)

            yield ctx
