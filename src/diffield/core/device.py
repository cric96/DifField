"""Single-device execution helpers for local aggregate execution."""

from __future__ import annotations

from contextlib import contextmanager
from typing import TYPE_CHECKING

import torch
from torch import Tensor

from ..pyg_backend import maybe_make_data
from .alignment import path_has_label
from .execution import AggregateContext
from .state import StateManager

if TYPE_CHECKING:
    from .context import RoundContext


class _MessageStateManager(StateManager):
    """Only row zero is computed locally; other rows must come from messages."""

    def __init__(self, num_nodes: int):
        super().__init__(num_nodes)
        self.neighbor_exports = {}

    def get_or_init(self, init_val: Tensor, *, name: str) -> Tensor:
        own_state = super().get_or_init(init_val, name=name)
        state = torch.cat([own_state[:1], init_val[1:]], dim=0).clone()
        for key, values in self.neighbor_exports.items():
            if key != name and not path_has_label(name, key):
                continue
            for position, value in enumerate(values):
                if value is not None:
                    state[position + 1] = torch.as_tensor(value, dtype=state.dtype, device=state.device)
        return state


class DeviceContext:
    """Run an aggregate program from the perspective of a single device.

    A debug and didactic view: the device is node 0 and its neighbours are
    nodes 1..k, which have no incoming edges of their own.  A program is
    reproducible here only if every ``gather`` reads a sensor, a constant, or
    an ``iterate`` state from the previous round; a ``gather`` of a
    gather-derived field has no neighbour values to read and yields fill
    values. ``mid()`` uses stable network IDs when ``device_id`` and
    ``neighbor_ids`` are supplied, otherwise it numbers the local view.

    ``self_loop`` follows the same convention as the graph builders and is off
    by default; pass ``include_self=True`` to ``gather`` to fold in the
    device's own value at range 0.
    """

    def __init__(
        self,
        num_neighbors: int,
        *,
        self_loop: bool = False,
        neighbor_ranges: float | list[float] | Tensor = 1.0,
        device_id: int | None = None,
        neighbor_ids: list[int] | Tensor | None = None,
        message_driven: bool = False,
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
        if message_driven:
            self._agg_ctx._ctx.state = _MessageStateManager(num_nodes)
        self._set_node_ids(device_id, neighbor_ids)

    def _set_node_ids(
        self, device_id: int | None, neighbor_ids: list[int] | Tensor | None
    ) -> None:
        """Give ``mid()`` the real network ids for this device and its neighbours."""
        if device_id is None and neighbor_ids is None:
            return
        ids = self._agg_ctx._ctx.node_ids.clone()
        if device_id is not None:
            ids[0] = float(device_id)
        if neighbor_ids is not None:
            tensor = torch.as_tensor(neighbor_ids, dtype=torch.float32).flatten()
            if tensor.shape[0] != self._num_neighbors:
                raise ValueError(
                    f"Expected {self._num_neighbors} neighbour ids, "
                    f"got {tensor.shape[0]}"
                )
            ids[1:] = tensor
        self._agg_ctx._ctx.node_ids = ids

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
        self, own: float, *, scatter: float | list[float] | Tensor = 0.0
    ) -> Tensor:
        """Build a field with *own* for this device and *scatter* for neighbours."""
        num_nodes = self._num_neighbors + 1
        if isinstance(scatter, (list, Tensor)):
            tensor = torch.as_tensor(scatter, dtype=torch.float32)
            field = torch.zeros(num_nodes, dtype=torch.float32)
            field[1 : 1 + tensor.shape[0]] = tensor
        else:
            field = torch.full((num_nodes,), scatter, dtype=torch.float32)
        field[0] = own
        return field

    @staticmethod
    def result(tensor: Tensor) -> Tensor:
        """Extract this device's value (node 0) from a full field."""
        return tensor[0]

    def get_state(self, name: str) -> float:
        """Return this device's ``iterate`` state for *name*.

        *name* may be a full alignment path or a bare label.  A label used in
        both arms of a ``branch`` is resolved to the arm this device is aligned
        with, since a single device is only ever in one partition.
        """
        manager = self.state
        try:
            key = manager.resolve_for_node(name, 0)
        except KeyError as exc:
            raise KeyError(f"State '{name}' not found") from exc
        state_tensor = manager.get_state(name=key)
        if state_tensor is None:
            raise KeyError(f"State '{name}' not found")
        return state_tensor[0].item()

    @property
    def state(self):
        """This device's state store, keyed by alignment path."""
        return self._agg_ctx._ctx.state

    def export_bundle(self) -> dict[str, Tensor]:
        """This device's own row of every ``iterate`` state slot.

        The outbound message of a decentralised round: keys are alignment
        paths, which carry no filenames, line numbers or object ids and are
        identical across processes, so they are usable directly on the wire.
        Feed the collected bundles of a device's neighbours back as
        ``neighbor_exports`` on the next round.
        """
        manager = self.state
        return {
            key: manager.get_state(name=key)[0].detach().clone()
            for key in manager.keys()
        }

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
            for key in ctx.state.resolve_all(name):
                state = ctx.state.get_state(name=key)
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

            if isinstance(ctx.state, _MessageStateManager):
                # Applied lazily by get_or_init, including a message received
                # before this device has ever evaluated that occurrence.
                ctx.state.neighbor_exports = neighbor_exports or {}
            elif neighbor_exports:
                self._inject_neighbor_exports(ctx, neighbor_exports)

            if neighbor_messages:
                self._inject_neighbor_messages(ctx, neighbor_messages)

            yield ctx
