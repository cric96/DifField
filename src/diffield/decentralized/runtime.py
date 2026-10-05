"""The per-device virtual machine, independent of any simulator.

One :class:`DeviceRuntime` is one device.  It holds that device's
:class:`~diffield.core.device.DeviceContext` -- a star graph with the device at
node 0 and its neighbours at 1..k -- and exposes the same surface a program sees
centrally (``signals``, ``metadata``, ``round_idx``, ``scenario``), so the very
same ``program(runtime)`` closure runs in both settings.

The wire format is the device's ``iterate`` state table keyed by alignment path
(see :meth:`~diffield.core.device.DeviceContext.export_bundle`).  Sensor values
travel too: a signal is localised by index, so a device sees its neighbours'
real sensor readings.  That is what makes ``branch`` exact -- domain restriction
keeps a link only when *both* endpoints are in the partition, so a device cannot
decide its own in-edges without knowing its neighbours' condition.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import Any

import torch
from torch import Tensor

from ..core.device import DeviceContext

#: A neighbour that has not spoken yet contributes ``None``.
InboxSlot = Sequence["Tensor | None"]


def _resize_states(
    carried: dict[str, Tensor],
    initializers: dict[str, Tensor],
    num_nodes: int,
) -> dict[str, Tensor]:
    """Re-create a state store at *num_nodes*, keeping only row 0 of each slot."""
    resized: dict[str, Tensor] = {}
    for key, own in carried.items():
        init = initializers.get(key)
        fill = init[0] if init is not None else own
        slot = fill.unsqueeze(0).expand(num_nodes, *fill.shape).clone()
        slot[0] = own
        resized[key] = slot
    return resized


class DeviceRuntime:
    """Run one device's rounds of an aggregate program."""

    def __init__(
        self,
        node_id: int,
        neighbor_ids: Sequence[int],
        neighbor_ranges: Sequence[float],
        signals: Mapping[str, Any],
        program: Callable[[DeviceRuntime], Tensor],
        *,
        metadata: Mapping[str, Any] | None = None,
        self_loop: bool = False,
    ) -> None:
        self.node_id = int(node_id)
        self.neighbor_ids = [int(n) for n in neighbor_ids]
        self.program = program
        self.round_idx = 0
        self.metadata = dict(metadata or {})
        # Kept for parity with SimulationRuntime: a device has no global view.
        self.scenario = None

        self._device = DeviceContext(
            num_neighbors=len(self.neighbor_ids),
            self_loop=self_loop,
            neighbor_ranges=list(neighbor_ranges),
            device_id=self.node_id,
            neighbor_ids=self.neighbor_ids,
            message_driven=True,
        )
        self._self_loop = self_loop
        self._global_signals = dict(signals)
        self._local_index = torch.tensor(
            [self.node_id, *self.neighbor_ids], dtype=torch.long
        )
        self.signals = {
            name: self._localize(v) for name, v in self._global_signals.items()
        }
        self._last_inbox: dict[str, list[Tensor | None]] = {}
        self._ranges = list(neighbor_ranges)

    # ------------------------------------------------------------------
    # local views
    # ------------------------------------------------------------------
    def _localize(self, value: Any) -> Any:
        """Slice a global node field down to this device's star graph.

        Row 0 is the device, rows 1..k its neighbours in ``neighbor_ids`` order
        -- the order :class:`DeviceContext` builds its star edges in.  Index
        selection preserves dtype and feature dimensions, so boolean masks stay
        boolean and vector payloads stay vectors.
        """
        if not isinstance(value, Tensor) or value.dim() == 0:
            return value
        index = self._local_index.to(value.device)
        return value.index_select(0, index)

    # ------------------------------------------------------------------
    # moving devices
    # ------------------------------------------------------------------
    def sync_topology(
        self,
        neighbor_ids: Sequence[int],
        neighbor_ranges: Sequence[float],
        signals: Mapping[str, Any] | None = None,
    ) -> bool:
        """Adopt a new neighbourhood, carrying this device's own state across.

        Returns whether the neighbour *set* changed.  When only the ranges moved
        the device keeps its star graph and simply measures its links anew; when
        the set changed the star graph is rebuilt at the new degree and every
        ``iterate`` slot is re-created at that size, preserving row 0 -- the only
        row that is genuinely this device's -- and filling the neighbour rows
        with the slot's initialiser, so a device not yet heard from reads as
        unknown rather than as whoever used to occupy that row.
        """
        if signals is not None:
            self._global_signals = dict(signals)

        neighbor_ids = [int(n) for n in neighbor_ids]
        self._ranges = list(neighbor_ranges)
        if neighbor_ids == self.neighbor_ids:
            if signals is not None:
                self._relocalize()
            return False

        carried = self._device.export_bundle()
        initializers = self._device.state.initializers()

        self.neighbor_ids = neighbor_ids
        self._device = DeviceContext(
            num_neighbors=len(neighbor_ids),
            self_loop=self._self_loop,
            neighbor_ranges=self._ranges,
            device_id=self.node_id,
            neighbor_ids=neighbor_ids,
            message_driven=True,
        )
        self._device.state.restore(
            _resize_states(carried, initializers, len(neighbor_ids) + 1)
        )

        # Rows have been renumbered, so remembered messages no longer line up.
        self._last_inbox.clear()
        self._relocalize()
        return True

    def _relocalize(self) -> None:
        self._local_index = torch.tensor(
            [self.node_id, *self.neighbor_ids], dtype=torch.long
        )
        self.signals = {
            name: self._localize(v) for name, v in self._global_signals.items()
        }

    # ------------------------------------------------------------------
    # message handling
    # ------------------------------------------------------------------
    def _resolve_inbox(self, inbox: Mapping[str, InboxSlot] | None) -> dict[str, InboxSlot]:
        """Turn per-neighbour messages into one dense row block per state slot.

        A neighbour that has not spoken this round keeps the value it last sent;
        if it has never spoken, the whole slot is left alone, so the device
        reads that neighbour's initialiser.  Under the synchronous barrier every
        neighbour speaks every round and neither fallback is ever taken.
        """
        for key, values in (inbox or {}).items():
            previous = self._last_inbox.get(key, [None] * len(self.neighbor_ids))
            if len(values) != len(self.neighbor_ids):
                raise ValueError("Inbox must contain one slot per neighbour")
            self._last_inbox[key] = [value if value is not None else previous[position]
                                     for position, value in enumerate(values)]
        return dict(self._last_inbox)

    # ------------------------------------------------------------------
    # execution
    # ------------------------------------------------------------------
    def step(
        self, inbox: Mapping[str, InboxSlot] | None = None
    ) -> tuple[Tensor, dict[str, Tensor]]:
        """Run one round; return this device's output and its outbound bundle."""
        exports = self._resolve_inbox(inbox)
        with self._device.round(
            neighbor_exports=exports or None, neighbor_ranges=self._ranges
        ):
            output = self.program(self)
        self.round_idx += 1
        return self._device.result(output).detach().clone(), self._device.export_bundle()

    def get_state(self, name: str) -> float:
        """This device's ``iterate`` state for *name* (full path or bare label)."""
        return self._device.get_state(name)
