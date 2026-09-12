"""Drive a decentralised run and hand back something comparable to a central one."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from dataclasses import field as dataclass_field
from typing import Any

import torch
from torch import Tensor

from ..sim.events import SimulationRuntime
from .runtime import DeviceRuntime
from .topology import all_in_edges, edge_index_to_networkx, in_edges_of, require_symmetric


@dataclass
class DecentralizedResult:
    """Per-round global views of a run that never had a global view."""

    fields: list[Tensor]
    model: Any = None
    runtimes: dict[int, DeviceRuntime] = dataclass_field(default_factory=dict)
    #: Rounds at which at least one device's neighbour set changed.
    retopologized_rounds: list[int] = dataclass_field(default_factory=list)

    @property
    def final(self) -> Tensor:
        return self.fields[-1]

    def __len__(self) -> int:
        return len(self.fields)


def build_runtimes(
    *,
    edge_index: Tensor,
    num_nodes: int,
    program: Callable[[DeviceRuntime], Tensor],
    signals: Mapping[str, Any],
    edge_weight: Tensor | None = None,
    metadata: Mapping[str, Any] | None = None,
    self_loop: bool = False,
) -> dict[int, DeviceRuntime]:
    """One :class:`DeviceRuntime` per node, each seeing only its own links."""
    runtimes: dict[int, DeviceRuntime] = {}
    for node_id in range(num_nodes):
        neighbor_ids, ranges = in_edges_of(node_id, edge_index, edge_weight)
        runtimes[node_id] = DeviceRuntime(
            node_id=node_id,
            neighbor_ids=neighbor_ids,
            neighbor_ranges=ranges,
            signals=signals,
            program=program,
            metadata=metadata,
            self_loop=self_loop,
        )
    return runtimes


def _topology_of(scenario) -> tuple[Tensor, Tensor | None]:
    edge_weight = getattr(scenario, "edge_weight", None)
    if edge_weight is not None:
        edge_weight = edge_weight.detach().cpu()
    return scenario.edge_index.detach().cpu(), edge_weight


def _cpu_signals(signals: Mapping[str, Any]) -> dict[str, Any]:
    return {
        name: value.detach().cpu() if isinstance(value, Tensor) else value
        for name, value in signals.items()
    }


def run_decentralized(
    *,
    scenario,
    program: Callable[[DeviceRuntime], Tensor],
    signals: Mapping[str, Any],
    rounds: int,
    mode: str = "sync",
    activation_prob: float = 1.0,
    seed: int | None = 0,
    metadata: Mapping[str, Any] | None = None,
    schedule=None,
) -> DecentralizedResult:
    """Run *program* device by device inside Mesa.

    ``scenario`` only supplies the topology; the devices themselves are Mesa
    agents on a ``NetworkGrid`` and never consult it again.  ``signals`` are
    global node fields, sliced down to each device's local view.

    ``schedule`` is an :class:`~diffield.sim.events.EventSchedule` -- the same
    object a centralised :meth:`~diffield.sim.engine.SimulationEngine.run`
    takes, so one schedule can drive both runs.  Its callbacks may move the
    devices (``scenario.update_positions(...)``) or rewrite ``signals``; either
    way the new neighbourhoods reach the devices before the round executes,
    matching the engine's event-then-topology-then-round ordering.
    """
    from .mesa_backend import DecentralizedModel

    if getattr(scenario, "self_loops", False):
        raise NotImplementedError(
            "decentralised execution does not model self-loops yet; "
            "build the scenario with self_loops=False"
        )

    edge_index, edge_weight = _topology_of(scenario)
    num_nodes = int(scenario.num_nodes)

    runtimes = build_runtimes(
        edge_index=edge_index,
        num_nodes=num_nodes,
        program=program,
        signals=_cpu_signals(signals),
        edge_weight=edge_weight,
        metadata=metadata,
    )
    graph = edge_index_to_networkx(edge_index, num_nodes)
    model = DecentralizedModel(
        graph,
        runtimes,
        mode=mode,
        activation_prob=activation_prob,
        seed=seed,
    )

    # What a schedule sees: the same shape a centralised run hands its events.
    world = SimulationRuntime(
        scenario=scenario,
        signals=dict(signals),
        metadata=dict(metadata or {}),
    )
    retopologized: list[int] = []

    with torch.no_grad():
        for round_idx in range(rounds):
            world.round_idx = round_idx
            if schedule is not None:
                schedule.apply(round_idx, world)
                if _resync(model, runtimes, scenario, world.signals, edge_index):
                    retopologized.append(round_idx)
                edge_index, _ = _topology_of(scenario)
            model.step()

    return DecentralizedResult(
        fields=model.history,
        model=model,
        runtimes=runtimes,
        retopologized_rounds=retopologized,
    )


def _resync(
    model,
    runtimes: Mapping[int, DeviceRuntime],
    scenario,
    signals: Mapping[str, Any],
    previous_edge_index: Tensor,
) -> bool:
    """Push the current topology and signals down to every device.

    Returns whether any device's neighbour set changed, which is also when the
    simulator's own adjacency has to be rewritten.
    """
    edge_index, edge_weight = _topology_of(scenario)
    num_nodes = int(scenario.num_nodes)
    local_signals = _cpu_signals(signals)

    if edge_index.shape == previous_edge_index.shape and torch.equal(
        edge_index, previous_edge_index
    ):
        # Devices may still have drifted within the same links: ranges change
        # even when nobody gains or loses a neighbour.
        table = all_in_edges(edge_index, num_nodes, edge_weight)
        for node_id, runtime in runtimes.items():
            runtime.sync_topology(*table[node_id], signals=local_signals)
        return False

    require_symmetric(edge_index)
    table = all_in_edges(edge_index, num_nodes, edge_weight)
    changed = False
    for node_id, runtime in runtimes.items():
        changed |= runtime.sync_topology(*table[node_id], signals=local_signals)
    model.set_edges(edge_index)
    return changed
