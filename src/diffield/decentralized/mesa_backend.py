"""Mesa agents that execute a diffield program device by device.

Mesa knows nothing about field calculus: it supplies the population, the
adjacency (``NetworkGrid``) and the activation order, and an agent's whole
behaviour is "read my neighbours' last message, run one round, publish a new
message".  Everything aggregate-specific lives in :class:`DeviceRuntime`.

Two activation regimes:

``sync``
    Every device computes from a double-buffered previous message, then a
    barrier publishes.  That is Gauss-Jacobi -- exactly what a centralised
    :class:`~diffield.sim.engine.SimulationEngine` round does -- so activation
    order provably cannot matter.

``async``
    Devices are shuffled and publish as they go (Gauss-Seidel), optionally
    skipping rounds with probability ``1 - activation_prob``.  Messages a device
    reads may then be from this round or several rounds old.
"""

from __future__ import annotations

from collections.abc import Mapping

import mesa
import torch
from mesa.space import NetworkGrid
from torch import Tensor

from .runtime import DeviceRuntime

SYNC = "sync"
ASYNC = "async"


class DeviceAgent(mesa.Agent):
    """One device of the aggregate system."""

    def __init__(self, model: DecentralizedModel, node_id: int, runtime: DeviceRuntime):
        super().__init__(model)
        self.node_id = node_id
        self.runtime = runtime
        self.published: dict[str, Tensor] = {}
        self.pending: dict[str, Tensor] = {}
        self.output: Tensor | None = None
        self.rounds_run = 0

    # ------------------------------------------------------------------
    def _inbox(self) -> dict[str, list[Tensor | None]]:
        """Collect neighbours' published messages, in this device's link order."""
        by_id = {
            agent.node_id: agent
            for agent in self.model.grid.get_neighbors(
                self.node_id, include_center=False
            )
        }
        ordered = [by_id[node_id] for node_id in self.runtime.neighbor_ids]

        keys: set[str] = set()
        for agent in ordered:
            keys |= agent.published.keys()

        return {
            key: [agent.published.get(key) for agent in ordered] for key in keys
        }

    # ------------------------------------------------------------------
    def compute(self) -> None:
        """Run one round into the pending buffer."""
        self.output, self.pending = self.runtime.step(self._inbox())
        self.rounds_run += 1

    def publish(self) -> None:
        """Make this round's message visible to neighbours."""
        if self.pending:
            self.published = self.pending

    def compute_and_publish(self) -> None:
        """Asynchronous activation: no barrier, and rounds may be skipped."""
        if self.model.random.random() >= self.model.activation_prob:
            return
        self.compute()
        self.publish()


class DecentralizedModel(mesa.Model):
    """A population of devices running one aggregate program."""

    def __init__(
        self,
        graph,
        runtimes: Mapping[int, DeviceRuntime],
        *,
        mode: str = SYNC,
        activation_prob: float = 1.0,
        seed: int | None = None,
    ) -> None:
        # `rng` seeds both Mesa's numpy generator and the stdlib `random` that
        # drives `shuffle_do`, so an asynchronous run is reproducible.
        super().__init__(rng=seed)
        if mode not in (SYNC, ASYNC):
            raise ValueError(f"mode must be {SYNC!r} or {ASYNC!r}, got {mode!r}")
        self.mode = mode
        self.activation_prob = activation_prob
        self.num_nodes = graph.number_of_nodes()
        self.grid = NetworkGrid(graph)

        for node_id in sorted(runtimes):
            agent = DeviceAgent(self, node_id, runtimes[node_id])
            self.grid.place_agent(agent, node_id)

        self.history: list[Tensor] = []

    # ------------------------------------------------------------------
    def step(self) -> None:
        if self.mode == SYNC:
            self.agents.do("compute")
            self.agents.do("publish")
        else:
            self.agents.shuffle_do("compute_and_publish")
        self.history.append(self.field())

    def set_edges(self, edge_index) -> None:
        """Replace the adjacency in place when the devices have moved.

        The agents stay where they are -- a device keeps its identity and its
        node -- so only the links are rewritten, and nothing needs re-placing.
        """
        self.grid.G.clear_edges()
        self.grid.G.add_edges_from(
            (int(source), int(target))
            for source, target in edge_index.t().tolist()
            if source != target
        )

    def field(self) -> Tensor:
        """Assemble the global view of a decentralised run, for comparison only.

        No device ever sees this.  A device that has not run yet contributes
        ``nan``, which stays ``nan`` through any comparison rather than quietly
        reading as agreement.
        """
        exemplar = next((agent.output for agent in self.agents if agent.output is not None), None)
        shape = () if exemplar is None else exemplar.shape
        if shape:
            # Before the first asynchronous activation the payload shape is
            # unknown. Once known, give earlier empty snapshots that shape too.
            self.history[:] = [torch.full((self.num_nodes, *shape), float("nan"))
                            if prior.shape == (self.num_nodes,) and torch.isnan(prior).all()
                            else prior for prior in self.history]
        values = torch.full((self.num_nodes, *shape), float("nan"))
        for agent in self.agents:
            if agent.output is not None:
                values[agent.node_id] = agent.output.float()
        return values
