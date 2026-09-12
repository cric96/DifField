"""Run a diffield program device by device in a third-party simulator.

The centralised path (:class:`~diffield.sim.engine.SimulationEngine`) evaluates
one round as a single batched operation over the whole graph.  This package
evaluates the *same program closure* the way the model claims it should be
executable: one device at a time, each seeing only its neighbours' messages,
scheduled by `Mesa <https://mesa.readthedocs.io>`_ -- an agent-based modelling
framework that knows nothing about field calculus.

Requires the optional extra::

    uv sync --extra cpu --extra decentralized
"""

from __future__ import annotations

from .runner import DecentralizedResult, build_runtimes, run_decentralized
from .runtime import DeviceRuntime
from .topology import edge_index_to_networkx, in_edges_of, require_symmetric

__all__ = [
    "DecentralizedResult",
    "DeviceRuntime",
    "build_runtimes",
    "edge_index_to_networkx",
    "in_edges_of",
    "require_symmetric",
    "run_decentralized",
]
