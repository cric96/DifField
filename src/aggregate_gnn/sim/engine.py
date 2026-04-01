"""Simulation engine wrapping AggregateContext execution."""

from __future__ import annotations

import time
from typing import Callable

import torch

from ..dsl import AggregateContext
from ..pyg_backend import maybe_make_data
from .events import EventSchedule, SimulationRuntime
from .recording import SnapshotRecorder


ProgramStep = Callable[[SimulationRuntime], torch.Tensor]


class SimulationEngine:
    """Run aggregate programs with optional event scheduling and recording."""

    def __init__(self, edge_index: torch.Tensor, num_nodes: int):
        self.ctx = AggregateContext(maybe_make_data(edge_index, num_nodes))
        self.device = edge_index.device
        self.num_nodes = num_nodes
        self.scenario = None

    @classmethod
    def from_scenario(cls, scenario) -> "SimulationEngine":
        engine = cls(scenario.edge_index, scenario.num_nodes)
        if hasattr(scenario, "edge_weight"):
            engine.ctx._ctx.edge_weight = scenario.edge_weight
            engine.ctx._ctx.data = maybe_make_data(
                scenario.edge_index,
                scenario.num_nodes,
                scenario.edge_weight,
            )
        engine.scenario = scenario
        return engine

    def run(
        self,
        *,
        rounds: int,
        program: ProgramStep,
        signals: dict[str, object],
        metadata: dict[str, object] | None = None,
        recorder: SnapshotRecorder | None = None,
        schedule: EventSchedule | None = None,
    ) -> tuple[torch.Tensor, SimulationRuntime]:
        runtime = self.init_runtime(signals=signals, metadata=metadata)
        output = torch.full((self.num_nodes,), float("inf"), device=self.device)
        for _ in range(rounds):
            output = self.step(
                runtime=runtime,
                program=program,
                recorder=recorder,
                schedule=schedule,
            )

        return output, runtime

    def init_runtime(
        self,
        *,
        signals: dict[str, object],
        metadata: dict[str, object] | None = None,
    ) -> SimulationRuntime:
        return SimulationRuntime(
            scenario=self.scenario,
            signals=signals,
            metadata=metadata or {},
        )

    def step(
        self,
        *,
        runtime: SimulationRuntime,
        program: ProgramStep,
        recorder: SnapshotRecorder | None = None,
        schedule: EventSchedule | None = None,
    ) -> torch.Tensor:
        round_idx = runtime.round_idx
        if schedule is not None:
            schedule.apply(round_idx, runtime)

        # Allow scenarios to update topology/weights before this round executes.
        if runtime.scenario is not None:
            sync_fn = getattr(runtime.scenario, "sync_context", None)
            if callable(sync_fn):
                sync_fn(self.ctx._ctx)

        started_at = time.time()
        with self.ctx.round():
            output = program(runtime)

        if recorder is not None:
            recorder.record(
                round_idx=round_idx,
                round_ctx=self.ctx._ctx,
                output=output,
                started_at=started_at,
            )

        runtime.round_idx += 1
        return output
