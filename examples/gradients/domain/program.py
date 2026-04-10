"""Aggregate program for distance gradient computation."""

from __future__ import annotations

import torch
from autofield import foldhood, mux, nbr, rep, SimulationEngine
from autofield.dsl import field

try:
    from autofield import SnapshotRecorder
except ImportError:
    SnapshotRecorder = None


def auto_rounds(rows: int, cols: int, rounds: int) -> int:
    """Determine computation rounds automatically based on grid size."""
    return rounds if rounds > 0 else rows + cols


def run_gradient_program(
    scenario,
    source: torch.Tensor,
    *,
    rounds: int,
    weight: torch.Tensor,
    aggr: str | torch.nn.Module = "min",
    recorder=None,
) -> tuple[torch.Tensor, SimulationEngine]:
    """Execute the aggregate distance gradient program."""
    engine = SimulationEngine.from_scenario(scenario)

    def program(_runtime):
        return rep(
            field.inf(),
            lambda dist_old: mux(
                source, field.of(0.0), foldhood(nbr(dist_old + weight), aggr=aggr)
            ),
            name="dist",
        )

    output, _ = engine.run(
        rounds=rounds, program=program, signals={"source": source}, recorder=recorder
    )
    return output, engine
