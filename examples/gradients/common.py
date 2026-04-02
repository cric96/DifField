"""Shared helpers for gradient example scripts."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

import torch

from aggregate_gnn import GridScenario, SimulationEngine, mux, nbr, rep, SnapshotRecorder
from aggregate_gnn.dsl import field
from aggregate_gnn.utils import get_grid_distances, get_device


def auto_rounds(rows: int, cols: int, rounds: int) -> int:
    return rounds if rounds > 0 else rows + cols


def build_corner_source_grid(
    rows: int,
    cols: int,
    *,
    connectivity: int = 4,
    device: torch.device | None = None,
) -> tuple[GridScenario, torch.Tensor, torch.Tensor]:
    scenario_kwargs = {"connectivity": connectivity}
    if device is not None:
        scenario_kwargs["device"] = device

    scenario = GridScenario(rows, cols, **scenario_kwargs)
    source_kwargs = {"dtype": torch.float32}
    if device is not None:
        source_kwargs["device"] = device

    source = torch.zeros(scenario.num_nodes, **source_kwargs)
    source[0] = 1.0
    expected = get_grid_distances(rows, cols, src_r=0, src_c=0, connectivity=connectivity)
    if device is not None:
        expected = expected.to(device)
    return scenario, source, expected


def run_gradient_program(
    scenario: GridScenario,
    source: torch.Tensor,
    *,
    rounds: int,
    weight: torch.Tensor,
    aggr: str | torch.nn.Module = "min",
    recorder: SnapshotRecorder | None = None,
) -> tuple[torch.Tensor, SimulationEngine]:
    engine = SimulationEngine.from_scenario(scenario)

    def program(_runtime):
        return rep(
            "dist",
            float("inf"),
            lambda dist_old: mux(source, field.of(0.0), nbr(dist_old + weight, aggr=aggr)),
        )

    output, _ = engine.run(rounds=rounds, program=program, signals={"source": source}, recorder=recorder)
    return output, engine