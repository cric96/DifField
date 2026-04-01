#!/usr/bin/env python3
"""Gradient simulation with a moving source."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from aggregate_gnn import EventSchedule, GridScenario, ScheduledEvent, SimulationEngine, SnapshotRecorder, mux, nbr, rep
from aggregate_gnn.dsl import field


def parse_args():
    parser = argparse.ArgumentParser(description="Moving-source Gradient simulation")
    parser.add_argument("--rows", type=int, default=10, help="Grid rows")
    parser.add_argument("--cols", type=int, default=10, help="Grid cols")
    parser.add_argument("--rounds", type=int, default=40, help="Simulation rounds")
    parser.add_argument("--hop", type=float, default=1.0, help="Hop cost")
    return parser.parse_args()


def make_source(scenario: GridScenario, row: int, col: int) -> torch.Tensor:
    return scenario.marker(row, col)


def move_source_event(row: int, col: int):
    def _event(runtime):
        scenario = runtime.scenario
        runtime.signals["source"] = make_source(scenario, row, col)
        runtime.metadata["source_pos"] = (row, col)

    return _event


def main():
    args = parse_args()
    scenario = GridScenario(args.rows, args.cols, connectivity=4)
    engine = SimulationEngine.from_scenario(scenario)

    source = make_source(scenario, 0, 0)
    schedule = EventSchedule(
        [
            ScheduledEvent(round_idx=args.rounds // 3, callback=move_source_event(args.rows // 2, args.cols // 2), name="move_to_center"),
            ScheduledEvent(
                round_idx=(2 * args.rounds) // 3,
                callback=move_source_event(args.rows - 1, args.cols - 1),
                name="move_to_bottom_right",
            ),
        ]
    )

    record_steps = {0, args.rounds // 3, (2 * args.rounds) // 3, args.rounds - 1}
    recorder = SnapshotRecorder(state_fields=["dist"], capture_output=True, record_rounds=record_steps)
    weight = torch.tensor(args.hop)

    def program(runtime):
        source_field = runtime.signals["source"]
        return rep("dist", float("inf"), lambda dist_old: mux(source_field, field.of(0.0), nbr(dist_old + weight, aggr="min")))

    output, runtime = engine.run(
        rounds=args.rounds,
        program=program,
        signals={"source": source},
        metadata={"source_pos": (0, 0)},
        recorder=recorder,
        schedule=schedule,
    )

    print("=== Moving Source Gradient ===")
    print(f"Grid: {args.rows}x{args.cols}  rounds: {args.rounds}")
    print(f"Final source position: {runtime.metadata['source_pos']}")
    print("Recorded snapshots:")
    for round_idx in sorted(recorder.records.keys()):
        center_idx = scenario.pos_to_idx(args.rows // 2, args.cols // 2)
        center_dist = recorder.records[round_idx]["output"][center_idx].item()
        print(f"  round {round_idx + 1:3d}: center distance={center_dist:.2f}")

    print("Final distance field:")
    print(output.view(args.rows, args.cols).detach().cpu().numpy())


if __name__ == "__main__":
    main()