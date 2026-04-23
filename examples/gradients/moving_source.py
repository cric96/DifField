#!/usr/bin/env python3
"""Gradient simulation with a moving source."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "examples"))

import torch
from diffield.utils import get_device

from diffield.sim import (
    EventSchedule,
    GridScenario,
    ScheduledEvent,
    SimulationEngine,
    SnapshotRecorder,
)
from diffield.dsl import mux, scatter, iterate, gather_min, field
from shared.plotting import save_grid_simulation_gif


def parse_args():
    parser = argparse.ArgumentParser(description="Moving-source Gradient simulation")
    parser.add_argument("--rows", type=int, default=10, help="Grid rows")
    parser.add_argument("--cols", type=int, default=10, help="Grid cols")
    parser.add_argument("--rounds", type=int, default=40, help="Simulation rounds")
    parser.add_argument("--seed", type=int, default=7, help="Random seed")
    parser.add_argument("--hop", type=float, default=1.0, help="Hop cost")
    parser.add_argument(
        "--device", type=str, default="", help="Device (cuda/cpu) [auto if empty]"
    )
    parser.add_argument(
        "--viz-prefix", type=str, default="generated/gradient_moving_source"
    )
    parser.add_argument("--gif-fps", type=int, default=10)
    parser.add_argument("--no-viz", action="store_true", help="Disable figure export")
    parser.add_argument("--no-gif", action="store_true", help="Disable gif export")
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
    torch.manual_seed(args.seed)
    device = get_device(args.device)
    scenario = GridScenario(args.rows, args.cols, connectivity=4, device=device)
    engine = SimulationEngine.from_scenario(scenario)

    source = make_source(scenario, 0, 0)
    schedule = EventSchedule(
        [
            ScheduledEvent(
                round_idx=args.rounds // 3,
                callback=move_source_event(args.rows // 2, args.cols // 2),
                name="move_to_center",
            ),
            ScheduledEvent(
                round_idx=(2 * args.rounds) // 3,
                callback=move_source_event(args.rows - 1, args.cols - 1),
                name="move_to_bottom_right",
            ),
        ]
    )

    record_steps = {0, args.rounds // 3, (2 * args.rounds) // 3, args.rounds - 1}
    record_rounds = None if not args.no_gif else record_steps
    recorder = SnapshotRecorder(
        state_fields=["dist"], capture_output=True, record_rounds=record_rounds
    )
    weight = torch.tensor(args.hop, device=device)

    def program(runtime):
        source_field = runtime.signals["source"]
        return iterate(
            field.inf(),
            lambda dist_old: mux(
                source_field, field.of(0.0), gather_min(scatter(dist_old + weight))
            ),
            name="dist",
        )

    output, runtime = engine.run(
        rounds=args.rounds,
        program=program,
        signals={"source": source},
        metadata={"source_pos": (0, 0)},
        recorder=recorder,
        schedule=schedule,
    )

    if not args.no_gif and not args.no_viz:
        save_grid_simulation_gif(
            recorder.records,
            "dist",
            args.rows,
            args.cols,
            f"{args.viz_prefix}_evolution.gif",
            title="Moving Source Gradient Evolution",
            fps=args.gif_fps,
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
