#!/usr/bin/env python3
"""Moving-node gradient simulation with dynamic topology."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "examples"))

from aggregate_gnn import (
    EventSchedule,
    ScheduledEvent,
    SimulationEngine,
    SnapshotRecorder,
    SpatialScenario,
    boids_acceleration_dense,
    bounce_in_box,
    limit_speed,
    normalize_vectors,
    mux,
    nbr,
    rep,
)
from aggregate_gnn.dsl import field
from shared.plotting import export_moving_gif, plot_moving_snapshots, plot_node_trajectories


def parse_args():
    parser = argparse.ArgumentParser(description="Moving-node DSL gradient simulation")
    parser.add_argument("--num-nodes", type=int, default=60)
    parser.add_argument("--rounds", type=int, default=80)
    parser.add_argument("--radius", type=float, default=0.23)
    parser.add_argument("--dt", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--motion", choices=["simple", "boids"], default="boids")
    parser.add_argument("--source", type=int, default=0)
    parser.add_argument("--hop", type=float, default=1.0)
    parser.add_argument("--speed", type=float, default=0.014)
    parser.add_argument("--sep", type=float, default=0.06)
    parser.add_argument("--w-sep", type=float, default=1.4)
    parser.add_argument("--w-align", type=float, default=0.8)
    parser.add_argument("--w-cohesion", type=float, default=0.6)
    parser.add_argument("--damping", type=float, default=0.96)
    parser.add_argument("--record-every", type=int, default=10)
    parser.add_argument("--viz-prefix", type=str, default="examples/moving_nodes")
    parser.add_argument("--gif-fps", type=int, default=8)
    parser.add_argument("--no-viz", action="store_true", help="Disable figure export")
    parser.add_argument("--no-gif", action="store_true", help="Disable gif export")
    parser.add_argument("--hide-links", action="store_true", help="Do not draw graph links in visual outputs")
    parser.add_argument("--links-alpha", type=float, default=0.15)
    parser.add_argument("--links-width", type=float, default=0.6)
    return parser.parse_args()


def simple_step(runtime, dt: float) -> None:
    scenario = runtime.scenario
    velocities = runtime.metadata["velocities"]
    positions = scenario.positions + velocities * dt
    positions, velocities = bounce_in_box(positions, velocities)
    runtime.metadata["velocities"] = velocities * runtime.metadata["damping"]
    scenario.update_positions(positions, refresh_topology=True)
    runtime.metadata["positions_by_round"][runtime.round_idx] = scenario.positions.detach().clone()
    runtime.metadata["edge_index_by_round"][runtime.round_idx] = scenario.edge_index.detach().clone()


def boids_step(runtime, dt: float) -> None:
    scenario = runtime.scenario
    positions = scenario.positions
    velocities = runtime.metadata["velocities"]

    acc = boids_acceleration_dense(
        positions,
        velocities,
        radius=runtime.metadata["radius"],
        sep=runtime.metadata["sep"],
        w_sep=runtime.metadata["w_sep"],
        w_align=runtime.metadata["w_align"],
        w_cohesion=runtime.metadata["w_cohesion"],
    )

    new_vel = runtime.metadata["damping"] * velocities + dt * acc
    new_vel = limit_speed(new_vel, runtime.metadata["speed"])

    new_pos = positions + dt * new_vel
    new_pos, new_vel = bounce_in_box(new_pos, new_vel)

    runtime.metadata["velocities"] = new_vel
    scenario.update_positions(new_pos, refresh_topology=True)
    runtime.metadata["positions_by_round"][runtime.round_idx] = scenario.positions.detach().clone()
    runtime.metadata["edge_index_by_round"][runtime.round_idx] = scenario.edge_index.detach().clone()


def movement_event(motion: str, dt: float):
    def _event(runtime):
        if motion == "simple":
            simple_step(runtime, dt)
        else:
            boids_step(runtime, dt)

    return _event


def main():
    args = parse_args()
    torch.manual_seed(args.seed)

    positions = torch.rand(args.num_nodes, 2)
    velocities = (torch.rand(args.num_nodes, 2) * 2.0 - 1.0)
    velocities = normalize_vectors(velocities) * args.speed

    scenario = SpatialScenario(positions=positions, edge_radius=args.radius)
    engine = SimulationEngine.from_scenario(scenario)

    source = scenario.marker(args.source)
    weight = torch.tensor(args.hop)

    schedule = EventSchedule(
        [
            ScheduledEvent(round_idx=round_idx, callback=movement_event(args.motion, args.dt), name=f"move_{round_idx}")
            for round_idx in range(args.rounds)
        ]
    )

    record_rounds = set(range(0, args.rounds, max(1, args.record_every))) | {args.rounds - 1}
    recorder = SnapshotRecorder(state_fields=["dist"], capture_output=True, record_rounds=record_rounds)

    def program(runtime):
        src = source.to(runtime.scenario.device)
        return rep("dist", float("inf"), lambda dist_old: mux(src, field.of(0.0), nbr(dist_old + weight, aggr="min")))

    output, runtime = engine.run(
        rounds=args.rounds,
        program=program,
        signals={"source": source},
        metadata={
            "velocities": velocities,
            "positions_by_round": {},
            "edge_index_by_round": {},
            "radius": args.radius,
            "sep": args.sep,
            "w_sep": args.w_sep,
            "w_align": args.w_align,
            "w_cohesion": args.w_cohesion,
            "damping": args.damping,
            "speed": args.speed,
        },
        recorder=recorder,
        schedule=schedule,
    )

    center = torch.tensor([0.5, 0.5])
    closest_idx = torch.argmin(torch.norm(runtime.scenario.positions - center, dim=1)).item()

    print("=== Moving Nodes Gradient ===")
    print(f"motion={args.motion} nodes={args.num_nodes} rounds={args.rounds} radius={args.radius}")
    print(f"source={args.source} center-nearest-node={closest_idx} dist={output[closest_idx].item():.3f}")
    print(f"final_edges={runtime.scenario.edge_index.shape[1]}")
    print("recorded rounds:", sorted(recorder.records.keys()))

    if not args.no_viz:
        positions_by_round = runtime.metadata["positions_by_round"]
        edge_index_by_round = runtime.metadata["edge_index_by_round"]
        values_by_round = {
            round_idx: payload["output"]
            for round_idx, payload in recorder.records.items()
            if "output" in payload
        }
        positions_over_time = [positions_by_round[round_idx] for round_idx in sorted(positions_by_round.keys())]

        plot_moving_snapshots(
            positions_by_round=positions_by_round,
            values_by_round=values_by_round,
            source_idx=args.source,
            output_path=f"{args.viz_prefix}_snapshots.png",
            title=f"Moving nodes ({args.motion}) distance snapshots",
            edge_index_by_round=edge_index_by_round,
            show_links=not args.hide_links,
            links_alpha=args.links_alpha,
            links_width=args.links_width,
        )
        plot_node_trajectories(
            positions_over_time=positions_over_time,
            source_idx=args.source,
            output_path=f"{args.viz_prefix}_trajectories.png",
            title=f"Moving nodes ({args.motion}) trajectories",
        )
        if not args.no_gif:
            export_moving_gif(
                positions_by_round=positions_by_round,
                values_by_round=values_by_round,
                source_idx=args.source,
                output_path=f"{args.viz_prefix}.gif",
                title=f"Moving nodes ({args.motion}) distance",
                fps=max(1, args.gif_fps),
                edge_index_by_round=edge_index_by_round,
                show_links=not args.hide_links,
                links_alpha=args.links_alpha,
                links_width=args.links_width,
            )


if __name__ == "__main__":
    main()