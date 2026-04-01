#!/usr/bin/env python3
"""Channel with obstacles — classic AC self-organising construct.

Demonstrates the aggregate computing *channel*: a Boolean field identifying
nodes on the shortest path between source and destination, routing around
a wall of obstacles using ``branch`` for communication isolation.
"""

import sys
import argparse
import torch
try:
    import matplotlib.pyplot as plt
except ImportError:
    plt = None

sys.path.insert(0, "src")

from aggregate_gnn import GridScenario, SimulationEngine, SnapshotRecorder, rep, nbr, mux, branch, broadcast
from aggregate_gnn.dsl import field
from channel_viz import (
    plot_channel_evolution,
    plot_channel_final_fields,
    plot_channel_overlay,
    plot_channel_setup,
)


def parse_args():
    parser = argparse.ArgumentParser(description="AC Channel with obstacles")
    parser.add_argument("--rows", type=int, default=15, help="Grid rows")
    parser.add_argument("--cols", type=int, default=15, help="Grid columns")
    parser.add_argument("--rounds", type=int, default=100, help="Number of compute rounds")
    parser.add_argument("--noise-scale", type=float, default=0.01, help="Hop cost noise")
    parser.add_argument("--tolerance", type=float, default=0.5, help="Path tolerance")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    return parser.parse_args()


def enforce_small_scale(args):
    """Keep this example focused on the original small-scale scenario."""
    max_rows, max_cols = 20, 20
    if args.rows > max_rows or args.cols > max_cols:
        print(
            "channel.py is the small-scale demo. "
            "Falling back to 15x15. "
            "Use channel_large.py for larger grids."
        )
        args.rows = 15
        args.cols = 15


def channel_body(source: torch.Tensor, dest: torch.Tensor, noise: torch.Tensor, tolerance: float):
    """Compute channel inside the obstacle-free partition."""
    # Gradient from source  (hop cost = 1 + tiny noise)
    dist_src = rep("dist_src", float("inf"), lambda d:
        mux(source, field.of(0.0), nbr(d + 1.0 + noise, aggr="min")))

    # Gradient from destination
    dist_dst = rep("dist_dst", float("inf"), lambda d:
        mux(dest, field.of(0.0), nbr(d + 1.0 + noise, aggr="min")))

    # Broadcast dist_dst from source to all nodes
    dist_sd = broadcast(source > 0.5, dist_dst, name="dist_channel")

    # Channel: nodes on shortest path (tolerance absorbs the noise)
    on_path = (dist_src + dist_dst <= dist_sd + tolerance)
    finite  = torch.isfinite(dist_src) & torch.isfinite(dist_dst) & torch.isfinite(dist_sd)
    return (on_path & finite).float()


def build_scenario(args):
    """Setup graph, source, dest, and obstacles."""
    scenario = GridScenario(args.rows, args.cols, connectivity=8)
    num_nodes = scenario.num_nodes

    # Source (middle-left) and destination (middle-right)
    src_pos = (args.rows // 2, 2)
    dst_pos = (args.rows // 2, args.cols - 3)
    
    source = scenario.marker(src_pos[0], src_pos[1])
    
    dest = scenario.marker(dst_pos[0], dst_pos[1])

    # Obstacle: vertical wall completely blocking direct horizontal path
    wall_col = args.cols // 2
    obstacle = scenario.mask_from_predicate(
        lambda r, c: c == wall_col and r < args.rows - 2
    )

    torch.manual_seed(args.seed)
    noise = torch.rand(num_nodes) * args.noise_scale

    return scenario, source, dest, obstacle, noise, src_pos, dst_pos, wall_col


def plot_results(args, snapshots, final, src_pos, dst_pos, obstacle, wall_col):
    """Generate and save visualisations using common_plot utilities."""
    if plt is None:
        print("Matplotlib not installed. Skipping plots.")
        return

    rows, cols = args.rows, args.cols
    sd_dist = final["dist_src"][dst_pos[0] * cols + dst_pos[1]].item()
    plot_channel_setup(rows, cols, src_pos, dst_pos, obstacle)
    plot_channel_evolution(rows, cols, snapshots, src_pos, dst_pos, obstacle)
    plot_channel_final_fields(rows, cols, final, src_pos, dst_pos, obstacle, args.rounds)
    plot_channel_overlay(rows, cols, final, src_pos, dst_pos, obstacle, sd_dist)


def main():
    args = parse_args()
    enforce_small_scale(args)
    
    scenario, source, dest, obstacle, noise, src_pos, dst_pos, wall_col = build_scenario(args)
    num_nodes = scenario.num_nodes
    engine = SimulationEngine.from_scenario(scenario)
    
    snapshot_steps = [5, 15, 30, 60, args.rounds - 1]
    recorder = SnapshotRecorder(
        state_fields=["dist_src", "dist_dst", "_bc_dist_channel"],
        capture_output=True,
        record_rounds=set(snapshot_steps),
    )

    def program(_runtime):
        return branch(
            ~obstacle,
            lambda: channel_body(source, dest, noise, args.tolerance),
            lambda: field.of(0.0),
            branch_name="obstacle",
        )

    engine.run(
        rounds=args.rounds,
        program=program,
        signals={"source": source, "dest": dest, "obstacle": obstacle},
        recorder=recorder,
    )

    snapshots = {}
    for t, payload in recorder.records.items():
        ds = payload.get("dist_src", torch.full((num_nodes,), float("inf")))
        dd = payload.get("dist_dst", torch.full((num_nodes,), float("inf")))
        dsd = payload.get("_bc_dist_channel", torch.full((num_nodes,), float("inf")))
        snapshots[t] = {
            "dist_src": ds,
            "dist_dst": dd,
            "sum": ds + dd,
            "dist_sd": dsd,
            "channel": payload["output"],
        }

    final = snapshots[args.rounds - 1]
    dst_id = dst_pos[0] * args.cols + dst_pos[1]
    sd_dist = final["dist_src"][dst_id].item()

    print("=== Channel with Obstacles ===")
    print(f"Grid: {args.rows}×{args.cols}   Source: {src_pos}   Dest: {dst_pos}")
    print(f"Wall: column {wall_col}")
    print(f"Rounds: {args.rounds}")
    print(f"Shortest-path distance (around wall): {sd_dist:.0f}")
    print(f"Channel nodes: {(final['channel'] > 0.5).sum().item()}")
    print()

    plot_results(args, snapshots, final, src_pos, dst_pos, obstacle, wall_col)


if __name__ == "__main__":
    main()
