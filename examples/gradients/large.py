#!/usr/bin/env python3
"""Large-scale gradient example."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import torch

try:
    import matplotlib.pyplot as plt
except ImportError:
    plt = None

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "examples"))

from autofield import SnapshotRecorder
from shared.plotting import save_grid_simulation_gif

try:
    from .domain.program import auto_rounds, run_gradient_program
except ImportError:
    from gradients.domain.program import auto_rounds, run_gradient_program

from autofield import GridScenario, SimulationEngine, mux, scatter, iterate, scatter_range, gather_min
from autofield.dsl import field
from autofield.utils import get_device


def parse_args():
    parser = argparse.ArgumentParser(description="Large-scale Gradient")
    parser.add_argument("--rows", type=int, default=500, help="Grid rows")
    parser.add_argument("--cols", type=int, default=500, help="Grid cols")
    parser.add_argument(
        "--rounds", type=int, default=0, help="Number of compute rounds (0 = auto)"
    )
    parser.add_argument("--seed", type=int, default=7, help="Random seed")
    parser.add_argument(
        "--device", type=str, default="", help="Device (cuda/cpu) [auto if empty]"
    )
    parser.add_argument("--viz-prefix", type=str, default="generated/gradient_large")
    parser.add_argument("--gif-fps", type=int, default=10)
    parser.add_argument("--no-viz", action="store_true", help="Disable visualization")
    parser.add_argument("--no-gif", action="store_true", help="Disable GIF generation")
    return parser.parse_args()


def build_large_data(args, device: torch.device):
    scenario = GridScenario(args.rows, args.cols, connectivity=4, device=device)
    center_r = args.rows // 2
    center_c = args.cols // 2
    source = scenario.marker(center_r, center_c)
    return scenario, source, center_r, center_c


def run_large_gradient(args, scenario, source, device: torch.device):
    rounds = auto_rounds(args.rows, args.cols, args.rounds)
    weight = torch.tensor(1.0, device=device, requires_grad=True)
    scenario.set_edge_weight(weight)
    engine = SimulationEngine.from_scenario(scenario)

    print(f"Running {rounds} rounds of aggregate computation...")
    start_time = time.time()

    record_at = [max(1, rounds // 10), max(1, rounds // 2), rounds]
    record_rounds = None if not args.no_gif else {step - 1 for step in record_at}

    recorder = SnapshotRecorder(
        state_fields=["dist"],
        capture_output=True,
        record_rounds=record_rounds,
    )

    def program(_runtime):
        return iterate(
            field.inf(),
            lambda dist_old: mux(
                source, field.of(0.0), gather_min(scatter(dist_old + weight))
            ),
            name="dist",
        )

    output, _ = engine.run(
        rounds=rounds, program=program, signals={"source": source}, recorder=recorder
    )

    if not args.no_gif and not args.no_viz:
        print("Generating evolution GIF...")
        # Subsample frames for very large simulations to keep GIF size reasonable
        gif_records = recorder.records
        if len(gif_records) > 50:
            step = len(gif_records) // 50
            gif_records = {
                k: v
                for i, (k, v) in enumerate(sorted(gif_records.items()))
                if i % step == 0
            }

        save_grid_simulation_gif(
            gif_records,
            "dist",
            args.rows,
            args.cols,
            f"{args.viz_prefix}_evolution.gif",
            title="Large-Scale Gradient Evolution",
            vmax=float(args.rows + args.cols) / 2.0,
            fps=args.gif_fps,
        )

    snapshots = {
        round_idx + 1: payload["output"]
        .detach()
        .cpu()
        .view(args.rows, args.cols)
        .clone()
        for round_idx, payload in recorder.records.items()
        if not (not args.no_gif) or (round_idx + 1) in record_at
    }

    if device.type == "cuda":
        torch.cuda.synchronize()

    compute_time = time.time() - start_time
    print(
        f"Computation finished in {compute_time:.4f}s ({compute_time / rounds:.6f}s per round)"
    )
    return output, weight, snapshots, compute_time, rounds


def plot_results(dist: torch.Tensor, snapshots: dict[int, torch.Tensor], args) -> None:
    if plt is None or args.no_viz:
        return

    fig, axes = plt.subplots(1, len(snapshots), figsize=(20, 4))
    if len(snapshots) == 1:
        axes = [axes]
    for index, (round_idx, snapshot) in enumerate(sorted(snapshots.items())):
        im = axes[index].imshow(snapshot.numpy(), cmap="magma")
        axes[index].set_title(f"Round {round_idx}")
        axes[index].axis("off")
        fig.colorbar(im, ax=axes[index], fraction=0.046, pad=0.04)

    plt.tight_layout()
    evolution_path = f"{args.viz_prefix}_evolution.png"
    Path(evolution_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(evolution_path)
    print(f"Evolution visualization saved to {evolution_path}")

    plt.figure(figsize=(8, 6))
    plt.imshow(dist.numpy(), cmap="magma")
    plt.colorbar(label="Distance")
    plt.title(
        f"Final Gradient field on {args.rows}x{args.cols} grid (Source at center)"
    )
    final_path = f"{args.viz_prefix}.png"
    Path(final_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(final_path)
    print(f"Final visualization saved to {final_path}")


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    device = get_device(args.device)

    print(
        f"=== Large-scale Gradient ({args.rows}x{args.cols} grid, {args.rows * args.cols} nodes) ==="
    )
    print(f"Device: {device}")

    print("Building graph...")
    start_time = time.time()
    scenario, source, center_r, center_c = build_large_data(args, device)
    graph_time = time.time() - start_time
    print(f"Graph built in {graph_time:.4f}s (Edges: {scenario.edge_index.shape[1]})")

    output, weight, snapshots, _, rounds = run_large_gradient(
        args, scenario, source, device
    )

    dist = output.detach().cpu().view(args.rows, args.cols)
    center_val = dist[center_r, center_c].item()
    print(f"Distance at center: {center_val:.1f} (expected 0.0)")

    corner_val = dist[0, 0].item()
    expected_corner = center_r + center_c
    print(f"Distance at corner (0,0): {corner_val:.1f} (expected {expected_corner}.0)")

    print("Computing gradient w.r.t. weight w...")
    start_time = time.time()
    loss = output[output.isfinite()].sum()
    loss.backward()
    backward_time = time.time() - start_time
    print(f"Backward pass in {backward_time:.4f}s")
    print(f"d(loss)/dw = {weight.grad.item():.1f}")

    plot_results(dist, snapshots, args)


if __name__ == "__main__":
    main()
