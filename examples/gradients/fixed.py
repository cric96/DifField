#!/usr/bin/env python3
"""Gradient (hop-distance) with fixed weights."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "examples"))

import torch
from diffield.dsl import field
from diffield.utils import get_device

try:
    import matplotlib.pyplot as plt
except ImportError:
    plt = None

try:
    from .domain.program import auto_rounds, run_gradient_program
    from .domain.grid import build_corner_source_grid
except ImportError:
    from gradients.domain.program import auto_rounds, run_gradient_program
    from gradients.domain.grid import build_corner_source_grid

from diffield.sim import SnapshotRecorder
from shared.plotting import save_grid_simulation_gif


def parse_args():
    parser = argparse.ArgumentParser(description="Fixed Gradient AC program")
    parser.add_argument("--rows", type=int, default=5, help="Grid rows")
    parser.add_argument("--cols", type=int, default=5, help="Grid cols")
    parser.add_argument(
        "--rounds", type=int, default=0, help="Number of compute rounds (0 = auto)"
    )
    parser.add_argument("--seed", type=int, default=7, help="Random seed")
    parser.add_argument("--weight", type=float, default=1.0, help="Fixed edge weight")
    parser.add_argument(
        "--device", type=str, default="", help="Device (cuda/cpu) [auto if empty]"
    )
    parser.add_argument("--viz-prefix", type=str, default="generated/gradient_fixed")
    parser.add_argument("--gif-fps", type=int, default=10)
    parser.add_argument("--no-viz", action="store_true", help="Disable figure export")
    parser.add_argument("--no-gif", action="store_true", help="Disable gif export")
    return parser.parse_args()


def plot_results(
    dist: torch.Tensor,
    expected: torch.Tensor,
    viz_prefix: str = "generated/gradient_fixed",
) -> None:
    if plt is None:
        return

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    im0 = axes[0].imshow(dist.detach().cpu().numpy(), cmap="viridis")
    axes[0].set_title("Computed (iterate + mux + scatter)")
    plt.colorbar(im0, ax=axes[0])

    im1 = axes[1].imshow(expected.detach().cpu().numpy(), cmap="viridis")
    axes[1].set_title("Expected (Manhattan)")
    plt.colorbar(im1, ax=axes[1])

    plt.tight_layout()
    output_path = f"{viz_prefix}.png"
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150)
    print(f"Saved figure to {output_path}")


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    device = get_device(args.device)
    scenario, source, expected = build_corner_source_grid(
        args.rows, args.cols, device=device
    )
    rounds = auto_rounds(args.rows, args.cols, args.rounds)

    recorder = None
    if not args.no_gif:
        recorder = SnapshotRecorder(state_fields=["dist"], capture_output=True)

    dist, _ = run_gradient_program(
        scenario,
        source,
        rounds=rounds,
        weight=torch.tensor(args.weight, device=device),
        recorder=recorder,
    )

    if not args.no_gif and not args.no_viz and recorder:
        save_grid_simulation_gif(
            recorder.records,
            "dist",
            args.rows,
            args.cols,
            f"{args.viz_prefix}_evolution.gif",
            title="Fixed Gradient Evolution",
            fps=args.gif_fps,
        )

    print(
        f"Final distance (0,0)->({args.rows - 1},{args.cols - 1}): {dist[scenario.num_nodes - 1].item():.2f}"
    )
    if not args.no_viz:
        plot_results(
            dist.view(args.rows, args.cols),
            expected.view(args.rows, args.cols),
            viz_prefix=args.viz_prefix,
        )


if __name__ == "__main__":
    main()
