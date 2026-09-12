#!/usr/bin/env python3
"""Gradient-based distance computation with obstacles using branch."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "examples"))

import numpy as np  # noqa: E402
import torch  # noqa: E402

try:
    import matplotlib.patches as mpatches
    import matplotlib.pyplot as plt
    from shared.plotting import apply_paper_style

    apply_paper_style()
except ImportError:
    mpatches = None
    plt = None

from shared.plotting import (  # noqa: E402
    draw_markers,
    draw_obstacles,
    save_grid_simulation_gif,
    to_grid,
)
from shared.plotting.style import MUTED  # noqa: E402

from diffield.dsl import field, gather_min, iterate, mux, scatter  # noqa: E402
from diffield.sim import GridScenario, SimulationEngine, SnapshotRecorder  # noqa: E402
from diffield.utils import get_device  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser(description="Gradient with obstacles")
    parser.add_argument("--rows", type=int, default=15, help="Grid rows")
    parser.add_argument("--cols", type=int, default=15, help="Grid cols")
    parser.add_argument("--rounds", type=int, default=100, help="Number of compute rounds")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--device", type=str, default="", help="Device (cuda/cpu) [auto if empty]")
    parser.add_argument("--viz-prefix", type=str, default="generated/gradient_obstacle")
    parser.add_argument("--gif-fps", type=int, default=10)
    parser.add_argument("--no-viz", action="store_true", help="Disable visualization")
    parser.add_argument("--no-gif", action="store_true", help="Disable GIF generation")
    return parser.parse_args()


def build_obstacle_wall(rows: int, cols: int, device: torch.device) -> torch.Tensor:
    """Create a vertical wall obstacle in the middle of the grid."""
    obstacle = torch.zeros(rows * cols, dtype=torch.bool, device=device)
    wall_col = cols // 2
    # Leave a gap at the bottom for passage
    for row in range(rows):
        if row < rows - 3:  # Leave last 3 rows open
            obstacle[row * cols + wall_col] = True
    return obstacle


def run_gradient_with_obstacle(
    scenario: GridScenario,
    source: torch.Tensor,
    obstacle: torch.Tensor,
    rounds: int,
    weight: float = 1.0,
    recorder: SnapshotRecorder | None = None,
) -> torch.Tensor:
    """Run gradient computation that avoids obstacles using branch."""
    engine = SimulationEngine.from_scenario(scenario)
    weight_tensor = torch.tensor(weight, device=scenario.device)

    def program(_runtime):
        # Use branch to skip obstacle cells
        # create a matrix that where obstacle is true, the result in infity, where is false, 1
        obstacle_tensor_with_inf = mux(obstacle, field.inf(), field.of(1.0))
        return mux(
            ~obstacle,
            iterate(
                field.inf(),
                lambda dist_old: mux(
                    source,
                    field.of(0.0),
                    gather_min(scatter(dist_old + weight_tensor * obstacle_tensor_with_inf))
                ),
                name="dist"
            ),
            field.of(0.0),
        )

    output, _ = engine.run(
        rounds=rounds,
        program=program,
        signals={"source": source, "obstacle": obstacle},
        recorder=recorder,
    )
    return output


def plot_setup(
    rows: int,
    cols: int,
    src_pos: tuple[int, int],
    obstacle: torch.Tensor,
    viz_prefix: str = "generated/gradient_obstacle",
):
    """Plot the initial grid setup with source and obstacles."""
    if plt is None or mpatches is None:
        print("matplotlib not available; skipping setup plot")
        return

    _fig, ax = plt.subplots(figsize=(7, 7))
    grid_rgb = np.full((rows, cols, 3), 0.92)
    obstacle_cpu = obstacle.detach().cpu().numpy()

    for row in range(rows):
        for col in range(cols):
            if obstacle_cpu[row * cols + col]:
                grid_rgb[row, col] = [0.15, 0.15, 0.15]

    grid_rgb[src_pos] = [0.0, 0.75, 0.0]

    ax.imshow(grid_rgb, interpolation="nearest")
    for row in range(rows + 1):
        ax.axhline(row - 0.5, color="white", lw=0.4)
    for col in range(cols + 1):
        ax.axvline(col - 0.5, color="white", lw=0.4)

    ax.set_xlabel("Column")
    ax.set_ylabel("Row")
    ax.legend(
        handles=[
            mpatches.Patch(color="green", label="Source"),
            mpatches.Patch(color="black", label="Obstacle wall"),
        ],
        loc="lower right",
        fontsize=9,
    )
    plt.tight_layout()
    output_path = f"{viz_prefix}_setup.png"
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150)
    print(f"Saved {output_path}")


def plot_final_field(
    rows: int,
    cols: int,
    dist: torch.Tensor,
    src_pos: tuple[int, int],
    obstacle: torch.Tensor,
    rounds: int,
    viz_prefix: str = "generated/gradient_obstacle",
):
    """Plot the final distance field."""
    if plt is None:
        return

    _fig, ax = plt.subplots(figsize=(8, 7))
    grid = to_grid(dist, rows, cols, obstacle)
    im = ax.imshow(grid, cmap="viridis", interpolation="nearest")

    draw_obstacles(ax, obstacle, rows, cols)
    draw_markers(ax, src_pos, ms=14)

    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    # Annotate finite cells with distance values
    dist_cpu = dist.detach().cpu()
    obstacle_cpu = obstacle.detach().cpu()
    for row in range(rows):
        for col in range(cols):
            node_id = row * cols + col
            if not obstacle_cpu[node_id]:
                value = dist_cpu[node_id].item()
                if np.isfinite(value):
                    color = "white" if value > 15 else "black"
                    ax.text(
                        col, row, f"{value:.0f}",
                        ha="center", va="center", fontsize=5, color=color,
                    )

    plt.tight_layout()
    output_path = f"{viz_prefix}_final.png"
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150)
    print(f"Saved {output_path}")


def plot_evolution(
    rows: int,
    cols: int,
    records: dict[int, dict[str, torch.Tensor]],
    src_pos: tuple[int, int],
    obstacle: torch.Tensor,
    viz_prefix: str = "generated/gradient_obstacle",
):
    """Plot selected evolution snapshots."""
    if plt is None:
        return

    snapshot_steps = sorted(records.keys())
    field_key = "dist"

    fig, axes = plt.subplots(1, len(snapshot_steps), figsize=(3.5 * len(snapshot_steps), 3.5))
    if len(snapshot_steps) == 1:
        axes = [axes]

    for col_idx, step in enumerate(snapshot_steps):
        ax = axes[col_idx]
        grid = to_grid(records[step][field_key], rows, cols, obstacle)
        im = ax.imshow(grid, cmap="viridis", interpolation="nearest")
        draw_obstacles(ax, obstacle, rows, cols)
        draw_markers(ax, src_pos, ms=10)
        ax.text(
            0.03, 0.96, f"t={step}",
            transform=ax.transAxes, fontsize=9, color=MUTED, ha="left", va="top",
        )
        ax.set_xticks([])
        ax.set_yticks([])
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    plt.tight_layout()
    output_path = f"{viz_prefix}_evolution.png"
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150)
    print(f"Saved {output_path}")


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    device = get_device(args.device)

    # Build scenario
    scenario = GridScenario(args.rows, args.cols, connectivity=8, device=device)
    src_pos = (args.rows // 2, 2)
    source = scenario.marker(src_pos[0], src_pos[1])

    # Build obstacle
    obstacle = build_obstacle_wall(args.rows, args.cols, device=device)

    # Setup recording for GIF
    recorder = None
    if not args.no_gif:
        recorder = SnapshotRecorder(state_fields=["dist"], capture_output=True)

    print("=== Gradient with Obstacles ===")
    print(f"Grid: {args.rows}x{args.cols}   Source: {src_pos}")
    print(f"Wall: column {args.cols // 2} (with gap at bottom)")
    print(f"Device: {device}")

    # Run simulation
    dist = run_gradient_with_obstacle(scenario, source, obstacle, args.rounds, recorder=recorder)

    # Statistics
    finite_mask = torch.isfinite(dist) & ~obstacle
    max_dist = dist[finite_mask].max().item() if finite_mask.any() else float("inf")
    print(f"Max finite distance: {max_dist:.2f}")

    # Visualizations
    if not args.no_viz:
        plot_setup(args.rows, args.cols, src_pos, obstacle, viz_prefix=args.viz_prefix)
        plot_final_field(
            args.rows, args.cols, dist, src_pos, obstacle,
            args.rounds, viz_prefix=args.viz_prefix,
        )

        if recorder and recorder.records:
            # Filter snapshots for evolution plot
            snapshot_rounds = [5, 15, 30, 60, args.rounds - 1]
            filtered_records = {k: v for k, v in recorder.records.items() if k in snapshot_rounds}
            if not filtered_records:
                filtered_records = recorder.records

            plot_evolution(
                args.rows, args.cols, filtered_records, src_pos,
                obstacle, viz_prefix=args.viz_prefix,
            )

            # Generate GIF
            if not args.no_gif:
                gif_path = f"{args.viz_prefix}_evolution.gif"
                save_grid_simulation_gif(
                    recorder.records,
                    "dist",
                    args.rows,
                    args.cols,
                    gif_path,
                    obstacle=obstacle,
                    src_pos=src_pos,
                    cmap="viridis",
                    vmin=0.0,
                    vmax=max(45.0, max_dist),
                    title="Gradient Evolution",
                    fps=args.gif_fps,
                )
                print(f"Saved {gif_path}")


if __name__ == "__main__":
    main()
