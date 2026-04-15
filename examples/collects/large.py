#!/usr/bin/env python3
"""Large-scale collect example on spatial layouts."""

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

from autofield import (
    SnapshotRecorder,
    SpatialScenario,
    SimulationEngine,
    collect_cast,
    gradient,
)
from autofield.dsl import field
from autofield.utils import get_device
from shared.plotting import save_grid_simulation_gif, to_grid


def parse_args():
    parser = argparse.ArgumentParser(
        description="Large-scale collect OR on spatial layouts"
    )
    parser.add_argument("--rows", type=int, default=40, help="Visual layout rows")
    parser.add_argument("--cols", type=int, default=40, help="Visual layout cols")
    parser.add_argument(
        "--rounds", type=int, default=0, help="Number of compute rounds (0 = auto)"
    )
    parser.add_argument("--seed", type=int, default=7, help="Random seed")
    parser.add_argument(
        "--device", type=str, default="", help="Device (cuda/cpu) [auto if empty]"
    )
    parser.add_argument(
        "--topology",
        type=str,
        default="radius",
        choices=["radius", "knn", "full"],
        help="Spatial topology used by the aggregate program",
    )
    parser.add_argument(
        "--k-neighbors",
        type=int,
        default=8,
        help="Neighbors per node when --topology knn",
    )
    parser.add_argument(
        "--edge-radius",
        type=float,
        default=0.0,
        help="Radius when --topology radius (0 = auto from grid spacing)",
    )
    parser.add_argument("--viz-prefix", type=str, default="generated/collects_large")
    parser.add_argument("--gif-fps", type=int, default=8)
    parser.add_argument("--no-viz", action="store_true", help="Disable visualization")
    parser.add_argument("--no-gif", action="store_true", help="Disable GIF generation")
    return parser.parse_args()


def auto_rounds(args) -> int:
    if args.rounds > 0:
        return args.rounds
    if args.topology == "full":
        return 4
    return 2 * max(args.rows, args.cols)


def build_positions(rows: int, cols: int, device: torch.device) -> torch.Tensor:
    y_coords = torch.linspace(0.0, 1.0, rows, device=device)
    x_coords = torch.linspace(0.0, 1.0, cols, device=device)
    grid_y, grid_x = torch.meshgrid(y_coords, x_coords, indexing="ij")
    return torch.stack((grid_x.reshape(-1), grid_y.reshape(-1)), dim=-1)


def central_region_spec(rows: int, cols: int) -> tuple[float, float, float]:
    radius = max(3.0, min(rows, cols) / 8.0)
    center_row = 0.5 * (rows - 1)
    center_col = 0.5 * (cols - 1)
    return center_row, center_col, radius


def default_edge_radius(rows: int, cols: int) -> float:
    row_step = 1.0 / max(rows - 1, 1)
    col_step = 1.0 / max(cols - 1, 1)
    return 2.01 * max(row_step, col_step)


def effective_edge_radius(args) -> float:
    if args.edge_radius > 0:
        return args.edge_radius
    return default_edge_radius(args.rows, args.cols)


def topology_label(args) -> str:
    if args.topology == "full":
        return "fully connected"
    if args.topology == "knn":
        return f"k-NN (k={args.k_neighbors})"
    return f"radius (r={effective_edge_radius(args):.3f})"


def build_scenario(
    args, positions: torch.Tensor, device: torch.device
) -> SpatialScenario:
    kwargs = {
        "positions": positions,
        "self_loops": False,
        "edge_weight_mode": "distance",
        "device": device,
    }
    if args.topology == "full":
        return SpatialScenario(fully_connected=True, **kwargs)
    if args.topology == "knn":
        return SpatialScenario(k_neighbors=args.k_neighbors, **kwargs)
    return SpatialScenario(edge_radius=effective_edge_radius(args), **kwargs)


def build_example(args, device: torch.device):
    positions = build_positions(args.rows, args.cols, device)
    scenario = build_scenario(args, positions, device)

    source_row = max(0, args.rows - 2)
    source_col = 1 if args.cols > 1 else 0
    source_idx = source_row * args.cols + source_col

    source = scenario.marker(source_idx)

    center_row, center_col, radius = central_region_spec(args.rows, args.cols)
    row_coords = torch.arange(args.rows, device=device, dtype=torch.float32).unsqueeze(
        1
    )
    col_coords = torch.arange(args.cols, device=device, dtype=torch.float32).unsqueeze(
        0
    )
    data_grid = (row_coords - center_row).square() + (
        col_coords - center_col
    ).square() <= radius * radius
    data = data_grid.reshape(-1)

    metadata = {
        "source_row": source_row,
        "source_col": source_col,
        "source_idx": source_idx,
        "center_region": {
            "center_row": center_row,
            "center_col": center_col,
            "radius": radius,
        },
        "topology_label": topology_label(args),
    }
    return scenario, source, data, metadata


def run_collect_example(args, scenario, source, data):
    engine = SimulationEngine.from_scenario(scenario)
    rounds = auto_rounds(args)
    recorder = (
        SnapshotRecorder(capture_output=True)
        if (not args.no_viz or not args.no_gif)
        else None
    )

    def program(runtime):
        potential = gradient(runtime.signals["source"], name="spatial_potential")
        collect_or = collect_cast(
            potential,
            runtime.signals["data"],
            field.of(False),
            torch.logical_or,
            name="collect_or",
            mode="hard",
        )
        return torch.stack((potential, collect_or.float()), dim=-1)

    start_time = time.perf_counter()
    output, _ = engine.run(
        rounds=rounds,
        program=program,
        signals={"source": source, "data": data},
        recorder=recorder,
    )
    elapsed = time.perf_counter() - start_time
    snapshots = {}
    if recorder is not None:
        snapshots = {
            round_idx + 1: payload["output"].detach().cpu().clone()
            for round_idx, payload in sorted(recorder.records.items())
        }
    return output, snapshots, rounds, elapsed


def cumulative_collect_cone(snapshots: dict[int, torch.Tensor]) -> torch.Tensor | None:
    if not snapshots:
        return None
    history = torch.stack(
        [snapshot[:, 1].float() for _, snapshot in sorted(snapshots.items())], dim=0
    )
    return history.mean(dim=0)


def plot_results(
    final_output: torch.Tensor,
    snapshots: dict[int, torch.Tensor],
    data: torch.Tensor,
    metadata: dict[str, int],
    args,
) -> None:
    if plt is None or args.no_viz:
        return

    final_cpu = final_output.detach().cpu()
    data_cpu = data.detach().cpu().float()
    potential = final_cpu[:, 0]
    collected = final_cpu[:, 1]
    cone = cumulative_collect_cone(snapshots)
    cone_field = collected if cone is None else cone.detach().cpu()
    source_row = metadata["source_row"]
    source_col = metadata["source_col"]
    finite_potential = potential[torch.isfinite(potential)]
    potential_vmax = (
        float(finite_potential.max().item()) if finite_potential.numel() > 0 else 1.0
    )

    fig, axes = plt.subplots(1, 4, figsize=(20, 5.2))

    panels = [
        (
            to_grid(data_cpu, args.rows, args.cols, replace_inf=False),
            "Central data region",
            "cividis",
            0.0,
            1.0,
        ),
        (
            to_grid(potential, args.rows, args.cols),
            "Potential = spatial gradient(source)",
            "magma",
            0.0,
            potential_vmax,
        ),
        (
            to_grid(collected, args.rows, args.cols, replace_inf=False),
            "Hard collect OR",
            "viridis",
            0.0,
            1.0,
        ),
        (
            to_grid(cone_field, args.rows, args.cols, replace_inf=False),
            "Aggregate cone = mean(round-wise collect)",
            "inferno",
            0.0,
            1.0,
        ),
    ]

    for ax, (grid, title, cmap, vmin, vmax) in zip(axes, panels):
        im = ax.imshow(grid, cmap=cmap, vmin=vmin, vmax=vmax, interpolation="nearest")
        ax.plot(
            source_col,
            source_row,
            marker="*",
            markersize=14,
            color="red",
            markeredgecolor="white",
            markeredgewidth=1.0,
        )
        ax.set_title(title)
        ax.set_xticks([])
        ax.set_yticks([])
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    fig.suptitle(f"Large-scale collect OR on a {metadata['topology_label']} layout")
    plt.tight_layout()
    output_path = f"{args.viz_prefix}.png"
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=160)
    plt.close(fig)
    print(f"Visualization saved to {output_path}")

    if args.no_gif:
        return

    if not snapshots:
        return

    gif_records = {
        round_idx - 1: {"collect": snapshot[:, 1]}
        for round_idx, snapshot in snapshots.items()
    }
    save_grid_simulation_gif(
        gif_records,
        "collect",
        args.rows,
        args.cols,
        f"{args.viz_prefix}_evolution.gif",
        src_pos=(source_row, source_col),
        cmap="viridis",
        vmin=0.0,
        vmax=1.0,
        fps=args.gif_fps,
        title=f"Collect OR evolution ({metadata['topology_label']})",
    )


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    device = get_device(args.device)

    scenario, source, data, metadata = build_example(args, device)
    output, snapshots, rounds, elapsed = run_collect_example(
        args, scenario, source, data
    )

    potential = output[:, 0]
    collected = output[:, 1] > 0.5
    cone = cumulative_collect_cone(snapshots)
    source_idx = metadata["source_idx"]
    source_value = bool(collected[source_idx].item())
    expected = bool(data.any().item())
    source_cone = float(cone[source_idx].item()) if cone is not None else None
    reached_nodes = int(collected.sum().item())

    center_region = metadata["center_region"]
    print("=== Large-scale spatial collect OR ===")
    print(f"Device: {device}")
    print(
        f"Layout: {args.rows}x{args.cols}  Topology: {metadata['topology_label']}  "
        f"Nodes: {scenario.num_nodes}  Edges: {scenario.edge_index.shape[1]}"
    )
    print(
        f"Rounds: {rounds}  Compute time: {elapsed:.3f}s  Source: ({metadata['source_row']}, {metadata['source_col']})"
    )
    print(
        "Central true region: "
        f"circle centered at ({center_region['center_row']:.1f}, {center_region['center_col']:.1f}) "
        f"with radius {center_region['radius']:.1f}"
    )
    print(f"True nodes in data region: {int(data.sum().item())}")
    print(f"Potential at source: {potential[source_idx].item():.4f}")
    print(f"Collect OR at source: {source_value}  (expected: {expected})")
    print(f"Nodes in hard collect support: {reached_nodes}")
    if source_cone is not None:
        print(f"Aggregate cone intensity at source: {source_cone:.4f}")

    plot_results(output, snapshots, data, metadata, args)


if __name__ == "__main__":
    main()
