#!/usr/bin/env python3
"""Scaling benchmark for the spatial channel example."""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "examples"))

from autofield import SimulationEngine, SpatialScenario, branch
from autofield.dsl import field
from autofield.utils import get_device
from channel.core import CHANNEL_THRESHOLD, channel_body
from channel.spatial import create_obstacle_mask
from shared.training import parse_int_csv

try:
    import matplotlib.pyplot as plt
except ImportError:
    plt = None


@dataclass(frozen=True)
class BenchmarkConfig:
    num_nodes: int
    k_neighbors: int
    rounds: int
    tolerance: float
    obstacle_ratio: float
    seed: int
    device: str


@dataclass(frozen=True)
class BenchmarkArtifacts:
    raw_csv: Path
    aggregated_csv: Path
    fixed_k_growth_csv: Path
    fixed_node_growth_csv: Path
    report_markdown: Path
    summary_json: Path
    density_sweep_plot: Path | None
    neighborhood_sweep_plot: Path | None
    surface_plot: Path | None


def default_seeds(repetitions: int, start: int) -> list[int]:
    return list(range(start, start + repetitions))


def synchronize_device(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def select_source_index(positions: torch.Tensor) -> int:
    pos = positions.detach().cpu()
    corner_score = pos[:, 0] - pos[:, 1]
    return int(torch.argmin(corner_score).item())


def select_destination_index(positions: torch.Tensor) -> int:
    pos = positions.detach().cpu()
    corner_score = (1.0 - pos[:, 0]) + pos[:, 1]
    return int(torch.argmin(corner_score).item())


def run_single_trial(config: BenchmarkConfig) -> dict[str, Any]:
    device = get_device(config.device)
    torch.manual_seed(config.seed)

    positions = torch.rand(config.num_nodes, 2, device=device)
    scenario = SpatialScenario(
        positions=positions,
        k_neighbors=config.k_neighbors,
        edge_weight_mode="distance",
        device=device,
    )
    source_idx = select_source_index(scenario.positions)
    dest_idx = select_destination_index(scenario.positions)
    source = scenario.marker(source_idx)
    dest = scenario.marker(dest_idx)
    obstacle = create_obstacle_mask(scenario.positions, config.obstacle_ratio)
    engine = SimulationEngine.from_scenario(scenario)

    def program(runtime):
        src = source.to(runtime.scenario.device)
        dst = dest.to(runtime.scenario.device)
        obs = obstacle.to(runtime.scenario.device)
        return branch(
            ~obs,
            lambda: channel_body(src, dst, config.tolerance),
            lambda: field.of(0.0),
            branch_name="obstacle",
        )

    mean_degree = float(scenario.edge_index.shape[1]) / float(scenario.num_nodes)
    obstacle_fraction = float(obstacle.float().mean().item())

    try:
        with torch.no_grad():
            synchronize_device(device)
            started = time.perf_counter()
            output, _runtime = engine.run(
                rounds=config.rounds,
                program=program,
                signals={"source": source, "dest": dest, "obstacle": obstacle},
            )
            synchronize_device(device)
            elapsed = time.perf_counter() - started

        channel_nodes = int((output > CHANNEL_THRESHOLD).sum().item())
        finite_output_fraction = float(torch.isfinite(output).float().mean().item())
        reached_target = bool(output[dest_idx].item() > CHANNEL_THRESHOLD)
        status = "ok"
        error_message = ""
    except Exception as exc:
        elapsed = math.nan
        channel_nodes = 0
        finite_output_fraction = 0.0
        reached_target = False
        status = "error"
        error_message = str(exc)

    return {
        "num_nodes": config.num_nodes,
        "k_neighbors": config.k_neighbors,
        "rounds": config.rounds,
        "tolerance": config.tolerance,
        "obstacle_ratio": config.obstacle_ratio,
        "seed": config.seed,
        "device": device.type,
        "source_idx": source_idx,
        "dest_idx": dest_idx,
        "edge_count": int(scenario.edge_index.shape[1]),
        "mean_degree": mean_degree,
        "obstacle_fraction": obstacle_fraction,
        "time_seconds": elapsed,
        "channel_nodes": channel_nodes,
        "finite_output_fraction": finite_output_fraction,
        "target_on_channel": reached_target,
        "status": status,
        "error_message": error_message,
    }


def aggregate_rows(raw_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[int, int], list[dict[str, Any]]] = {}
    for row in raw_rows:
        key = (int(row["num_nodes"]), int(row["k_neighbors"]))
        grouped.setdefault(key, []).append(row)

    aggregated: list[dict[str, Any]] = []
    for (num_nodes, k_neighbors), rows in sorted(grouped.items()):
        successful = [row for row in rows if row["status"] == "ok" and math.isfinite(float(row["time_seconds"]))]
        times = [float(row["time_seconds"]) for row in successful]
        mean_degree_values = [float(row["mean_degree"]) for row in successful]
        edge_count_values = [float(row["edge_count"]) for row in successful]
        channel_nodes_values = [float(row["channel_nodes"]) for row in successful]
        obstacle_values = [float(row["obstacle_fraction"]) for row in successful]
        target_rate_values = [1.0 if bool(row["target_on_channel"]) else 0.0 for row in successful]

        aggregated.append(
            {
                "num_nodes": num_nodes,
                "k_neighbors": k_neighbors,
                "trials": len(rows),
                "successful_trials": len(successful),
                "success_rate": (len(successful) / len(rows)) if rows else 0.0,
                "mean_time_seconds": statistics.fmean(times) if times else math.nan,
                "std_time_seconds": statistics.stdev(times) if len(times) > 1 else 0.0,
                "median_time_seconds": statistics.median(times) if times else math.nan,
                "min_time_seconds": min(times) if times else math.nan,
                "max_time_seconds": max(times) if times else math.nan,
                "mean_degree": statistics.fmean(mean_degree_values) if mean_degree_values else math.nan,
                "mean_edge_count": statistics.fmean(edge_count_values) if edge_count_values else math.nan,
                "mean_channel_nodes": statistics.fmean(channel_nodes_values) if channel_nodes_values else math.nan,
                "mean_obstacle_fraction": statistics.fmean(obstacle_values) if obstacle_values else math.nan,
                "target_on_channel_rate": statistics.fmean(target_rate_values) if target_rate_values else 0.0,
            }
        )
    return aggregated


def build_fixed_k_growth_rows(aggregated_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[int, list[dict[str, Any]]] = {}
    for row in aggregated_rows:
        grouped.setdefault(int(row["k_neighbors"]), []).append(row)

    growth_rows: list[dict[str, Any]] = []
    for k_neighbors, rows in sorted(grouped.items()):
        ordered = sorted(rows, key=lambda row: int(row["num_nodes"]))
        baseline = ordered[0]
        baseline_time = float(baseline["mean_time_seconds"])
        baseline_nodes = int(baseline["num_nodes"])
        for row in ordered:
            current_time = float(row["mean_time_seconds"])
            time_ratio = current_time / baseline_time if baseline_time > 0.0 and math.isfinite(current_time) else math.nan
            growth_rows.append(
                {
                    "sweep": "fixed_k_neighbors",
                    "k_neighbors": k_neighbors,
                    "baseline_num_nodes": baseline_nodes,
                    "num_nodes": int(row["num_nodes"]),
                    "mean_degree": float(row["mean_degree"]),
                    "mean_time_seconds": current_time,
                    "baseline_time_seconds": baseline_time,
                    "time_ratio_vs_baseline": time_ratio,
                    "time_delta_vs_baseline": current_time - baseline_time if math.isfinite(current_time) else math.nan,
                    "percent_increase_vs_baseline": (time_ratio - 1.0) * 100.0 if math.isfinite(time_ratio) else math.nan,
                }
            )
    return growth_rows


def build_fixed_node_growth_rows(aggregated_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[int, list[dict[str, Any]]] = {}
    for row in aggregated_rows:
        grouped.setdefault(int(row["num_nodes"]), []).append(row)

    growth_rows: list[dict[str, Any]] = []
    for num_nodes, rows in sorted(grouped.items()):
        ordered = sorted(rows, key=lambda row: int(row["k_neighbors"]))
        baseline = ordered[0]
        baseline_time = float(baseline["mean_time_seconds"])
        baseline_k = int(baseline["k_neighbors"])
        baseline_degree = float(baseline["mean_degree"])
        for row in ordered:
            current_time = float(row["mean_time_seconds"])
            time_ratio = current_time / baseline_time if baseline_time > 0.0 and math.isfinite(current_time) else math.nan
            growth_rows.append(
                {
                    "sweep": "fixed_num_nodes",
                    "num_nodes": num_nodes,
                    "baseline_k_neighbors": baseline_k,
                    "baseline_mean_degree": baseline_degree,
                    "k_neighbors": int(row["k_neighbors"]),
                    "mean_degree": float(row["mean_degree"]),
                    "mean_time_seconds": current_time,
                    "baseline_time_seconds": baseline_time,
                    "time_ratio_vs_baseline": time_ratio,
                    "time_delta_vs_baseline": current_time - baseline_time if math.isfinite(current_time) else math.nan,
                    "percent_increase_vs_baseline": (time_ratio - 1.0) * 100.0 if math.isfinite(time_ratio) else math.nan,
                }
            )
    return growth_rows


def format_markdown_table(rows: list[dict[str, Any]], columns: list[tuple[str, str]]) -> str:
    if not rows:
        headers = [label for _key, label in columns]
        separator = ["---" for _ in headers]
        return "| " + " | ".join(headers) + " |\n| " + " | ".join(separator) + " |\n"

    def stringify(value: Any) -> str:
        if isinstance(value, float):
            if not math.isfinite(value):
                return "nan"
            return f"{value:.4f}"
        return str(value)

    headers = [label for _key, label in columns]
    separator = ["---" for _ in headers]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(separator) + " |",
    ]
    for row in rows:
        lines.append(
            "| "
            + " | ".join(stringify(row[key]) for key, _label in columns)
            + " |"
        )
    return "\n".join(lines) + "\n"


def write_rows_csv(rows: list[dict[str, Any]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        output_path.write_text("", encoding="utf-8")
        return
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_report_markdown(
    aggregated_rows: list[dict[str, Any]],
    fixed_k_growth: list[dict[str, Any]],
    fixed_node_growth: list[dict[str, Any]],
    output_path: Path,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    overview_rows = sorted(aggregated_rows, key=lambda row: (int(row["num_nodes"]), int(row["k_neighbors"])))
    fixed_k_rows = sorted(fixed_k_growth, key=lambda row: (int(row["k_neighbors"]), int(row["num_nodes"])))
    fixed_node_rows = sorted(fixed_node_growth, key=lambda row: (int(row["num_nodes"]), int(row["k_neighbors"])))

    text = "# Spatial Channel Scaling Benchmark\n\n"
    text += "## Aggregated Runs\n\n"
    text += format_markdown_table(
        overview_rows,
        [
            ("num_nodes", "Nodes"),
            ("k_neighbors", "k neighbors"),
            ("mean_degree", "Mean degree"),
            ("mean_time_seconds", "Mean time (s)"),
            ("std_time_seconds", "Std (s)"),
            ("success_rate", "Success rate"),
        ],
    )
    text += "\n## Growth At Fixed k Neighbors\n\n"
    text += format_markdown_table(
        fixed_k_rows,
        [
            ("k_neighbors", "k neighbors"),
            ("num_nodes", "Nodes"),
            ("mean_degree", "Mean degree"),
            ("mean_time_seconds", "Mean time (s)"),
            ("time_ratio_vs_baseline", "Time ratio"),
            ("percent_increase_vs_baseline", "% increase"),
        ],
    )
    text += "\n## Growth At Fixed Node Count\n\n"
    text += format_markdown_table(
        fixed_node_rows,
        [
            ("num_nodes", "Nodes"),
            ("k_neighbors", "k neighbors"),
            ("mean_degree", "Mean degree"),
            ("mean_time_seconds", "Mean time (s)"),
            ("time_ratio_vs_baseline", "Time ratio"),
            ("percent_increase_vs_baseline", "% increase"),
        ],
    )
    output_path.write_text(text, encoding="utf-8")


def plot_density_sweep(aggregated_rows: list[dict[str, Any]], output_path: Path) -> None:
    if plt is None:
        return

    grouped: dict[int, list[dict[str, Any]]] = {}
    for row in aggregated_rows:
        grouped.setdefault(int(row["k_neighbors"]), []).append(row)

    fig, ax = plt.subplots(figsize=(10, 6))
    fig.patch.set_facecolor("#fbf7ef")
    ax.set_facecolor("#fffdf8")
    for k_neighbors, rows in sorted(grouped.items()):
        ordered = sorted(rows, key=lambda row: int(row["num_nodes"]))
        ax.plot(
            [int(row["num_nodes"]) for row in ordered],
            [float(row["mean_time_seconds"]) for row in ordered],
            marker="o",
            linewidth=2.0,
            label=f"k={k_neighbors}",
        )

    ax.set_title("Scaling With More Nodes At Fixed k")
    ax.set_xlabel("Number of nodes")
    ax.set_ylabel("Mean runtime (s)")
    ax.grid(alpha=0.25)
    ax.legend(frameon=True)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close(fig)


def plot_neighborhood_sweep(aggregated_rows: list[dict[str, Any]], output_path: Path) -> None:
    if plt is None:
        return

    grouped: dict[int, list[dict[str, Any]]] = {}
    for row in aggregated_rows:
        grouped.setdefault(int(row["num_nodes"]), []).append(row)

    fig, ax = plt.subplots(figsize=(10, 6))
    fig.patch.set_facecolor("#fbf7ef")
    ax.set_facecolor("#fffdf8")
    for num_nodes, rows in sorted(grouped.items()):
        ordered = sorted(rows, key=lambda row: int(row["k_neighbors"]))
        ax.plot(
            [int(row["k_neighbors"]) for row in ordered],
            [float(row["mean_time_seconds"]) for row in ordered],
            marker="o",
            linewidth=2.0,
            label=f"nodes={num_nodes}",
        )

    ax.set_title("Scaling With Denser k-NN Neighborhoods At Fixed Node Count")
    ax.set_xlabel("k nearest neighbors")
    ax.set_ylabel("Mean runtime (s)")
    ax.grid(alpha=0.25)
    ax.legend(frameon=True)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close(fig)


def plot_surface(aggregated_rows: list[dict[str, Any]], output_path: Path) -> None:
    if plt is None:
        return

    valid_rows = [
        row
        for row in aggregated_rows
        if math.isfinite(float(row["mean_time_seconds"])) and math.isfinite(float(row["mean_degree"]))
    ]
    if len(valid_rows) < 3:
        return

    node_values = np.array([int(row["num_nodes"]) for row in valid_rows], dtype=float)
    degree_values = np.array([float(row["mean_degree"]) for row in valid_rows], dtype=float)
    time_values = np.array([float(row["mean_time_seconds"]) for row in valid_rows], dtype=float)

    fig = plt.figure(figsize=(11, 8))
    fig.patch.set_facecolor("#fbf7ef")
    ax = fig.add_subplot(111, projection="3d")
    surface = ax.plot_trisurf(
        node_values,
        degree_values,
        time_values,
        cmap="viridis",
        linewidth=0.2,
        antialiased=True,
        alpha=0.9,
    )
    ax.scatter(node_values, degree_values, time_values, color="#1f1b18", s=18)
    ax.set_title("Spatial Channel Runtime Surface")
    ax.set_xlabel("Number of nodes")
    ax.set_ylabel("Mean neighborhood degree")
    ax.set_zlabel("Mean runtime (s)")
    fig.colorbar(surface, shrink=0.7, pad=0.1, label="Mean runtime (s)")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=170)
    plt.close(fig)


def maybe_warmup(config: BenchmarkConfig) -> None:
    warmup_config = BenchmarkConfig(
        num_nodes=max(32, min(config.num_nodes, 256)),
        k_neighbors=min(config.k_neighbors, max(2, min(config.num_nodes - 1, 8))),
        rounds=min(config.rounds, 5),
        tolerance=config.tolerance,
        obstacle_ratio=config.obstacle_ratio,
        seed=config.seed,
        device=config.device,
    )
    run_single_trial(warmup_config)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark scaling for the spatial channel example")
    parser.add_argument("--node-counts", type=str, default="2000,4000,8000,16000")
    parser.add_argument("--k-neighbors", type=str, default="4,8,12,16")
    parser.add_argument("--rounds", type=int, default=1000)
    parser.add_argument("--tolerance", type=float, default=0.01)
    parser.add_argument("--obstacle-ratio", type=float, default=0.15)
    parser.add_argument("--repetitions", type=int, default=16)
    parser.add_argument("--seed-start", type=int, default=101)
    parser.add_argument("--seeds", type=str, default="")
    parser.add_argument("--out-dir", type=str, default="generated/results/channel_scaling")
    parser.add_argument("--device", type=str, default="")
    parser.add_argument("--skip-plots", action="store_true")
    parser.add_argument("--skip-warmup", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    node_counts = parse_int_csv(args.node_counts)
    k_neighbors_values = parse_int_csv(args.k_neighbors)
    seeds = parse_int_csv(args.seeds) if args.seeds.strip() else default_seeds(args.repetitions, args.seed_start)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not node_counts:
        raise ValueError("node-counts must not be empty")
    if not k_neighbors_values:
        raise ValueError("k-neighbors must not be empty")
    if not seeds:
        raise ValueError("At least one seed is required")

    configs = [
        BenchmarkConfig(
            num_nodes=num_nodes,
            k_neighbors=k_neighbors,
            rounds=args.rounds,
            tolerance=args.tolerance,
            obstacle_ratio=args.obstacle_ratio,
            seed=seed,
            device=args.device,
        )
        for num_nodes in node_counts
        for k_neighbors in k_neighbors_values
        for seed in seeds
    ]

    print("=== Spatial Channel Scaling Benchmark ===")
    print(f"node_counts={node_counts}")
    print(f"k_neighbors={k_neighbors_values}")
    print(f"seeds={seeds}")
    print(f"total_trials={len(configs)}")
    print(f"out_dir={out_dir}")

    if not args.skip_warmup:
        print("Running warmup...")
        maybe_warmup(configs[0])

    raw_rows: list[dict[str, Any]] = []
    benchmark_start = time.perf_counter()
    for index, config in enumerate(configs, start=1):
        print(
            f"[{index}/{len(configs)}] nodes={config.num_nodes} k={config.k_neighbors} seed={config.seed}"
        )
        row = run_single_trial(config)
        raw_rows.append(row)
        if row["status"] == "ok":
            print(
                f"  time={row['time_seconds']:.3f}s mean_degree={row['mean_degree']:.2f} "
                f"edges={row['edge_count']} channel_nodes={row['channel_nodes']}"
            )
        else:
            print(f"  FAILED: {row['error_message']}")

    total_elapsed = time.perf_counter() - benchmark_start
    aggregated_rows = aggregate_rows(raw_rows)
    fixed_k_growth = build_fixed_k_growth_rows(aggregated_rows)
    fixed_node_growth = build_fixed_node_growth_rows(aggregated_rows)

    artifacts = BenchmarkArtifacts(
        raw_csv=out_dir / "raw_runs.csv",
        aggregated_csv=out_dir / "aggregated.csv",
        fixed_k_growth_csv=out_dir / "growth_fixed_k.csv",
        fixed_node_growth_csv=out_dir / "growth_fixed_nodes.csv",
        report_markdown=out_dir / "growth_report.md",
        summary_json=out_dir / "summary.json",
        density_sweep_plot=None if args.skip_plots else out_dir / "density_sweep.png",
        neighborhood_sweep_plot=None if args.skip_plots else out_dir / "neighborhood_sweep.png",
        surface_plot=None if args.skip_plots else out_dir / "runtime_surface_3d.png",
    )

    write_rows_csv(raw_rows, artifacts.raw_csv)
    write_rows_csv(aggregated_rows, artifacts.aggregated_csv)
    write_rows_csv(fixed_k_growth, artifacts.fixed_k_growth_csv)
    write_rows_csv(fixed_node_growth, artifacts.fixed_node_growth_csv)
    write_report_markdown(
        aggregated_rows,
        fixed_k_growth,
        fixed_node_growth,
        artifacts.report_markdown,
    )

    if not args.skip_plots and plt is not None:
        plot_density_sweep(aggregated_rows, artifacts.density_sweep_plot)
        plot_neighborhood_sweep(aggregated_rows, artifacts.neighborhood_sweep_plot)
        plot_surface(aggregated_rows, artifacts.surface_plot)

    summary_payload = {
        "configuration": {
            "node_counts": node_counts,
            "k_neighbors": k_neighbors_values,
            "seeds": seeds,
            "rounds": args.rounds,
            "tolerance": args.tolerance,
            "obstacle_ratio": args.obstacle_ratio,
            "device": str(get_device(args.device)),
            "skip_plots": args.skip_plots,
        },
        "runtime": {
            "total_benchmark_seconds": total_elapsed,
            "trial_count": len(configs),
            "successful_trial_count": sum(1 for row in raw_rows if row["status"] == "ok"),
        },
        "artifacts": {key: str(value) if value is not None else None for key, value in asdict(artifacts).items()},
    }
    artifacts.summary_json.write_text(json.dumps(summary_payload, indent=2), encoding="utf-8")

    print(f"Saved raw runs to {artifacts.raw_csv}")
    print(f"Saved aggregates to {artifacts.aggregated_csv}")
    print(f"Saved growth tables to {artifacts.fixed_k_growth_csv} and {artifacts.fixed_node_growth_csv}")
    print(f"Saved report to {artifacts.report_markdown}")
    if artifacts.surface_plot is not None and plt is not None:
        print(f"Saved plots to {artifacts.density_sweep_plot}, {artifacts.neighborhood_sweep_plot}, and {artifacts.surface_plot}")
    print(f"Benchmark finished in {total_elapsed:.2f}s")


if __name__ == "__main__":
    main()
