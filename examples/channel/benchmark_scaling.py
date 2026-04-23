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

from diffield.sim import SimulationEngine, SpatialScenario
from diffield.dsl import branch
from diffield.dsl import field
from diffield.utils import get_device
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
    report_markdown: Path
    summary_json: Path
    runtime_matrix_plot: Path | None
    scaling_trend_plot: Path | None
    relative_scaling_plot: Path | None
    heatmap_plot: Path | None
    plot_3d_bar_plot: Path | None
    plot_linearity_plot: Path | None


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


def build_runtime_matrix(aggregated_rows: list[dict[str, Any]]) -> tuple[list[int], list[int], np.ndarray, np.ndarray]:
    node_counts = sorted({int(row["num_nodes"]) for row in aggregated_rows})
    k_values = sorted({int(row["k_neighbors"]) for row in aggregated_rows})
    matrix = np.full((len(node_counts), len(k_values)), np.nan, dtype=float)
    std_matrix = np.full((len(node_counts), len(k_values)), np.nan, dtype=float)
    node_to_row = {value: index for index, value in enumerate(node_counts)}
    k_to_row = {value: index for index, value in enumerate(k_values)}
    for row in aggregated_rows:
        n_idx = node_to_row[int(row["num_nodes"])]
        k_idx = k_to_row[int(row["k_neighbors"])]
        matrix[n_idx][k_idx] = float(row["mean_time_seconds"])
        std_matrix[n_idx][k_idx] = float(row.get("std_time_seconds", 0.0))
    return node_counts, k_values, matrix, std_matrix


def format_runtime_matrix_markdown(aggregated_rows: list[dict[str, Any]]) -> str:
    node_counts, k_values, matrix, std_matrix = build_runtime_matrix(aggregated_rows)
    headers = ["Nodes", *[f"k={value}" for value in k_values]]
    separator = ["---" for _ in headers]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(separator) + " |",
    ]
    for row_index, node_count in enumerate(node_counts):
        values = [str(node_count)]
        for col_index in range(len(k_values)):
            m = matrix[row_index, col_index]
            s = std_matrix[row_index, col_index]
            if not math.isfinite(m):
                values.append("nan")
            else:
                values.append(f"{m:.4f} ± {s:.4f}")
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines) + "\n"


def print_runtime_matrix(title: str, aggregated_rows: list[dict[str, Any]]) -> None:
    print(f"\n=== {title} ===")
    print(format_runtime_matrix_markdown(aggregated_rows), end="")


def write_rows_csv(rows: list[dict[str, Any]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        output_path.write_text("", encoding="utf-8")
        return
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_report_markdown(aggregated_rows: list[dict[str, Any]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    text = "# Spatial Channel Scaling Benchmark\n\n"
    text += "## Runtime Matrix\n\n"
    text += format_runtime_matrix_markdown(aggregated_rows)
    text += (
        "\nI plot principali usano asse x logaritmico sui nodi e asse y lineare stretto sui tempi, "
        "cosi si vede meglio che la crescita resta contenuta anche quando il numero di nodi aumenta molto.\n"
    )
    output_path.write_text(text, encoding="utf-8")


def plot_scaling_trend(aggregated_rows: list[dict[str, Any]], output_path: Path) -> None:
    if plt is None:
        return

    grouped: dict[int, list[dict[str, Any]]] = {}
    for row in aggregated_rows:
        grouped.setdefault(int(row["k_neighbors"]), []).append(row)

    fig, ax = plt.subplots(figsize=(10, 6))
    fig.patch.set_facecolor("#fbf7ef")
    ax.set_facecolor("#fffdf8")
    finite_times: list[float] = []
    for k_neighbors, rows in sorted(grouped.items()):
        ordered = sorted(rows, key=lambda row: int(row["num_nodes"]))
        nodes = [int(row["num_nodes"]) for row in ordered]
        times = [float(row["mean_time_seconds"]) for row in ordered]
        stds = [float(row.get("std_time_seconds", 0.0)) for row in ordered]
        finite_times.extend(time for time in times if math.isfinite(time))
        line, = ax.plot(
            nodes,
            times,
            marker="o",
            linewidth=2.0,
            label=f"k={k_neighbors}",
        )
        ax.fill_between(
            nodes,
            [m - s for m, s in zip(times, stds)],
            [m + s for m, s in zip(times, stds)],
            color=line.get_color(),
            alpha=0.15,
        )

    if finite_times:
        ymin = min(finite_times)
        ymax = max(finite_times)
        pad = max((ymax - ymin) * 0.18, ymax * 0.03, 1e-6)
        ax.set_ylim(max(0.0, ymin - pad), ymax + pad)

    node_counts = sorted({int(row["num_nodes"]) for row in aggregated_rows})
    if len(node_counts) >= 2:
        ax.set_xscale("log", base=2)
        ax.set_xticks(node_counts)
        ax.set_xticklabels([str(value) for value in node_counts])

    ax.set_title("Runtime vs Nodes")
    ax.set_xlabel("Number of nodes (log2 scale)")
    ax.set_ylabel("Mean runtime (s)")
    ax.grid(alpha=0.25)
    ax.legend(frameon=True)
    ax.text(
        0.02,
        0.98,
        "Zoomed y-axis to highlight small runtime growth",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=9,
        color="#5c5347",
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close(fig)


def plot_runtime_matrix_table(aggregated_rows: list[dict[str, Any]], output_path: Path) -> None:
    if plt is None or not aggregated_rows:
        return

    node_counts, k_values, matrix, std_matrix = build_runtime_matrix(aggregated_rows)
    headers = ["Nodes", *[f"k={value}" for value in k_values]]
    cells = []
    for row_index, node_count in enumerate(node_counts):
        row_values = [str(node_count)]
        for col_index in range(len(k_values)):
            m = matrix[row_index, col_index]
            s = std_matrix[row_index, col_index]
            if not math.isfinite(m):
                row_values.append("nan")
            else:
                row_values.append(f"{m:.4f}\n± {s:.4f}")
        cells.append(row_values)

    figure_height = max(3.0, 0.6 * (len(cells) + 2))
    figure_width = max(8.0, 1.4 * len(headers) + 1.5)
    fig, ax = plt.subplots(figsize=(figure_width, figure_height))
    fig.patch.set_facecolor("#fbf7ef")
    ax.set_facecolor("#fffdf8")
    ax.axis("off")
    table = ax.table(
        cellText=cells,
        colLabels=headers,
        loc="center",
        cellLoc="center",
        colLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.0, 1.35)
    for (row_idx, col_idx), cell in table.get_celld().items():
        cell.set_edgecolor("#d8ccb8")
        if row_idx == 0:
            cell.set_facecolor("#ead9b6")
            cell.set_text_props(weight="bold", color="#1f1b18")
        else:
            cell.set_facecolor("#fffdf8" if row_idx % 2 else "#f7f0e3")
            cell.set_text_props(color="#2f2a24")

    ax.set_title("Runtime Matrix: nodes x neighborhood", fontsize=14, pad=16)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=170, bbox_inches="tight")
    plt.close(fig)


def plot_relative_scaling(aggregated_rows: list[dict[str, Any]], output_path: Path) -> None:
    if plt is None:
        return

    grouped: dict[int, list[dict[str, Any]]] = {}
    for row in aggregated_rows:
        grouped.setdefault(int(row["k_neighbors"]), []).append(row)

    fig, ax = plt.subplots(figsize=(10, 6))
    fig.patch.set_facecolor("#fbf7ef")
    ax.set_facecolor("#fffdf8")
    all_ratios: list[float] = []
    for k_neighbors, rows in sorted(grouped.items()):
        ordered = sorted(rows, key=lambda row: int(row["num_nodes"]))
        nodes = [int(row["num_nodes"]) for row in ordered]
        times = [float(row["mean_time_seconds"]) for row in ordered]
        stds = [float(row.get("std_time_seconds", 0.0)) for row in ordered]

        baseline = times[0]
        baseline_std = stds[0]

        ratios = [t / baseline if baseline > 0 else 0.0 for t in times]
        # Uncertainty propagation for ratio R = T/T0: sigma_R = R * sqrt((sigma_T/T)^2 + (sigma_T0/T0)^2)
        ratio_stds = [
            r * math.sqrt((s / t) ** 2 + (baseline_std / baseline) ** 2)
            if t > 0 and baseline > 0
            else 0.0
            for r, t, s in zip(ratios, times, stds)
        ]

        all_ratios.extend(ratios)
        line, = ax.plot(
            nodes,
            ratios,
            marker="o",
            linewidth=2.0,
            label=f"k={k_neighbors}",
        )
        ax.fill_between(
            nodes,
            [r - rs for r, rs in zip(ratios, ratio_stds)],
            [r + rs for r, rs in zip(ratios, ratio_stds)],
            color=line.get_color(),
            alpha=0.15,
        )

    node_counts = sorted({int(row["num_nodes"]) for row in aggregated_rows})
    if len(node_counts) >= 2:
        ax.set_xscale("log", base=2)
        ax.set_xticks(node_counts)
        ax.set_xticklabels([str(value) for value in node_counts])

    if all_ratios:
        ymin = min(all_ratios)
        ymax = max(all_ratios)
        pad = max((ymax - ymin) * 0.15, 0.05)
        ax.set_ylim(max(0.9, ymin - pad), ymax + pad)

    ax.set_title("Runtime Relative To Smallest Graph")
    ax.set_xlabel("Number of nodes (log2 scale)")
    ax.set_ylabel("Runtime / runtime at smallest node count")
    ax.grid(alpha=0.25)
    ax.legend(frameon=True)
    ax.axhline(1.0, color="#8f8577", linestyle="--", linewidth=1.0)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close(fig)


def plot_runtime_heatmap(aggregated_rows: list[dict[str, Any]], output_path: Path) -> None:
    if plt is None or not aggregated_rows:
        return

    node_counts, k_values, matrix, _ = build_runtime_matrix(aggregated_rows)

    masked = np.ma.masked_invalid(matrix.T)
    fig, ax = plt.subplots(figsize=(9.5, 6.5))
    fig.patch.set_facecolor("#fbf7ef")
    ax.set_facecolor("#fffdf8")
    image = ax.imshow(masked, cmap="YlGnBu", aspect="auto", interpolation="nearest")
    ax.set_title("Mean Runtime Heatmap")
    ax.set_xlabel("Number of nodes")
    ax.set_ylabel("k nearest neighbors")
    ax.set_xticks(np.arange(len(node_counts)), labels=[str(value) for value in node_counts])
    ax.set_yticks(np.arange(len(k_values)), labels=[str(value) for value in k_values])

    for row_index, k_value in enumerate(k_values):
        for col_index, node_count in enumerate(node_counts):
            value = matrix[col_index, row_index]
            label = "nan" if not math.isfinite(value) else f"{value:.3f}"
            ax.text(col_index, row_index, label, ha="center", va="center", color="#1f1b18", fontsize=9)

    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04, label="Mean runtime (s)")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=170)
    plt.close(fig)


def plot_3d_bar_time(aggregated_rows: list[dict[str, Any]], output_path: Path) -> None:
    if plt is None or not aggregated_rows:
        return

    node_counts, k_values, matrix, _ = build_runtime_matrix(aggregated_rows)

    fig = plt.figure(figsize=(12, 10))
    fig.patch.set_facecolor("#ffffff")
    ax = fig.add_subplot(111, projection='3d')
    ax.set_facecolor("#ffffff")

    _x = np.arange(len(k_values))
    _y = np.arange(len(node_counts))
    _xx, _yy = np.meshgrid(_x, _y)
    x, y = _xx.ravel(), _yy.ravel()

    top = matrix.ravel()
    mask = np.isfinite(top)
    x = x[mask]
    y = y[mask]
    top = top[mask]
    
    max_z = 2.0
    top_capped = np.minimum(top, max_z)
    
    bottom = np.zeros_like(top)
    width = depth = 0.8  # Increased width to reduce gaps between bars

    cmap = plt.get_cmap('inferno')
    norm = plt.Normalize(0, max_z)
    colors = cmap(norm(top_capped))

    # Use x - width/2 and y - depth/2 so the bars are centered on the ticks
    # Adding black edges helps distinguish the bars
    ax.bar3d(x - width/2, y - depth/2, bottom, width, depth, top_capped, shade=True, color=colors, edgecolor='black', linewidth=0.1, alpha=0.95)
    
    ax.set_title("Runtime Scaling (3D)")
    ax.set_xlabel("k nearest neighbors")
    ax.set_ylabel("Number of nodes")
    ax.set_zlabel("Mean runtime (s)")

    ax.set_xticks(_x)
    ax.set_xticklabels([str(k) for k in k_values])
    ax.set_yticks(_y)
    ax.set_yticklabels([str(n) for n in node_counts])

    ax.set_zlim(0, max_z)
    
    # Adjust viewing angle to make smaller bars in front
    ax.view_init(elev=25, azim=-50)

    ax.bar3d(x, y, bottom, width, depth, top, shade=True, color=colors)
    
    ax.set_title("Runtime Scaling (3D)")
    ax.set_xlabel("k nearest neighbors")
    ax.set_ylabel("Number of nodes")
    ax.set_zlabel("Mean runtime (s)")

    ax.set_xticks(_x + 0.4)
    ax.set_xticklabels([str(k) for k in k_values])
    ax.set_yticks(_y + 0.4)
    ax.set_yticklabels([str(n) for n in node_counts])

    ax.set_zlim(0, 2.0)
    
    ax.view_init(elev=30, azim=-60)
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    
    variation_path = output_path.with_name(output_path.stem + "_variation" + output_path.suffix)
    ax.view_init(elev=20, azim=45)
    plt.savefig(variation_path, dpi=160)
    plt.close(fig)


def plot_linearity(aggregated_rows: list[dict[str, Any]], output_path: Path) -> None:
    if plt is None or not aggregated_rows:
        return

    rows_k32 = [row for row in aggregated_rows if int(row["k_neighbors"]) == 32]
    if not rows_k32:
        return
        
    ordered = sorted(rows_k32, key=lambda row: int(row["num_nodes"]))
    nodes = np.array([int(row["num_nodes"]) for row in ordered])
    times = np.array([float(row["mean_time_seconds"]) for row in ordered])
    
    fig, ax = plt.subplots(figsize=(8, 5))
    fig.patch.set_facecolor("#fbf7ef")
    ax.set_facecolor("#fffdf8")
    
    ax.plot(nodes, times, marker="o", linewidth=2.0, label="Actual runtime (k=32)")
    
    if len(nodes) >= 2:
        m, c = np.polyfit(nodes, times, 1)
        ax.plot(nodes, m * nodes + c, linestyle="--", color="gray", label="Linear fit")
        
    ax.set_title("Linearity Check for k=32")
    ax.set_xlabel("Number of nodes")
    ax.set_ylabel("Mean runtime (s)")
    ax.grid(alpha=0.25)
    ax.legend(frameon=True)
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
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
    parser = argparse.ArgumentParser(
        description="Benchmark scaling for the spatial channel example",
        epilog=(
            "Example:\n"
            "  uv run python examples/channel/benchmark_scaling.py "
            "--node-counts 1000,2000,4000,8000 --k-neighbors 4,8,12,16,24 "
            "--repetitions 5 --rounds 300 --out-dir generated/results/channel_scaling_large"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
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
    artifacts = BenchmarkArtifacts(
        raw_csv=out_dir / "raw_runs.csv",
        aggregated_csv=out_dir / "aggregated.csv",
        report_markdown=out_dir / "growth_report.md",
        summary_json=out_dir / "summary.json",
        runtime_matrix_plot=None if args.skip_plots else out_dir / "runtime_matrix.png",
        scaling_trend_plot=None if args.skip_plots else out_dir / "scaling_trend.png",
        relative_scaling_plot=None if args.skip_plots else out_dir / "relative_scaling.png",
        heatmap_plot=None if args.skip_plots else out_dir / "runtime_heatmap.png",
        plot_3d_bar_plot=None if args.skip_plots else out_dir / "runtime_3d_bar.png",
        plot_linearity_plot=None if args.skip_plots else out_dir / "linearity_check.png",
    )

    write_rows_csv(raw_rows, artifacts.raw_csv)
    write_rows_csv(aggregated_rows, artifacts.aggregated_csv)
    write_report_markdown(aggregated_rows, artifacts.report_markdown)

    print_runtime_matrix("Runtime Matrix", aggregated_rows)

    if not args.skip_plots and plt is not None:
        try:
            plot_runtime_matrix_table(aggregated_rows, artifacts.runtime_matrix_plot)
            plot_scaling_trend(aggregated_rows, artifacts.scaling_trend_plot)
            plot_relative_scaling(aggregated_rows, artifacts.relative_scaling_plot)
            plot_runtime_heatmap(aggregated_rows, artifacts.heatmap_plot)
            plot_3d_bar_time(aggregated_rows, artifacts.plot_3d_bar_plot)
            plot_linearity(aggregated_rows, artifacts.plot_linearity_plot)
        except Exception as e:
            print(f"Warning: failed to generate plots: {e}")

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
    print(f"Saved report to {artifacts.report_markdown}")
    if not args.skip_plots and plt is not None:
        print(
            "Saved plots to:\n"
            f"  - {artifacts.runtime_matrix_plot}\n"
            f"  - {artifacts.scaling_trend_plot}\n"
            f"  - {artifacts.relative_scaling_plot}\n"
            f"  - {artifacts.heatmap_plot}\n"
            f"  - {artifacts.plot_3d_bar_plot}\n"
            f"  - {artifacts.plot_linearity_plot}"
        )
    print(f"Benchmark finished in {total_elapsed:.2f}s")


if __name__ == "__main__":
    main()
