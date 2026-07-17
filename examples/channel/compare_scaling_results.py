#!/usr/bin/env python3
"""Compare CPU vs CUDA scaling benchmark results (plots from saved CSVs).

Reads the ``aggregated.csv`` written by ``benchmark_scaling.py`` for two
devices and produces paper-styled comparison figures:

  - ``scaling_overview.png``  -- (a) runtime vs. nodes at a representative k,
    (b) CPU/CUDA speedup vs. nodes for every k (the headline figure)
  - ``speedup_heatmap.png``   -- full nodes x k speedup grid, annotated
  - ``comparison_k<k>.png``   -- per-k runtime curves (single-column)
  - ``speedup_3d_bar.png``    -- 3D bar variant (slides, not paper)
"""

import argparse
import csv
import sys
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "examples"))

from shared.plotting.style import (  # noqa: E402
    BLUE,
    FIG_WIDTH_1COL,
    FIG_WIDTH_2COL,
    INK,
    MUTED,
    ORANGE,
    apply_paper_style,
    panel_label,
    savefig,
)

apply_paper_style()

# CPU/CUDA are devices, not policy roles: fixed local pairing with distinct
# markers/linestyles as the second (colour-independent) channel.
DEVICE_STYLE = {
    "CPU": (ORANGE, "s", "--"),
    "CUDA": (BLUE, "o", "-"),
}


def load_aggregated(csv_path: Path) -> dict[tuple[int, int], dict[str, float]]:
    data = {}
    with open(csv_path, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            key = (int(row["num_nodes"]), int(row["k_neighbors"]))
            data[key] = {
                "mean": float(row["mean_time_seconds"]),
                "std": float(row.get("std_time_seconds", 0.0)),
                "trials": float(row.get("trials", 1.0)),
            }
    return data


def series_for(
    data: dict[tuple[int, int], dict[str, float]], k: int, nodes: list[int]
) -> tuple[np.ndarray, np.ndarray]:
    means = np.array([data[(n, k)]["mean"] for n in nodes])
    stds = np.array([data[(n, k)]["std"] for n in nodes])
    trials = np.array([data[(n, k)]["trials"] for n in nodes])
    cis = 1.96 * stds / np.sqrt(trials)
    return means, cis


def draw_runtime_axis(
    ax, cpu_data, cuda_data, k: int, nodes: list[int], *, legend: bool = True
) -> None:
    for label, data in (("CPU", cpu_data), ("CUDA", cuda_data)):
        color, marker, linestyle = DEVICE_STYLE[label]
        means, cis = series_for(data, k, nodes)
        ax.plot(
            nodes, means, color=color, marker=marker, linestyle=linestyle,
            markersize=4.5, linewidth=1.8, label=label,
        )
        ax.fill_between(nodes, means - cis, means + cis, color=color, alpha=0.18, linewidth=0)
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xticks(nodes)
    ax.set_xticklabels([f"{n:,}" for n in nodes], rotation=45, ha="right", fontsize=8.5)
    ax.set_xlabel("nodes")
    ax.set_ylabel("runtime (s)")
    if legend:
        ax.legend()
    ax.grid(True, which="both", alpha=0.4)


def plot_overview(cpu_data, cuda_data, keys, out_path: Path) -> None:
    node_counts = sorted({key[0] for key in keys})
    k_values = sorted({key[1] for key in keys})
    k_repr = k_values[len(k_values) // 2]

    fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(FIG_WIDTH_2COL, 2.6))

    draw_runtime_axis(ax_a, cpu_data, cuda_data, k_repr, node_counts)
    ax_a.text(
        0.97, 0.04, f"k = {k_repr}", transform=ax_a.transAxes,
        ha="right", va="bottom", fontsize=9, color=MUTED,
    )
    panel_label(ax_a, "a")

    cmap = plt.get_cmap("viridis")
    shades = [cmap(v) for v in np.linspace(0.15, 0.8, len(k_values))]
    markers = ("o", "s", "D", "^", "v", "P")
    for idx, k in enumerate(k_values):
        nodes = sorted([n for (n, k_val) in keys if k_val == k])
        cpu_means, _ = series_for(cpu_data, k, nodes)
        cuda_means, _ = series_for(cuda_data, k, nodes)
        speedup = cpu_means / cuda_means
        ax_b.plot(
            nodes, speedup, color=shades[idx], marker=markers[idx % len(markers)],
            markersize=4, linewidth=1.6, label=f"k = {k}",
        )
    ax_b.axhline(1.0, color=INK, linewidth=0.8, linestyle=":", alpha=0.7)
    ax_b.set_xscale("log", base=2)
    ax_b.set_xticks(sorted({key[0] for key in keys}))
    ax_b.set_xticklabels(
        [f"{n:,}" for n in sorted({key[0] for key in keys})],
        rotation=45, ha="right", fontsize=8.5,
    )
    ax_b.set_xlabel("nodes")
    ax_b.set_ylabel("CPU / CUDA speedup")
    ax_b.legend(fontsize=8.5)
    ax_b.grid(True, which="both", alpha=0.4)
    panel_label(ax_b, "b")

    savefig(fig, out_path)


def plot_speedup_heatmap(speedup_matrix, node_counts, k_values, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(FIG_WIDTH_1COL, 2.8))
    finite = speedup_matrix[np.isfinite(speedup_matrix)]
    vmax = max(2.0, float(finite.max())) if finite.size else 2.0
    norm = TwoSlopeNorm(vmin=0.0, vcenter=1.0, vmax=vmax)
    im = ax.imshow(
        speedup_matrix.T, cmap="RdBu_r", norm=norm, origin="lower", aspect="auto"
    )

    ax.set_xticks(np.arange(len(node_counts)))
    ax.set_xticklabels([f"{n:,}" for n in node_counts], rotation=45, ha="right", fontsize=8)
    ax.set_yticks(np.arange(len(k_values)))
    ax.set_yticklabels([str(k) for k in k_values], fontsize=8.5)
    ax.set_xlabel("nodes")
    ax.set_ylabel("k neighbours")
    ax.grid(False)

    for i in range(len(node_counts)):
        for j in range(len(k_values)):
            val = speedup_matrix[i, j]
            if np.isfinite(val):
                frac = norm(val)
                ax.text(
                    i, j, f"{val:.1f}x", ha="center", va="center", fontsize=7.5,
                    color="white" if abs(frac - 0.5) > 0.28 else INK,
                )

    cbar = fig.colorbar(im, ax=ax, shrink=0.9)
    cbar.set_label("CPU / CUDA speedup", fontsize=9)
    cbar.ax.tick_params(labelsize=8)
    savefig(fig, out_path)


def plot_speedup_3d(speedup_matrix, node_counts, k_values, out_path: Path) -> None:
    fig = plt.figure(figsize=(6.5, 5.0))
    ax = fig.add_subplot(111, projection="3d")

    _x = np.arange(len(k_values))
    _y = np.arange(len(node_counts))
    _xx, _yy = np.meshgrid(_x, _y)
    x, y = _xx.ravel(), _yy.ravel()

    top = speedup_matrix.ravel()
    mask = np.isfinite(top)
    x, y, top = x[mask], y[mask], top[mask]
    width = depth = 0.8

    cmap = plt.get_cmap("plasma")
    norm = plt.Normalize(vmin=1.0, vmax=max(2.0, top.max()))
    ax.bar3d(
        x - width / 2, y - depth / 2, np.zeros_like(top), width, depth, top,
        shade=True, color=cmap(norm(top)), edgecolor=INK, linewidth=0.1,
    )
    ax.set_xlabel("k neighbours", labelpad=10)
    ax.set_ylabel("nodes", labelpad=10)
    ax.set_zlabel("speedup (x)")
    ax.set_xticks(_x)
    ax.set_xticklabels([str(k) for k in k_values], fontsize=8.5)
    ax.set_yticks(_y)
    ax.set_yticklabels([f"{n:,}" for n in node_counts], fontsize=8.5)
    ax.set_zlim(0, max(2.0, top.max() * 1.1))
    ax.view_init(elev=30, azim=-60)
    fig.savefig(out_path, dpi=200, bbox_inches="tight", pad_inches=0.2)
    plt.close(fig)
    print(f"Saved {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Compare CPU vs CUDA scaling benchmark results")
    parser.add_argument("--cpu-csv", type=str, required=True, help="Path to CPU aggregated.csv")
    parser.add_argument("--cuda-csv", type=str, required=True, help="Path to CUDA aggregated.csv")
    parser.add_argument(
        "--out-dir", type=str, default="generated/results/comparison",
        help="Output directory",
    )
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    cpu_data = load_aggregated(Path(args.cpu_csv))
    cuda_data = load_aggregated(Path(args.cuda_csv))

    keys = sorted(set(cpu_data.keys()) & set(cuda_data.keys()))
    if not keys:
        print("No overlapping configuration found between CPU and CUDA results.")
        return

    node_counts = sorted({key[0] for key in keys})
    k_values = sorted({key[1] for key in keys})

    speedup_matrix = np.full((len(node_counts), len(k_values)), np.nan)
    node_to_idx = {n: i for i, n in enumerate(node_counts)}
    k_to_idx = {k: i for i, k in enumerate(k_values)}
    for (n, k) in keys:
        cpu_time = cpu_data[(n, k)]["mean"]
        cuda_time = cuda_data[(n, k)]["mean"]
        speedup_matrix[node_to_idx[n], k_to_idx[k]] = (
            cpu_time / cuda_time if cuda_time > 0 else np.nan
        )

    plot_overview(cpu_data, cuda_data, keys, out_dir / "scaling_overview.png")
    plot_speedup_heatmap(speedup_matrix, node_counts, k_values, out_dir / "speedup_heatmap.png")
    plot_speedup_3d(speedup_matrix, node_counts, k_values, out_dir / "speedup_3d_bar.png")

    for k in k_values:
        nodes = sorted([n for (n, k_val) in keys if k_val == k])
        fig, ax = plt.subplots(figsize=(FIG_WIDTH_1COL, 2.6))
        draw_runtime_axis(ax, cpu_data, cuda_data, k, nodes)
        savefig(fig, out_dir / f"comparison_k{k}.png")

    print(f"Generated comparison plots in {out_dir}")


if __name__ == "__main__":
    main()
