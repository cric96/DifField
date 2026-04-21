#!/usr/bin/env python3
import argparse
import csv
from pathlib import Path
from typing import Any

import numpy as np
import matplotlib.pyplot as plt

def load_aggregated(csv_path: Path) -> dict[tuple[int, int], dict[str, float]]:
    data = {}
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            key = (int(row["num_nodes"]), int(row["k_neighbors"]))
            data[key] = {
                "mean": float(row["mean_time_seconds"]),
                "std": float(row.get("std_time_seconds", 0.0)),
                "trials": float(row.get("trials", 1.0))
            }
    return data

def main():
    parser = argparse.ArgumentParser(description="Compare CPU vs CUDA scaling benchmark results")
    parser.add_argument("--cpu-csv", type=str, required=True, help="Path to CPU aggregated.csv")
    parser.add_argument("--cuda-csv", type=str, required=True, help="Path to CUDA aggregated.csv")
    parser.add_argument("--out-dir", type=str, default="generated/results/comparison", help="Output directory")
    args = parser.parse_args()

    cpu_path = Path(args.cpu_csv)
    cuda_path = Path(args.cuda_csv)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    cpu_data = load_aggregated(cpu_path)
    cuda_data = load_aggregated(cuda_path)

    # Intersection of keys
    keys = sorted(set(cpu_data.keys()) & set(cuda_data.keys()))
    if not keys:
        print("No overlapping configuration found between CPU and CUDA results.")
        return

    node_counts = sorted({k[0] for k in keys})
    k_values = sorted({k[1] for k in keys})

    speedup_matrix = np.full((len(node_counts), len(k_values)), np.nan)
    node_to_idx = {n: i for i, n in enumerate(node_counts)}
    k_to_idx = {k: i for i, k in enumerate(k_values)}

    for (n, k) in keys:
        cpu_time = cpu_data[(n, k)]["mean"]
        cuda_time = cuda_data[(n, k)]["mean"]
        speedup = cpu_time / cuda_time if cuda_time > 0 else np.nan
        speedup_matrix[node_to_idx[n], k_to_idx[k]] = speedup

    # 1. Speedup Heatmap
    fig, ax = plt.subplots(figsize=(10, 8))
    fig.patch.set_facecolor("#ffffff")
    im = ax.imshow(speedup_matrix.T, cmap="RdYlGn", origin="lower", aspect="auto")
    
    ax.set_xticks(np.arange(len(node_counts)))
    ax.set_xticklabels([str(n) for n in node_counts])
    ax.set_yticks(np.arange(len(k_values)))
    ax.set_yticklabels([str(k) for k in k_values])
    
    ax.set_xlabel("Number of nodes")
    ax.set_ylabel("k nearest neighbors")
    # ax.set_title("Speedup (CPU Time / CUDA Time)")
    
    # Add text annotations
    for i in range(len(node_counts)):
        for j in range(len(k_values)):
            val = speedup_matrix[i, j]
            if not np.isnan(val):
                ax.text(i, j, f"{val:.1f}x", ha="center", va="center", color="black" if 0.5 < val < 2.0 else "white", fontsize=14)

    fig.colorbar(im, ax=ax, label="Speedup factor")
    plt.tight_layout()
    plt.savefig(out_dir / "speedup_heatmap.png", dpi=160)
    plt.close(fig)

    # 1b. 3D Speedup Bar Plot
    fig = plt.figure(figsize=(10, 8))
    fig.patch.set_facecolor("#ffffff")
    ax = fig.add_subplot(111, projection='3d')
    ax.set_facecolor("#ffffff")

    _x = np.arange(len(k_values))
    _y = np.arange(len(node_counts))
    _xx, _yy = np.meshgrid(_x, _y)
    x, y = _xx.ravel(), _yy.ravel()

    top = speedup_matrix.ravel()
    mask = np.isfinite(top)
    x = x[mask]
    y = y[mask]
    top = top[mask]
    bottom = np.zeros_like(top)
    width = depth = 0.8

    cmap = plt.get_cmap('plasma')
    norm = plt.Normalize(vmin=1.0, vmax=max(2.0, top.max()))
    colors = cmap(norm(top))

    ax.bar3d(x - width/2, y - depth/2, bottom, width, depth, top, shade=True, color=colors, edgecolor='black', linewidth=0.1)

    # ax.set_title("CPU / CUDA Speedup", fontsize=20, pad=20)
    ax.set_xlabel("k neighbors", fontsize=20, labelpad=15)
    ax.set_ylabel("Nodes", fontsize=20, labelpad=15)
    ax.set_zlabel("Speedup (x)", fontsize=20, labelpad=3)

    ax.set_xticks(_x)
    ax.set_xticklabels([str(k) for k in k_values], fontsize=16)
    ax.set_yticks(_y)
    ax.set_yticklabels([str(n) for n in node_counts], fontsize=16)
    
    # Force Z axis to show the full range clearly
    ax.set_zlim(0, max(2.0, top.max() * 1.1))
    ax.tick_params(axis='z', labelsize=16)

    # Add a colorbar to make the speedup values explicit
    #mappable = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    #mappable.set_array(top)
    # cbar = fig.colorbar(mappable, ax=ax, shrink=0.5, aspect=10, pad=0.05)
    # cbar.ax.tick_params(labelsize=14)
    # cbar.set_label("Speedup Factor", fontsize=16)

    ax.view_init(elev=30, azim=-60)
    
    # Use bbox_inches='tight' to ensure labels are included in the saved image.
    # Aggressive subplots_adjust often clips 3D labels.
    plt.savefig(out_dir / "speedup_3d_bar.png", dpi=160, bbox_inches='tight', pad_inches=0.2)
    plt.close(fig)

    # 2. Side-by-side comparison for specific k
    for k in k_values:
        fig, ax = plt.subplots(figsize=(10, 8))
        fig.patch.set_facecolor("#ffffff")
        
        relevant_nodes = sorted([n_val for (n_val, k_val) in keys if k_val == k])
        
        def get_series(data_map, k_val, nodes):
            means = np.array([data_map[(n, k_val)]["mean"] for n in nodes])
            stds = np.array([data_map[(n, k_val)]["std"] for n in nodes])
            trials = np.array([data_map[(n, k_val)]["trials"] for n in nodes])
            # 95% Confidence Interval: 1.96 * std / sqrt(n)
            cis = 1.96 * stds / np.sqrt(trials)
            return means, cis

        cpu_means, cpu_cis = get_series(cpu_data, k, relevant_nodes)
        cuda_means, cuda_cis = get_series(cuda_data, k, relevant_nodes)
        
        ax.plot(relevant_nodes, cpu_means, marker='o', label="CPU Runtime", color="#d62728", linewidth=4, markersize=10)
        ax.fill_between(relevant_nodes, cpu_means - cpu_cis, cpu_means + cpu_cis, color="#d62728", alpha=0.15)
        
        ax.plot(relevant_nodes, cuda_means, marker='s', label="CUDA Runtime", color="#1f77b4", linewidth=4, markersize=10)
        ax.fill_between(relevant_nodes, cuda_means - cuda_cis, cuda_means + cuda_cis, color="#1f77b4", alpha=0.15)
        
        ax.set_xscale("log", base=2)
        ax.set_xticks(relevant_nodes)
        ax.set_xticklabels([str(n) for n in relevant_nodes], fontsize=16)
        ax.tick_params(axis='y', labelsize=16)
        
        ax.set_xlabel("Number of nodes", fontsize=24, labelpad=15)
        ax.set_ylabel("Mean runtime (s)", fontsize=24, labelpad=15)
        # ax.set_title(f"CPU vs CUDA Scaling Comparison (k={k})", fontsize=20, pad=20)
        ax.legend(fontsize=18)
        ax.grid(True, which="both", ls="-", alpha=0.2)
           
        filename = f"comparison_k{k}.png"
        plt.tight_layout()
        plt.savefig(out_dir / filename, dpi=160)
        plt.close(fig)

    print(f"Generated comparison plots in {out_dir}")

if __name__ == "__main__":
    main()
