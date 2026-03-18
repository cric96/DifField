#!/usr/bin/env python3
"""Channel with obstacles — classic AC self-organising construct.

Demonstrates the aggregate computing *channel*: a Boolean field identifying
nodes on the shortest path between source and destination, routing around
a wall of obstacles using ``branch`` for communication isolation.
"""

import sys
import argparse
import torch
import numpy as np
try:
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
except ImportError:
    plt = None

sys.path.insert(0, "src")

from aggregate_gnn import AggregateContext, rep, nbr, mux, branch, broadcast
from aggregate_gnn.dsl import field
from aggregate_gnn.utils import make_grid_graph
from common_plot import to_grid, draw_obstacles, draw_markers


def parse_args():
    parser = argparse.ArgumentParser(description="AC Channel with obstacles")
    parser.add_argument("--rows", type=int, default=15, help="Grid rows")
    parser.add_argument("--cols", type=int, default=15, help="Grid columns")
    parser.add_argument("--rounds", type=int, default=100, help="Number of compute rounds")
    parser.add_argument("--noise-scale", type=float, default=0.01, help="Hop cost noise")
    parser.add_argument("--tolerance", type=float, default=0.5, help="Path tolerance")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    return parser.parse_args()


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
    edge_index, N = make_grid_graph(args.rows, args.cols, connectivity=8)

    # Source (middle-left) and destination (middle-right)
    src_pos = (args.rows // 2, 2)
    dst_pos = (args.rows // 2, args.cols - 3)
    
    source = torch.zeros(N)
    source[src_pos[0] * args.cols + src_pos[1]] = 1.0
    
    dest = torch.zeros(N)
    dest[dst_pos[0] * args.cols + dst_pos[1]] = 1.0

    # Obstacle: vertical wall completely blocking direct horizontal path
    obstacle = torch.zeros(N, dtype=torch.bool)
    wall_col = args.cols // 2
    for r in range(0, args.rows - 2):
        obstacle[r * args.cols + wall_col] = True

    torch.manual_seed(args.seed)
    noise = torch.rand(N) * args.noise_scale

    return edge_index, N, source, dest, obstacle, noise, src_pos, dst_pos, wall_col


def plot_results(args, snapshots, final, src_pos, dst_pos, obstacle, wall_col):
    """Generate and save visualisations using common_plot utilities."""
    if plt is None:
        print("Matplotlib not installed. Skipping plots.")
        return

    rows, cols = args.rows, args.cols
    snapshot_steps = sorted(snapshots.keys())
    sd_dist = final["dist_src"][dst_pos[0] * cols + dst_pos[1]].item()

    # ── Figure 1: Setup ─────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(7, 7))
    grid_rgb = np.full((rows, cols, 3), 0.92)
    for r in range(rows):
        for c in range(cols):
            if obstacle[r * cols + c]:
                grid_rgb[r, c] = [0.15, 0.15, 0.15]
    grid_rgb[src_pos] = [0.0, 0.75, 0.0]
    grid_rgb[dst_pos] = [0.85, 0.0, 0.0]

    ax.imshow(grid_rgb, interpolation="nearest")
    for i in range(rows + 1):
        ax.axhline(i - 0.5, color="white", lw=0.4)
    for j in range(cols + 1):
        ax.axvline(j - 0.5, color="white", lw=0.4)
    ax.set_title("Grid Layout — Source ▲  Destination ▼  Wall ■", fontsize=12)
    ax.set_xlabel("Column"); ax.set_ylabel("Row")

    patches = [
        mpatches.Patch(color="green",  label="Source"),
        mpatches.Patch(color="red",    label="Destination"),
        mpatches.Patch(color="black",  label="Obstacle wall"),
    ]
    ax.legend(handles=patches, loc="lower right", fontsize=9)
    plt.tight_layout()
    plt.savefig("examples/channel_setup.png", dpi=150)
    print("Saved examples/channel_setup.png")

    # ── Figure 2: Evolution ─────────────────────────────────────────────
    field_keys   = ["dist_src", "dist_dst", "sum", "dist_sd", "channel"]
    field_labels = [
        "Distance from Source",
        "Distance from Dest",
        "Sum  (dist_src + dist_dst)",
        "Broadcast (dist S→D)",
        "Channel",
    ]

    n_steps = len(snapshot_steps)
    fig, axes = plt.subplots(len(field_keys), n_steps,
                             figsize=(3.2 * n_steps, 3.1 * len(field_keys)))

    for ri, (fk, fl) in enumerate(zip(field_keys, field_labels)):
        for ci, step in enumerate(snapshot_steps):
            ax = axes[ri, ci]
            gd = to_grid(snapshots[step][fk], rows, cols, obstacle)

            if fk == "channel":
                im = ax.imshow(gd, cmap="Oranges", vmin=0, vmax=1,
                               interpolation="nearest")
            else:
                im = ax.imshow(gd, cmap="viridis", vmin=0, vmax=45,
                               interpolation="nearest")

            draw_obstacles(ax, obstacle, rows, cols)
            draw_markers(ax, src_pos, dst_pos, ms=6)
            ax.set_title(f"t = {step + 1}", fontsize=9)
            if ci == 0:
                ax.set_ylabel(fl, fontsize=9)
            ax.set_xticks([]); ax.set_yticks([])
            if ci == n_steps - 1:
                plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    fig.suptitle("Channel with Obstacles — Field Evolution", fontsize=13, y=0.99)
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.savefig("examples/channel_evolution.png", dpi=150)
    print("Saved examples/channel_evolution.png")

    # ── Figure 3: Final fields (3×2 grid, 5 panels) ────────────────────
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    axes_flat = axes.flatten()

    for idx, (fk, fl) in enumerate(zip(field_keys, field_labels)):
        ax = axes_flat[idx]
        data = final[fk]
        gd = to_grid(data, rows, cols, obstacle)

        if fk == "channel":
            im = ax.imshow(gd, cmap="Oranges", vmin=0, vmax=1,
                           interpolation="nearest")
        else:
            im = ax.imshow(gd, cmap="viridis", interpolation="nearest")

        draw_obstacles(ax, obstacle, rows, cols)
        draw_markers(ax, src_pos, dst_pos, ms=11)
        ax.set_title(fl, fontsize=13)
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

        # Annotate distance values on free cells
        if fk in ("dist_src", "dist_dst"):
            for r in range(rows):
                for c in range(cols):
                    nid = r * cols + c
                    v = data[nid].item()
                    if not obstacle[nid] and np.isfinite(v):
                        colour = "white" if v > 18 else "black"
                        ax.text(c, r, f"{v:.0f}", ha="center", va="center",
                                fontsize=5, color=colour)

    # Hide the unused 6th subplot
    axes_flat[-1].set_visible(False)

    fig.suptitle(f"Channel with Obstacles — Converged (t = {args.rounds})", fontsize=14)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig("examples/channel_final.png", dpi=150)
    print("Saved examples/channel_final.png")

    # ── Figure 4: Clean channel overlay ─────────────────────────────────
    fig, ax = plt.subplots(figsize=(8, 8))
    overlay = np.full((rows, cols, 4), [0.92, 0.92, 0.92, 1.0])
    ch = final["channel"]
    for r in range(rows):
        for c in range(cols):
            nid = r * cols + c
            if obstacle[nid]:
                overlay[r, c] = [0.15, 0.15, 0.15, 1.0]
            elif ch[nid] > 0.5:
                overlay[r, c] = [1.0, 0.55, 0.0, 0.9]   # orange channel
    overlay[src_pos] = [0.0, 0.80, 0.0, 1.0]
    overlay[dst_pos] = [0.85, 0.0, 0.0, 1.0]

    ax.imshow(overlay, interpolation="nearest")
    for i in range(rows + 1):
        ax.axhline(i - 0.5, color="white", lw=0.3)
    for j in range(cols + 1):
        ax.axvline(j - 0.5, color="white", lw=0.3)
    draw_markers(ax, src_pos, dst_pos, ms=14)
    ax.set_title(
        f"Shortest-path channel around obstacle  "
        f"(distance = {sd_dist:.0f} hops)",
        fontsize=12,
    )
    ax.set_xlabel("Column"); ax.set_ylabel("Row")
    patches = [
        mpatches.Patch(color="green",  label=f"Source {src_pos}"),
        mpatches.Patch(color="red",    label=f"Destination {dst_pos}"),
        mpatches.Patch(color="black",  label="Obstacle wall"),
        mpatches.Patch(color="orange", label="Channel (shortest path)"),
    ]
    ax.legend(handles=patches, loc="upper right", fontsize=9,
              framealpha=0.9)
    plt.tight_layout()
    plt.savefig("examples/channel_path.png", dpi=150)
    print("Saved examples/channel_path.png")


def main():
    args = parse_args()
    
    edge_index, N, source, dest, obstacle, noise, src_pos, dst_pos, wall_col = build_scenario(args)

    ctx = AggregateContext(edge_index, N)
    
    snapshot_steps = [5, 15, 30, 60, args.rounds - 1]
    snapshots = {}

    for t in range(args.rounds):
        with ctx.round():
            channel_field = branch(
                ~obstacle,
                lambda: channel_body(source, dest, noise, args.tolerance),
                lambda: field.of(0.0),
                branch_name="obstacle",
            )

        if t in snapshot_steps:
            ds = ctx._ctx.state._states.get("dist_src", torch.full((N,), float("inf"))).detach().clone()
            dd = ctx._ctx.state._states.get("dist_dst", torch.full((N,), float("inf"))).detach().clone()
            dsd = ctx._ctx.state._states.get("_bc_dist_channel", torch.full((N,), float("inf"))).detach().clone()
            snapshots[t] = {
                "dist_src": ds,
                "dist_dst": dd,
                "sum": ds + dd,
                "dist_sd": dsd,
                "channel": channel_field.detach().clone(),
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
