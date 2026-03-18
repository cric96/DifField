#!/usr/bin/env python3
"""Large-scale channel with multiple obstacles.

Same AC program as ``channel.py`` but on a configurable grid (default 50×80,
4 000 devices) with a maze-like obstacle layout: two vertical walls with
staggered gaps that force the channel through a winding corridor.

8-connected grid → Chebyshev (chessboard) distance.

Produces three PNG figures:
  1. channel_large_setup.png     — grid layout
  2. channel_large_evolution.png — field snapshots at selected rounds
  3. channel_large_final.png     — converged overlay (channel path on grid)
"""

import sys, time, argparse

sys.path.insert(0, "src")

import torch
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

from aggregate_gnn import AggregateContext, rep, nbr, mux, branch, broadcast
from aggregate_gnn.dsl import field
from aggregate_gnn.utils import make_grid_graph
# ── Constants ──────────────────────────────────────────────────────────────

CHANNEL_THRESHOLD = 0.5    # classify node as "on channel" when value > threshold


# ── CLI ────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(description="Large-scale AC Channel")
    parser.add_argument("--rows",      type=int,   default=50,  help="Grid rows")
    parser.add_argument("--cols",      type=int,   default=80,  help="Grid columns")
    parser.add_argument("--rounds",    type=int,   default=300, help="Number of compute rounds")
    parser.add_argument("--tolerance", type=float, default=0.5, help="Path tolerance")
    return parser.parse_args()

# ── Obstacle builder ────────────────────────────────────────────────────

def build_obstacles(rows, cols):
    """Two vertical walls with staggered gaps → Z-shaped corridor.

    Wall 1: col 25, rows  0–39  (gap at rows 40–49, bottom)
    Wall 2: col 55, rows 10–49  (gap at rows  0–9,  top)

    Forces: left → down → right → up → right.
    """
    obstacle = torch.zeros(rows * cols, dtype=torch.bool)

    # Wall 1
    for r in range(0, 40):
        obstacle[r * cols + 25] = True

    # Wall 2
    for r in range(10, rows):
        obstacle[r * cols + 55] = True

    return obstacle


# ── Channel body ────────────────────────────────────────────────────────

def channel_body(source, dest, tolerance):
    dist_src = rep("dist_src", float("inf"), lambda d:
        mux(source, field.of(0.0), nbr(d + 1.0, aggr="min")))
    dist_dst = rep("dist_dst", float("inf"), lambda d:
        mux(dest, field.of(0.0), nbr(d + 1.0, aggr="min")))
    dist_sd = broadcast(source > 0.5, dist_dst, name="dist_channel")

    on_path = (dist_src + dist_dst <= dist_sd + tolerance)
    finite  = torch.isfinite(dist_src) & torch.isfinite(dist_dst) & torch.isfinite(dist_sd)
    return (on_path & finite).float()


# ── Visualisation helpers ───────────────────────────────────────────────

def to_grid(tensor, rows, cols, obstacle_mask):
    arr = tensor.detach().float().clone()
    arr[obstacle_mask] = float("nan")
    arr[arr == float("inf")] = float("nan")
    return arr.view(rows, cols).numpy()


# ── Snapshot helper ─────────────────────────────────────────────────────

def _take_snapshot(ctx, snapshots, t, N, ch, obstacle):
    ds = ctx._ctx.state._states.get(
        "dist_src", torch.full((N,), float("inf"))).detach().clone()
    dd = ctx._ctx.state._states.get(
        "dist_dst", torch.full((N,), float("inf"))).detach().clone()
    dsd = ctx._ctx.state._states.get(
        "_bc_dist_channel", torch.full((N,), float("inf"))).detach().clone()
    snapshots[t] = {
        "dist_src": ds, "dist_dst": dd,
        "sum": ds + dd, "dist_sd": dsd,
        "channel": ch.detach().clone(),
    }


# ── Main ────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    rows, cols = args.rows, args.cols
    T = args.rounds
    N = rows * cols
    print(f"Building {rows}×{cols} grid ({N} devices) …")
    edge_index, N = make_grid_graph(args.rows, args.cols, connectivity=8)
    
    # Source: middle-left;  Destination: middle-right
    src_pos = (rows // 2, 5)
    dst_pos = (rows // 2, cols - 6)
    source = torch.zeros(N); source[src_pos[0] * cols + src_pos[1]] = 1.0
    dest   = torch.zeros(N); dest[dst_pos[0] * cols + dst_pos[1]]   = 1.0

    obstacle = build_obstacles(rows, cols)
    print(f"Obstacle cells: {obstacle.sum().item()}")

    # ── Run ─────────────────────────────────────────────────────────────
    ctx = AggregateContext(edge_index, N)
    snapshot_at_seconds = [0.1, 0.3, 0.7, 1.5]
    snapshot_steps: list[int] = []
    snapshots: dict[int, dict[str, torch.Tensor]] = {}
    next_snap_idx = 0

    t0 = time.time()
    for t in range(T):
        with ctx.round():
            ch = branch(
                ~obstacle,
                lambda: channel_body(source, dest, args.tolerance),
                lambda: field.of(0.0),
                branch_name="obstacle",
            )

        elapsed_now = time.time() - t0

        # Time-based snapshots
        if next_snap_idx < len(snapshot_at_seconds) and elapsed_now >= snapshot_at_seconds[next_snap_idx]:
            snapshot_steps.append(t)
            next_snap_idx += 1
            _take_snapshot(ctx, snapshots, t, N, ch, obstacle)

        if (t + 1) % 100 == 0:
            print(f"  round {t + 1}/{T}  ({elapsed_now:.1f}s)")

    # Always include last round
    if T - 1 not in snapshot_steps:
        snapshot_steps.append(T - 1)
        _take_snapshot(ctx, snapshots, T - 1, N, ch, obstacle)

    elapsed = time.time() - t0
    final = snapshots[T - 1]
    sd_dist = final["dist_src"][dst_pos[0] * cols + dst_pos[1]].item()
    n_ch = (final["channel"] > 0.5).sum().item()
    print(f"\nDone in {elapsed:.1f}s  |  distance = {sd_dist:.0f}  |  channel nodes = {n_ch}")

    sec_per_round = elapsed / T
    snap_labels = []
    for s in snapshot_steps:
        t_sec = (s + 1) * sec_per_round
        snap_labels.append(f"t={s+1}  ({t_sec:.1f}s)")

    obs_np = obstacle.numpy().reshape(rows, cols)

    # ── Figure 1: Setup ─────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(18, 14))
    grid_rgb = np.full((rows, cols, 3), 0.92)
    grid_rgb[obs_np] = [0.12, 0.12, 0.12]
    grid_rgb[src_pos] = [0.0, 0.8, 0.0]
    grid_rgb[dst_pos] = [0.85, 0.0, 0.0]
    ax.imshow(grid_rgb, interpolation="nearest", aspect="equal")
    ax.plot(src_pos[1], src_pos[0], "g^", ms=14, mec="white", mew=1.5)
    ax.plot(dst_pos[1], dst_pos[0], "rv", ms=14, mec="white", mew=1.5)
    ax.set_title(f"Large-scale channel — {rows}×{cols} grid ({N} devices)", fontsize=14)
    ax.set_xlabel("Column"); ax.set_ylabel("Row")
    patches = [
        mpatches.Patch(color="green", label=f"Source {src_pos}"),
        mpatches.Patch(color="red",   label=f"Dest {dst_pos}"),
        mpatches.Patch(color="black", label="Obstacles"),
    ]
    ax.legend(handles=patches, loc="upper right", fontsize=10, framealpha=0.9)
    plt.tight_layout()
    plt.savefig("examples/channel_large_setup.png", dpi=150)
    print("Saved examples/channel_large_setup.png")

    # ── Figure 2: Evolution (selected fields) ───────────────────────────
    evo_keys   = ["dist_src", "dist_dst", "sum", "dist_sd", "channel"]
    evo_labels = ["dist_src", "dist_dst", "sum (src+dst)", "broadcast", "channel"]
    n_rows_fig = len(evo_keys)
    n_cols_fig = len(snapshot_steps)

    fig, axes = plt.subplots(n_rows_fig, n_cols_fig,
                             figsize=(3.5 * n_cols_fig, 2.5 * n_rows_fig))

    for ri, (fk, fl) in enumerate(zip(evo_keys, evo_labels)):
        for ci, step in enumerate(snapshot_steps):
            ax = axes[ri, ci]
            gd = to_grid(snapshots[step][fk], rows, cols, obstacle)
            if fk == "channel":
                im = ax.imshow(gd, cmap="Oranges", vmin=0, vmax=1,
                               interpolation="nearest", aspect="equal")
            else:
                im = ax.imshow(gd, cmap="viridis",
                               interpolation="nearest", aspect="equal")
            ax.set_title(snap_labels[ci], fontsize=8)
            if ci == 0:
                ax.set_ylabel(fl, fontsize=8)
            ax.set_xticks([]); ax.set_yticks([])
            if ci == n_cols_fig - 1:
                plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    fig.suptitle(f"Large-scale channel — Evolution ({N} devices, {T} rounds, {elapsed:.1f}s)",
                 fontsize=13, y=0.99)
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.savefig("examples/channel_large_evolution.png", dpi=150)
    print("Saved examples/channel_large_evolution.png")

    # ── Figure 3: Final overlay ─────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(18, 14))
    overlay = np.full((rows, cols, 4), [0.92, 0.92, 0.92, 1.0])
    overlay[obs_np] = [0.12, 0.12, 0.12, 1.0]

    channel_field = final["channel"]
    for r in range(rows):
        for c_idx in range(cols):
            nid = r * cols + c_idx
            if not obstacle[nid] and channel_field[nid] > CHANNEL_THRESHOLD:
                overlay[r, c_idx] = [1.0, 0.50, 0.0, 0.95]

    overlay[src_pos] = [0.0, 0.80, 0.0, 1.0]
    overlay[dst_pos] = [0.85, 0.0, 0.0, 1.0]

    ax.imshow(overlay, interpolation="nearest", aspect="equal")
    ax.plot(src_pos[1], src_pos[0], "g^", ms=14, mec="white", mew=1.5)
    ax.plot(dst_pos[1], dst_pos[0], "rv", ms=14, mec="white", mew=1.5)
    ax.set_title(
        f"Channel path — {N} devices, {T} rounds in {elapsed:.1f}s, "
        f"distance = {sd_dist:.0f} hops, {n_ch} channel nodes",
        fontsize=13,
    )
    ax.set_xlabel("Column"); ax.set_ylabel("Row")
    patches = [
        mpatches.Patch(color="green",  label=f"Source {src_pos}"),
        mpatches.Patch(color="red",    label=f"Dest {dst_pos}"),
        mpatches.Patch(color="black",  label="Obstacles"),
        mpatches.Patch(color="orange", label="Channel"),
    ]
    ax.legend(handles=patches, loc="upper right", fontsize=10, framealpha=0.9)
    plt.tight_layout()
    plt.savefig("examples/channel_large_final.png", dpi=150)
    print("Saved examples/channel_large_final.png")

    print("\nDone.")


if __name__ == "__main__":
    main()