"""Visualization helpers for channel-based aggregate simulations."""

from __future__ import annotations

import numpy as np

try:
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
except ImportError:
    plt = None

from common_plot import to_grid, draw_obstacles, draw_markers


def plot_channel_setup(rows, cols, src_pos, dst_pos, obstacle):
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
    ax.set_xlabel("Column")
    ax.set_ylabel("Row")

    patches = [
        mpatches.Patch(color="green", label="Source"),
        mpatches.Patch(color="red", label="Destination"),
        mpatches.Patch(color="black", label="Obstacle wall"),
    ]
    ax.legend(handles=patches, loc="lower right", fontsize=9)
    plt.tight_layout()
    plt.savefig("examples/channel_setup.png", dpi=150)
    print("Saved examples/channel_setup.png")


def plot_channel_evolution(rows, cols, snapshots, src_pos, dst_pos, obstacle):
    snapshot_steps = sorted(snapshots.keys())
    field_keys = ["dist_src", "dist_dst", "sum", "dist_sd", "channel"]
    field_labels = [
        "Distance from Source",
        "Distance from Dest",
        "Sum  (dist_src + dist_dst)",
        "Broadcast (dist S→D)",
        "Channel",
    ]

    n_steps = len(snapshot_steps)
    fig, axes = plt.subplots(len(field_keys), n_steps, figsize=(3.2 * n_steps, 3.1 * len(field_keys)))

    for ri, (fk, fl) in enumerate(zip(field_keys, field_labels)):
        for ci, step in enumerate(snapshot_steps):
            ax = axes[ri, ci]
            gd = to_grid(snapshots[step][fk], rows, cols, obstacle)

            if fk == "channel":
                im = ax.imshow(gd, cmap="Oranges", vmin=0, vmax=1, interpolation="nearest")
            else:
                im = ax.imshow(gd, cmap="viridis", vmin=0, vmax=45, interpolation="nearest")

            draw_obstacles(ax, obstacle, rows, cols)
            draw_markers(ax, src_pos, dst_pos, ms=6)
            ax.set_title(f"t = {step + 1}", fontsize=9)
            if ci == 0:
                ax.set_ylabel(fl, fontsize=9)
            ax.set_xticks([])
            ax.set_yticks([])
            if ci == n_steps - 1:
                plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    fig.suptitle("Channel with Obstacles — Field Evolution", fontsize=13, y=0.99)
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.savefig("examples/channel_evolution.png", dpi=150)
    print("Saved examples/channel_evolution.png")


def plot_channel_final_fields(rows, cols, final, src_pos, dst_pos, obstacle, rounds):
    field_keys = ["dist_src", "dist_dst", "sum", "dist_sd", "channel"]
    field_labels = [
        "Distance from Source",
        "Distance from Dest",
        "Sum  (dist_src + dist_dst)",
        "Broadcast (dist S→D)",
        "Channel",
    ]

    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    axes_flat = axes.flatten()

    for idx, (fk, fl) in enumerate(zip(field_keys, field_labels)):
        ax = axes_flat[idx]
        data = final[fk]
        gd = to_grid(data, rows, cols, obstacle)

        if fk == "channel":
            im = ax.imshow(gd, cmap="Oranges", vmin=0, vmax=1, interpolation="nearest")
        else:
            im = ax.imshow(gd, cmap="viridis", interpolation="nearest")

        draw_obstacles(ax, obstacle, rows, cols)
        draw_markers(ax, src_pos, dst_pos, ms=11)
        ax.set_title(fl, fontsize=13)
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

        if fk in ("dist_src", "dist_dst"):
            for r in range(rows):
                for c in range(cols):
                    nid = r * cols + c
                    v = data[nid].item()
                    if not obstacle[nid] and np.isfinite(v):
                        colour = "white" if v > 18 else "black"
                        ax.text(c, r, f"{v:.0f}", ha="center", va="center", fontsize=5, color=colour)

    axes_flat[-1].set_visible(False)

    fig.suptitle(f"Channel with Obstacles — Converged (t = {rounds})", fontsize=14)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig("examples/channel_final.png", dpi=150)
    print("Saved examples/channel_final.png")


def plot_channel_overlay(rows, cols, final, src_pos, dst_pos, obstacle, sd_dist):
    fig, ax = plt.subplots(figsize=(8, 8))
    overlay = np.full((rows, cols, 4), [0.92, 0.92, 0.92, 1.0])
    ch = final["channel"]
    for r in range(rows):
        for c in range(cols):
            nid = r * cols + c
            if obstacle[nid]:
                overlay[r, c] = [0.15, 0.15, 0.15, 1.0]
            elif ch[nid] > 0.5:
                overlay[r, c] = [1.0, 0.55, 0.0, 0.9]
    overlay[src_pos] = [0.0, 0.80, 0.0, 1.0]
    overlay[dst_pos] = [0.85, 0.0, 0.0, 1.0]

    ax.imshow(overlay, interpolation="nearest")
    for i in range(rows + 1):
        ax.axhline(i - 0.5, color="white", lw=0.3)
    for j in range(cols + 1):
        ax.axvline(j - 0.5, color="white", lw=0.3)
    draw_markers(ax, src_pos, dst_pos, ms=14)
    ax.set_title(f"Shortest-path channel around obstacle  (distance = {sd_dist:.0f} hops)", fontsize=12)
    ax.set_xlabel("Column")
    ax.set_ylabel("Row")
    patches = [
        mpatches.Patch(color="green", label=f"Source {src_pos}"),
        mpatches.Patch(color="red", label=f"Destination {dst_pos}"),
        mpatches.Patch(color="black", label="Obstacle wall"),
        mpatches.Patch(color="orange", label="Channel (shortest path)"),
    ]
    ax.legend(handles=patches, loc="upper right", fontsize=9, framealpha=0.9)
    plt.tight_layout()
    plt.savefig("examples/channel_path.png", dpi=150)
    print("Saved examples/channel_path.png")


def plot_channel_large_setup(rows, cols, num_nodes, src_pos, dst_pos, obstacle):
    obs_np = obstacle.numpy().reshape(rows, cols)
    fig, ax = plt.subplots(figsize=(18, 14))
    grid_rgb = np.full((rows, cols, 3), 0.92)
    grid_rgb[obs_np] = [0.12, 0.12, 0.12]
    grid_rgb[src_pos] = [0.0, 0.8, 0.0]
    grid_rgb[dst_pos] = [0.85, 0.0, 0.0]
    ax.imshow(grid_rgb, interpolation="nearest", aspect="equal")
    ax.plot(src_pos[1], src_pos[0], "g^", ms=14, mec="white", mew=1.5)
    ax.plot(dst_pos[1], dst_pos[0], "rv", ms=14, mec="white", mew=1.5)
    ax.set_title(f"Large-scale channel — {rows}×{cols} grid ({num_nodes} devices)", fontsize=14)
    ax.set_xlabel("Column")
    ax.set_ylabel("Row")
    patches = [
        mpatches.Patch(color="green", label=f"Source {src_pos}"),
        mpatches.Patch(color="red", label=f"Dest {dst_pos}"),
        mpatches.Patch(color="black", label="Obstacles"),
    ]
    ax.legend(handles=patches, loc="upper right", fontsize=10, framealpha=0.9)
    plt.tight_layout()
    plt.savefig("examples/channel_large_setup.png", dpi=150)
    print("Saved examples/channel_large_setup.png")


def plot_channel_large_evolution(rows, cols, snapshots, snapshot_steps, snap_labels, num_nodes, rounds, elapsed, obstacle):
    from mpl_toolkits.axes_grid1 import make_axes_locatable

    evo_keys = ["dist_src", "dist_dst", "sum", "dist_sd", "channel"]
    evo_labels = ["dist_src", "dist_dst", "sum (src+dst)", "broadcast", "channel"]
    n_rows_fig = len(evo_keys)
    n_cols_fig = len(snapshot_steps)

    fig, axes = plt.subplots(n_rows_fig, n_cols_fig, figsize=(3.5 * n_cols_fig, 2.5 * n_rows_fig))

    row_images = []
    for ri, (fk, fl) in enumerate(zip(evo_keys, evo_labels)):
        last_im = None
        for ci, step in enumerate(snapshot_steps):
            ax = axes[ri, ci]
            gd = to_grid(snapshots[step][fk], rows, cols, obstacle)
            if fk == "channel":
                im = ax.imshow(gd, cmap="Oranges", vmin=0, vmax=1, interpolation="nearest", aspect="equal")
            else:
                im = ax.imshow(gd, cmap="viridis", interpolation="nearest", aspect="equal")
            last_im = im
            ax.set_title(snap_labels[ci], fontsize=8)
            if ci == 0:
                ax.set_ylabel(fl, fontsize=8)
            ax.set_xticks([])
            ax.set_yticks([])
        row_images.append(last_im)

    for ri, im in enumerate(row_images):
        last_ax = axes[ri, n_cols_fig - 1]
        divider = make_axes_locatable(last_ax)
        cax = divider.append_axes("right", size="3%", pad=0.08)
        plt.colorbar(im, cax=cax)

    fig.suptitle(f"Large-scale channel — Evolution ({num_nodes} devices, {rounds} rounds, {elapsed:.1f}s)", fontsize=13, y=0.99)
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.savefig("examples/channel_large_evolution.png", dpi=150)
    print("Saved examples/channel_large_evolution.png")


def plot_channel_large_final(rows, cols, final, src_pos, dst_pos, obstacle, channel_threshold, num_nodes, rounds, elapsed, sd_dist, channel_nodes):
    obs_np = obstacle.numpy().reshape(rows, cols)
    fig, ax = plt.subplots(figsize=(18, 14))
    overlay = np.full((rows, cols, 4), [0.92, 0.92, 0.92, 1.0])
    overlay[obs_np] = [0.12, 0.12, 0.12, 1.0]

    channel_field = final["channel"]
    for r in range(rows):
        for c_idx in range(cols):
            nid = r * cols + c_idx
            if not obstacle[nid] and channel_field[nid] > channel_threshold:
                overlay[r, c_idx] = [1.0, 0.50, 0.0, 0.95]

    overlay[src_pos] = [0.0, 0.80, 0.0, 1.0]
    overlay[dst_pos] = [0.85, 0.0, 0.0, 1.0]

    ax.imshow(overlay, interpolation="nearest", aspect="equal")
    ax.plot(src_pos[1], src_pos[0], "g^", ms=14, mec="white", mew=1.5)
    ax.plot(dst_pos[1], dst_pos[0], "rv", ms=14, mec="white", mew=1.5)
    ax.set_title(
        f"Channel path — {num_nodes} devices, {rounds} rounds in {elapsed:.1f}s, "
        f"distance = {sd_dist:.0f} hops, {channel_nodes} channel nodes",
        fontsize=13,
    )
    ax.set_xlabel("Column")
    ax.set_ylabel("Row")
    patches = [
        mpatches.Patch(color="green", label=f"Source {src_pos}"),
        mpatches.Patch(color="red", label=f"Dest {dst_pos}"),
        mpatches.Patch(color="black", label="Obstacles"),
        mpatches.Patch(color="orange", label="Channel"),
    ]
    ax.legend(handles=patches, loc="upper right", fontsize=10, framealpha=0.9)
    plt.tight_layout()
    plt.savefig("examples/channel_large_final.png", dpi=150)
    print("Saved examples/channel_large_final.png")
