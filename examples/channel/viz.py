"""Visualization helpers for channel-based aggregate simulations."""

from __future__ import annotations

import numpy as np

try:
    import matplotlib.patches as mpatches
    import matplotlib.pyplot as plt
except ImportError:
    mpatches = None
    plt = None

from shared.plotting import draw_markers, draw_obstacles, to_grid


def plot_channel_setup(rows, cols, src_pos, dst_pos, obstacle):
    if plt is None or mpatches is None:
        print("matplotlib not available; skipping channel setup plot")
        return
    fig, ax = plt.subplots(figsize=(7, 7))
    grid_rgb = np.full((rows, cols, 3), 0.92)
    for row in range(rows):
        for col in range(cols):
            if obstacle[row * cols + col]:
                grid_rgb[row, col] = [0.15, 0.15, 0.15]
    grid_rgb[src_pos] = [0.0, 0.75, 0.0]
    grid_rgb[dst_pos] = [0.85, 0.0, 0.0]
    ax.imshow(grid_rgb, interpolation="nearest")
    for row in range(rows + 1):
        ax.axhline(row - 0.5, color="white", lw=0.4)
    for col in range(cols + 1):
        ax.axvline(col - 0.5, color="white", lw=0.4)
    ax.set_title("Grid Layout - Source ▲  Destination ▼  Wall ■", fontsize=12)
    ax.set_xlabel("Column")
    ax.set_ylabel("Row")
    ax.legend(
        handles=[
            mpatches.Patch(color="green", label="Source"),
            mpatches.Patch(color="red", label="Destination"),
            mpatches.Patch(color="black", label="Obstacle wall"),
        ],
        loc="lower right",
        fontsize=9,
    )
    plt.tight_layout()
    plt.savefig("examples/channel_setup.png", dpi=150)
    print("Saved examples/channel_setup.png")


def plot_channel_evolution(rows, cols, snapshots, src_pos, dst_pos, obstacle):
    if plt is None:
        print("matplotlib not available; skipping channel evolution plot")
        return
    snapshot_steps = sorted(snapshots.keys())
    field_keys = ["dist_src", "dist_dst", "sum", "dist_sd", "channel"]
    field_labels = [
        "Distance from Source",
        "Distance from Dest",
        "Sum  (dist_src + dist_dst)",
        "Broadcast (dist S->D)",
        "Channel",
    ]
    num_steps = len(snapshot_steps)
    fig, axes = plt.subplots(len(field_keys), num_steps, figsize=(3.2 * num_steps, 3.1 * len(field_keys)))
    for row_idx, (field_key, field_label) in enumerate(zip(field_keys, field_labels)):
        for col_idx, step in enumerate(snapshot_steps):
            ax = axes[row_idx, col_idx]
            grid = to_grid(snapshots[step][field_key], rows, cols, obstacle)
            if field_key == "channel":
                image = ax.imshow(grid, cmap="Oranges", vmin=0, vmax=1, interpolation="nearest")
            else:
                image = ax.imshow(grid, cmap="viridis", vmin=0, vmax=45, interpolation="nearest")
            draw_obstacles(ax, obstacle, rows, cols)
            draw_markers(ax, src_pos, dst_pos, ms=6)
            ax.set_title(f"t = {step + 1}", fontsize=9)
            if col_idx == 0:
                ax.set_ylabel(field_label, fontsize=9)
            ax.set_xticks([])
            ax.set_yticks([])
            if col_idx == num_steps - 1:
                plt.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    fig.suptitle("Channel with Obstacles - Field Evolution", fontsize=13, y=0.99)
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.savefig("examples/channel_evolution.png", dpi=150)
    print("Saved examples/channel_evolution.png")


def plot_channel_final_fields(rows, cols, final, src_pos, dst_pos, obstacle, rounds):
    if plt is None:
        print("matplotlib not available; skipping channel final fields plot")
        return
    field_keys = ["dist_src", "dist_dst", "sum", "dist_sd", "channel"]
    field_labels = [
        "Distance from Source",
        "Distance from Dest",
        "Sum  (dist_src + dist_dst)",
        "Broadcast (dist S->D)",
        "Channel",
    ]
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    axes_flat = axes.flatten()
    for idx, (field_key, field_label) in enumerate(zip(field_keys, field_labels)):
        ax = axes_flat[idx]
        data = final[field_key]
        grid = to_grid(data, rows, cols, obstacle)
        if field_key == "channel":
            image = ax.imshow(grid, cmap="Oranges", vmin=0, vmax=1, interpolation="nearest")
        else:
            image = ax.imshow(grid, cmap="viridis", interpolation="nearest")
        draw_obstacles(ax, obstacle, rows, cols)
        draw_markers(ax, src_pos, dst_pos, ms=11)
        ax.set_title(field_label, fontsize=13)
        plt.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
        if field_key in ("dist_src", "dist_dst"):
            for row in range(rows):
                for col in range(cols):
                    node_id = row * cols + col
                    value = data[node_id].item()
                    if not obstacle[node_id] and np.isfinite(value):
                        colour = "white" if value > 18 else "black"
                        ax.text(col, row, f"{value:.0f}", ha="center", va="center", fontsize=5, color=colour)
    axes_flat[-1].set_visible(False)
    fig.suptitle(f"Channel with Obstacles - Converged (t = {rounds})", fontsize=14)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig("examples/channel_final.png", dpi=150)
    print("Saved examples/channel_final.png")


def plot_channel_overlay(rows, cols, final, src_pos, dst_pos, obstacle, sd_dist):
    if plt is None or mpatches is None:
        print("matplotlib not available; skipping channel overlay plot")
        return
    fig, ax = plt.subplots(figsize=(8, 8))
    overlay = np.full((rows, cols, 4), [0.92, 0.92, 0.92, 1.0])
    channel = final["channel"]
    for row in range(rows):
        for col in range(cols):
            node_id = row * cols + col
            if obstacle[node_id]:
                overlay[row, col] = [0.15, 0.15, 0.15, 1.0]
            elif channel[node_id] > 0.5:
                overlay[row, col] = [1.0, 0.55, 0.0, 0.9]
    overlay[src_pos] = [0.0, 0.80, 0.0, 1.0]
    overlay[dst_pos] = [0.85, 0.0, 0.0, 1.0]
    ax.imshow(overlay, interpolation="nearest")
    for row in range(rows + 1):
        ax.axhline(row - 0.5, color="white", lw=0.3)
    for col in range(cols + 1):
        ax.axvline(col - 0.5, color="white", lw=0.3)
    draw_markers(ax, src_pos, dst_pos, ms=14)
    ax.set_title(f"Shortest-path channel around obstacle  (distance = {sd_dist:.0f} hops)", fontsize=12)
    ax.set_xlabel("Column")
    ax.set_ylabel("Row")
    ax.legend(
        handles=[
            mpatches.Patch(color="green", label=f"Source {src_pos}"),
            mpatches.Patch(color="red", label=f"Destination {dst_pos}"),
            mpatches.Patch(color="black", label="Obstacle wall"),
            mpatches.Patch(color="orange", label="Channel (shortest path)"),
        ],
        loc="upper right",
        fontsize=9,
        framealpha=0.9,
    )
    plt.tight_layout()
    plt.savefig("examples/channel_path.png", dpi=150)
    print("Saved examples/channel_path.png")


def plot_channel_large_setup(rows, cols, num_nodes, src_pos, dst_pos, obstacle):
    if plt is None or mpatches is None:
        print("matplotlib not available; skipping large channel setup plot")
        return
    obstacle_np = obstacle.numpy().reshape(rows, cols)
    fig, ax = plt.subplots(figsize=(18, 14))
    grid_rgb = np.full((rows, cols, 3), 0.92)
    grid_rgb[obstacle_np] = [0.12, 0.12, 0.12]
    grid_rgb[src_pos] = [0.0, 0.8, 0.0]
    grid_rgb[dst_pos] = [0.85, 0.0, 0.0]
    ax.imshow(grid_rgb, interpolation="nearest", aspect="equal")
    ax.plot(src_pos[1], src_pos[0], "g^", ms=14, mec="white", mew=1.5)
    ax.plot(dst_pos[1], dst_pos[0], "rv", ms=14, mec="white", mew=1.5)
    ax.set_title(f"Large-scale channel - {rows}x{cols} grid ({num_nodes} devices)", fontsize=14)
    ax.set_xlabel("Column")
    ax.set_ylabel("Row")
    ax.legend(
        handles=[
            mpatches.Patch(color="green", label=f"Source {src_pos}"),
            mpatches.Patch(color="red", label=f"Dest {dst_pos}"),
            mpatches.Patch(color="black", label="Obstacles"),
        ],
        loc="upper right",
        fontsize=10,
        framealpha=0.9,
    )
    plt.tight_layout()
    plt.savefig("examples/channel_large_setup.png", dpi=150)
    print("Saved examples/channel_large_setup.png")


def plot_channel_large_evolution(rows, cols, snapshots, snapshot_steps, snapshot_labels, num_nodes, rounds, elapsed, obstacle):
    if plt is None:
        print("matplotlib not available; skipping large channel evolution plot")
        return
    from mpl_toolkits.axes_grid1 import make_axes_locatable

    field_keys = ["dist_src", "dist_dst", "sum", "dist_sd", "channel"]
    field_labels = ["dist_src", "dist_dst", "sum (src+dst)", "broadcast", "channel"]
    fig, axes = plt.subplots(len(field_keys), len(snapshot_steps), figsize=(3.5 * len(snapshot_steps), 2.5 * len(field_keys)))
    row_images = []
    for row_idx, (field_key, field_label) in enumerate(zip(field_keys, field_labels)):
        last_image = None
        for col_idx, step in enumerate(snapshot_steps):
            ax = axes[row_idx, col_idx]
            grid = to_grid(snapshots[step][field_key], rows, cols, obstacle)
            if field_key == "channel":
                image = ax.imshow(grid, cmap="Oranges", vmin=0, vmax=1, interpolation="nearest", aspect="equal")
            else:
                image = ax.imshow(grid, cmap="viridis", interpolation="nearest", aspect="equal")
            last_image = image
            ax.set_title(snapshot_labels[col_idx], fontsize=8)
            if col_idx == 0:
                ax.set_ylabel(field_label, fontsize=8)
            ax.set_xticks([])
            ax.set_yticks([])
        row_images.append(last_image)
    for row_idx, image in enumerate(row_images):
        last_ax = axes[row_idx, len(snapshot_steps) - 1]
        divider = make_axes_locatable(last_ax)
        color_ax = divider.append_axes("right", size="3%", pad=0.05)
        fig.colorbar(image, cax=color_ax)
    fig.suptitle(
        f"Large-scale channel evolution - {rows}x{cols}, {num_nodes} devices, {rounds} rounds, {elapsed:.1f}s",
        fontsize=12,
        y=0.995,
    )
    plt.tight_layout(rect=[0, 0, 1, 0.98])
    plt.savefig("examples/channel_large_evolution.png", dpi=160)
    print("Saved examples/channel_large_evolution.png")


def plot_channel_large_final(rows, cols, final, src_pos, dst_pos, obstacle, channel_threshold, num_nodes, rounds, elapsed, sd_dist, channel_nodes):
    if plt is None or mpatches is None:
        print("matplotlib not available; skipping large channel final plot")
        return
    channel = final["channel"]
    overlay = np.full((rows, cols, 4), [0.94, 0.94, 0.94, 1.0])
    for row in range(rows):
        for col in range(cols):
            node_id = row * cols + col
            if obstacle[node_id]:
                overlay[row, col] = [0.10, 0.10, 0.10, 1.0]
            elif channel[node_id] > channel_threshold:
                overlay[row, col] = [1.0, 0.55, 0.0, 0.95]
    overlay[src_pos] = [0.0, 0.80, 0.0, 1.0]
    overlay[dst_pos] = [0.85, 0.0, 0.0, 1.0]
    fig, ax = plt.subplots(figsize=(18, 14))
    ax.imshow(overlay, interpolation="nearest", aspect="equal")
    ax.plot(src_pos[1], src_pos[0], "g^", ms=15, mec="white", mew=1.5)
    ax.plot(dst_pos[1], dst_pos[0], "rv", ms=15, mec="white", mew=1.5)
    ax.set_title(
        f"Converged channel path - distance={sd_dist:.0f}, channel_nodes={channel_nodes}, {rounds} rounds in {elapsed:.1f}s",
        fontsize=14,
    )
    ax.set_xlabel("Column")
    ax.set_ylabel("Row")
    ax.legend(
        handles=[
            mpatches.Patch(color="green", label="Source"),
            mpatches.Patch(color="red", label="Destination"),
            mpatches.Patch(color="black", label="Obstacle"),
            mpatches.Patch(color="orange", label="Channel"),
        ],
        loc="upper right",
        fontsize=10,
        framealpha=0.92,
    )
    plt.tight_layout()
    plt.savefig("examples/channel_large_final.png", dpi=160)
    print("Saved examples/channel_large_final.png")