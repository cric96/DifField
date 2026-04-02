"""Common plotting utilities for example scripts."""

from __future__ import annotations

from typing import Optional

import numpy as np
import torch

try:
    import matplotlib.patches as mpatches
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation, PillowWriter
    from matplotlib.axes import Axes
    from matplotlib.collections import LineCollection
except ImportError:
    plt = None
    mpatches = None
    FuncAnimation = None
    PillowWriter = None
    LineCollection = None
    Axes = None


def to_grid(
    tensor: torch.Tensor,
    rows: int,
    cols: int,
    obstacle_mask: Optional[torch.Tensor] = None,
    replace_inf: bool = True,
) -> np.ndarray:
    """Reshape a flat tensor to a 2D grid, masking obstacles and infinities."""
    arr = tensor.detach().cpu().float().clone()
    if obstacle_mask is not None:
        arr[obstacle_mask.cpu()] = float("nan")
    if replace_inf:
        arr[torch.isinf(arr)] = float("nan")
    return arr.view(rows, cols).numpy()


def save_gif(
    render_frame_fn: Callable[[Axes, int], None],
    frames: list[int],
    output_path: str,
    fps: int = 10,
    figsize: tuple[float, float] = (5, 5),
) -> None:
    """Reusable utility for creating a GIF from a sequence of frames."""
    if plt is None or FuncAnimation is None or PillowWriter is None:
        print("matplotlib animation tools not available; skipping GIF generation")
        return

    fig, ax = plt.subplots(figsize=figsize)

    def update(frame_idx):
        ax.clear()
        render_frame_fn(ax, frame_idx)

    anim = FuncAnimation(fig, update, frames=frames)
    writer = PillowWriter(fps=fps)
    anim.save(output_path, writer=writer)
    plt.close(fig)
    print(f"Saved GIF to {output_path}")


def save_grid_simulation_gif(
    records: dict[int, dict[str, torch.Tensor]],
    field_key: str,
    rows: int,
    cols: int,
    output_path: str,
    obstacle: Optional[torch.Tensor] = None,
    src_pos: Optional[tuple[int, int]] = None,
    dst_pos: Optional[tuple[int, int]] = None,
    cmap: str = "viridis",
    vmin: float = 0.0,
    vmax: float = 45.0,
    fps: int = 10,
    figsize: tuple[float, float] = (5, 5),
    title: Optional[str] = None,
) -> None:
    """Save a GIF of a specific field from a grid simulation's records."""
    frames = sorted(records.keys())

    def render_frame(ax, frame_idx):
        grid = to_grid(records[frame_idx][field_key], rows, cols, obstacle)
        im = ax.imshow(grid, cmap=cmap, vmin=vmin, vmax=vmax, interpolation="nearest")
        if obstacle is not None:
            draw_obstacles(ax, obstacle, rows, cols)
        if src_pos is not None:
            draw_markers(ax, src_pos, dst_pos)
        if title:
            ax.set_title(f"{title} - Round {frame_idx + 1}")
        else:
            ax.set_title(f"Field: {field_key} - Round {frame_idx + 1}")
        ax.set_xticks([])
        ax.set_yticks([])

    save_gif(render_frame, frames, output_path, fps=fps, figsize=figsize)


def draw_obstacles(ax: Axes | None, obstacle_mask: torch.Tensor, rows: int, cols: int, color: str = "black") -> None:
    """Draw solid rectangles over obstacle cells in a grid plot."""
    if plt is None or ax is None:
        return
    obstacle = obstacle_mask.cpu().view(rows, cols)
    for row in range(rows):
        for col in range(cols):
            if obstacle[row, col]:
                ax.add_patch(
                    plt.Rectangle(
                        (col - 0.5, row - 0.5),
                        1,
                        1,
                        facecolor=color,
                        edgecolor=color,
                    )
                )


def draw_markers(
    ax: Axes | None,
    src_pos: tuple[int, int],
    dst_pos: Optional[tuple[int, int]] = None,
    ms: int = 10,
) -> None:
    """Draw source and destination markers."""
    if plt is None or ax is None:
        return
    ax.plot(src_pos[1], src_pos[0], "g^", markersize=ms, markeredgecolor="white", markeredgewidth=1.2)
    if dst_pos is not None:
        ax.plot(dst_pos[1], dst_pos[0], "rv", markersize=ms, markeredgecolor="white", markeredgewidth=1.2)


def _to_color_values(values: torch.Tensor) -> np.ndarray:
    arr = values.detach().cpu().float().numpy()
    arr = np.where(np.isfinite(arr), arr, np.nan)
    return arr


def _scatter_with_unreachable(
    ax: Axes,
    pos: np.ndarray,
    vals: np.ndarray,
    *,
    vmin: float,
    vmax: float,
    size: float,
):
    finite_mask = np.isfinite(vals)
    mappable = None
    if finite_mask.any():
        mappable = ax.scatter(
            pos[finite_mask, 0],
            pos[finite_mask, 1],
            c=vals[finite_mask],
            cmap="viridis",
            vmin=vmin,
            vmax=vmax,
            s=size,
            zorder=3,
        )
    if (~finite_mask).any():
        ax.scatter(
            pos[~finite_mask, 0],
            pos[~finite_mask, 1],
            c="#b8b8b8",
            edgecolors="#666666",
            linewidths=0.4,
            s=size,
            zorder=3,
        )
    return mappable


def _edge_segments_from_round(
    positions: torch.Tensor,
    edge_index: torch.Tensor,
) -> list[tuple[tuple[float, float], tuple[float, float]]]:
    pos = positions.detach().cpu().numpy()
    edges = edge_index.detach().cpu().numpy()
    segments: list[tuple[tuple[float, float], tuple[float, float]]] = []
    for src, tgt in zip(edges[0], edges[1]):
        segments.append(((float(pos[src, 0]), float(pos[src, 1])), (float(pos[tgt, 0]), float(pos[tgt, 1]))))
    return segments


def plot_moving_snapshots(
    *,
    positions_by_round: dict[int, torch.Tensor],
    values_by_round: dict[int, torch.Tensor],
    source_idx: int,
    output_path: str,
    title: str = "Moving Nodes Distance Snapshots",
    edge_index_by_round: dict[int, torch.Tensor] | None = None,
    show_links: bool = True,
    links_alpha: float = 0.15,
    links_width: float = 0.6,
) -> None:
    if plt is None:
        print("matplotlib not available; skipping snapshot plot")
        return
    rounds = sorted(set(positions_by_round.keys()) & set(values_by_round.keys()))
    if not rounds:
        print("no overlapping rounds to plot")
        return

    num_rounds = len(rounds)
    ncols = min(4, num_rounds)
    nrows = int(np.ceil(num_rounds / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.0 * ncols, 3.8 * nrows), squeeze=False)

    finite_all = []
    for round_idx in rounds:
        arr = _to_color_values(values_by_round[round_idx])
        finite = arr[np.isfinite(arr)]
        if finite.size:
            finite_all.append(finite)
    if finite_all:
        flat = np.concatenate(finite_all)
        vmin, vmax = float(flat.min()), float(flat.max())
    else:
        vmin, vmax = 0.0, 1.0

    mappable = None
    for index, round_idx in enumerate(rounds):
        ax = axes[index // ncols][index % ncols]
        pos = positions_by_round[round_idx].detach().cpu().numpy()
        vals = _to_color_values(values_by_round[round_idx])
        if show_links and edge_index_by_round is not None and round_idx in edge_index_by_round and LineCollection is not None:
            segments = _edge_segments_from_round(positions_by_round[round_idx], edge_index_by_round[round_idx])
            if segments:
                ax.add_collection(LineCollection(segments, colors="black", linewidths=links_width, alpha=links_alpha, zorder=1))
        scatter = _scatter_with_unreachable(ax, pos, vals, vmin=vmin, vmax=vmax, size=38)
        if scatter is not None:
            mappable = scatter
        ax.scatter(pos[source_idx, 0], pos[source_idx, 1], marker="o", s=140, c="white", edgecolors="black", linewidths=0.8, zorder=9)
        ax.scatter(pos[source_idx, 0], pos[source_idx, 1], marker="*", s=220, c="red", edgecolors="white", linewidths=1.0, zorder=10)
        ax.set_title(f"round {round_idx + 1}")
        ax.set_xlim(0.0, 1.0)
        ax.set_ylim(0.0, 1.0)
        ax.set_aspect("equal")
        ax.grid(alpha=0.2)
        ax.set_xlabel("x")
        ax.set_ylabel("y")

    for index in range(num_rounds, nrows * ncols):
        axes[index // ncols][index % ncols].set_visible(False)

    fig.subplots_adjust(left=0.07, right=0.88, bottom=0.08, top=0.90, wspace=0.30, hspace=0.35)
    fig.suptitle(title)
    if mappable is not None:
        cax = fig.add_axes([0.90, 0.14, 0.018, 0.70])
        fig.colorbar(mappable, cax=cax)
    plt.savefig(output_path, dpi=150)
    print(f"Saved {output_path}")


def plot_node_trajectories(
    *,
    positions_over_time: list[torch.Tensor],
    source_idx: int,
    output_path: str,
    title: str = "Node Trajectories",
) -> None:
    if plt is None:
        print("matplotlib not available; skipping trajectory plot")
        return
    if not positions_over_time:
        print("no trajectory data to plot")
        return

    traj = torch.stack(positions_over_time, dim=0).detach().cpu().numpy()
    _, num_nodes, _ = traj.shape

    fig, ax = plt.subplots(figsize=(7.5, 7.0))
    for node_idx in range(num_nodes):
        linewidth = 2.2 if node_idx == source_idx else 0.8
        alpha = 0.95 if node_idx == source_idx else 0.35
        color = "crimson" if node_idx == source_idx else "steelblue"
        ax.plot(traj[:, node_idx, 0], traj[:, node_idx, 1], color=color, alpha=alpha, linewidth=linewidth)

    ax.scatter(traj[0, :, 0], traj[0, :, 1], c="black", s=18, alpha=0.7, label="start")
    ax.scatter(traj[-1, :, 0], traj[-1, :, 1], c="orange", s=18, alpha=0.7, label="end")
    ax.scatter(traj[-1, source_idx, 0], traj[-1, source_idx, 1], marker="o", s=140, c="white", edgecolors="black", linewidths=0.8, zorder=9)
    ax.scatter(traj[-1, source_idx, 0], traj[-1, source_idx, 1], marker="*", s=220, c="red", edgecolors="white", linewidths=1.0, zorder=10, label="source")

    ax.set_title(title)
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.set_aspect("equal")
    ax.grid(alpha=0.25)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.legend(loc="upper right")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    print(f"Saved {output_path}")


def export_moving_gif(
    *,
    positions_by_round: dict[int, torch.Tensor],
    values_by_round: dict[int, torch.Tensor],
    source_idx: int,
    output_path: str,
    title: str = "Moving Nodes Distance Evolution",
    fps: int = 8,
    edge_index_by_round: dict[int, torch.Tensor] | None = None,
    show_links: bool = True,
    links_alpha: float = 0.15,
    links_width: float = 0.6,
) -> None:
    if plt is None or FuncAnimation is None or PillowWriter is None:
        print("matplotlib animation/pillow not available; skipping gif export")
        return

    rounds = sorted(set(positions_by_round.keys()) & set(values_by_round.keys()))
    if not rounds:
        print("no overlapping rounds to animate")
        return

    finite_all = []
    for round_idx in rounds:
        arr = _to_color_values(values_by_round[round_idx])
        finite = arr[np.isfinite(arr)]
        if finite.size:
            finite_all.append(finite)
    if finite_all:
        flat = np.concatenate(finite_all)
        vmin, vmax = float(flat.min()), float(flat.max())
    else:
        vmin, vmax = 0.0, 1.0

    fig, ax = plt.subplots(figsize=(6.6, 6.0))
    pos0 = positions_by_round[rounds[0]].detach().cpu().numpy()
    vals0 = _to_color_values(values_by_round[rounds[0]])
    finite0 = np.isfinite(vals0)

    scatter = ax.scatter(
        pos0[finite0, 0],
        pos0[finite0, 1],
        c=vals0[finite0] if finite0.any() else np.array([]),
        cmap="viridis",
        vmin=vmin,
        vmax=vmax,
        s=42,
        zorder=3,
    )
    unreachable = ax.scatter(
        pos0[~finite0, 0],
        pos0[~finite0, 1],
        c="#b8b8b8",
        edgecolors="#666666",
        linewidths=0.4,
        s=42,
        zorder=3,
    )
    src_halo = ax.scatter(pos0[source_idx, 0], pos0[source_idx, 1], marker="o", s=140, c="white", edgecolors="black", linewidths=0.8, zorder=9)
    src_star = ax.scatter(pos0[source_idx, 0], pos0[source_idx, 1], marker="*", s=220, c="red", edgecolors="white", linewidths=1.0, zorder=10)
    edge_collection = None
    if show_links and edge_index_by_round is not None and rounds[0] in edge_index_by_round and LineCollection is not None:
        edge_collection = LineCollection(
            _edge_segments_from_round(positions_by_round[rounds[0]], edge_index_by_round[rounds[0]]),
            colors="black",
            linewidths=links_width,
            alpha=links_alpha,
            zorder=1,
        )
        ax.add_collection(edge_collection)
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.set_aspect("equal")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.grid(alpha=0.2)
    title_obj = ax.set_title(f"{title} (round {rounds[0] + 1})")
    fig.colorbar(scatter, ax=ax, fraction=0.046, pad=0.04)

    def _update(frame_idx: int):
        round_idx = rounds[frame_idx]
        pos = positions_by_round[round_idx].detach().cpu().numpy()
        vals = _to_color_values(values_by_round[round_idx])
        finite_mask = np.isfinite(vals)
        if finite_mask.any():
            scatter.set_offsets(pos[finite_mask])
            scatter.set_array(vals[finite_mask])
        else:
            scatter.set_offsets(np.empty((0, 2)))
            scatter.set_array(np.array([], dtype=float))
        unreachable.set_offsets(pos[~finite_mask])
        if edge_collection is not None and edge_index_by_round is not None and round_idx in edge_index_by_round:
            edge_collection.set_segments(_edge_segments_from_round(positions_by_round[round_idx], edge_index_by_round[round_idx]))
        src_halo.set_offsets(pos[source_idx : source_idx + 1])
        src_star.set_offsets(pos[source_idx : source_idx + 1])
        title_obj.set_text(f"{title} (round {round_idx + 1})")
        return scatter, unreachable, src_halo, src_star, title_obj

    animation = FuncAnimation(fig, _update, frames=len(rounds), interval=max(1, int(1000 / fps)), blit=False)
    try:
        animation.save(output_path, writer=PillowWriter(fps=fps))
        print(f"Saved {output_path}")
    except Exception as exc:
        print(f"Failed to export gif: {exc}")
    finally:
        plt.close(fig)


def plot_trajectory_comparison(
    *,
    predicted_positions_over_time: list[torch.Tensor],
    teacher_positions_over_time: list[torch.Tensor],
    source_idx: int,
    output_path: str,
    title: str = "Predicted vs Teacher Trajectories",
) -> None:
    if plt is None:
        print("matplotlib not available; skipping comparison plot")
        return
    if not predicted_positions_over_time or not teacher_positions_over_time:
        print("no trajectory data to compare")
        return

    predicted = torch.stack(predicted_positions_over_time, dim=0).detach().cpu().numpy()
    teacher = torch.stack(teacher_positions_over_time, dim=0).detach().cpu().numpy()
    if predicted.shape[1] != teacher.shape[1]:
        print("trajectory mismatch: predicted and teacher node counts differ")
        return

    _, num_nodes, _ = predicted.shape
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharex=True, sharey=True)
    for ax, (traj, panel_title) in zip(axes, ((predicted, "Predicted"), (teacher, "Teacher"))):
        for node_idx in range(num_nodes):
            linewidth = 2.2 if node_idx == source_idx else 0.8
            alpha = 0.95 if node_idx == source_idx else 0.35
            color = "crimson" if node_idx == source_idx else "steelblue"
            ax.plot(traj[:, node_idx, 0], traj[:, node_idx, 1], color=color, alpha=alpha, linewidth=linewidth)

        ax.scatter(traj[0, :, 0], traj[0, :, 1], c="black", s=18, alpha=0.7, label="start")
        ax.scatter(traj[-1, :, 0], traj[-1, :, 1], c="orange", s=18, alpha=0.7, label="end")
        ax.scatter(traj[-1, source_idx, 0], traj[-1, source_idx, 1], marker="o", s=140, c="white", edgecolors="black", linewidths=0.8, zorder=9)
        ax.scatter(traj[-1, source_idx, 0], traj[-1, source_idx, 1], marker="*", s=220, c="red", edgecolors="white", linewidths=1.0, zorder=10, label="source")
        ax.set_title(panel_title)
        ax.set_xlim(0.0, 1.0)
        ax.set_ylim(0.0, 1.0)
        ax.set_aspect("equal")
        ax.grid(alpha=0.25)
        ax.set_xlabel("x")
        ax.set_ylabel("y")

    axes[1].legend(loc="upper right")
    fig.suptitle(title)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    print(f"Saved {output_path}")