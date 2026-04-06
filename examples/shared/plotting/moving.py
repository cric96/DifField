"""Plotting utilities for simulations with moving nodes (e.g., boids)."""

from __future__ import annotations

from pathlib import Path
import numpy as np
import torch
from .common import plt, FuncAnimation, PillowWriter, LineCollection, Axes


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
        segments.append(
            (
                (float(pos[src, 0]), float(pos[src, 1])),
                (float(pos[tgt, 0]), float(pos[tgt, 1])),
            )
        )
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
    fig, axes = plt.subplots(
        nrows, ncols, figsize=(4.0 * ncols, 3.8 * nrows), squeeze=False
    )

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
        if (
            show_links
            and edge_index_by_round is not None
            and round_idx in edge_index_by_round
            and LineCollection is not None
        ):
            segments = _edge_segments_from_round(
                positions_by_round[round_idx], edge_index_by_round[round_idx]
            )
            if segments:
                ax.add_collection(
                    LineCollection(
                        segments,
                        colors="black",
                        linewidths=links_width,
                        alpha=links_alpha,
                        zorder=1,
                    )
                )
        scatter = _scatter_with_unreachable(
            ax, pos, vals, vmin=vmin, vmax=vmax, size=38
        )
        if scatter is not None:
            mappable = scatter
        ax.scatter(
            pos[source_idx, 0],
            pos[source_idx, 1],
            marker="o",
            s=140,
            c="white",
            edgecolors="black",
            linewidths=0.8,
            zorder=9,
        )
        ax.scatter(
            pos[source_idx, 0],
            pos[source_idx, 1],
            marker="*",
            s=220,
            c="red",
            edgecolors="white",
            linewidths=1.0,
            zorder=10,
        )
        ax.set_title(f"round {round_idx + 1}")
        ax.set_xlim(0.0, 1.0)
        ax.set_ylim(0.0, 1.0)
        ax.set_aspect("equal")
        ax.grid(alpha=0.2)
        ax.set_xlabel("x")
        ax.set_ylabel("y")

    for index in range(num_rounds, nrows * ncols):
        axes[index // ncols][index % ncols].set_visible(False)

    fig.subplots_adjust(
        left=0.07, right=0.88, bottom=0.08, top=0.90, wspace=0.30, hspace=0.35
    )
    fig.suptitle(title)
    if mappable is not None:
        cax = fig.add_axes([0.90, 0.14, 0.018, 0.70])
        fig.colorbar(mappable, cax=cax)
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
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
        ax.plot(
            traj[:, node_idx, 0],
            traj[:, node_idx, 1],
            color=color,
            alpha=alpha,
            linewidth=linewidth,
        )

    ax.scatter(traj[0, :, 0], traj[0, :, 1], c="black", s=18, alpha=0.7, label="start")
    ax.scatter(traj[-1, :, 0], traj[-1, :, 1], c="orange", s=18, alpha=0.7, label="end")
    ax.scatter(
        traj[-1, source_idx, 0],
        traj[-1, source_idx, 1],
        marker="o",
        s=140,
        c="white",
        edgecolors="black",
        linewidths=0.8,
        zorder=9,
    )
    ax.scatter(
        traj[-1, source_idx, 0],
        traj[-1, source_idx, 1],
        marker="*",
        s=220,
        c="red",
        edgecolors="white",
        linewidths=1.0,
        zorder=10,
        label="source",
    )

    ax.set_title(title)
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.set_aspect("equal")
    ax.grid(alpha=0.25)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.legend(loc="upper right")
    plt.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
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
    src_halo = ax.scatter(
        pos0[source_idx, 0],
        pos0[source_idx, 1],
        marker="o",
        s=140,
        c="white",
        edgecolors="black",
        linewidths=0.8,
        zorder=9,
    )
    src_star = ax.scatter(
        pos0[source_idx, 0],
        pos0[source_idx, 1],
        marker="*",
        s=220,
        c="red",
        edgecolors="white",
        linewidths=1.0,
        zorder=10,
    )
    edge_collection = None
    if (
        show_links
        and edge_index_by_round is not None
        and rounds[0] in edge_index_by_round
        and LineCollection is not None
    ):
        edge_collection = LineCollection(
            _edge_segments_from_round(
                positions_by_round[rounds[0]], edge_index_by_round[rounds[0]]
            ),
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
        if (
            edge_collection is not None
            and edge_index_by_round is not None
            and round_idx in edge_index_by_round
        ):
            edge_collection.set_segments(
                _edge_segments_from_round(
                    positions_by_round[round_idx], edge_index_by_round[round_idx]
                )
            )
        src_halo.set_offsets(pos[source_idx : source_idx + 1])
        src_star.set_offsets(pos[source_idx : source_idx + 1])
        title_obj.set_text(f"{title} (round {round_idx + 1})")
        return scatter, unreachable, src_halo, src_star, title_obj

    animation = FuncAnimation(
        fig, _update, frames=len(rounds), interval=max(1, int(1000 / fps)), blit=False
    )
    try:
        animation.save(output_path, writer=PillowWriter(fps=fps))
        print(f"Saved {output_path}")
    except Exception as exc:
        print(f"Failed to export gif: {exc}")
    finally:
        plt.close(fig)
        plt.close()
