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


def _get_identity_colors(num_nodes: int, cmap_name: str = "tab20") -> np.ndarray:
    cmap = plt.get_cmap(cmap_name)
    return cmap(np.linspace(0, 1, num_nodes))


def _scatter_with_unreachable(
    ax: Axes,
    pos: np.ndarray,
    vals: np.ndarray,
    *,
    vmin: float,
    vmax: float,
    size: float,
    color_override: np.ndarray | None = None,
):
    finite_mask = np.isfinite(vals)
    mappable = None
    if finite_mask.any():
        c = color_override[finite_mask] if color_override is not None else vals[finite_mask]
        mappable = ax.scatter(
            pos[finite_mask, 0],
            pos[finite_mask, 1],
            c=c,
            cmap="viridis" if color_override is None else None,
            vmin=vmin if color_override is None else None,
            vmax=vmax if color_override is None else None,
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
    color_by_id: bool = True,
    show_source: bool = True,
) -> None:
    if plt is None:
        print("matplotlib not available; skipping snapshot plot")
        return
    rounds = sorted(set(positions_by_round.keys()) & set(values_by_round.keys()))
    if not rounds:
        print("no overlapping rounds to plot")
        return

    num_nodes = positions_by_round[rounds[0]].shape[0]
    id_colors = _get_identity_colors(num_nodes) if color_by_id else None

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
                edge_colors = "black"
                if color_by_id:
                    edges = edge_index_by_round[round_idx].detach().cpu().numpy()
                    edge_colors = id_colors[edges[0]]

                ax.add_collection(
                    LineCollection(
                        segments,
                        colors=edge_colors,
                        linewidths=links_width,
                        alpha=links_alpha,
                        zorder=1,
                    )
                )
        scatter = _scatter_with_unreachable(
            ax, pos, vals, vmin=vmin, vmax=vmax, size=38, color_override=id_colors
        )
        if scatter is not None:
            mappable = scatter
        
        if show_source:
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


def _draw_trajectory_on_ax(
    ax: Axes,
    traj: np.ndarray,
    num_rounds: int,
    num_nodes: int,
    id_colors: np.ndarray | None,
    color_by_id: bool,
    show_source: bool,
    source_idx: int,
) -> None:
    for node_idx in range(num_nodes):
        is_source = show_source and node_idx == source_idx
        color = id_colors[node_idx] if color_by_id else ("crimson" if is_source else "steelblue")
        alpha_base = 0.9 if is_source else 0.7
        
        # Use segments for gradient trails
        points = traj[:, node_idx, :].reshape(-1, 1, 2)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)
        
        # Line width and alpha progression
        widths = np.linspace(0.4, 2.5 if is_source else 1.8, num_rounds - 1)
        alphas = np.linspace(0.1, alpha_base, num_rounds - 1)
        
        lc = LineCollection(segments, linewidths=widths, colors=color, alpha=alphas, zorder=2)
        ax.add_collection(lc)

        # Final position marker
        ax.scatter(
            traj[-1, node_idx, 0],
            traj[-1, node_idx, 1],
            color=color,
            s=40 if is_source else 25,
            edgecolors="white",
            linewidths=0.5,
            zorder=5 if is_source else 3,
        )

    # Markers for Start and End (Legend purpose)
    if color_by_id and id_colors is not None:
        ax.scatter(traj[0, :, 0], traj[0, :, 1], c=id_colors, s=12, alpha=0.25, label="start", zorder=1)
        ax.scatter(traj[-1, :, 0], traj[-1, :, 1], c=id_colors, marker="x", s=45, alpha=1.0, label="end", zorder=10)
    else:
        ax.scatter(traj[0, :, 0], traj[0, :, 1], c="black", s=12, alpha=0.25, label="start", zorder=1)
        ax.scatter(traj[-1, :, 0], traj[-1, :, 1], c="black", marker="x", s=35, alpha=0.8, label="end", zorder=10)
    
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.set_aspect("equal")
    ax.grid(alpha=0.25)
    ax.set_xlabel("x")
    ax.set_ylabel("y")


def plot_node_trajectories(
    *,
    positions_over_time: list[torch.Tensor],
    source_idx: int,
    output_path: str,
    title: str = "Node Trajectories",
    color_by_id: bool = True,
    show_source: bool = True,
    phantom_pos_seq: torch.Tensor | None = None,
) -> None:
    if plt is None:
        print("matplotlib not available; skipping trajectory plot")
        return
    if not positions_over_time:
        print("no trajectory data to plot")
        return

    traj = torch.stack(positions_over_time, dim=0).detach().cpu().numpy()
    num_rounds, num_nodes, _ = traj.shape
    id_colors = _get_identity_colors(num_nodes) if color_by_id else None

    if phantom_pos_seq is not None:
        fig, axes = plt.subplots(1, 2, figsize=(15.0, 7.5))
        phantom = phantom_pos_seq.detach().cpu().numpy()
        
        # Plot Teacher Reference
        _draw_trajectory_on_ax(axes[0], phantom, num_rounds, num_nodes, id_colors, color_by_id, show_source, source_idx)
        axes[0].set_title("Teacher Ground Truth")
        
        # Plot Prediction
        _draw_trajectory_on_ax(axes[1], traj, num_rounds, num_nodes, id_colors, color_by_id, show_source, source_idx)
        axes[1].set_title("Learned Model")
        
        fig.suptitle(title)
        axes[1].legend(loc="upper right")
    else:
        fig, ax = plt.subplots(figsize=(8.0, 7.5))
        _draw_trajectory_on_ax(ax, traj, num_rounds, num_nodes, id_colors, color_by_id, show_source, source_idx)
        ax.set_title(title)
        ax.legend(loc="upper right")

    plt.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"Saved {output_path}")


def _draw_snap_on_ax(
    ax: Axes,
    pos: np.ndarray,
    vals: np.ndarray,
    vmin: float,
    vmax: float,
    id_colors: np.ndarray | None,
    edge_index: torch.Tensor | None,
    show_links: bool,
    links_width: float,
    links_alpha: float,
) -> None:
    if show_links and edge_index is not None and LineCollection is not None:
        segments = _edge_segments_from_round(torch.from_numpy(pos), edge_index)
        if segments:
            edge_colors = "black"
            if id_colors is not None:
                edges = edge_index.detach().cpu().numpy()
                edge_colors = id_colors[edges[0]]
            ax.add_collection(
                LineCollection(segments, colors=edge_colors, linewidths=links_width, alpha=links_alpha, zorder=1)
            )
    _scatter_with_unreachable(ax, pos, vals, vmin=vmin, vmax=vmax, size=38, color_override=id_colors)
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.set_aspect("equal")
    ax.grid(alpha=0.2)
    ax.set_xlabel("x")
    ax.set_ylabel("y")


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
    color_by_id: bool = True,
    trail_length: int = 12,
    show_source: bool = True,
    phantom_pos_seq: torch.Tensor | None = None,
) -> None:
    if plt is None or FuncAnimation is None or PillowWriter is None:
        print("matplotlib animation/pillow not available; skipping gif export")
        return

    rounds = sorted(set(positions_by_round.keys()) & set(values_by_round.keys()))
    if not rounds:
        print("no overlapping rounds to animate")
        return

    num_nodes = positions_by_round[rounds[0]].shape[0]
    id_colors = _get_identity_colors(num_nodes) if color_by_id else None

    # Global min/max for color scaling if not using ID colors
    vmin, vmax = 0.0, 1.0
    if not color_by_id:
        finite_all = []
        for r in rounds:
            arr = _to_color_values(values_by_round[r])
            finite = arr[np.isfinite(arr)]
            if finite.size: finite_all.append(finite)
        if finite_all:
            flat = np.concatenate(finite_all)
            vmin, vmax = float(flat.min()), float(flat.max())

    # Create subplots if phantom is provided
    is_split = phantom_pos_seq is not None
    ncols = 2 if is_split else 1
    fig, axes = plt.subplots(1, ncols, figsize=(6.6 * ncols, 6.0), squeeze=False)
    ax_pred = axes[0][0 if not is_split else 1]
    ax_ref = axes[0][0] if is_split else None

    def setup_ax(ax, title_str):
        ax.set_xlim(0.0, 1.0)
        ax.set_ylim(0.0, 1.0)
        ax.set_aspect("equal")
        ax.grid(alpha=0.2)
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        return ax.set_title(title_str)

    setup_ax(ax_pred, "Learned Model" if is_split else title)
    if is_split: setup_ax(ax_ref, "Teacher Reference")

    # Artists containers
    def create_artists(ax):
        # Trails
        trails = []
        for i in range(trail_length):
            alpha = 0.35 * (0.75 ** (i + 1))
            size = 38 * (0.88 ** (i + 1))
            ts = ax.scatter([], [], c=[], s=size, alpha=alpha, zorder=2)
            trails.append(ts)
        # Main scatter
        main = ax.scatter([], [], c=[], s=42, zorder=4)
        unreachable = ax.scatter([], [], c="#b8b8b8", edgecolors="#666666", linewidths=0.4, s=42, zorder=4)
        # Links
        links = None
        if show_links and LineCollection is not None:
            links = LineCollection([], linewidths=links_width, alpha=links_alpha, zorder=1)
            ax.add_collection(links)
        return {"trails": trails, "main": main, "unreachable": unreachable, "links": links}

    pred_artists = create_artists(ax_pred)
    ref_artists = create_artists(ax_ref) if is_split else None

    def update_ax(round_idx, artists, pos_data, val_data, edge_data, frame_idx):
        pos = pos_data[round_idx].detach().cpu().numpy()
        vals = _to_color_values(val_data[round_idx])
        finite = np.isfinite(vals)
        
        # Trails
        for i in range(trail_length):
            prev_idx = frame_idx - (i + 1)
            if prev_idx >= 0:
                p_round = rounds[prev_idx]
                p_pos = pos_data[p_round].detach().cpu().numpy()
                p_vals = _to_color_values(val_data[p_round])
                p_finite = np.isfinite(p_vals)
                artists["trails"][i].set_offsets(p_pos[p_finite])
                if color_by_id: artists["trails"][i].set_color(id_colors[p_finite])
                else: artists["trails"][i].set_array(p_vals[p_finite])
            else:
                artists["trails"][i].set_offsets(np.empty((0, 2)))

        artists["main"].set_offsets(pos[finite])
        if color_by_id: artists["main"].set_color(id_colors[finite])
        else: artists["main"].set_array(vals[finite])
        artists["unreachable"].set_offsets(pos[~finite])
        
        if artists["links"] is not None and edge_data is not None and round_idx in edge_data:
            segs = _edge_segments_from_round(pos_data[round_idx], edge_data[round_idx])
            artists["links"].set_segments(segs)
            if color_by_id:
                edges = edge_data[round_idx].detach().cpu().numpy()
                artists["links"].set_colors(id_colors[edges[0]])
            else:
                artists["links"].set_colors("black")
        
        res = [artists["main"], artists["unreachable"], *artists["trails"]]
        if artists["links"]: res.append(artists["links"])
        return res

    def update(frame_idx: int):
        round_idx = rounds[frame_idx]
        res = update_ax(round_idx, pred_artists, positions_by_round, values_by_round, edge_index_by_round, frame_idx)
        if is_split:
            # For reference, we reuse values_by_round (identity) or we can assume teacher always reachable
            # Values for teacher: we can just pass the same as pred since identity is same
            res += update_ax(round_idx, ref_artists, {r: phantom_pos_seq[r] for r in rounds}, values_by_round, edge_index_by_round, frame_idx)
        return res

    fig.suptitle(title)
    ani = FuncAnimation(fig, update, frames=len(rounds), interval=1000 / fps, blit=True)
    writer = PillowWriter(fps=fps)
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    ani.save(output_path, writer=writer)
    plt.close(fig)
    print(f"Saved {output_path}")
