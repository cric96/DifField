#!/usr/bin/env python3
"""Channel with fixed random nodes and obstacles (non-grid topology)."""

from __future__ import annotations

import argparse
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "examples"))

from channel.core import (  # noqa: E402
    CHANNEL_THRESHOLD,
    build_snapshot_payloads,
    channel_body,
    count_channel_nodes,
)
from shared.plotting.common import LineCollection, plt, save_gif  # noqa: E402

from diffield.dsl import branch, field  # noqa: E402
from diffield.sim import (  # noqa: E402
    SimulationEngine,
    SnapshotRecorder,
    SpatialScenario,
)
from diffield.utils import get_device  # noqa: E402


@dataclass(frozen=True)
class ChannelVizStyle:
    node_color: str = "steelblue"
    node_size: int = 50
    node_alpha: float = 0.7
    channel_color: str = "orange"
    channel_size: int = 70
    channel_alpha: float = 0.9
    obstacle_color: str = "darkred"
    obstacle_size: int = 80
    obstacle_alpha: float = 0.8
    source_ring_color: str = "white"
    source_ring_size: int = 200
    source_star_color: str = "green"
    source_star_size: int = 150
    dest_ring_color: str = "white"
    dest_ring_size: int = 200
    dest_star_color: str = "red"
    dest_star_size: int = 150
    edge_color: str = "gray"
    edge_alpha: float = 0.12
    edge_width: float = 0.5


STYLE = ChannelVizStyle()


def parse_args():
    parser = argparse.ArgumentParser(
        description="Channel with fixed random nodes and obstacles"
    )
    parser.add_argument("--num-nodes", type=int, default=150)
    parser.add_argument("--rounds", type=int, default=100)
    parser.add_argument(
        "--radius", type=float, default=None,
        help="Connectivity radius (auto-computed if not set)",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--source", type=int, default=-1)
    parser.add_argument("--dest", type=int, default=-1)
    parser.add_argument("--tolerance", type=float, default=0.01)
    parser.add_argument("--obstacle-ratio", type=float, default=0.18)
    parser.add_argument("--record-every", type=int, default=20)
    parser.add_argument("--viz-prefix", type=str, default="generated/channel_random")
    parser.add_argument("--gif-fps", type=int, default=10)
    parser.add_argument("--no-viz", action="store_true", help="Disable figure export")
    parser.add_argument("--no-gif", action="store_true", help="Disable gif export")
    parser.add_argument(
        "--hide-links",
        action="store_true",
        help="Do not draw graph links in visual outputs",
    )
    parser.add_argument("--links-alpha", type=float, default=0.12)
    parser.add_argument("--links-width", type=float, default=0.5)
    parser.add_argument(
        "--device", type=str, default="", help="Device (cuda/cpu) [auto if empty]"
    )
    return parser.parse_args()


def _to_numpy(tensor: torch.Tensor) -> np.ndarray:
    return tensor.detach().cpu().numpy()


def _ensure_parent_dir(output_path: str) -> None:
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)


def _edge_segments(
    positions: np.ndarray, edge_index: np.ndarray
) -> list[tuple[tuple[float, float], tuple[float, float]]]:
    return [
        (
            (float(positions[src, 0]), float(positions[src, 1])),
            (float(positions[tgt, 0]), float(positions[tgt, 1])),
        )
        for src, tgt in zip(edge_index[0], edge_index[1], strict=False)
    ]


def _draw_edges(
    ax,
    positions: np.ndarray,
    edge_index: np.ndarray,
    *,
    show_links: bool,
    alpha: float,
    width: float,
    color: str = "gray",
):
    if not show_links or LineCollection is None:
        return None

    segments = _edge_segments(positions, edge_index)
    if not segments:
        return None

    collection = LineCollection(
        segments,
        colors=color,
        linewidths=width,
        alpha=alpha,
        zorder=1,
    )
    ax.add_collection(collection)
    return collection


def _draw_nodes(
    ax,
    positions: np.ndarray,
    mask: np.ndarray,
    *,
    color: str,
    size: int,
    alpha: float,
    marker: str = "o",
    label: str | None = None,
    zorder: int = 3,
):
    if not mask.any():
        return None
    return ax.scatter(
        positions[mask, 0],
        positions[mask, 1],
        c=color,
        s=size,
        alpha=alpha,
        marker=marker,
        zorder=zorder,
        label=label,
    )


def _draw_source_dest(
    ax,
    positions: np.ndarray,
    source_idx: int,
    dest_idx: int,
    *,
    source_label: str | None = None,
    dest_label: str | None = None,
    source_ring_size: int = STYLE.source_ring_size,
    source_star_size: int = STYLE.source_star_size,
    dest_ring_size: int = STYLE.dest_ring_size,
    dest_star_size: int = STYLE.dest_star_size,
):
    source_point = positions[source_idx : source_idx + 1]
    dest_point = positions[dest_idx : dest_idx + 1]

    source_ring = ax.scatter(
        source_point[:, 0],
        source_point[:, 1],
        c=STYLE.source_ring_color,
        s=source_ring_size,
        edgecolors="black",
        linewidths=2.2,
        zorder=9,
        marker="o",
        clip_on=False,
    )
    source_star = ax.scatter(
        source_point[:, 0],
        source_point[:, 1],
        c=STYLE.source_star_color,
        s=source_star_size,
        marker="*",
        zorder=10,
        label=source_label,
        edgecolors="#123b12",
        linewidths=1.1,
        clip_on=False,
    )
    dest_ring = ax.scatter(
        dest_point[:, 0],
        dest_point[:, 1],
        c=STYLE.dest_ring_color,
        s=dest_ring_size,
        edgecolors="black",
        linewidths=2.2,
        zorder=9,
        marker="o",
        clip_on=False,
    )
    dest_star = ax.scatter(
        dest_point[:, 0],
        dest_point[:, 1],
        c=STYLE.dest_star_color,
        s=dest_star_size,
        marker="*",
        zorder=10,
        label=dest_label,
        edgecolors="#5c0d0d",
        linewidths=1.1,
        clip_on=False,
    )

    return source_ring, source_star, dest_ring, dest_star


def build_record_rounds(total_rounds: int, requested_every: int) -> set[int]:
    """Front-load a small number of snapshots around channel formation."""

    if total_rounds <= 1:
        return {0}

    frame_budget = max(6, min(18, total_rounds // max(1, requested_every) + 4))
    front_loaded = np.square(np.linspace(0.0, 1.0, frame_budget))
    front_loaded_rounds = np.rint(front_loaded * (total_rounds - 1)).astype(int)

    formation_window = max(2, int(0.2 * total_rounds))
    formation_samples = np.linspace(0, formation_window, num=min(8, total_rounds))
    formation_rounds = np.rint(formation_samples).astype(int)

    return {
        int(round_idx)
        for round_idx in np.unique(
            np.concatenate(
                [front_loaded_rounds, formation_rounds, np.array([total_rounds - 1])]
            )
        )
    }


def _finalize_axes(
    ax, title: str, *, legend_loc: str = "upper left",
    show_legend: bool = False,
) -> None:
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_title(title, fontsize=13)
    if show_legend:
        ax.legend(loc=legend_loc)


def _draw_channel_overlay(
    ax,
    positions: torch.Tensor,
    edge_index: torch.Tensor,
    obstacle: torch.Tensor,
    source_idx: int,
    dest_idx: int,
    *,
    channel_values: torch.Tensor | None,
    title: str,
    show_links: bool,
    links_alpha: float,
    links_width: float,
    node_size: int = STYLE.node_size,
    channel_size: int = STYLE.channel_size,
    obstacle_size: int = STYLE.obstacle_size,
    source_ring_size: int = STYLE.source_ring_size,
    source_star_size: int = STYLE.source_star_size,
    dest_ring_size: int = STYLE.dest_ring_size,
    dest_star_size: int = STYLE.dest_star_size,
    node_alpha: float = STYLE.node_alpha,
    channel_alpha: float = STYLE.channel_alpha,
    obstacle_alpha: float = STYLE.obstacle_alpha,
    legend_loc: str = "upper left",
    show_legend: bool = False,
    normal_label: str | None = None,
    channel_label: str | None = None,
    obstacle_label: str | None = None,
    source_label: str | None = None,
    dest_label: str | None = None,
):
    pos = _to_numpy(positions)
    obs = _to_numpy(obstacle).astype(bool)
    if channel_values is None:
        channel_mask = np.zeros_like(obs, dtype=bool)
    else:
        channel_mask = _to_numpy(channel_values) > CHANNEL_THRESHOLD
    normal_mask = ~obs & ~channel_mask

    edge_collection = _draw_edges(
        ax,
        pos,
        _to_numpy(edge_index),
        show_links=show_links,
        alpha=links_alpha,
        width=links_width,
    )

    normal_scatter = _draw_nodes(
        ax,
        pos,
        normal_mask,
        color=STYLE.node_color,
        size=node_size,
        alpha=node_alpha,
        label=normal_label,
        zorder=3,
    )
    channel_scatter = _draw_nodes(
        ax,
        pos,
        channel_mask,
        color=STYLE.channel_color,
        size=channel_size,
        alpha=channel_alpha,
        label=channel_label,
        zorder=4,
    )
    obstacle_scatter = _draw_nodes(
        ax,
        pos,
        obs,
        color=STYLE.obstacle_color,
        size=obstacle_size,
        alpha=obstacle_alpha,
        marker="X",
        label=obstacle_label,
        zorder=5,
    )
    source_ring, source_star, dest_ring, dest_star = _draw_source_dest(
        ax,
        pos,
        source_idx,
        dest_idx,
        source_label=source_label,
        dest_label=dest_label,
        source_ring_size=source_ring_size,
        source_star_size=source_star_size,
        dest_ring_size=dest_ring_size,
        dest_star_size=dest_star_size,
    )

    _finalize_axes(ax, title, legend_loc=legend_loc, show_legend=show_legend)
    return {
        "positions": pos,
        "obstacle_mask": obs,
        "channel_mask": channel_mask,
        "normal_mask": normal_mask,
        "normal_scatter": normal_scatter,
        "obstacle_scatter": obstacle_scatter,
        "edge_collection": edge_collection,
        "channel_scatter": channel_scatter,
        "source_ring": source_ring,
        "source_star": source_star,
        "dest_ring": dest_ring,
        "dest_star": dest_star,
    }


def _draw_scalar_field_panel(
    ax,
    positions: torch.Tensor,
    edge_index: torch.Tensor,
    obstacle: torch.Tensor,
    source_idx: int,
    dest_idx: int,
    values: torch.Tensor,
    *,
    cmap: str,
    title: str,
    show_links: bool,
    links_alpha: float,
    links_width: float,
):
    pos = _to_numpy(positions)
    obs = _to_numpy(obstacle).astype(bool)
    scalar = _to_numpy(values)
    finite_mask = np.isfinite(scalar)

    _draw_edges(
        ax,
        pos,
        _to_numpy(edge_index),
        show_links=show_links,
        alpha=links_alpha,
        width=links_width,
    )

    mappable = ax.scatter(
        pos[finite_mask, 0],
        pos[finite_mask, 1],
        c=scalar[finite_mask],
        cmap=cmap,
        s=STYLE.node_size,
        zorder=3,
    )
    if (~finite_mask).any():
        _draw_nodes(
            ax,
            pos,
            ~finite_mask,
            color="gray",
            size=STYLE.node_size,
            alpha=0.3,
            zorder=2,
        )
    _draw_nodes(
        ax,
        pos,
        obs,
        color=STYLE.obstacle_color,
        size=STYLE.obstacle_size,
        alpha=STYLE.obstacle_alpha,
        marker="X",
        zorder=5,
    )
    _draw_source_dest(ax, pos, source_idx, dest_idx)
    _finalize_axes(ax, title)
    return mappable


def create_obstacle_mask(
    positions: torch.Tensor,
    obstacle_ratio: float,
) -> torch.Tensor:
    """Create alternating vertical walls with narrow gaps.

    Even-indexed walls leave a bottom opening; odd-indexed walls leave a top
    opening. The resulting maze forces a narrow zig-zag channel from left to
    right while keeping the same obstacle_ratio control.
    """

    x_coords = positions[:, 0]
    y_coords = positions[:, 1]

    ratio = float(obstacle_ratio)
    wall_count = max(2, min(6, 2 + round(ratio * 12.0)))
    wall_width = 0.04
    gap_size = 0.02
    margin = 0.02
    wall_centers = torch.linspace(
        0.18,
        0.82,
        steps=wall_count,
        device=positions.device,
        dtype=positions.dtype,
    )

    obstacle = torch.zeros_like(x_coords, dtype=torch.bool)
    for wall_idx, wall_center in enumerate(wall_centers):
        in_wall_band = (x_coords > (wall_center - wall_width)) & (
            x_coords < (wall_center + wall_width)
        )
        if wall_idx % 2 == 0:
            blocked_segment = y_coords > (margin + gap_size)
        else:
            blocked_segment = y_coords < (1.0 - margin - gap_size)
        obstacle |= in_wall_band & blocked_segment

    return obstacle


def plot_channel_setup(
    positions: torch.Tensor,
    edge_index: torch.Tensor,
    obstacle: torch.Tensor,
    source_idx: int,
    dest_idx: int,
    output_path: str = "generated/channel_random_setup.png",
    show_links: bool = True,
    links_alpha: float = 0.12,
    links_width: float = 0.5,
) -> None:
    if plt is None:
        print("matplotlib not available; skipping setup plot")
        return

    fig, ax = plt.subplots(figsize=(10, 8))
    _draw_channel_overlay(
        ax,
        positions,
        edge_index,
        obstacle,
        source_idx,
        dest_idx,
        channel_values=None,
        title="Channel Setup - Fixed Random Nodes with Obstacles",
        show_links=show_links,
        links_alpha=links_alpha,
        links_width=links_width,
        normal_label="Nodes",
        obstacle_label="Obstacles",
        source_label="Source",
        dest_label="Dest",
        show_legend=True,
    )

    _ensure_parent_dir(output_path)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Saved {output_path}")
    plt.close(fig)


def plot_channel_final(
    positions: torch.Tensor,
    edge_index: torch.Tensor,
    obstacle: torch.Tensor,
    source_idx: int,
    dest_idx: int,
    final: dict[str, torch.Tensor],
    output_path: str = "generated/channel_random_final.png",
    show_links: bool = True,
    links_alpha: float = 0.12,
    links_width: float = 0.5,
) -> None:
    if plt is None:
        print("matplotlib not available; skipping final plot")
        return

    fig = plt.figure(figsize=(18.5, 6.8), facecolor="#f5f1e8")
    grid = fig.add_gridspec(1, 5, width_ratios=[1.0, 1.0, 1.0, 0.045, 0.045], wspace=0.18)
    axes = [fig.add_subplot(grid[0, idx]) for idx in range(3)]
    colorbar_axes = [fig.add_subplot(grid[0, 3]), fig.add_subplot(grid[0, 4])]

    for ax in axes:
        ax.set_facecolor("#fbf8f3")
        for spine in ax.spines.values():
            spine.set_color("#d8d1c2")
            spine.set_linewidth(1.0)

    overlay = _draw_channel_overlay(
        axes[0],
        positions,
        edge_index,
        obstacle,
        source_idx,
        dest_idx,
        channel_values=final["channel"],
        title="Channel Corridor",
        show_links=show_links,
        links_alpha=links_alpha * 0.5,
        links_width=links_width,
        channel_label="Channel",
        obstacle_label="Obstacles",
        source_label="Source",
        dest_label="Dest",
        show_legend=False,
    )

    source_distance = _draw_scalar_field_panel(
        axes[1],
        positions,
        edge_index,
        obstacle,
        source_idx,
        dest_idx,
        final["dist_src"],
        cmap="viridis",
        title="Weighted Distance From Source",
        show_links=show_links,
        links_alpha=links_alpha * 0.5,
        links_width=links_width,
    )
    distance_bar = fig.colorbar(source_distance, cax=colorbar_axes[0])
    distance_bar.set_label("Distance", fontsize=10, color="#3a342b")
    distance_bar.ax.tick_params(colors="#3a342b")
    colorbar_axes[0].set_facecolor("#f5f1e8")

    path_sum = _draw_scalar_field_panel(
        axes[2],
        positions,
        edge_index,
        obstacle,
        source_idx,
        dest_idx,
        final["sum"],
        cmap="plasma",
        title="Combined Path Distance",
        show_links=show_links,
        links_alpha=links_alpha * 0.5,
        links_width=links_width,
    )
    sum_bar = fig.colorbar(path_sum, cax=colorbar_axes[1])
    sum_bar.set_label("Distance", fontsize=10, color="#3a342b")
    sum_bar.ax.tick_params(colors="#3a342b")
    colorbar_axes[1].set_facecolor("#f5f1e8")

    legend_handles = []
    legend_labels = []
    for handle, label in (
        (overlay["channel_scatter"], "Channel"),
        (overlay["obstacle_scatter"], "Obstacles"),
        (overlay["source_star"], "Source"),
        (overlay["dest_star"], "Target"),
    ):
        if handle is not None:
            legend_handles.append(handle)
            legend_labels.append(label)

    if legend_handles:
        fig.legend(
            legend_handles,
            legend_labels,
            loc="upper center",
            bbox_to_anchor=(0.5, 0.94),
            ncol=len(legend_handles),
            frameon=True,
            facecolor="#fffaf0",
            edgecolor="#d8d1c2",
            fontsize=10,
        )

    fig.suptitle(
        "Channel Through Alternating Walls",
        fontsize=18,
        fontweight="semibold",
        color="#2f2a24",
        y=0.995,
    )
    fig.text(
        0.5,
        0.955,
        "Weighted shortest-path distance field through a narrow zig-zag corridor",
        ha="center",
        va="center",
        fontsize=11,
        color="#6a6155",
    )
    fig.subplots_adjust(left=0.04, right=0.98, bottom=0.11, top=0.83)
    _ensure_parent_dir(output_path)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Saved {output_path}")
    plt.close(fig)


def plot_channel_evolution(
    positions_by_round: dict[int, torch.Tensor],
    values_by_round: dict[int, torch.Tensor],
    edge_index_by_round: dict[int, torch.Tensor],
    obstacle: torch.Tensor,
    source_idx: int,
    dest_idx: int,
    output_dir: str = "generated/channel_random_evolution_frames",
    show_links: bool = True,
    links_alpha: float = 0.12,
    links_width: float = 0.5,
) -> None:
    if plt is None:
        print("matplotlib not available; skipping evolution plot")
        return

    rounds = sorted(set(positions_by_round) & set(values_by_round) & set(edge_index_by_round))
    if not rounds:
        print("no overlapping rounds to plot")
        return

    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    for round_idx in rounds:
        fig, ax = plt.subplots(figsize=(8, 7))
        _draw_channel_overlay(
            ax,
            positions_by_round[round_idx],
            edge_index_by_round[round_idx],
            obstacle,
            source_idx,
            dest_idx,
            channel_values=values_by_round[round_idx],
            title="Channel Evolution",
            show_links=show_links,
            links_alpha=links_alpha,
            links_width=links_width,
            node_size=40,
            channel_size=60,
            obstacle_size=70,
            source_ring_size=150,
            source_star_size=120,
            dest_ring_size=150,
            dest_star_size=120,
        )
        fig.tight_layout()
        frame_path = out_path / f"frame_{round_idx:04d}.png"
        plt.savefig(frame_path, dpi=150, bbox_inches="tight")
        plt.close(fig)

    print(f"Saved {len(rounds)} evolution frames to {out_path}")


def export_channel_gif(
    positions_by_round: dict[int, torch.Tensor],
    values_by_round: dict[int, torch.Tensor],
    edge_index_by_round: dict[int, torch.Tensor],
    obstacle: torch.Tensor,
    source_idx: int,
    dest_idx: int,
    output_path: str = "generated/channel_random_evolution.gif",
    fps: int = 10,
    show_links: bool = True,
    links_alpha: float = 0.12,
    links_width: float = 0.5,
) -> None:
    if plt is None:
        print("matplotlib not available; skipping GIF export")
        return

    rounds = sorted(set(positions_by_round) & set(values_by_round) & set(edge_index_by_round))
    if not rounds:
        print("no overlapping rounds to animate")
        return

    def render_frame(ax, frame_idx: int) -> None:
        round_idx = rounds[frame_idx]
        _draw_channel_overlay(
            ax,
            positions_by_round[round_idx],
            edge_index_by_round[round_idx],
            obstacle,
            source_idx,
            dest_idx,
            channel_values=values_by_round[round_idx],
            title="Channel Evolution",
            show_links=show_links,
            links_alpha=links_alpha,
            links_width=links_width,
            node_size=50,
            channel_size=70,
            obstacle_size=80,
            source_ring_size=200,
            source_star_size=150,
            dest_ring_size=200,
            dest_star_size=150,
            show_legend=False,
        )

    try:
        _ensure_parent_dir(output_path)
        save_gif(render_frame, list(range(len(rounds))), output_path, fps=fps, figsize=(8, 7))
    except Exception as exc:
        print(f"Failed to export GIF: {exc}")


def _select_destination_index(
    scenario: SpatialScenario, source_idx: int, requested_dest: int,
) -> int:
    if requested_dest != -1:
        return requested_dest

    pos = scenario.positions.detach().cpu()
    corner_score = (1.0 - pos[:, 0]) + pos[:, 1]
    dest_idx = torch.argmin(corner_score).item()
    dest_pos = pos[dest_idx]
    print(
        "Auto-selected destination node: "
        f"{dest_idx} (bottom-right at x={dest_pos[0]:.3f}, y={dest_pos[1]:.3f})"
    )
    return dest_idx


def _select_source_index(positions: torch.Tensor, requested_source: int) -> int:
    if requested_source != -1:
        return requested_source

    pos = positions.detach().cpu()
    corner_score = pos[:, 0] - pos[:, 1]
    source_idx = torch.argmin(corner_score).item()
    source_pos = pos[source_idx]
    print(
        "Auto-selected source node: "
        f"{source_idx} (top-left at x={source_pos[0]:.3f}, y={source_pos[1]:.3f})"
    )
    return source_idx


def main():  # noqa: PLR0915
    args = parse_args()
    device = get_device(args.device)
    torch.manual_seed(args.seed)

    positions = torch.rand(args.num_nodes, 2, device=device)

    if args.radius is None:
        # Heuristic: aim for average degree ~22 in a unit square
        # k = n * pi * r^2  => r = sqrt(k / (n * pi))
        args.radius = float(np.sqrt(18.0 / (args.num_nodes * np.pi)))
        print(f"Auto-computed radius: {args.radius:.4f} (targeting average degree ~22)")

    scenario = SpatialScenario(
        positions=positions,
        edge_radius=args.radius,
        edge_weight_mode="distance",
        device=device,
    )
    engine = SimulationEngine.from_scenario(scenario)

    source_idx = _select_source_index(positions, args.source)
    source = scenario.marker(source_idx)
    dest_idx = _select_destination_index(scenario, source_idx, args.dest)
    dest = scenario.marker(dest_idx)

    obstacle = create_obstacle_mask(positions, args.obstacle_ratio)
    print(f"Obstacle nodes: {obstacle.sum().item()} / {args.num_nodes}")

    if not args.no_gif:
        record_rounds = build_record_rounds(args.rounds, args.record_every)
    else:
        record_rounds = {args.rounds - 1}

    recorder = SnapshotRecorder(
        state_fields=["dist_src", "dist_dst", "_gc_dist_channel"],
        capture_output=True,
        record_rounds=record_rounds,
    )

    def program(runtime):
        src = source.to(runtime.scenario.device)
        dst = dest.to(runtime.scenario.device)
        obs = obstacle.to(runtime.scenario.device)

        return branch(
            ~obs,
            lambda: channel_body(src, dst, args.tolerance),
            lambda: field.of(0.0),
            branch_name="obstacle",
        )

    start_time = time.perf_counter()
    _output, _runtime = engine.run(
        rounds=args.rounds,
        program=program,
        signals={"source": source, "dest": dest, "obstacle": obstacle},
        recorder=recorder,
    )
    elapsed_time = time.perf_counter() - start_time

    snapshots = build_snapshot_payloads(recorder.records, num_nodes=args.num_nodes)
    values_by_round = {round_idx: payload["channel"] for round_idx, payload in snapshots.items()}
    positions_by_round = dict.fromkeys(snapshots, scenario.positions)
    edge_index_by_round = dict.fromkeys(snapshots, scenario.edge_index)

    final_step = args.rounds - 1
    final = snapshots.get(final_step, snapshots[max(snapshots)])
    sd_dist = final["dist_src"][dest_idx].item()

    print("=== Channel with Fixed Random Nodes ===")
    print(f"nodes={args.num_nodes} rounds={args.rounds} radius={args.radius}")
    print(f"source={source_idx} dest={dest_idx}")
    print(
        f"obstacles={obstacle.sum().item()} tolerance={args.tolerance} edge_weight_mode=distance"
    )
    print(f"Weighted distance source->dest: {sd_dist:.2f}")
    print(f"Channel nodes: {count_channel_nodes(final)}")
    print(f"Simulation time: {elapsed_time:.3f}s")
    print("recorded rounds:", sorted(recorder.records.keys()))

    if not args.no_viz:
        plot_channel_setup(
            scenario.positions,
            scenario.edge_index,
            obstacle,
            source_idx,
            dest_idx,
            output_path=f"{args.viz_prefix}_setup.png",
            show_links=not args.hide_links,
            links_alpha=args.links_alpha,
            links_width=args.links_width,
        )

        plot_channel_final(
            scenario.positions,
            scenario.edge_index,
            obstacle,
            source_idx,
            dest_idx,
            final,
            output_path=f"{args.viz_prefix}_final.png",
            show_links=not args.hide_links,
            links_alpha=args.links_alpha,
            links_width=args.links_width,
        )

        if not args.no_gif:
            plot_channel_evolution(
                positions_by_round,
                values_by_round,
                edge_index_by_round,
                obstacle,
                source_idx,
                dest_idx,
                output_dir=f"{args.viz_prefix}_evolution_frames",
                show_links=not args.hide_links,
                links_alpha=args.links_alpha,
                links_width=args.links_width,
            )

        # Always render the last frame as a standalone image in evolution style
        last_round = max(values_by_round.keys())
        fig, ax = plt.subplots(figsize=(8, 7))
        _draw_channel_overlay(
            ax,
            positions_by_round[last_round],
            edge_index_by_round[last_round],
            obstacle,
            source_idx,
            dest_idx,
            channel_values=values_by_round[last_round],
            title="",
            show_links=not args.hide_links,
            links_alpha=args.links_alpha,
            links_width=args.links_width,
            node_size=50,
            channel_size=70,
            obstacle_size=80,
            source_ring_size=200,
            source_star_size=150,
            dest_ring_size=200,
            dest_star_size=150,
            show_legend=False,
        )
        last_frame_path = f"{args.viz_prefix}_last_frame.png"
        plt.savefig(last_frame_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved last frame to {last_frame_path}")

        if not args.no_gif:
            export_channel_gif(
                positions_by_round,
                values_by_round,
                edge_index_by_round,
                obstacle,
                source_idx,
                dest_idx,
                output_path=f"{args.viz_prefix}.gif",
                fps=args.gif_fps,
                show_links=not args.hide_links,
                links_alpha=args.links_alpha,
                links_width=args.links_width,
            )


if __name__ == "__main__":
    main()
