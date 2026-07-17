#!/usr/bin/env python3
"""Collect-cast with OR: circle of True nodes propagating toward a sink.

Demonstrates density convergence: as node count increases (with constant
average degree), the boolean OR collect-cast better approximates the ideal
continuous result.
"""

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

from shared.plotting.common import LineCollection, plt  # noqa: E402
from shared.plotting.style import (  # noqa: E402
    FIG_WIDTH_1COL,
    FIG_WIDTH_2COL,
    MUTED,
    apply_paper_style,
    savefig,
)

apply_paper_style()

from diffield.dsl import collect_cast, field, gradient  # noqa: E402
from diffield.sim import SimulationEngine, SpatialScenario  # noqa: E402
from diffield.sim.recording import SnapshotRecorder  # noqa: E402
from diffield.utils import get_device  # noqa: E402

CIRCLE_CENTER = np.array([0.5, 0.5])
CIRCLE_RADIUS = 0.2
SINK_CORNER = np.array([0.05, 0.05])
TRUE_COLOR = "#FFD700"
FALSE_COLOR = "steelblue"
SINK_COLOR = "red"


@dataclass(frozen=True)
class VizStyle:
    node_size: int = 30
    node_alpha: float = 0.7
    sink_size: int = 250
    sink_alpha: float = 0.95
    edge_color: str = "gray"
    edge_alpha: float = 0.08
    edge_width: float = 0.4
    bg_color: str = "#fafafa"


STYLE = VizStyle()


def parse_args():
    parser = argparse.ArgumentParser(
        description="Collect-cast OR with varying node density"
    )
    parser.add_argument(
        "--node-counts",
        type=int,
        nargs="+",
        default=[1024, 4096, 10000, 19600],
    )
    parser.add_argument("--rounds", type=int, default=200)
    parser.add_argument("--radius-scale", type=float, default=20.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--record-every", type=int, default=10)
    parser.add_argument("--output-dir", type=str, default="generated/collect_cast")
    parser.add_argument("--no-viz", action="store_true")
    parser.add_argument("--hide-links", action="store_true")
    parser.add_argument("--links-alpha", type=float, default=0.0)
    parser.add_argument("--links-width", type=float, default=0.4)
    parser.add_argument("--device", type=str, default="")
    return parser.parse_args()


def _to_numpy(tensor: torch.Tensor) -> np.ndarray:
    return tensor.detach().cpu().numpy()


def _ensure_parent_dir(output_path: str) -> None:
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)


def _compute_radius(num_nodes: int, scale: float) -> float:
    return float(np.sqrt(scale / (num_nodes * np.pi)))


def _find_sink_index(positions: torch.Tensor) -> int:
    sink = torch.tensor(
        SINK_CORNER, dtype=positions.dtype, device=positions.device,
    ).reshape(1, 2)
    dist = torch.cdist(positions, sink)
    return int(torch.argmin(dist).item())


def _circle_mask(positions: torch.Tensor) -> torch.Tensor:
    center = torch.tensor(
        CIRCLE_CENTER, dtype=positions.dtype, device=positions.device,
    ).reshape(1, 2)
    dist = torch.cdist(positions, center)
    return (dist.squeeze(-1) <= CIRCLE_RADIUS).to(positions.device)


def _edge_segments(positions: np.ndarray, edge_index: np.ndarray):
    return [
        (
            (float(positions[src, 0]), float(positions[src, 1])),
            (float(positions[tgt, 0]), float(positions[tgt, 1])),
        )
        for src, tgt in zip(edge_index[0], edge_index[1], strict=False)
    ]


def _draw_edges(ax, positions, edge_index, *, show_links, alpha, width, color="gray"):
    if not show_links or LineCollection is None:
        return
    segments = _edge_segments(positions, edge_index)
    if not segments:
        return
    collection = LineCollection(segments, colors=color, linewidths=width, alpha=alpha, zorder=1)
    ax.add_collection(collection)


def _finalize_axes(ax) -> None:
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)


def _draw_collect_cast_result(
    ax,
    positions: np.ndarray,
    edge_index: np.ndarray,
    result: np.ndarray,
    sink_idx: int,
    *,
    show_links: bool,
    links_alpha: float,
    links_width: float,
    node_size: int | None = None,
    sink_size: int | None = None,
):
    size = node_size or STYLE.node_size
    sk_size = sink_size or STYLE.sink_size
    is_true = result.astype(bool)
    is_false = ~is_true

    _draw_edges(
        ax, positions, edge_index,
        show_links=show_links, alpha=links_alpha, width=links_width,
    )

    if is_false.any():
        ax.scatter(
            positions[is_false, 0], positions[is_false, 1],
            c=FALSE_COLOR, s=size, alpha=STYLE.node_alpha, zorder=2,
        )
    if is_true.any():
        ax.scatter(
            positions[is_true, 0], positions[is_true, 1],
            c=TRUE_COLOR, s=size, alpha=STYLE.node_alpha, zorder=3,
        )

    ax.scatter(
        positions[sink_idx : sink_idx + 1, 0],
        positions[sink_idx : sink_idx + 1, 1],
        c=SINK_COLOR, s=sk_size, alpha=STYLE.sink_alpha,
        marker="*", zorder=5, edgecolors="#5c0d0d", linewidths=1.0,
    )

    _finalize_axes(ax)


def _compute_node_size(num_nodes: int, panel_size_inches: float) -> int:
    side = int(np.sqrt(num_nodes))
    spacing = panel_size_inches / side
    return max(2, min(50, int(spacing * 100)))


def _make_grid_positions(num_nodes: int, device: torch.device) -> torch.Tensor:
    side = int(np.sqrt(num_nodes))
    if side * side != num_nodes:
        raise ValueError(f"num_nodes={num_nodes} is not a perfect square")
    coords = np.linspace(0.02, 0.98, side)
    x, y = np.meshgrid(coords, coords)
    positions = np.stack([x.ravel(), y.ravel()], axis=-1)
    return torch.tensor(positions, dtype=torch.float32, device=device)


def run_single_density(
    num_nodes: int,
    rounds: int,
    radius: float,
    seed: int,
    device: torch.device,
    record_rounds: set[int] | None = None,
):
    positions = _make_grid_positions(num_nodes, device)

    scenario = SpatialScenario(
        positions=positions,
        edge_radius=radius,
        edge_weight_mode="distance",
        device=device,
    )
    engine = SimulationEngine.from_scenario(scenario)

    sink_idx = _find_sink_index(positions)
    circle = _circle_mask(positions)

    def program(runtime):
        sink = runtime.signals["sink"]
        local = runtime.signals["local"]
        potential = gradient(sink, name="sink_dist")
        return collect_cast(
            potential,
            local,
            field.zeros(),
            torch.logical_or,
            name="or_collect",
        )

    recorder = SnapshotRecorder(
        state_fields=["_potential"],
        capture_output=True,
        record_rounds=record_rounds or {rounds - 1},
    )

    sink_field = scenario.zeros()
    sink_field[sink_idx] = 1.0
    local_field = circle.float()

    start = time.perf_counter()
    output, _runtime = engine.run(
        rounds=rounds,
        program=program,
        signals={"sink": sink_field, "local": local_field},
        recorder=recorder,
    )
    elapsed = time.perf_counter() - start

    return {
        "positions": _to_numpy(positions),
        "edge_index": _to_numpy(scenario.edge_index),
        "sink_idx": sink_idx,
        "output": _to_numpy(output),
        "true_count": int((_to_numpy(output) > 0.5).sum()),
        "total_nodes": num_nodes,
        "elapsed": elapsed,
        "recorder": recorder,
    }


def plot_comparison_separate(
    results: dict[int, dict],
    output_dir: str,
    *,
    show_links: bool,
    links_alpha: float,
    links_width: float,
):
    if plt is None:
        print("matplotlib not available; skipping comparison plots")
        return

    for n_nodes, res in sorted(results.items()):
        fig, ax = plt.subplots(figsize=(7, 7))
        node_size = _compute_node_size(n_nodes, panel_size_inches=7)
        _draw_collect_cast_result(
            ax,
            res["positions"],
            res["edge_index"],
            res["output"],
            res["sink_idx"],
            show_links=show_links,
            links_alpha=links_alpha,
            links_width=links_width,
            node_size=node_size,
        )
        fig.tight_layout()

        out_path = Path(output_dir) / f"collect_cast_comparison_N{n_nodes}.png"
        _ensure_parent_dir(str(out_path))
        plt.savefig(out_path, dpi=150, bbox_inches="tight")
        print(f"Saved comparison N={n_nodes:,} to {out_path}")
        plt.close(fig)


def plot_evolution_separate(
    results: dict[int, dict],
    output_dir: str,
    *,
    show_links: bool,
    links_alpha: float,
    links_width: float,
    snapshot_rounds: list[int],
):
    if plt is None:
        print("matplotlib not available; skipping evolution plots")
        return

    n_snapshots = len(snapshot_rounds)

    for n_nodes, res in sorted(results.items()):
        recorder = res["recorder"]
        snapshots = recorder.records
        rounds = sorted(snapshots.keys())
        positions = res["positions"]
        edge_index = res["edge_index"]
        sink_idx = res["sink_idx"]
        node_size = _compute_node_size(n_nodes, panel_size_inches=4)

        fig, axes = plt.subplots(1, n_snapshots, figsize=(4 * n_snapshots, 4))

        for col_idx, target_round in enumerate(snapshot_rounds):
            actual_round = min(rounds, key=lambda r: abs(r - target_round))
            snap = snapshots[actual_round]
            output_val = _to_numpy(snap["output"])

            _draw_collect_cast_result(
                axes[col_idx],
                positions,
                edge_index,
                output_val,
                sink_idx,
                show_links=show_links,
                links_alpha=links_alpha,
                links_width=links_width,
                node_size=node_size,
            )
            axes[col_idx].set_title(
                f"t = {actual_round}", fontsize=9.5, fontweight="normal", color=MUTED
            )

        out_path = Path(output_dir) / f"collect_cast_evolution_N{n_nodes}.png"
        _ensure_parent_dir(str(out_path))
        savefig(fig, out_path)


def plot_convergence(
    results: dict[int, dict],
    output_path: str,
    *,
    rounds: int,
):
    """One panel, one curve per density: true-node ratio over rounds.

    Density is ordinal, so the curves use a sequential (viridis) ramp — the
    same family as the field heatmaps elsewhere in the repo.
    """
    if plt is None:
        print("matplotlib not available; skipping convergence plot")
        return

    fig, ax = plt.subplots(figsize=(FIG_WIDTH_1COL, 2.4))
    cmap = plt.get_cmap("viridis")
    shades = [cmap(v) for v in np.linspace(0.15, 0.8, len(results))]
    markers = ("o", "s", "D", "^", "v", "P")

    for idx, (n_nodes, res) in enumerate(sorted(results.items())):
        recorder = res["recorder"]
        snapshots = recorder.records
        sorted_rounds = sorted(snapshots.keys())
        ratios = [
            int((_to_numpy(snapshots[r]["output"]) > 0.5).sum()) / n_nodes
            for r in sorted_rounds
        ]
        ax.plot(
            sorted_rounds, ratios,
            color=shades[idx], marker=markers[idx % len(markers)],
            markersize=4, markevery=max(1, len(sorted_rounds) // 10),
            linewidth=1.8, label=f"N = {n_nodes:,}",
        )

    ax.set_xlabel("round")
    ax.set_ylabel("true-node ratio")
    ax.set_xlim(0, rounds - 1)
    ax.legend()
    ax.grid(True, alpha=0.5)

    _ensure_parent_dir(output_path)
    savefig(fig, Path(output_path))


def plot_density_row(
    results: dict[int, dict],
    output_path: str,
    *,
    show_links: bool,
    links_alpha: float,
    links_width: float,
):
    """Paper composite: the final collected region at every density, one row.

    The visual argument of the example — the boolean OR collect-cast converges
    to the ideal continuous region as density grows — in a single full-width
    figure instead of four separate files.
    """
    if plt is None:
        print("matplotlib not available; skipping density row")
        return

    n_panels = len(results)
    panel_w = FIG_WIDTH_2COL / n_panels
    fig, axes = plt.subplots(1, n_panels, figsize=(FIG_WIDTH_2COL, panel_w * 1.25))

    for ax, (n_nodes, res) in zip(axes, sorted(results.items()), strict=True):
        node_size = _compute_node_size(n_nodes, panel_size_inches=panel_w)
        _draw_collect_cast_result(
            ax,
            res["positions"],
            res["edge_index"],
            res["output"],
            res["sink_idx"],
            show_links=show_links,
            links_alpha=links_alpha,
            links_width=links_width,
            node_size=node_size,
        )
        ax.set_title(f"N = {n_nodes:,}", fontsize=9.5, fontweight="normal", color=MUTED)

    _ensure_parent_dir(output_path)
    savefig(fig, Path(output_path))


def main():
    args = parse_args()
    device = get_device(args.device)

    results = {}

    for n_nodes in args.node_counts:
        radius = _compute_radius(n_nodes, args.radius_scale)
        print(f"\n{'='*60}")
        print(f"Running N = {n_nodes:,}  radius = {radius:.4f}")
        print(f"{'='*60}")

        frame_budget = max(8, min(20, args.rounds // args.record_every + 4))
        front_loaded = np.square(np.linspace(0.0, 1.0, frame_budget))
        record_set = {round(r) for r in front_loaded * (args.rounds - 1)}
        record_set.add(args.rounds - 1)

        res = run_single_density(
            num_nodes=n_nodes,
            rounds=args.rounds,
            radius=radius,
            seed=args.seed,
            device=device,
            record_rounds=record_set,
        )
        results[n_nodes] = res

        tc = res["true_count"]
        tn = res["total_nodes"]
        print(f"  True nodes: {tc:,} / {tn:,} ({tc / tn:.1%})")
        print(f"  Time: {res['elapsed']:.2f}s")

    if not args.no_viz:
        out_dir = Path(args.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        plot_comparison_separate(
            results,
            str(out_dir),
            show_links=not args.hide_links,
            links_alpha=args.links_alpha,
            links_width=args.links_width,
        )

        snapshot_rounds = [0, args.rounds // 5, args.rounds // 4, args.rounds - 1]
        plot_evolution_separate(
            results,
            str(out_dir),
            show_links=not args.hide_links,
            links_alpha=args.links_alpha,
            links_width=args.links_width,
            snapshot_rounds=snapshot_rounds,
        )

        plot_density_row(
            results,
            str(out_dir / "collect_cast_density.png"),
            show_links=not args.hide_links,
            links_alpha=args.links_alpha,
            links_width=args.links_width,
        )

        plot_convergence(
            results,
            str(out_dir / "collect_cast_convergence.png"),
            rounds=args.rounds,
        )


if __name__ == "__main__":
    main()
