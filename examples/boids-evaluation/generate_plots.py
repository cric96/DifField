#!/usr/bin/env python3
"""Standalone script to regenerate boids plots from saved evaluation data."""

import argparse
import math
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))
if str(ROOT / "examples") not in sys.path:
    sys.path.insert(0, str(ROOT / "examples"))

import matplotlib  # noqa: E402

matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
from shared.metrics import mean, std  # noqa: E402
from shared.plotting import (  # noqa: E402
    FIG_WIDTH_1COL,
    apply_paper_style,
    export_moving_gif,
    panel_label,
    plot_moving_snapshots,
    plot_node_trajectories,
)
from shared.plotting import savefig as save_figure  # noqa: E402
from shared.plotting.moving import _draw_trajectory_on_ax, _get_identity_colors  # noqa: E402
from shared.plotting.style import AQUA, BLUE, ORANGE  # noqa: E402

apply_paper_style()

FIGSIZE_1COL = (FIG_WIDTH_1COL, 2.4)
# fixed, CVD-safe style per series (colour + marker + linestyle: two channels)
SERIES_STYLE = {
    "train": (BLUE, "o", "-"),
    "val": (ORANGE, "s", "--"),
    "w_sep": (BLUE, "o", "-"),
    "w_align": (ORANGE, "s", "--"),
    "w_cohesion": (AQUA, "D", "-."),
}
SERIES_LABEL = {
    "train": "train",
    "val": "validation",
    "w_sep": "$w_{\\mathrm{sep}}$",
    "w_align": "$w_{\\mathrm{align}}$",
    "w_cohesion": "$w_{\\mathrm{coh}}$",
}


def _plot_band(ax, x_vals, mean_vals, std_vals, series):
    valid_indices = [i for i, v in enumerate(mean_vals) if not math.isnan(v)]
    if not valid_indices:
        return
    x_valid = [x_vals[i] for i in valid_indices]
    m_valid = [mean_vals[i] for i in valid_indices]
    s_valid = [std_vals[i] for i in valid_indices]

    color, marker, linestyle = SERIES_STYLE.get(series, (None, "o", "-"))
    line, = ax.plot(
        x_valid, m_valid, linewidth=1.8, label=SERIES_LABEL.get(series, series),
        marker=marker, markersize=4, markevery=max(1, len(x_valid) // 12),
        linestyle=linestyle, color=color,
    )
    if len(x_valid) > 1:
        lower = [v - d for v, d in zip(m_valid, s_valid, strict=False)]
        upper = [v + d for v, d in zip(m_valid, s_valid, strict=False)]
        ax.fill_between(
            x_valid, lower, upper, alpha=0.18, color=line.get_color(), linewidth=0,
        )

def plot_train_val_metrics(
    histories: list[dict[str, list[float]]],
    out_dir: Path,
) -> None:
    if plt is None or not histories:
        return

    epochs = histories[0].get("epoch", [])
    if not epochs:
        return

    def series_mean_std(key: str):
        series = [h[key] for h in histories if h.get(key)]
        if not series:
            return None
        min_len = min(len(v) for v in series)
        trimmed = [v[:min_len] for v in series]

        def _compute(agg_fn):
            return [
                agg_fn([v[i] for v in trimmed])
                if not all(math.isnan(v[i]) for v in trimmed)
                else float("nan")
                for i in range(min_len)
            ]

        return _compute(mean), _compute(std)

    # --- Training Metrics ---
    train_loss = series_mean_std("total_loss")
    train_pos_err = series_mean_std("pos_error")
    val_loss = series_mean_std("val_total_loss")
    val_pos_err = series_mean_std("val_pos_error")

    if not (train_loss or val_loss):
        return

    fig, ax = plt.subplots(figsize=FIGSIZE_1COL)
    for series, data in (("train", train_loss), ("val", val_loss)):
        if data:
            min_len = min(len(epochs), len(data[0]))
            _plot_band(ax, epochs[:min_len], data[0][:min_len], data[1][:min_len], series)
    ax.set_xlabel("epoch")
    ax.set_ylabel("total loss")
    ax.grid(alpha=0.5, which="both")
    ax.legend()
    save_figure(fig, out_dir / "metrics_loss.png")

    fig, ax = plt.subplots(figsize=FIGSIZE_1COL)
    for series, data in (("train", train_pos_err), ("val", val_pos_err)):
        if data:
            min_len = min(len(epochs), len(data[0]))
            _plot_band(ax, epochs[:min_len], data[0][:min_len], data[1][:min_len], series)
    ax.set_xlabel("epoch")
    ax.set_ylabel("position error (L2)")
    ax.grid(alpha=0.5)
    ax.legend()
    save_figure(fig, out_dir / "metrics_pos_error.png")

def plot_parameter_recovery_bands(
    histories: list[dict[str, list[float]]],
    teacher_params: dict[str, float],
    out_dir: Path,
) -> None:
    if plt is None or not histories:
        return

    epochs = histories[0].get("epoch", [])
    if not epochs:
        return

    fig_abs, ax_abs = plt.subplots(figsize=FIGSIZE_1COL)
    fig_rel, ax_rel = plt.subplots(figsize=FIGSIZE_1COL)
    plotted = False

    for name in teacher_params:
        abs_key = f"{name}_abs_error"
        rel_key = f"{name}_rel_error"
        if abs_key not in histories[0] or rel_key not in histories[0]:
            continue

        abs_series = [h[abs_key] for h in histories if h.get(abs_key)]
        rel_series = [h[rel_key] for h in histories if h.get(rel_key)]
        if not abs_series or not rel_series:
            continue

        min_len = min(*(len(v) for v in abs_series), *(len(v) for v in rel_series))
        abs_trimmed = [v[:min_len] for v in abs_series]
        rel_trimmed = [v[:min_len] for v in rel_series]

        abs_mean = [mean([v[i] for v in abs_trimmed]) for i in range(min_len)]
        abs_std = [std([v[i] for v in abs_trimmed]) for i in range(min_len)]
        rel_mean = [mean([v[i] for v in rel_trimmed]) for i in range(min_len)]
        rel_std = [std([v[i] for v in rel_trimmed]) for i in range(min_len)]

        _plot_band(ax_abs, epochs[:min_len], abs_mean, abs_std, name)
        _plot_band(ax_rel, epochs[:min_len], rel_mean, rel_std, name)
        plotted = True

    if not plotted:
        plt.close(fig_abs)
        plt.close(fig_rel)
        return

    ax_abs.set_xlabel("epoch")
    ax_abs.set_ylabel("absolute weight error")
    ax_abs.grid(alpha=0.5)
    ax_abs.legend()
    save_figure(fig_abs, out_dir / "param_recovery_abs.png")

    ax_rel.set_xlabel("epoch")
    ax_rel.set_ylabel("relative weight error (%)")
    ax_rel.grid(alpha=0.5)
    ax_rel.legend()
    save_figure(fig_rel, out_dir / "param_recovery_rel.png")

def generate_boids_plots(  # noqa: PLR0915
    data: dict, seed: int, seed_dir: Path, hide_links: bool,
    links_alpha: float, links_width: float, gif_fps: int,
):
    pred_pos = data["pred_pos"]
    pred_vel = data["pred_vel"]
    eval_trace_pos = data["eval_trace_pos"]
    pred_edge_seq = data["pred_edge_seq"]
    highlight_idx = data["highlight_node"]
    rounds = data["rounds"]

    positions_over_time = [pred_pos[r] for r in range(rounds)]

    plot_node_trajectories(
        positions_over_time=positions_over_time,
        source_idx=highlight_idx,
        output_path=str(seed_dir / "trajectories.png"),
        show_source=False,
        phantom_pos_seq=eval_trace_pos,
    )

    positions_by_round = {r: pred_pos[r] for r in range(rounds)}
    values_by_round = {r: pred_vel[r].norm(dim=1) for r in range(rounds)}
    edge_index_by_round = {r: pred_edge_seq[r] for r in range(rounds)}

    plot_moving_snapshots(
        positions_by_round=positions_by_round,
        values_by_round=values_by_round,
        edge_index_by_round=edge_index_by_round,
        source_idx=highlight_idx,
        output_path=str(seed_dir / "validation_positions.png"),
        show_links=not hide_links,
        links_alpha=links_alpha,
        links_width=links_width,
        show_source=False,
    )

    export_moving_gif(
        positions_by_round=positions_by_round,
        values_by_round=values_by_round,
        edge_index_by_round=edge_index_by_round,
        source_idx=highlight_idx,
        output_path=str(seed_dir / "validation_final.gif"),
        title=f"Final model prediction (seed {seed})",
        fps=gif_fps,
        show_links=not hide_links,
        links_alpha=links_alpha,
        links_width=links_width,
        show_source=False,
        phantom_pos_seq=eval_trace_pos,
    )

    # Progression plot: start, middle, end, reference
    if "eval_trajectories" in data and plt is not None:
        eval_trajectories = data["eval_trajectories"]
        epochs = sorted(eval_trajectories.keys())

        if len(epochs) > 0:
            start_ep = epochs[0]
            end_ep = epochs[-1]

            # Find a middle epoch
            mid_ep = epochs[(epochs[-1] - epochs[0]) // 15]
            if len(epochs) >= 3:
                print(f"Using epoch {mid_ep} as middle point for progression plot.")
                print(len(epochs))
            elif len(epochs) == 2:
                mid_ep = epochs[0]

            # Progression plots: grid and line versions
            id_colors = _get_identity_colors(data["num_nodes"])

            def _plot_panel(ax, pos_seq, tag, bg_color="white"):
                traj = pos_seq.numpy()
                _draw_trajectory_on_ax(
                    ax, traj,
                    num_rounds=rounds,
                    num_nodes=data["num_nodes"],
                    id_colors=id_colors,
                    color_by_id=True,
                    show_source=False,
                    source_idx=highlight_idx
                )
                panel_label(ax, tag)
                ax.set_facecolor(bg_color)
                # Remove legends to avoid clutter on subplots
                if ax.get_legend():
                    ax.get_legend().remove()

            # 1. Grid version (2x2): (a) start, (b) mid, (c) end, (d) teacher reference
            fig_grid, axes_grid = plt.subplots(2, 2, figsize=(10, 10))
            _plot_panel(axes_grid[0, 0], eval_trajectories[start_ep], "a")
            _plot_panel(axes_grid[0, 1], eval_trajectories[mid_ep], "b")
            _plot_panel(axes_grid[1, 0], eval_trajectories[end_ep], "c")
            _plot_panel(axes_grid[1, 1], eval_trace_pos, "d", bg_color="#f4f8ff")

            fig_grid.tight_layout()
            out_path_grid = seed_dir / "progression_trajectories_grid.png"
            fig_grid.savefig(str(out_path_grid), dpi=150)
            plt.close(fig_grid)

            # 2. Line version (1x4): (a) start, (b) mid, (c) end, (d) teacher reference
            fig_line, axes_line = plt.subplots(1, 4, figsize=(18, 5))
            _plot_panel(axes_line[0], eval_trajectories[start_ep], "a")
            _plot_panel(axes_line[1], eval_trajectories[mid_ep], "b")
            _plot_panel(axes_line[2], eval_trajectories[end_ep], "c")
            _plot_panel(axes_line[3], eval_trace_pos, "d", bg_color="#f4f8ff")

            fig_line.tight_layout()
            out_path_line = seed_dir / "progression_trajectories_line.png"
            fig_line.savefig(str(out_path_line), dpi=150)
            plt.close(fig_line)

def main():
    parser = argparse.ArgumentParser(
        description="Regenerate plots for boids evaluation from extracted data.",
    )
    parser.add_argument(
        "--data-dir", type=str, default="generated/boids-evaluation",
        help="Base directory containing seed_X subfolders",
    )
    parser.add_argument("--hide-links", action="store_true")
    parser.add_argument("--links-alpha", type=float, default=0.15)
    parser.add_argument("--links-width", type=float, default=0.6)
    parser.add_argument("--gif-fps", type=int, default=8)

    args = parser.parse_args()

    base_dir = Path(args.data_dir)
    if not base_dir.exists():
        print(f"Directory {base_dir} does not exist.")
        return

    # Find all eval_data.pt files
    data_files = list(base_dir.rglob("eval_data.pt"))
    if not data_files:
        print(f"No eval_data.pt files found in {base_dir}.")
        return

    print(f"Found {len(data_files)} data files.")
    all_histories = []
    teacher_params = None

    for data_path in data_files:
        seed_dir = data_path.parent
        # Try to extract seed from folder name, e.g., "seed_5"
        seed_str = seed_dir.name.replace("seed_", "")
        seed = int(seed_str) if seed_str.isdigit() else 0

        print(f"Processing {data_path}...")
        data = torch.load(data_path, map_location="cpu", weights_only=False)

        if "history" in data:
            all_histories.append(data["history"])
        if "teacher_params" in data and teacher_params is None:
            teacher_params = data["teacher_params"]

        generate_boids_plots(
            data=data,
            seed=seed,
            seed_dir=seed_dir,
            hide_links=args.hide_links,
            links_alpha=args.links_alpha,
            links_width=args.links_width,
            gif_fps=args.gif_fps
        )
        print(f"Plots regenerated in {seed_dir}")

    if all_histories and teacher_params:
        print(f"Generating aggregated metric plots in {base_dir}...")
        plot_train_val_metrics(all_histories, base_dir)
        plot_parameter_recovery_bands(all_histories, teacher_params, base_dir)

if __name__ == "__main__":
    main()
