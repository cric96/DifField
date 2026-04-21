#!/usr/bin/env python3
"""Standalone script to regenerate boids plots from saved evaluation data."""

import argparse
from pathlib import Path
import sys
import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))
if str(ROOT / "examples") not in sys.path:
    sys.path.insert(0, str(ROOT / "examples"))

from shared.plotting import export_moving_gif, plot_moving_snapshots, plot_node_trajectories
from shared.plotting.moving import _draw_trajectory_on_ax, _get_identity_colors
import numpy as np

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
except ImportError:
    plt = None

import sys
from shared.metrics import mean, std

def _plot_band(ax, x_vals, mean_vals, std_vals, label, color=None):
    import math
    valid_indices = [i for i, v in enumerate(mean_vals) if not math.isnan(v)]
    if not valid_indices:
        return
    x_valid = [x_vals[i] for i in valid_indices]
    m_valid = [mean_vals[i] for i in valid_indices]
    s_valid = [std_vals[i] for i in valid_indices]
    
    line, = ax.plot(x_valid, m_valid, linewidth=2.0, label=label, marker="o", markersize=3, color=color)
    if len(x_valid) > 1:
        lower = [v - d for v, d in zip(m_valid, s_valid)]
        upper = [v + d for v, d in zip(m_valid, s_valid)]
        ax.fill_between(x_valid, lower, upper, alpha=0.18, color=line.get_color())

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
        series = [h[key] for h in histories if key in h and h[key]]
        if not series:
            return None
        min_len = min(len(v) for v in series)
        trimmed = [v[:min_len] for v in series]
        import math
        return [mean([v[i] for v in trimmed]) if not all(math.isnan(v[i]) for v in trimmed) else float("nan") for i in range(min_len)], \
               [std([v[i] for v in trimmed]) if not all(math.isnan(v[i]) for v in trimmed) else float("nan") for i in range(min_len)]

    # --- Training Metrics ---
    train_loss = series_mean_std("total_loss")
    train_pos_err = series_mean_std("pos_error")
    val_loss = series_mean_std("val_total_loss")
    val_pos_err = series_mean_std("val_pos_error")

    if not (train_loss or val_loss):
        return

    # Use larger fonts globally for these plots
    plt.rcParams.update({'font.size': 14})

    # Plot Total Loss separately (squarish)
    fig, ax = plt.subplots(figsize=(5, 5))
    if train_loss:
        min_len = min(len(epochs), len(train_loss[0]))
        _plot_band(ax, epochs[:min_len], train_loss[0][:min_len], train_loss[1][:min_len], "train", color="tab:blue")
    if val_loss:
        min_len = min(len(epochs), len(val_loss[0]))
        _plot_band(ax, epochs[:min_len], val_loss[0][:min_len], val_loss[1][:min_len], "val", color="tab:orange")
    
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Total Loss")
    ax.grid(alpha=0.25, which="both")
    ax.legend(loc="best")
    fig.tight_layout()
    output_path = out_dir / "metrics_loss.png"
    fig.savefig(str(output_path), dpi=150)
    plt.close(fig)
    print(f"Saved {output_path}")

    # Plot Position Error separately (squarish)
    fig, ax = plt.subplots(figsize=(5, 5))
    if train_pos_err:
        min_len = min(len(epochs), len(train_pos_err[0]))
        _plot_band(ax, epochs[:min_len], train_pos_err[0][:min_len], train_pos_err[1][:min_len], "train", color="tab:blue")
    if val_pos_err:
        min_len = min(len(epochs), len(val_pos_err[0]))
        _plot_band(ax, epochs[:min_len], val_pos_err[0][:min_len], val_pos_err[1][:min_len], "val", color="tab:orange")

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Position Error (L2)")
    ax.grid(alpha=0.25)
    ax.legend(loc="best")
    fig.tight_layout()
    output_path = out_dir / "metrics_pos_error.png"
    fig.savefig(str(output_path), dpi=150)
    plt.close(fig)
    print(f"Saved {output_path}")
    
    # Reset font size
    plt.rcParams.update({'font.size': 10})

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

    # Use larger fonts globally
    plt.rcParams.update({'font.size': 14})

    fig_abs, ax_abs = plt.subplots(figsize=(5, 5))
    fig_rel, ax_rel = plt.subplots(figsize=(5, 5))
    plotted = False

    for name in teacher_params:
        abs_key = f"{name}_abs_error"
        rel_key = f"{name}_rel_error"
        if abs_key not in histories[0] or rel_key not in histories[0]:
            continue

        abs_series = [h[abs_key] for h in histories if abs_key in h and h[abs_key]]
        rel_series = [h[rel_key] for h in histories if rel_key in h and h[rel_key]]
        if not abs_series or not rel_series:
            continue

        min_len = min(min(len(v) for v in abs_series), min(len(v) for v in rel_series))
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

    ax_abs.set_xlabel("Epoch")
    ax_abs.set_ylabel("Absolute Error")
    ax_abs.grid(alpha=0.25)
    ax_abs.legend(loc="best")
    fig_abs.tight_layout()
    out_abs = out_dir / "param_recovery_abs.png"
    fig_abs.savefig(str(out_abs), dpi=150)
    plt.close(fig_abs)
    print(f"Saved {out_abs}")

    ax_rel.set_xlabel("Epoch")
    ax_rel.set_ylabel("Relative Error (%)")
    ax_rel.grid(alpha=0.25)
    ax_rel.legend(loc="best")
    fig_rel.tight_layout()
    out_rel = out_dir / "param_recovery_rel.png"
    fig_rel.savefig(str(out_rel), dpi=150)
    plt.close(fig_rel)
    print(f"Saved {out_rel}")

    plt.rcParams.update({'font.size': 10})

def generate_boids_plots(data: dict, seed: int, seed_dir: Path, hide_links: bool, links_alpha: float, links_width: float, gif_fps: int):
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
        title=f"Boids trajectories (seed {seed})",
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
        title=f"Validation position over time (seed {seed})",
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
                
            # Progression plot: start, middle, end, reference
            fig, axes = plt.subplots(2, 2, figsize=(8, 8))
            id_colors = _get_identity_colors(data["num_nodes"])
            
            def _plot_panel(ax, pos_seq, title, bg_color="white"):
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
                ax.set_title(title, fontsize=14)
                ax.set_facecolor(bg_color)
                # Remove legends to avoid clutter on subplots
                if ax.get_legend():
                    ax.get_legend().remove()
                    
            _plot_panel(axes[0, 0], eval_trajectories[start_ep], f"Start (Epoch {start_ep + 1})")
            _plot_panel(axes[0, 1], eval_trajectories[mid_ep], f"Middle (Epoch {mid_ep + 1})")
            _plot_panel(axes[1, 0], eval_trajectories[end_ep], f"End (Epoch {end_ep + 1})")
            
            # Reference plot with a different background for distinction
            _plot_panel(axes[1, 1], eval_trace_pos, "Reference (Teacher)", bg_color="#f4f8ff")
            
            fig.tight_layout()
            out_path = seed_dir / "progression_trajectories.png"
            fig.savefig(str(out_path), dpi=150)
            plt.close(fig)

def main():
    parser = argparse.ArgumentParser(description="Regenerate plots for boids evaluation from extracted data.")
    parser.add_argument("--data-dir", type=str, default="generated/boids-evaluation", help="Base directory containing seed_X subfolders")
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
