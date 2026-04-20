#!/usr/bin/env python3
"""Evaluation runner for the learnable boids family."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "examples"))

try:
    import matplotlib.pyplot as plt
except ImportError:
    plt = None

from boids.runner import LearnableRunOptions, run_learnable_subprocess
from shared.metrics import is_finite_number, mean, nested_get, std


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate learnable boids across seeds")
    parser.add_argument("--seeds", type=str, default="11,13,17,19,23,32")
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--rounds", type=int, default=24)
    parser.add_argument("--num-nodes", type=int, default=24)
    parser.add_argument("--lr", type=float, default=0.05)
    parser.add_argument("--eval-seeds", type=str, default="101")
    parser.add_argument("--eval-every", type=int, default=5)
    parser.add_argument("--out-dir", type=str, default="generated/results/evaluation")
    parser.add_argument("--python", type=str, default=sys.executable)
    parser.add_argument("--supervision-mode", choices=["teacher", "replay"], default="teacher")
    parser.add_argument("--replay-trace-dir", type=str, default="", help="Optional shared replay trace directory")
    parser.add_argument("--save-replay-traces", action="store_true", help="Persist teacher traces while evaluating")
    parser.add_argument("--skip-viz", action="store_true", help="Pass --no-viz --no-gif to each run")
    parser.add_argument("--device", type=str, default="", help="Device (cuda/cpu) [auto if empty]")
    return parser.parse_args()


def _parse_csv(text: str) -> list[str]:
    return [item.strip() for item in text.split(",") if item.strip()]


def _load_history(run_dir: Path) -> dict[str, list[float]] | None:
    history_path = run_dir / "history.csv"
    if not history_path.exists():
        return None
    with history_path.open("r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
        fieldnames = reader.fieldnames or []
    if not rows or not fieldnames:
        return None
    return {name: [float(row[name]) for row in rows] for name in fieldnames}


def _series_mean_std(histories: list[dict[str, list[float]]], key: str) -> tuple[list[float], list[float]] | None:
    series = [history[key] for history in histories if key in history and history[key]]
    if not series:
        return None
    min_len = min(len(values) for values in series)
    trimmed = [values[:min_len] for values in series]
    means = [mean([values[idx] for values in trimmed]) for idx in range(min_len)]
    stds = [std([values[idx] for values in trimmed]) for idx in range(min_len)]
    return means, stds


def _plot_band(ax, x_vals: list[float], mean_vals: list[float], std_vals: list[float], label: str) -> None:
    line, = ax.plot(x_vals, mean_vals, linewidth=2.0, label=label)
    lower = [value - delta for value, delta in zip(mean_vals, std_vals)]
    upper = [value + delta for value, delta in zip(mean_vals, std_vals)]
    ax.fill_between(x_vals, lower, upper, alpha=0.18, color=line.get_color())


def _plot_training_bands(mode: str, histories: list[dict[str, list[float]]], output_dir: Path) -> None:
    if plt is None or not histories:
        return

    epochs = histories[0].get("epoch", [])
    total_stats = _series_mean_std(histories, "per_step_loss")
    val_curr_total_stats = _series_mean_std(histories, "val_curriculum_total_loss")
    val_full_total_stats = _series_mean_std(histories, "val_full_total_loss")
    val_curr_step_stats = _series_mean_std(histories, "val_curriculum_per_step_loss")
    val_full_step_stats = _series_mean_std(histories, "val_full_per_step_loss")
    center_stats = _series_mean_std(histories, "center_error")
    val_curr_center_stats = _series_mean_std(histories, "val_curriculum_center_error")
    val_full_center_stats = _series_mean_std(histories, "val_full_center_error")

    if not epochs or total_stats is None or center_stats is None:
        return

    min_len = min(
        len(epochs), len(total_stats[0]), len(center_stats[0]),
        len(val_curr_step_stats[0]) if val_curr_step_stats is not None else len(epochs),
        len(val_full_step_stats[0]) if val_full_step_stats is not None else len(epochs),
        len(val_curr_total_stats[0]) if val_curr_total_stats is not None else len(epochs),
        len(val_full_total_stats[0]) if val_full_total_stats is not None else len(epochs),
        len(val_curr_center_stats[0]) if val_curr_center_stats is not None else len(epochs),
        len(val_full_center_stats[0]) if val_full_center_stats is not None else len(epochs),
    )
    epochs = epochs[:min_len]

    fig, axes = plt.subplots(2, 1, figsize=(9.5, 7.5), sharex=True)

    _plot_band(axes[0], epochs, total_stats[0][:min_len], total_stats[1][:min_len], "train objective (pre-clip)")
    if val_curr_step_stats is not None:
        _plot_band(axes[0], epochs, val_curr_step_stats[0][:min_len], val_curr_step_stats[1][:min_len], "val per-step (curriculum)")
    if val_full_step_stats is not None:
        _plot_band(axes[0], epochs, val_full_step_stats[0][:min_len], val_full_step_stats[1][:min_len], "val per-step (full)")
    axes[0].set_ylabel("loss / step")
    axes[0].set_title(f"{mode} Teacher-forced Objective Across Seeds")
    axes[0].grid(alpha=0.25)
    axes[0].legend(loc="best")

    _plot_band(axes[1], epochs, center_stats[0][:min_len], center_stats[1][:min_len], "train center")
    if val_curr_center_stats is not None:
        _plot_band(axes[1], epochs, val_curr_center_stats[0][:min_len], val_curr_center_stats[1][:min_len], "val center (curriculum)")
    if val_full_center_stats is not None:
        _plot_band(axes[1], epochs, val_full_center_stats[0][:min_len], val_full_center_stats[1][:min_len], "val center (full)")
    axes[1].set_xlabel("epoch")
    axes[1].set_ylabel("error")
    axes[1].set_title(f"{mode} Center Error Across Seeds")
    axes[1].grid(alpha=0.25)
    axes[1].legend(loc="best")

    fig.tight_layout()
    fig.savefig(output_dir / f"{mode}_bands.png", dpi=150)
    plt.close(fig)


def _plot_parameter_error_bands(
    mode: str,
    histories: list[dict[str, list[float]]],
    teacher_params: dict[str, float],
    output_dir: Path,
) -> None:
    if plt is None or not histories:
        return

    epochs = histories[0].get("epoch", [])
    if not epochs:
        return

    fig, axes = plt.subplots(2, 1, figsize=(9.5, 7.8), sharex=True)
    plotted = False
    for name, teacher_value in teacher_params.items():
        if name not in histories[0]:
            continue
        abs_error_histories = [
            [abs(value - teacher_value) for value in history[name]]
            for history in histories if name in history and history[name]
        ]
        rel_error_histories = [
            [abs(value - teacher_value) / max(abs(teacher_value), 1e-9) * 100.0 for value in history[name]]
            for history in histories if name in history and history[name]
        ]
        if not abs_error_histories or not rel_error_histories:
            continue
        abs_stats = _series_mean_std([{name: values} for values in abs_error_histories], name)
        rel_stats = _series_mean_std([{name: values} for values in rel_error_histories], name)
        if abs_stats is None or rel_stats is None:
            continue

        min_len = min(len(epochs), len(abs_stats[0]), len(rel_stats[0]))
        plot_epochs = epochs[:min_len]
        _plot_band(axes[0], plot_epochs, abs_stats[0][:min_len], abs_stats[1][:min_len], name)
        _plot_band(axes[1], plot_epochs, rel_stats[0][:min_len], rel_stats[1][:min_len], name)
        plotted = True

    if not plotted:
        plt.close(fig)
        return

    axes[0].set_ylabel("abs error")
    axes[0].set_title(f"{mode} Parameter Error Across Seeds")
    axes[0].grid(alpha=0.25)
    axes[0].legend(loc="best")

    axes[1].set_xlabel("epoch")
    axes[1].set_ylabel("rel error (%)")
    axes[1].grid(alpha=0.25)
    axes[1].legend(loc="best")

    fig.tight_layout()
    fig.savefig(output_dir / f"{mode}_param_error_bands.png", dpi=150)
    plt.close(fig)


def _export_plots(
    mode: str,
    summaries: list[dict],
    histories: list[dict[str, list[float]]],
    output_dir: Path,
) -> None:
    if plt is None or not summaries or not histories:
        return

    teacher_params = nested_get(summaries[0], "parameters.teacher", {}) or {}
    _plot_training_bands(mode, histories, output_dir)
    if isinstance(teacher_params, dict) and teacher_params:
        _plot_parameter_error_bands(mode, histories, teacher_params, output_dir)


def main() -> None:
    args = parse_args()
    seeds = _parse_csv(args.seeds)
    root = Path(args.out_dir)
    root.mkdir(parents=True, exist_ok=True)
    run_options = LearnableRunOptions(
        python=args.python, root=root, epochs=args.epochs, rounds=args.rounds,
        num_nodes=args.num_nodes, eval_seeds=args.eval_seeds, eval_every=args.eval_every,
        skip_viz=args.skip_viz, supervision_mode=args.supervision_mode,
        replay_trace_dir=args.replay_trace_dir, save_replay_traces=args.save_replay_traces,
        device=args.device,
    )

    label = "demo"
    print(f"=== Boids Demo Evaluation: {len(seeds)} seeds ===")
    summaries: list[dict] = []
    histories: list[dict[str, list[float]]] = []
    for seed in seeds:
        run_name = f"{label}_seed{seed}"
        print(f"  running {run_name} ...")
        outcome = run_learnable_subprocess(
            options=run_options, run_name=run_name, seed=int(seed), lr=args.lr,
        )
        if not outcome.ok:
            print(f"  FAILED {run_name}: exit={outcome.returncode}")
            if outcome.stderr:
                for line in outcome.stderr.strip().splitlines()[-5:]:
                    print(f"    {line}")
            continue
        summaries.append(outcome.summary or {})
        history = _load_history(outcome.run_dir)
        if history is not None:
            histories.append(history)

    metrics = {
        "train_total_loss": "training.final_total_loss",
        "train_per_step_loss": "training.final_per_step_loss",
        "train_center_error": "training.final_center_error",
        "val_curriculum_total_loss": "validation.curriculum_horizon.final_total_loss",
        "val_curriculum_per_step_loss": "validation.curriculum_horizon.final_per_step_loss",
        "val_full_total_loss": "validation.full_horizon.final_total_loss",
        "val_full_per_step_loss": "validation.full_horizon.final_per_step_loss",
        "val_full_center_error": "validation.full_horizon.final_center_error",
    }
    recovery_params = ["w_sep", "w_align", "w_cohesion"]
    rows = []
    if summaries:
        row: dict[str, str | float] = {"mode": label, "runs": len(summaries)}
        for metric_name, metric_path in metrics.items():
            values = [
                float(nested_get(summary, metric_path))
                for summary in summaries if is_finite_number(nested_get(summary, metric_path))
            ]
            row[f"{metric_name}_mean"] = mean(values) if values else float("nan")
            row[f"{metric_name}_std"] = std(values) if values else float("nan")
        for name in recovery_params:
            errors = []
            for summary in summaries:
                path = f"parameters.recovery.{name}.rel_error"
                rel_error = nested_get(summary, path)
                if is_finite_number(rel_error):
                    errors.append(float(rel_error))
            row[f"recovery_{name}_rel_mean"] = mean(errors) if errors else float("nan")
            row[f"recovery_{name}_rel_std"] = std(errors) if errors else float("nan")
        rows.append(row)

    if rows:
        csv_path = root / "evaluation_summary.csv"
        with csv_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
        print(f"\nSaved {csv_path}")

    print("\n### Results\n")
    header = f"| {'mode':^10} | {'runs':>4} | {'train_total':>12} | {'train_step':>12} | {'val_curr':>12} | {'val_full':>12} | {'center_err':>12} |"
    header += "".join(f" {name + '_rel':>12} |" for name in recovery_params)
    print(header)
    print("|" + "-" * 12 + "|" + ("-" * 6 + "|") + (("-" * 14 + "|") * 5) + (("-" * 14 + "|") * len(recovery_params)))

    def _fmt(row: dict[str, str | float], key: str) -> str:
        mean_value = row.get(f"{key}_mean", float("nan"))
        std_value = row.get(f"{key}_std", float("nan"))
        if not is_finite_number(mean_value):
            return "N/A"
        return f"{mean_value:.6f}±{std_value:.6f}" if is_finite_number(std_value) else f"{mean_value:.6f}"

    def _fmt_percent(row: dict[str, str | float], key: str) -> str:
        mean_value = row.get(f"{key}_mean", float("nan"))
        std_value = row.get(f"{key}_std", float("nan"))
        if not is_finite_number(mean_value):
            return "N/A"
        if is_finite_number(std_value):
            return f"{mean_value:.1%}±{std_value:.1%}"
        return f"{mean_value:.1%}"

    for row in rows:
        line = (
            f"| {row['mode']:^10} | {row['runs']:>4} | {_fmt(row, 'train_total_loss'):>12} |"
            f" {_fmt(row, 'train_per_step_loss'):>12} | {_fmt(row, 'val_curriculum_total_loss'):>12} |"
            f" {_fmt(row, 'val_full_total_loss'):>12} | {_fmt(row, 'val_full_center_error'):>12} |"
        )
        for name in recovery_params:
            line += f" {_fmt_percent(row, f'recovery_{name}_rel'):>12} |"
        print(line)

    with (root / "all_summaries.json").open("w", encoding="utf-8") as handle:
        json.dump({label: summaries}, handle, indent=2)

    _export_plots(label, summaries, histories, root)

    print("\nDone.")


if __name__ == "__main__":
    main()
