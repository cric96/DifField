#!/usr/bin/env python3
"""Batch evaluation runner for learnable territories experiments."""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "examples"))

try:
    import matplotlib.pyplot as plt
except ImportError:
    plt = None

from shared.metrics import is_finite_number, mean, nested_get, std


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate learnable territories across training seeds")
    parser.add_argument("--label", type=str, default="demo")
    parser.add_argument("--seeds", type=str, default="11,13,17,19,23,29")
    parser.add_argument("--rows", type=int, default=18)
    parser.add_argument("--cols", type=int, default=18)
    parser.add_argument("--connectivity", type=int, default=4, choices=[4, 8])
    parser.add_argument("--num-sinks", type=int, default=None)
    parser.add_argument(
        "--scenario-preset",
        type=str,
        default="asymmetric_canyon",
        choices=["balanced", "asymmetric_canyon"],
    )
    parser.add_argument("--sink-positions", type=str, default="")
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--lr", type=float, default=0.03)
    parser.add_argument("--risk-loss-weight", type=float, default=0.5)
    parser.add_argument("--eval-seeds", type=str, default="101,103,107")
    parser.add_argument("--eval-every", type=int, default=5)
    parser.add_argument("--print-every", type=int, default=10)
    parser.add_argument("--mode", type=str, default="scalars", choices=["scalars", "hybrid"])
    parser.add_argument("--out-dir", type=str, default="generated/territories/results/evaluation")
    parser.add_argument("--python", type=str, default=sys.executable)
    parser.add_argument("--skip-viz", action="store_true")
    parser.add_argument("--device", type=str, default="")
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

    return {
        name: [float(row[name]) for row in rows]
        for name in fieldnames
    }


def _build_command(args: argparse.Namespace, *, root: Path, run_name: str, seed: int) -> list[str]:
    cmd = [
        args.python,
        "examples/territories/learnable.py",
        "--rows",
        str(args.rows),
        "--cols",
        str(args.cols),
        "--connectivity",
        str(args.connectivity),
        "--scenario-preset",
        args.scenario_preset,
        "--epochs",
        str(args.epochs),
        "--lr",
        f"{args.lr:.4f}",
        "--risk-loss-weight",
        str(args.risk_loss_weight),
        "--seed",
        str(seed),
        "--eval-seeds",
        args.eval_seeds,
        "--eval-every",
        str(args.eval_every),
        "--print-every",
        str(args.print_every),
        "--mode",
        args.mode,
        "--out-dir",
        str(root),
        "--run-name",
        run_name,
    ]
    if args.num_sinks is not None:
        cmd.extend(["--num-sinks", str(args.num_sinks)])
    if args.sink_positions.strip():
        cmd.extend(["--sink-positions", args.sink_positions])
    if args.device:
        cmd.extend(["--device", args.device])
    if args.skip_viz:
        cmd.append("--no-viz")
    return cmd


def _run_subprocess(args: argparse.Namespace, *, root: Path, run_name: str, seed: int) -> tuple[int, str, str, dict[str, object] | None]:
    cmd = _build_command(args, root=root, run_name=run_name, seed=seed)
    proc = subprocess.run(cmd, capture_output=True, text=True)
    summary_path = root / run_name / "summary.json"
    summary = None
    if proc.returncode == 0 and summary_path.exists():
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
    return proc.returncode, proc.stdout, proc.stderr, summary


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


def _plot_training_bands(label: str, histories: list[dict[str, list[float]]], output_dir: Path) -> None:
    if plt is None or not histories:
        return

    epochs = histories[0].get("epoch", [])
    train_total = _series_mean_std(histories, "total")
    val_total = _series_mean_std(histories, "val_total")
    train_owner = _series_mean_std(histories, "owner_agreement")
    val_owner = _series_mean_std(histories, "val_owner_agreement")

    if not epochs or train_total is None or train_owner is None:
        return

    min_len = min(
        len(epochs),
        len(train_total[0]),
        len(train_owner[0]),
        len(val_total[0]) if val_total is not None else len(epochs),
        len(val_owner[0]) if val_owner is not None else len(epochs),
    )
    epochs = epochs[:min_len]

    fig, axes = plt.subplots(2, 1, figsize=(9.5, 7.5), sharex=True)

    _plot_band(axes[0], epochs, train_total[0][:min_len], train_total[1][:min_len], "train final-summary objective")
    if val_total is not None:
        _plot_band(axes[0], epochs, val_total[0][:min_len], val_total[1][:min_len], "val final-summary objective")
    axes[0].set_ylabel("loss")
    axes[0].set_title(f"{label} Territories Summary Loss Across Seeds")
    axes[0].grid(alpha=0.25)
    axes[0].legend(loc="best")

    _plot_band(axes[1], epochs, train_owner[0][:min_len], train_owner[1][:min_len], "train owner agreement")
    if val_owner is not None:
        _plot_band(axes[1], epochs, val_owner[0][:min_len], val_owner[1][:min_len], "val owner agreement")
    axes[1].set_xlabel("epoch")
    axes[1].set_ylabel("agreement")
    axes[1].set_title(f"{label} Territory Agreement Across Seeds")
    axes[1].grid(alpha=0.25)
    axes[1].legend(loc="best")

    fig.tight_layout()
    fig.savefig(output_dir / f"{label}_bands.png", dpi=150)
    plt.close(fig)


def _plot_parameter_error_bands(
    label: str,
    histories: list[dict[str, list[float]]],
    teacher_params: dict[str, float],
    output_dir: Path,
) -> None:
    if plt is None or not histories or not teacher_params:
        return

    epochs = histories[0].get("epoch", [])
    if not epochs:
        return

    fig, ax = plt.subplots(figsize=(9.5, 5.5))
    plotted = False
    for name, teacher_value in teacher_params.items():
        if name not in histories[0]:
            continue
        rel_error_histories = [
            [abs(value - teacher_value) / max(abs(teacher_value), 1e-9) * 100.0 for value in history[name]]
            for history in histories
            if name in history and history[name]
        ]
        if not rel_error_histories:
            continue
        stats = _series_mean_std([{name: values} for values in rel_error_histories], name)
        if stats is None:
            continue
        min_len = min(len(epochs), len(stats[0]))
        _plot_band(ax, epochs[:min_len], stats[0][:min_len], stats[1][:min_len], name)
        plotted = True

    if not plotted:
        plt.close(fig)
        return

    ax.set_xlabel("epoch")
    ax.set_ylabel("relative error (%)")
    ax.set_title(f"{label} Parameter Error Across Seeds")
    ax.grid(alpha=0.25)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(output_dir / f"{label}_param_error_bands.png", dpi=150)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    seeds = _parse_csv(args.seeds)
    root = Path(args.out_dir)
    root.mkdir(parents=True, exist_ok=True)

    print(f"=== Territories Evaluation: {len(seeds)} seeds ===")
    summaries: list[dict[str, object]] = []
    histories: list[dict[str, list[float]]] = []
    for seed_text in seeds:
        seed = int(seed_text)
        run_name = f"{args.label}_seed{seed}"
        print(f"  running {run_name} ...")
        returncode, _stdout, stderr, summary = _run_subprocess(args, root=root, run_name=run_name, seed=seed)
        if returncode != 0 or summary is None:
            print(f"  FAILED {run_name}: exit={returncode}")
            if stderr:
                for line in stderr.strip().splitlines()[-5:]:
                    print(f"    {line}")
            continue
        summaries.append(summary)
        history = _load_history(root / run_name)
        if history is not None:
            histories.append(history)

    metrics = {
        "train_total_loss": "training.final_total_loss",
        "train_owner_agreement": "training.final_owner_agreement",
        "eval_total_loss": "validation.mean.total",
        "eval_owner_agreement": "validation.mean.owner_agreement",
        "eval_conservation_error": "validation.mean.conservation_error",
    }
    recovery_params = ["range_weight", "risk_weight", "assignment_tau"]
    rows = []
    if summaries:
        resolved_num_sinks = nested_get(summaries[0], "run.num_sinks")
        row: dict[str, str | float | int] = {
            "label": args.label,
            "runs": len(summaries),
            "mode": args.mode,
            "scenario_preset": args.scenario_preset,
            "num_sinks": int(resolved_num_sinks) if is_finite_number(resolved_num_sinks) else "explicit",
        }
        for metric_name, metric_path in metrics.items():
            values = [
                float(nested_get(summary, metric_path))
                for summary in summaries
                if is_finite_number(nested_get(summary, metric_path))
            ]
            row[f"{metric_name}_mean"] = mean(values) if values else float("nan")
            row[f"{metric_name}_std"] = std(values) if values else float("nan")
        for name in recovery_params:
            errors = [
                float(nested_get(summary, f"parameters.recovery.{name}.rel_error"))
                for summary in summaries
                if is_finite_number(nested_get(summary, f"parameters.recovery.{name}.rel_error"))
            ]
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

    with (root / "all_summaries.json").open("w", encoding="utf-8") as handle:
        json.dump({args.label: summaries}, handle, indent=2)

    if summaries and histories:
        teacher_params = nested_get(summaries[0], "parameters.teacher", {}) or {}
        if isinstance(teacher_params, dict):
            _plot_training_bands(args.label, histories, root)
            _plot_parameter_error_bands(
                args.label,
                histories,
                {str(key): float(value) for key, value in teacher_params.items()},
                root,
            )

    print("\n### Results\n")
    if rows:
        row = rows[0]

        def _fmt(metric_key: str) -> str:
            mean_value = row.get(f"{metric_key}_mean", float("nan"))
            std_value = row.get(f"{metric_key}_std", float("nan"))
            if not is_finite_number(mean_value):
                return "N/A"
            if is_finite_number(std_value):
                return f"{float(mean_value):.6f}±{float(std_value):.6f}"
            return f"{float(mean_value):.6f}"

        def _fmt_percent(metric_key: str) -> str:
            mean_value = row.get(f"{metric_key}_mean", float("nan"))
            std_value = row.get(f"{metric_key}_std", float("nan"))
            if not is_finite_number(mean_value):
                return "N/A"
            if is_finite_number(std_value):
                return f"{float(mean_value):.1%}±{float(std_value):.1%}"
            return f"{float(mean_value):.1%}"

        print(
            f"| {'label':^10} | {'runs':>4} | {'train_total':>12} | {'eval_total':>12} | {'owner':>12} | {'conservation':>12} | {'range_rel':>12} | {'risk_rel':>12} | {'tau_rel':>12} |"
        )
        print("|------------|------|--------------|--------------|--------------|--------------|--------------|--------------|--------------|")
        print(
            f"| {str(row['label']):^10} | {int(row['runs']):>4} | {_fmt('train_total_loss'):>12} | {_fmt('eval_total_loss'):>12} | {_fmt('eval_owner_agreement'):>12} | {_fmt('eval_conservation_error'):>12} | {_fmt_percent('recovery_range_weight_rel'):>12} | {_fmt_percent('recovery_risk_weight_rel'):>12} | {_fmt_percent('recovery_assignment_tau_rel'):>12} |"
        )

    print("\nDone.")


if __name__ == "__main__":
    main()