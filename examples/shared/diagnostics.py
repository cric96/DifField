"""Diagnostics helpers for learnable aggregate experiments."""

from __future__ import annotations

import csv
import math
from pathlib import Path

try:
    import matplotlib.pyplot as plt
except ImportError:
    plt = None


def save_history_csv(history: dict[str, list[float]], output_path: str | Path) -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    keys = list(history.keys())
    length = len(history[keys[0]]) if keys else 0

    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        for index in range(length):
            row = {key: history[key][index] for key in keys}
            writer.writerow(row)


def save_summary_csv(summary: dict[str, float | int | str | bool], output_path: str | Path) -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary.keys()))
        writer.writeheader()
        writer.writerow(summary)


def _finite_pairs(xs: list[float], ys: list[float]) -> tuple[list[float], list[float]]:
    out_x = []
    out_y = []
    for x_val, y_val in zip(xs, ys):
        if math.isfinite(y_val):
            out_x.append(x_val)
            out_y.append(y_val)
    return out_x, out_y


def _plot_loss_curves(history: dict[str, list[float]], output_path: Path, title_prefix: str) -> None:
    if plt is None:
        print("matplotlib not available; skipping loss curves")
        return

    epochs = history["epoch"]
    marker = "o" if len(epochs) == 1 else None
    fig, ax = plt.subplots(figsize=(9.0, 5.0))
    train_key = "total"
    train_label = "train objective (pre-clip)"
    ax.plot(epochs, history[train_key], label=train_label, linewidth=2.0, marker=marker)

    for key, label, linestyle in [
        ("val_full_per_step_loss", "val per-step (full)", ":"),
    ]:
        if key not in history:
            continue
        val_x, val_y = _finite_pairs(epochs, history.get(key, []))
        if val_x:
            val_marker = "o" if len(val_x) == 1 else None
            ax.plot(val_x, val_y, label=label, linewidth=1.8, linestyle=linestyle, marker=val_marker)

    ax.set_title(f"{title_prefix}Teacher-forced Train Objective vs Full-rollout Validation Loss")
    ax.set_xlabel("epoch")
    ax.set_ylabel("loss")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def _plot_parameter_trajectories(history: dict[str, list[float]], output_path: Path, title_prefix: str) -> None:
    if plt is None:
        print("matplotlib not available; skipping parameter trajectories")
        return

    epochs = history["epoch"]
    marker = "o" if len(epochs) == 1 else None
    fig, axes = plt.subplots(2, 1, figsize=(9.0, 7.5), sharex=True)

    axes[0].plot(epochs, history["w_sep"], label="w_sep", marker=marker)
    axes[0].plot(epochs, history["w_align"], label="w_align", marker=marker)
    axes[0].plot(epochs, history["w_cohesion"], label="w_cohesion", marker=marker)
    axes[0].plot(epochs, history["max_speed"], label="max_speed", marker=marker)
    axes[0].plot(epochs, history["damping"], label="damping", marker=marker)
    axes[0].set_ylabel("parameter value")
    axes[0].set_title(f"{title_prefix}Model Parameter Trajectories")
    axes[0].grid(alpha=0.25)
    axes[0].legend(loc="best")

    tau_keys = [key for key in ("tau_align", "tau_cohesion") if key in history]
    for key in tau_keys:
        axes[1].plot(epochs, history[key], label=key, marker=marker)
    axes[1].set_xlabel("epoch")
    axes[1].set_ylabel("temperature")
    axes[1].grid(alpha=0.25)
    if tau_keys:
        axes[1].legend(loc="best")

    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def _plot_training_health(history: dict[str, list[float]], output_path: Path, title_prefix: str) -> None:
    if plt is None:
        print("matplotlib not available; skipping training health")
        return

    epochs = history["epoch"]
    marker = "o" if len(epochs) == 1 else None
    fig, axes = plt.subplots(2, 1, figsize=(9.0, 7.2), sharex=True)

    axes[0].plot(epochs, history["grad_norm"], color="tab:orange", label="grad_norm", marker=marker)
    axes[0].set_ylabel("norm")
    axes[0].set_title(f"{title_prefix}Gradient Norm")
    axes[0].grid(alpha=0.25)
    axes[0].legend(loc="best")

    axes[1].plot(epochs, history["center_error"], color="tab:green", label="train center_error", marker=marker)
    validation_curves = [
        ("val_curriculum_center_error", "val center (curriculum)", "tab:red", "--"),
        ("val_full_center_error", "val center (full)", "tab:purple", ":"),
        ("val_center_error", "val center_error", "tab:red", "--"),
    ]
    for key, label, color, linestyle in validation_curves:
        if key not in history:
            continue
        val_x, val_y = _finite_pairs(epochs, history.get(key, []))
        if val_x:
            val_marker = "o" if len(val_x) == 1 else None
            axes[1].plot(val_x, val_y, color=color, linestyle=linestyle, label=label, marker=val_marker)
    axes[1].set_xlabel("epoch")
    axes[1].set_ylabel("error")
    axes[1].set_title("Center Error")
    axes[1].grid(alpha=0.25)
    axes[1].legend(loc="best")

    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def _plot_parameter_recovery(
    teacher_params: dict[str, float],
    learned_params: dict[str, float],
    output_path: Path,
    title_prefix: str,
) -> None:
    if plt is None:
        print("matplotlib not available; skipping parameter recovery")
        return

    names = list(teacher_params.keys())
    teacher_vals = [teacher_params[name] for name in names]
    learned_vals = [learned_params[name] for name in names]

    import numpy as np

    x_pos = np.arange(len(names))
    width = 0.35

    fig, ax = plt.subplots(figsize=(9.0, 5.0))
    ax.bar(x_pos - width / 2, teacher_vals, width, label="Teacher (target)", color="tab:blue", alpha=0.8)
    ax.bar(x_pos + width / 2, learned_vals, width, label="Learned", color="tab:orange", alpha=0.8)

    ax.set_ylabel("Value")
    ax.set_title(f"{title_prefix}Parameter Recovery")
    ax.set_xticks(x_pos)
    ax.set_xticklabels(names, rotation=25, ha="right")
    ax.legend()
    ax.grid(axis="y", alpha=0.25)

    for index, name in enumerate(names):
        teacher_val = teacher_vals[index]
        learned_val = learned_vals[index]
        rel_err = abs(learned_val - teacher_val) / max(abs(teacher_val), 1e-9)
        ax.annotate(
            f"{rel_err:.0%}",
            xy=(x_pos[index] + width / 2, learned_val),
            xytext=(0, 4),
            textcoords="offset points",
            ha="center",
            fontsize=8,
            color="tab:red",
        )

    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def _plot_parameter_errors(
    history: dict[str, list[float]],
    teacher_params: dict[str, float],
    output_path: Path,
    title_prefix: str,
) -> None:
    if plt is None:
        print("matplotlib not available; skipping parameter errors")
        return

    epochs = history["epoch"]
    parameter_names = [name for name in teacher_params if name in history]
    if not parameter_names:
        return

    marker = "o" if len(epochs) == 1 else None
    fig, axes = plt.subplots(2, 1, figsize=(9.0, 7.5), sharex=True)

    for name in parameter_names:
        teacher_value = float(teacher_params[name])
        abs_errors = [abs(value - teacher_value) for value in history[name]]
        rel_errors = [
            abs(value - teacher_value) / max(abs(teacher_value), 1e-9) * 100.0
            for value in history[name]
        ]
        axes[0].plot(epochs, abs_errors, label=name, marker=marker)
        axes[1].plot(epochs, rel_errors, label=name, marker=marker)

    axes[0].set_ylabel("abs error")
    axes[0].set_title(f"{title_prefix}Parameter Error vs Teacher")
    axes[0].grid(alpha=0.25)
    axes[0].legend(loc="best")

    axes[1].set_xlabel("epoch")
    axes[1].set_ylabel("rel error (%)")
    axes[1].grid(alpha=0.25)
    axes[1].legend(loc="best")

    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def export_diagnostics(
    history: dict[str, list[float]],
    output_prefix: str | Path,
    *,
    title_prefix: str = "",
    teacher_params: dict[str, float] | None = None,
    learned_params: dict[str, float] | None = None,
    compact: bool = False,
) -> None:
    prefix = Path(output_prefix)
    prefix.parent.mkdir(parents=True, exist_ok=True)

    _plot_loss_curves(history, prefix.with_name(prefix.name + "_loss.png"), title_prefix)
    _plot_parameter_trajectories(history, prefix.with_name(prefix.name + "_params.png"), title_prefix)
    if compact:
        return
    _plot_training_health(history, prefix.with_name(prefix.name + "_health.png"), title_prefix)

    if teacher_params is not None:
        _plot_parameter_errors(
            history,
            teacher_params,
            prefix.with_name(prefix.name + "_param_error.png"),
            title_prefix,
        )

    if teacher_params is not None and learned_params is not None:
        _plot_parameter_recovery(
            teacher_params,
            learned_params,
            prefix.with_name(prefix.name + "_recovery.png"),
            title_prefix,
        )