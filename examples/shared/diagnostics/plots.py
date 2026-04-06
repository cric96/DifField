"""Internal plotting functions for experiment diagnostics."""

from __future__ import annotations

import math
from pathlib import Path
import numpy as np

try:
    import matplotlib.pyplot as plt
except ImportError:
    plt = None


def _finite_pairs(xs: list[float], ys: list[float]) -> tuple[list[float], list[float]]:
    """Filter pairs of values to include only finite ones."""
    out_x = []
    out_y = []
    for x_val, y_val in zip(xs, ys):
        if math.isfinite(y_val):
            out_x.append(x_val)
            out_y.append(y_val)
    return out_x, out_y


def plot_loss_curves(
    history: dict[str, list[float]], output_path: Path, title_prefix: str
) -> None:
    """Render training and validation loss curves."""
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
            ax.plot(
                val_x,
                val_y,
                label=label,
                linewidth=1.8,
                linestyle=linestyle,
                marker=val_marker,
            )

    ax.set_title(
        f"{title_prefix}Teacher-forced Train Objective vs Full-rollout Validation Loss"
    )
    ax.set_xlabel("epoch")
    ax.set_ylabel("loss")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def plot_parameter_trajectories(
    history: dict[str, list[float]], output_path: Path, title_prefix: str
) -> None:
    """Render the evolution of model parameters over epochs."""
    if plt is None:
        print("matplotlib not available; skipping parameter trajectories")
        return

    epochs = history["epoch"]
    marker = "o" if len(epochs) == 1 else None
    fig, axes = plt.subplots(2, 1, figsize=(9.0, 7.5), sharex=True)

    param_names = [
        "w_sep",
        "w_align",
        "w_cohesion",
        "max_speed",
        "damping",
        "range_weight",
        "risk_weight",
        "assignment_tau",
    ]
    for name in param_names:
        if name in history:
            axes[0].plot(epochs, history[name], label=name, marker=marker)

    axes[0].set_ylabel("parameter value")
    axes[0].set_title(f"{title_prefix}Model Parameter Trajectories")
    axes[0].grid(alpha=0.25)
    axes[0].legend(loc="best")

    tau_keys = [
        key for key in ("tau_align", "tau_cohesion", "assignment_tau") if key in history
    ]
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


def plot_training_health(
    history: dict[str, list[float]], output_path: Path, title_prefix: str
) -> None:
    """Render auxiliary health metrics like gradient norms and center error."""
    if plt is None:
        print("matplotlib not available; skipping training health")
        return

    epochs = history["epoch"]
    marker = "o" if len(epochs) == 1 else None
    fig, axes = plt.subplots(2, 1, figsize=(9.0, 7.2), sharex=True)

    if "grad_norm" in history:
        axes[0].plot(
            epochs,
            history["grad_norm"],
            color="tab:orange",
            label="grad_norm",
            marker=marker,
        )
        axes[0].set_ylabel("norm")
        axes[0].set_title(f"{title_prefix}Gradient Norm")
        axes[0].grid(alpha=0.25)
        axes[0].legend(loc="best")

    if "center_error" in history:
        axes[1].plot(
            epochs,
            history["center_error"],
            color="tab:green",
            label="train center_error",
            marker=marker,
        )
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
                axes[1].plot(
                    val_x,
                    val_y,
                    color=color,
                    linestyle=linestyle,
                    label=label,
                    marker=val_marker,
                )
        axes[1].set_xlabel("epoch")
        axes[1].set_ylabel("error")
        axes[1].set_title("Center Error")
        axes[1].grid(alpha=0.25)
        axes[1].legend(loc="best")

    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def plot_parameter_recovery(
    teacher_params: dict[str, float],
    learned_params: dict[str, float],
    output_path: Path,
    title_prefix: str,
) -> None:
    """Render a bar chart comparing final learned parameters to target values."""
    if plt is None:
        print("matplotlib not available; skipping parameter recovery")
        return

    names = list(teacher_params.keys())
    teacher_vals = [teacher_params[name] for name in names]
    learned_vals = [learned_params[name] for name in names]

    x_pos = np.arange(len(names))
    width = 0.35

    fig, ax = plt.subplots(figsize=(9.0, 5.0))
    ax.bar(
        x_pos - width / 2,
        teacher_vals,
        width,
        label="Teacher (target)",
        color="tab:blue",
        alpha=0.8,
    )
    ax.bar(
        x_pos + width / 2,
        learned_vals,
        width,
        label="Learned",
        color="tab:orange",
        alpha=0.8,
    )

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


def plot_parameter_errors(
    history: dict[str, list[float]],
    teacher_params: dict[str, float],
    output_path: Path,
    title_prefix: str,
) -> None:
    """Render absolute and relative parameter errors over epochs."""
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
