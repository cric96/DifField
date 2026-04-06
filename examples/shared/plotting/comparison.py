"""Visualization utilities for comparing two trajectories side-by-side."""

from __future__ import annotations

import torch
from .common import plt


def plot_trajectory_comparison(
    *,
    predicted_positions_over_time: list[torch.Tensor],
    teacher_positions_over_time: list[torch.Tensor],
    source_idx: int,
    output_path: str,
    title: str = "Predicted vs Teacher Trajectories",
) -> None:
    """Render a side-by-side comparison of predicted and teacher trajectories."""
    if plt is None:
        print("matplotlib not available; skipping comparison plot")
        return
    if not predicted_positions_over_time or not teacher_positions_over_time:
        print("no trajectory data to compare")
        return

    predicted = torch.stack(predicted_positions_over_time, dim=0).detach().cpu().numpy()
    teacher = torch.stack(teacher_positions_over_time, dim=0).detach().cpu().numpy()
    if predicted.shape[1] != teacher.shape[1]:
        print("trajectory mismatch: predicted and teacher node counts differ")
        return

    _, num_nodes, _ = predicted.shape
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharex=True, sharey=True)
    for ax, (traj, panel_title) in zip(
        axes, ((predicted, "Predicted"), (teacher, "Teacher"))
    ):
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

        ax.scatter(
            traj[0, :, 0], traj[0, :, 1], c="black", s=18, alpha=0.7, label="start"
        )
        ax.scatter(
            traj[-1, :, 0], traj[-1, :, 1], c="orange", s=18, alpha=0.7, label="end"
        )
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
        ax.set_title(panel_title)
        ax.set_xlim(0.0, 1.0)
        ax.set_ylim(0.0, 1.0)
        ax.set_aspect("equal")
        ax.grid(alpha=0.25)
        ax.set_xlabel("x")
        ax.set_ylabel("y")

    axes[1].legend(loc="upper right")
    fig.suptitle(title)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    print(f"Saved {output_path}")
    plt.close(fig)
