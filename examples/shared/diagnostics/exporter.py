"""High-level entry points for exporting experiment diagnostics."""

from __future__ import annotations

from pathlib import Path
from .plots import (
    plot_loss_curves,
    plot_parameter_trajectories,
    plot_training_health,
    plot_parameter_errors,
    plot_parameter_recovery,
)


def export_diagnostics(
    history: dict[str, list[float]],
    output_prefix: str | Path,
    *,
    title_prefix: str = "",
    teacher_params: dict[str, float] | None = None,
    learned_params: dict[str, float] | None = None,
    compact: bool = False,
) -> None:
    """Generate and save a standard suite of diagnostic plots."""
    prefix = Path(output_prefix)
    prefix.parent.mkdir(parents=True, exist_ok=True)

    plot_loss_curves(history, prefix.with_name(prefix.name + "_loss.png"), title_prefix)
    plot_parameter_trajectories(
        history, prefix.with_name(prefix.name + "_params.png"), title_prefix
    )

    if compact:
        return

    plot_training_health(
        history, prefix.with_name(prefix.name + "_health.png"), title_prefix
    )

    if teacher_params is not None:
        plot_parameter_errors(
            history,
            teacher_params,
            prefix.with_name(prefix.name + "_param_error.png"),
            title_prefix,
        )

    if teacher_params is not None and learned_params is not None:
        plot_parameter_recovery(
            teacher_params,
            learned_params,
            prefix.with_name(prefix.name + "_recovery.png"),
            title_prefix,
        )
