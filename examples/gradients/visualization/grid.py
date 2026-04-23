"""Visualization utilities for grid-based gradients."""

from __future__ import annotations

from typing import TYPE_CHECKING
from diffield.sim import SnapshotRecorder
from ..domain.program import run_gradient_program

try:
    from shared.plotting import save_grid_simulation_gif
except ImportError:
    from ...shared.plotting import save_grid_simulation_gif

if TYPE_CHECKING:
    import torch


def render_gradient_evolution(
    scenario,
    source: torch.Tensor,
    rounds: int,
    weight: torch.Tensor,
    rows: int,
    cols: int,
    output_path: str,
    aggr="min",
    title: str = "Gradient Evolution",
    fps: int = 10,
) -> None:
    """Run a rollout and save a GIF of the distance field evolution."""
    recorder = SnapshotRecorder(state_fields=["dist"], capture_output=True)
    run_gradient_program(
        scenario,
        source,
        rounds=rounds,
        weight=weight,
        aggr=aggr,
        recorder=recorder,
    )
    save_grid_simulation_gif(
        recorder.records,
        "dist",
        rows,
        cols,
        output_path,
        title=title,
        fps=fps,
    )
