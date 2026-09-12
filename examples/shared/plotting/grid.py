"""Plotting utilities for grid-based simulations."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from .common import Axes, plt, save_gif

if TYPE_CHECKING:
    import numpy as np


def to_grid(
    tensor: torch.Tensor,
    rows: int,
    cols: int,
    obstacle_mask: torch.Tensor | None = None,
    replace_inf: bool = True,
) -> np.ndarray:
    """Reshape a flat tensor to a 2D grid, masking obstacles and infinities."""
    arr = tensor.detach().cpu().float().clone()
    if obstacle_mask is not None:
        arr[obstacle_mask.cpu()] = float("nan")
    if replace_inf:
        arr[torch.isinf(arr)] = float("nan")
    return arr.view(rows, cols).numpy()


def save_grid_simulation_gif(
    records: dict[int, dict[str, torch.Tensor]],
    field_key: str,
    rows: int,
    cols: int,
    output_path: str,
    obstacle: torch.Tensor | None = None,
    src_pos: tuple[int, int] | None = None,
    dst_pos: tuple[int, int] | None = None,
    cmap: str = "viridis",
    vmin: float = 0.0,
    vmax: float = 45.0,
    fps: int = 10,
    figsize: tuple[float, float] = (5, 5),
    title: str | None = None,
) -> None:
    """Save a GIF of a specific field from a grid simulation's records."""
    frames = sorted(records.keys())

    def render_frame(ax, frame_idx):
        grid = to_grid(records[frame_idx][field_key], rows, cols, obstacle)
        ax.imshow(grid, cmap=cmap, vmin=vmin, vmax=vmax, interpolation="nearest")
        if obstacle is not None:
            draw_obstacles(ax, obstacle, rows, cols)
        if src_pos is not None:
            draw_markers(ax, src_pos, dst_pos)
        if title:
            ax.set_title(f"{title} - Round {frame_idx + 1}")
        else:
            ax.set_title(f"Field: {field_key} - Round {frame_idx + 1}")
        ax.set_xticks([])
        ax.set_yticks([])

    save_gif(render_frame, frames, output_path, fps=fps, figsize=figsize)


def draw_obstacles(
    ax: Axes | None,
    obstacle_mask: torch.Tensor,
    rows: int,
    cols: int,
    color: str = "black",
) -> None:
    """Draw solid rectangles over obstacle cells in a grid plot."""
    if plt is None or ax is None:
        return
    obstacle = obstacle_mask.cpu().view(rows, cols)
    for row in range(rows):
        for col in range(cols):
            if obstacle[row, col]:
                ax.add_patch(
                    plt.Rectangle(
                        (col - 0.5, row - 0.5),
                        1,
                        1,
                        facecolor=color,
                        edgecolor=color,
                    )
                )


def draw_markers(
    ax: Axes | None,
    src_pos: tuple[int, int],
    dst_pos: tuple[int, int] | None = None,
    ms: int = 10,
) -> None:
    """Draw source and destination markers."""
    if plt is None or ax is None:
        return
    ax.plot(
        src_pos[1],
        src_pos[0],
        "g^",
        markersize=ms,
        markeredgecolor="white",
        markeredgewidth=1.2,
    )
    if dst_pos is not None:
        ax.plot(
            dst_pos[1],
            dst_pos[0],
            "rv",
            markersize=ms,
            markeredgecolor="white",
            markeredgewidth=1.2,
        )
