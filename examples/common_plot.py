"""Common plotting utilities for Aggregate GNN examples.

This module isolates `matplotlib` dependencies so that the core
`aggregate_gnn` library remains free of visualisation-specific code.
"""

from typing import Optional, Tuple
import torch
import numpy as np

try:
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    from matplotlib.axes import Axes
except ImportError:
    plt = None
    mpatches = None
    Axes = None


def to_grid(
    tensor: torch.Tensor,
    rows: int,
    cols: int,
    obstacle_mask: Optional[torch.Tensor] = None,
    replace_inf: bool = True
) -> np.ndarray:
    """Reshape a flat tensor to a 2D NumPy grid array.
    
    Obstacles and infinite values are optionally converted to NaN
    so they appear transparent or distinct in typical colormaps.
    """
    arr = tensor.detach().cpu().float().clone()
    if obstacle_mask is not None:
        arr[obstacle_mask.cpu()] = float("nan")
    if replace_inf:
        arr[torch.isinf(arr)] = float("nan")
    return arr.view(rows, cols).numpy()


def draw_obstacles(ax: 'Axes', obstacle_mask: torch.Tensor, rows: int, cols: int, color: str = "black") -> None:
    """Draw solid rectangles over obstacle cells in a grid plot."""
    if plt is None:
        return
    obstacle = obstacle_mask.cpu().view(rows, cols)
    for r in range(rows):
        for c in range(cols):
            if obstacle[r, c]:
                ax.add_patch(plt.Rectangle(
                    (c - 0.5, r - 0.5), 1, 1,
                    facecolor=color, edgecolor=color
                ))


def draw_markers(
    ax: 'Axes',
    src_pos: Tuple[int, int],
    dst_pos: Optional[Tuple[int, int]] = None,
    ms: int = 10
) -> None:
    """Draw source (▲ green) and destination (▼ red) markers."""
    if plt is None:
        return
    ax.plot(src_pos[1], src_pos[0], "g^",
            markersize=ms, markeredgecolor="white", markeredgewidth=1.2)
    if dst_pos is not None:
        ax.plot(dst_pos[1], dst_pos[0], "rv",
                markersize=ms, markeredgecolor="white", markeredgewidth=1.2)
