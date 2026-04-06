"""Common plotting primitives and safe matplotlib imports."""

from __future__ import annotations

from pathlib import Path
from typing import Callable, Sequence

try:
    import matplotlib.patches as mpatches
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation, PillowWriter
    from matplotlib.axes import Axes
    from matplotlib.collections import LineCollection
except ImportError:
    plt = None
    mpatches = None
    FuncAnimation = None
    PillowWriter = None
    LineCollection = None
    Axes = None


def save_gif(
    render_frame_fn: Callable[[Axes, int], None],
    frames: Sequence[int],
    output_path: str,
    fps: int = 10,
    figsize: tuple[float, float] = (5, 5),
) -> None:
    """Reusable utility for creating a GIF from a sequence of frames."""
    if plt is None or FuncAnimation is None or PillowWriter is None:
        print("matplotlib animation tools not available; skipping GIF generation")
        return

    fig, ax = plt.subplots(figsize=figsize)

    def update(frame_idx):
        ax.clear()
        render_frame_fn(ax, frame_idx)

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    anim = FuncAnimation(fig, update, frames=frames)
    writer = PillowWriter(fps=fps)
    anim.save(output_path, writer=writer)
    plt.close(fig)
    print(f"Saved GIF to {output_path}")
