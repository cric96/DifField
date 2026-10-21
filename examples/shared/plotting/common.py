"""Common plotting primitives and safe matplotlib imports."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

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


def save_figure(
    fig, stem: Path, *, dpi: float | None = None, bbox_inches=None, formats=("pdf", "png")
) -> None:
    """Save ``fig`` as ``stem.<format>`` for every format, in order, then close it.

    ``None`` leaves ``dpi`` and ``bbox_inches`` to the ``savefig`` settings of matplotlib. The
    first format lays out a constrained figure, so the order shows in the other files.
    """
    stem.parent.mkdir(parents=True, exist_ok=True)
    for suffix in formats:
        fig.savefig(stem.parent / f"{stem.name}.{suffix}", dpi=dpi, bbox_inches=bbox_inches)
    plt.close(fig)


def save_animation(fig, draw, frames, path: Path, *, fps: int = 8, dpi: float | None = None):
    """Write the frames ``draw(0)`` to ``draw(frames - 1)`` of ``fig`` as the GIF ``path``.

    The GIF is written atomically, so an interrupted render never leaves a partial file; the
    figure stays open.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.stem + ".tmp.gif")
    animation = FuncAnimation(fig, draw, frames=frames, interval=1000 / fps)
    animation.save(temporary, writer=PillowWriter(fps=fps), dpi=dpi)
    temporary.replace(path)


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

    save_animation(fig, update, frames, Path(output_path), fps=fps)
    plt.close(fig)
    print(f"Saved GIF to {output_path}")
