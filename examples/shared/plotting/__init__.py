"""Shared plotting layer: grid and moving node visualizations."""

from .common import save_gif
from .comparison import plot_trajectory_comparison
from .grid import draw_markers, draw_obstacles, save_grid_simulation_gif, to_grid
from .moving import export_moving_gif, plot_moving_snapshots, plot_node_trajectories
from .style import (
    FIG_WIDTH_1COL,
    FIG_WIDTH_2COL,
    apply_paper_style,
    band,
    color_of,
    label_of,
    linestyle_of,
    marker_of,
    panel_label,
    savefig,
    style_axes,
)

__all__ = [
    "FIG_WIDTH_1COL",
    "FIG_WIDTH_2COL",
    "apply_paper_style",
    "band",
    "color_of",
    "draw_markers",
    "draw_obstacles",
    "export_moving_gif",
    "label_of",
    "linestyle_of",
    "marker_of",
    "panel_label",
    "plot_moving_snapshots",
    "plot_node_trajectories",
    "plot_trajectory_comparison",
    "save_gif",
    "save_grid_simulation_gif",
    "savefig",
    "style_axes",
    "to_grid",
]
