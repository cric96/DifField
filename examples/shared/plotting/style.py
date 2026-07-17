"""Paper-quality matplotlib style shared by the SHAC / imitation comparison plots.

One consistent look (typography, spines, grid, legend, CI bands) and one fixed,
colour-blind-safe categorical palette assigned *by role* (not by call order) so
the same policy always gets the same colour/marker across every figure and every
experiment script that imports this module.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from .common import plt

if TYPE_CHECKING:
    from collections.abc import Sequence
    from pathlib import Path

    from matplotlib.axes import Axes
    from matplotlib.figure import Figure

# Fixed-order categorical palette (validated for CVD-safety; see the `dataviz`
# skill's `references/palette.md`). Roles are assigned once, below, and must be
# looked up by role -- never by cycling through this list.
BLUE = "#2a78d6"
AQUA = "#1baf7a"
YELLOW = "#eda100"
GREEN = "#008300"
VIOLET = "#4a3aa7"
RED = "#e34948"
MAGENTA = "#e87ba4"
ORANGE = "#eb6834"
INK = "#0b0b0b"
MUTED = "#898781"
GRID = "#e1e0d9"

# Standard conference figure widths (inches, LNCS/ACM two-column layout): use
# these instead of ad hoc figsize tuples so every static figure in the paper
# sits at a consistent scale. Height is left to the caller (aspect depends on
# content); only width is standardized.
FIG_WIDTH_1COL = 3.4
FIG_WIDTH_2COL = 7.0

# Semantic roles, held constant across every DIFFIELD example script: the
# interpretable field-program family stays in the cool (blue/violet/aqua)
# range, the black-box baseline is always red, and ground truth/expert
# references are always ink-black. Do not reassign per-script.
ROLE_COLOR = {
    "parametric": BLUE,  # DIFFIELD: aggregate program, learned static weights
    "diffield": BLUE,  # alias used by single-policy DIFFIELD-vs-neural scripts
    "hybrid": AQUA,  # DIFFIELD: program weights controlled by a neural gate net
    "hybrid_res": VIOLET,  # ablation: program + free neural force residual
    "leader": VIOLET,  # legacy alias (leader-election program baseline)
    "neural": RED,  # black-box baseline (MLP / GNN / message passing)
    "gnn": RED,  # alias
    "neural_d2": "#c2452e",  # depth-2 GNN student (imitation expressivity)
    "neural_d3": "#8f2f1d",  # depth-3 GNN student
    "expert": INK,  # ground truth / teacher reference
}

# A distinct marker + linestyle per role as a second, colour-independent channel
# (grayscale printing, colour-blind reviewers) -- pairs with ROLE_COLOR above.
ROLE_MARKER = {
    "parametric": "o",
    "diffield": "o",
    "hybrid": "s",
    "hybrid_res": "v",
    "leader": "^",
    "neural": "D",
    "gnn": "D",
    "neural_d2": "d",
    "neural_d3": "P",
    "expert": "x",
}
ROLE_LINESTYLE = {
    "parametric": "-",
    "diffield": "-",
    "hybrid": "--",
    "hybrid_res": "-.",
    "leader": "-.",
    "neural": ":",
    "gnn": ":",
    "neural_d2": ":",
    "neural_d3": ":",
    "expert": "-",
}

# Human-readable labels for the legend (override per-script only when a role
# needs a scenario-specific caption; otherwise import and reuse this).
ROLE_LABEL = {
    "parametric": "DIFFIELD (parametric)",
    "diffield": "DIFFIELD",
    "hybrid": "DIFFIELD (hybrid)",
    "hybrid_res": "hybrid (residual, ablation)",
    "leader": "DIFFIELD (leader/aggregate)",
    "neural": "neural (GNN)",
    "gnn": "GNN",
    "neural_d2": "GNN depth 2",
    "neural_d3": "GNN depth 3",
    "expert": "expert (ground truth)",
}


def color_of(role: str) -> str:
    return ROLE_COLOR.get(role, MUTED)


def marker_of(role: str) -> str:
    return ROLE_MARKER.get(role, "o")


def linestyle_of(role: str) -> str:
    return ROLE_LINESTYLE.get(role, "-")


def label_of(role: str) -> str:
    return ROLE_LABEL.get(role, role)


_PANEL_LABEL_POSITIONS: dict[str, tuple[float, float, str, str]] = {
    "upper left": (0.03, 0.96, "left", "top"),
    "upper right": (0.97, 0.96, "right", "top"),
    "lower left": (0.03, 0.04, "left", "bottom"),
    "lower right": (0.97, 0.04, "right", "bottom"),
}


def panel_label(ax: Axes, label: str, *, loc: str = "upper left") -> None:
    """Small, non-bold panel tag, e.g. ``panel_label(ax, "a")`` -> "(a)".

    Figures never carry a title -- the paper caption/subfigure text names
    each panel. This is only the disambiguating tag a caption refers back
    to, so it stays small and out of the way instead of restating content.
    """
    x, y, ha, va = _PANEL_LABEL_POSITIONS[loc]
    ax.text(
        x,
        y,
        f"({label})",
        transform=ax.transAxes,
        fontsize=10.5,
        fontweight="semibold",
        color=INK,
        ha=ha,
        va=va,
        zorder=20,
        bbox={
            "boxstyle": "round,pad=0.15",
            "facecolor": "white",
            "edgecolor": "none",
            "alpha": 0.7,
        },
    )


def apply_paper_style() -> None:
    """Set matplotlib rcParams once, at import/startup, for every figure in a run."""
    if plt is None:
        return
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.size": 10.5,
            "axes.titlesize": 12,
            "axes.titleweight": "semibold",
            "axes.labelsize": 10.5,
            "axes.edgecolor": MUTED,
            "axes.labelcolor": INK,
            "axes.linewidth": 0.8,
            "axes.grid": True,
            "axes.axisbelow": True,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "grid.color": GRID,
            "grid.linewidth": 0.7,
            "grid.alpha": 0.6,
            "xtick.color": MUTED,
            "ytick.color": MUTED,
            "text.color": INK,
            "legend.frameon": False,
            "legend.fontsize": 9.5,
            "figure.dpi": 150,
            "savefig.dpi": 200,
            "savefig.bbox": "tight",
        }
    )


def band(
    ax: Axes,
    x: Sequence[float],
    means: Sequence[float],
    cis: Sequence[float],
    *,
    role: str,
    label: str | None = None,
) -> None:
    """Mean line + 95% CI ribbon for one role, styled consistently everywhere."""
    color = color_of(role)
    lower = [m - c for m, c in zip(means, cis, strict=True)]
    upper = [m + c for m, c in zip(means, cis, strict=True)]
    ax.plot(
        x,
        means,
        color=color,
        linewidth=1.8,
        linestyle=linestyle_of(role),
        marker=marker_of(role),
        markersize=4.5,
        markevery=max(1, len(x) // 12),
        label=label if label is not None else label_of(role),
    )
    ax.fill_between(x, lower, upper, color=color, alpha=0.18, linewidth=0)


def style_axes(ax: Axes, *, ylabel: str | None = None, xlabel: str | None = None) -> None:
    if xlabel is not None:
        ax.set_xlabel(xlabel)
    if ylabel is not None:
        ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.6, linewidth=0.7)


def savefig(fig: Figure, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    print(f"Saved {out_path}")
