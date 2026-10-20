"""Render the animations of the README with diffield, in a light and in a dark theme.

Each animation is written to ``docs/assets/<name>.gif`` and ``docs/assets/<name>-dark.gif``.
It runs on the library alone, not on the experiments in ``examples/``:

    uv run python docs/assets/animations.py
"""

from __future__ import annotations

import shutil
import subprocess
from collections.abc import Callable
from dataclasses import dataclass
from functools import partial
from pathlib import Path

import matplotlib as mpl
import torch
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.collections import LineCollection
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Circle

from diffield import AggregateContext
from diffield.dsl import (
    branch,
    broadcast,
    field,
    gather_min,
    gradient,
    iterate,
    mux,
    scatter,
    scatter_range,
)
from diffield.sim import build_spatial_graph

ASSETS = Path(__file__).resolve().parent
SEED = 7
SIDE = 26  # devices per side of the deformed lattice
RADIUS = 1.6  # communication range, in lattice spacings
FPS = 10
HIDDEN = ((SIDE - 1) / 2, (SIDE - 1) / 2)  # where the obstacle learning finds is
HIDDEN_RADIUS = SIDE / 6
# Distances along the colours of the logo, as in the figures drawn by diagrams.py.
RAMP = LinearSegmentedColormap.from_list(
    "diffield", ["#1f5fa8", "#1f7f74", "#3f7f2f", "#b45309", "#9a3412"]
)


@dataclass(frozen=True)
class Theme:
    """Page colours of one theme, and the suffix of its files."""

    suffix: str
    background: str
    text: str
    muted: str
    edge: str
    accent: str


THEMES = (
    Theme("", "#ffffff", "#1f2328", "#59636e", "#d1d9e0", "#b45309"),
    Theme("-dark", "#0d1117", "#f0f6fc", "#9198a1", "#30363d", "#f5a35c"),
)


@dataclass(frozen=True)
class Network:
    """Devices on a deformed lattice and the links between those in range."""

    positions: torch.Tensor
    edge_index: torch.Tensor
    edge_weight: torch.Tensor

    @property
    def size(self) -> int:
        """The number of devices."""
        return self.positions.shape[0]

    def context(self) -> AggregateContext:
        """A fresh execution context on this network."""
        return AggregateContext(self.edge_index, self.size, edge_weight=self.edge_weight)

    def near(self, x: float, y: float, radius: float) -> torch.Tensor:
        """A boolean field: the devices within ``radius`` of (x, y)."""
        return (self.positions - torch.tensor([x, y])).norm(dim=-1) <= radius


def deploy() -> Network:
    """The same deformed lattice at every run."""
    generator = torch.Generator().manual_seed(SEED)
    axis = torch.arange(SIDE, dtype=torch.float32)
    grid = torch.stack(torch.meshgrid(axis, axis, indexing="xy"), dim=-1).reshape(-1, 2)
    positions = grid + (torch.rand(grid.shape, generator=generator) - 0.5) * 0.7
    edge_index, edge_weight = build_spatial_graph(
        positions, edge_radius=RADIUS, edge_weight_mode="distance"
    )
    return Network(positions, edge_index, edge_weight)


def wave(network: Network) -> list[dict[str, torch.Tensor]]:
    """The distance from a corner spreading through the network, round by round."""
    source = network.near(0.0, 0.0, 1.0).float()
    ctx = network.context()
    frames = []
    for _ in range(36):
        with ctx.round():
            frames.append({"value": gradient(source, name="distance").detach()})
    return frames


def channel(network: Network) -> list[dict[str, torch.Tensor]]:
    """A channel between two corners, bending around an obstacle in the middle."""
    centre = (SIDE - 1) / 2
    source = network.near(0.0, 0.0, 1.0).float()
    target = network.near(SIDE - 1.0, SIDE - 1.0, 1.0).float()
    # Off the diagonal, so that one way around it is shorter than the other.
    obstacle = network.near(centre + 3.0, centre - 3.0, SIDE / 5)

    def body() -> torch.Tensor:
        to_source = gradient(source, name="to_source")
        to_target = gradient(target, name="to_target")
        between = broadcast(source > 0, to_target, name="between")
        reached = to_source.isfinite() & to_target.isfinite() & between.isfinite()
        return (reached & (to_source + to_target <= between + 1.5)).float()

    ctx = network.context()
    frames = []
    for _ in range(60):
        with ctx.round():
            on = branch(~obstacle, body, lambda: field.of(0.0), branch_name="free")
        frames.append({"value": on.detach(), "obstacle": obstacle})
    return frames


def distances(network: Network, cost: torch.Tensor, source: torch.Tensor,
              rounds: int) -> torch.Tensor:
    """The distance from ``source`` when crossing a device costs ``cost`` on top of the link."""
    ctx = network.context()
    out = torch.full((network.size,), float("inf"))
    for _ in range(rounds):
        with ctx.round():
            out = iterate(
                field.inf(),
                lambda d: mux(
                    source, field.of(0.0), gather_min(scatter(d + cost) + scatter_range())
                ),
                name="distance",
            )
    return out


def learning(network: Network) -> list[dict[str, torch.Tensor]]:
    """Learn a cost per device from distances alone, and find the obstacle they went around."""
    centre = (SIDE - 1) / 2
    source = network.near(0.0, centre, 1.0)
    hidden = network.near(*HIDDEN, HIDDEN_RADIUS).float() * 6.0
    rounds = 2 * SIDE
    target = distances(network, hidden, source, rounds).detach()
    logits = torch.full((network.size,), -4.0, requires_grad=True)
    optimizer = torch.optim.Adam([logits], lr=0.25)
    frames = []
    for _ in range(60):
        cost = torch.nn.functional.softplus(logits)
        predicted = distances(network, cost, source, rounds)
        loss = (predicted - target).pow(2).mean() + 0.05 * cost.mean()
        frames.append({"value": predicted.detach(), "cost": cost.detach(), "loss": loss.detach()})
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    return frames


def links(network: Network) -> list[tuple[tuple[float, float], tuple[float, float]]]:
    """Each undirected link once, as a segment."""
    points = network.positions.tolist()
    return [
        (tuple(points[a]), tuple(points[b]))
        for a, b in network.edge_index.t().tolist()
        if a < b
    ]


type Frame = dict[str, torch.Tensor]
type Painter = Callable[[Frame, Theme], tuple[list, str]]


def draw(network: Network, frames: list[Frame], theme: Theme, paint: Painter, path: Path,
         outline: tuple[float, float, float] | None = None) -> None:
    """Animate ``frames``, each coloured by ``paint``, and save them as an optimised GIF.

    ``outline`` is a circle (x, y, radius) drawn dashed over the network.
    """
    fig, ax = plt.subplots(figsize=(4.2, 4.2), dpi=100)
    fig.patch.set_facecolor(theme.background)
    fig.subplots_adjust(left=0.02, right=0.98, bottom=0.02, top=0.92)
    ax.set_facecolor(theme.background)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.add_collection(LineCollection(links(network), colors=theme.edge, linewidths=0.6, zorder=1))
    points = network.positions.numpy()
    dots = ax.scatter(points[:, 0], points[:, 1], s=14, linewidths=0, zorder=2)
    title = ax.set_title("", color=theme.text, fontsize=11, loc="left")
    if outline is not None:
        x, y, radius = outline
        ax.add_patch(Circle((x, y), radius, fill=False, linestyle="--", linewidth=1.2,
                            edgecolor=theme.muted, zorder=3))

    def update(index: int) -> tuple:
        colors, caption = paint(frames[index], theme)
        dots.set_color(colors)
        title.set_text(caption)
        return dots, title

    animation = FuncAnimation(fig, update, frames=len(frames), blit=False)
    animation.save(path, writer=PillowWriter(fps=FPS),
                   savefig_kwargs={"facecolor": theme.background})
    plt.close(fig)
    if shutil.which("gifsicle"):
        command = ["gifsicle", "-O3", "--colors", "64", "-b", str(path)]
        subprocess.run(command, check=True)  # noqa: S603


def ramp(values: torch.Tensor, theme: Theme, top: float) -> list:
    """Colours along the ramp of the logo; unreached devices in the colour of the links."""
    return [
        theme.edge if value == float("inf") else RAMP(min(value / top, 1.0))
        for value in values.tolist()
    ]


def paint_wave(frame: Frame, theme: Theme, top: float) -> tuple[list, str]:
    """Devices coloured by their distance."""
    return ramp(frame["value"], theme, top), "distance from a corner"


def paint_channel(frame: Frame, theme: Theme) -> tuple[list, str]:
    """The channel in the accent colour, the obstacle muted."""
    pairs = zip(frame["value"].tolist(), frame["obstacle"].tolist(), strict=True)
    colours = [
        theme.muted if blocked else theme.accent if on else theme.edge for on, blocked in pairs
    ]
    return colours, "a channel around an obstacle"


def paint_learning(frame: Frame, theme: Theme, top: float) -> tuple[list, str]:
    """Devices coloured by their distance, those that learnt a high cost in the text colour."""
    colours = ramp(frame["value"], theme, top)
    for index in torch.nonzero(frame["cost"] > 1.0).flatten().tolist():
        colours[index] = theme.text
    return colours, f"finding a hidden obstacle   loss {float(frame['loss']):.2f}"


def main() -> None:
    """Render every animation in both themes."""
    mpl.rcParams["font.family"] = "sans-serif"
    network = deploy()
    waves, channels, lessons = wave(network), channel(network), learning(network)
    paint_waves = partial(paint_wave, top=float(waves[-1]["value"].max()))
    paint_lessons = partial(paint_learning, top=float(lessons[-1]["value"].max()))
    for theme in THEMES:
        draw(network, waves, theme, paint_waves, ASSETS / f"wave{theme.suffix}.gif")
        draw(network, channels, theme, paint_channel, ASSETS / f"channel{theme.suffix}.gif")
        draw(network, lessons, theme, paint_lessons, ASSETS / f"learning{theme.suffix}.gif",
             outline=(*HIDDEN, HIDDEN_RADIUS))


if __name__ == "__main__":
    main()
