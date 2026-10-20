#!/usr/bin/env python3
"""Running-example figure of the SEAMS paper (Fig. 3): the channel and its gradients.

Top row: on a random deployment, the distance from the source, the distance from
the destination, and the channel that they compose. Bottom row: the channel
around obstacles at increasing density. Source and destination are drawn as
large stars, inset from the corners, so that they stay visible in every panel.

    uv run python examples/channel/running_example_figure.py \
        --out paper-2026-seams-diffield/figures/running-example.pdf
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "examples"))

import matplotlib.patheffects as pe  # noqa: E402
from channel.core import CHANNEL_THRESHOLD, build_snapshot_payloads, channel_body  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402
from shared.plotting.common import plt  # noqa: E402
from shared.plotting.style import FIG_WIDTH_1COL, GREEN, INK, ORANGE, RED  # noqa: E402

from diffield.dsl import branch, field  # noqa: E402
from diffield.sim import SimulationEngine, SnapshotRecorder, SpatialScenario  # noqa: E402
from diffield.utils import get_device  # noqa: E402

DEVICE_GREY = "#c9c8c2"
WALL_GREY = "#5f5e5a"
SOURCE_AT = (0.06, 0.94)
DEST_AT = (0.94, 0.06)
# four vertical walls, alternately open at the bottom and at the top
WALLS = (0.2, 0.4, 0.6, 0.8)
WALL_HALF_WIDTH = 0.035
WALL_GAP = 0.1


def walls(positions: torch.Tensor) -> torch.Tensor:
    x, y = positions[:, 0], positions[:, 1]
    mask = torch.zeros_like(x, dtype=torch.bool)
    for k, centre in enumerate(WALLS):
        band = (x - centre).abs() < WALL_HALF_WIDTH
        blocked = y > WALL_GAP if k % 2 == 0 else y < 1.0 - WALL_GAP
        mask |= band & blocked
    return mask


def nearest(positions: torch.Tensor, point: tuple[float, float]) -> int:
    target = torch.tensor(point, device=positions.device, dtype=positions.dtype)
    return int(torch.argmin((positions - target).norm(dim=-1)).item())


def simulate(num_nodes, rounds, *, obstacles, degree, seed, tolerance, device):
    torch.manual_seed(seed)
    positions = torch.rand(num_nodes, 2, device=device)
    scenario = SpatialScenario(
        positions=positions,
        edge_radius=float(np.sqrt(degree / (num_nodes * np.pi))),
        edge_weight_mode="distance",
        device=device,
    )
    engine = SimulationEngine.from_scenario(scenario)
    src_idx, dst_idx = nearest(positions, SOURCE_AT), nearest(positions, DEST_AT)
    source, dest = scenario.marker(src_idx), scenario.marker(dst_idx)
    blocked = walls(positions)
    if not obstacles:
        blocked = torch.zeros(num_nodes, dtype=torch.bool, device=device)
    recorder = SnapshotRecorder(
        state_fields=["dist_src", "dist_dst", "dist_channel"],
        capture_output=True,
        record_rounds={rounds - 1},
    )

    def program(runtime):
        return branch(
            ~blocked,
            lambda: channel_body(source, dest, tolerance),
            lambda: field.of(0.0),
            branch_name="obstacle",
        )

    engine.run(
        rounds=rounds,
        program=program,
        signals={"source": source, "dest": dest, "obstacle": blocked},
        recorder=recorder,
    )
    final = build_snapshot_payloads(recorder.records, num_nodes=num_nodes)[rounds - 1]
    as_np = {key: value.detach().cpu().numpy() for key, value in final.items()}
    print(
        f"N={num_nodes} obstacles={obstacles}: "
        f"d(S,T)={as_np['dist_src'][dst_idx]:.3f}, "
        f"channel={(as_np['channel'] > CHANNEL_THRESHOLD).sum()} devices"
    )
    return {
        "positions": positions.cpu().numpy(),
        "blocked": blocked.cpu().numpy(),
        "source": src_idx,
        "dest": dst_idx,
        **as_np,
    }


def frame(ax, letter, title):
    ax.set_xlim(-0.01, 1.01)
    ax.set_ylim(-0.01, 1.01)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.grid(False)
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color("#9a9993")
        spine.set_linewidth(0.6)
    ax.set_title(rf"$\mathit{{({letter})}}$ {title}", fontsize=7, fontweight="normal", color=INK,
                 pad=2.5)  # fmt: skip


def endpoints(ax, run, size):
    halo = [pe.withStroke(linewidth=1.6, foreground="white")]
    for idx, colour, label, offset, align in (
        (run["source"], GREEN, "S", (0.075, -0.02), "left"),
        (run["dest"], RED, "T", (-0.075, 0.02), "right"),
    ):
        x, y = run["positions"][idx]
        ax.scatter([x], [y], s=size, marker="*", c=colour, edgecolors="white",
                   linewidths=0.6, zorder=10)
        ax.text(x + offset[0], y + offset[1], label, fontsize=6.5, fontweight="bold",
                color=colour, ha=align, va="center", zorder=11, path_effects=halo)


def scalar_panel(ax, run, key, letter, title, dot):  # noqa: PLR0917
    values = run[key]
    finite = np.isfinite(values)
    ax.scatter(*run["positions"][finite].T, c=values[finite], cmap="viridis",
               s=dot, linewidths=0, rasterized=True, zorder=2)
    frame(ax, letter, title)
    endpoints(ax, run, 65)


def channel_panel(ax, run, letter, title, dot):
    on = run["channel"] > CHANNEL_THRESHOLD
    free = ~on & ~run["blocked"]
    if run["blocked"].any():
        for k, centre in enumerate(WALLS):
            bottom, top = (WALL_GAP, 1.01) if k % 2 == 0 else (-0.01, 1.0 - WALL_GAP)
            ax.add_patch(Rectangle((centre - WALL_HALF_WIDTH, bottom), 2 * WALL_HALF_WIDTH,
                                   top - bottom, facecolor=WALL_GREY, edgecolor="none", zorder=1))
    ax.scatter(*run["positions"][free].T, c=DEVICE_GREY, s=dot, linewidths=0,
               rasterized=True, zorder=2)
    ax.scatter(*run["positions"][on].T, c=ORANGE, s=dot * 1.5, linewidths=0,
               rasterized=True, zorder=3)
    frame(ax, letter, title)
    endpoints(ax, run, 65)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="generated/running-example.pdf")
    parser.add_argument("--degree", type=float, default=20.0)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--tolerance", type=float, default=0.02)
    parser.add_argument("--device", default="")
    args = parser.parse_args()
    device = get_device(args.device)
    common = {
        "degree": args.degree,
        "seed": args.seed,
        "tolerance": args.tolerance,
        "device": device,
    }

    top = simulate(1000, 200, obstacles=False, **common)
    obs = simulate(5000, 2000, obstacles=True, **{**common, "tolerance": 0.01})

    fig, axes = plt.subplots(1, 4, figsize=(FIG_WIDTH_1COL, 0.94))
    scalar_panel(axes[0], top, "dist_src", "a", "$d_S$", 2.0)
    scalar_panel(axes[1], top, "dist_dst", "b", "$d_T$", 2.0)
    channel_panel(axes[2], top, "c", "channel", 2.0)
    channel_panel(axes[3], obs, "d", "obstacles", 0.9)
    fig.subplots_adjust(left=0.005, right=0.995, bottom=0.01, top=0.84, wspace=0.07)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=600, bbox_inches="tight", pad_inches=0.01)
    if str(args.out).endswith(".pdf"):
        fig.savefig(str(args.out).removesuffix(".pdf") + ".png", dpi=300, bbox_inches="tight",
                    pad_inches=0.01)
    print(f"Saved {args.out}")


if __name__ == "__main__":
    main()
