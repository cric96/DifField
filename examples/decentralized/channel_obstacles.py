#!/usr/bin/env python3
"""One channel program, executed centrally and device by device.

The same ``program`` closure is run twice:

* **centralized** -- :class:`~diffield.sim.engine.SimulationEngine`, one batched
  round over the whole graph;
* **decentralized** -- one device at a time inside `Mesa
  <https://mesa.readthedocs.io>`_, each device seeing only its own neighbours'
  messages and never the global field.

Mesa knows nothing about field calculus, so agreement between the two is
evidence about the semantics rather than about a shared implementation.

The program is the channel *with obstacles*: ``branch`` restricts the domain,
and because domain restriction keeps a link only when both of its endpoints are
in the partition, a device cannot even decide its own in-edges without knowing
its neighbours' obstacle flag.  Sensor values therefore travel on the wire
alongside state.

Reproducible only for programs whose every ``gather`` scatters a sensor, a
constant, or a previous-round ``iterate`` state -- the condition documented on
:class:`~diffield.core.device.DeviceContext`.  It holds for every building block
used here.  ``branch`` conditions must likewise be sensors or previous-round
state.  Devices may move: when a link forms or breaks, the affected star
graphs are rebuilt and each device carries its own state across (see
``tests/decentralized/test_moving_devices.py``).

Usage::

    uv run python examples/decentralized/channel_obstacles.py
    uv run python examples/decentralized/channel_obstacles.py --async
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "examples"))

import torch
from channel.core import channel_body
from shared.plotting.common import plt
from shared.plotting.grid import draw_markers, draw_obstacles, to_grid

from diffield import GridScenario, SimulationEngine, SnapshotRecorder, branch
from diffield.decentralized import run_decentralized
from diffield.dsl import field


def parse_args():
    parser = argparse.ArgumentParser(
        description="Channel with obstacles, run centrally and decentrally"
    )
    parser.add_argument("--rows", type=int, default=12, help="Grid rows")
    parser.add_argument("--cols", type=int, default=12, help="Grid cols")
    parser.add_argument("--rounds", type=int, default=40, help="Compute rounds")
    parser.add_argument("--tolerance", type=float, default=1.0, help="Channel width")
    parser.add_argument(
        "--async",
        dest="asynchronous",
        action="store_true",
        help="Drop the round barrier: shuffled activation, devices may skip rounds",
    )
    parser.add_argument(
        "--activation-prob",
        type=float,
        default=0.6,
        help="Per-round activation probability of a device in async mode",
    )
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    parser.add_argument("--no-viz", action="store_true", help="Skip plots")
    parser.add_argument(
        "--out-prefix",
        type=str,
        default="generated/decentralized_channel",
        help="Output path prefix for figures",
    )
    return parser.parse_args()


# ----------------------------------------------------------------------
# scenario
# ----------------------------------------------------------------------
def build_scenario(args):
    """Grid, source, destination and a wall the channel has to route around."""
    scenario = GridScenario(args.rows, args.cols, connectivity=8)
    src_pos = (args.rows // 2, 2)
    dst_pos = (args.rows // 2, args.cols - 3)
    wall_col = args.cols // 2

    signals = {
        "source": scenario.marker(*src_pos),
        "dest": scenario.marker(*dst_pos),
        "obstacle": scenario.mask_from_predicate(
            lambda row, col: col == wall_col and row < args.rows - 2
        ),
        # A per-node cost perturbation; zero here so the two runs are compared
        # on the program, not on a random field.
        "noise": torch.zeros(scenario.num_nodes),
    }
    return scenario, signals, src_pos, dst_pos, wall_col


def make_program(tolerance: float):
    """The shared closure.

    Everything it needs comes from ``runtime.signals``, which is the whole
    trick: centrally those are global ``[N]`` fields, on a device they are that
    device's local ``[1 + k]`` view.  The closure itself cannot tell.
    """

    def program(runtime):
        signals = runtime.signals
        return branch(
            ~signals["obstacle"],
            lambda: channel_body(
                signals["source"], signals["dest"], tolerance, signals["noise"]
            ),
            lambda: field.of(0.0),
            branch_name="obstacle",
        )

    return program


# ----------------------------------------------------------------------
# runs
# ----------------------------------------------------------------------
def run_centralized(scenario, program, signals, rounds):
    """The batched reference run: one global round per step."""
    engine = SimulationEngine.from_scenario(scenario)
    recorder = SnapshotRecorder(
        state_fields=["dist_src", "dist_dst"],
        capture_output=True,
        record_rounds=None,
    )
    with torch.no_grad():
        engine.run(
            rounds=rounds, program=program, signals=signals, recorder=recorder
        )
    return [recorder.get(idx, "output").cpu() for idx in range(rounds)]


# ----------------------------------------------------------------------
# comparison
# ----------------------------------------------------------------------
def mismatches(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    """Nodes where the two runs disagree.

    ``inf == inf`` already holds, so only ``nan`` -- a device that has not run
    yet -- needs spelling out; it counts as a mismatch unless both sides have it.
    """
    agree = (left == right) | (torch.isnan(left) & torch.isnan(right))
    return ~agree


def max_finite_diff(left: torch.Tensor, right: torch.Tensor) -> float:
    finite = torch.isfinite(left) & torch.isfinite(right)
    if not finite.any():
        return 0.0
    return float((left[finite] - right[finite]).abs().max())


def report(central: list[torch.Tensor], local: list[torch.Tensor]) -> int:
    """Per-round table; returns how many rounds disagreed.

    "silent" counts devices that had not run yet, which is why a round can show
    mismatches and still a zero numeric difference.
    """
    print()
    print(f"  {'round':>5}  {'mismatched':>10}  {'silent':>6}  {'max |diff|':>10}")
    print(f"  {'-' * 5}  {'-' * 10}  {'-' * 6}  {'-' * 10}")
    bad_rounds = 0
    for idx, (c, d) in enumerate(zip(central, local)):
        wrong = int(mismatches(c, d).sum())
        bad_rounds += wrong > 0
        silent = int(torch.isnan(d).sum())
        print(
            f"  {idx:>5}  {wrong:>10}  {silent:>6}  {max_finite_diff(c, d):>10.3g}"
        )
    return bad_rounds


# ----------------------------------------------------------------------
# plots
# ----------------------------------------------------------------------
def plot_comparison(central, local, signals, args, src_pos, dst_pos, path):
    if plt is None:
        print("matplotlib unavailable; skipping figure")
        return
    rows, cols = args.rows, args.cols
    obstacle = signals["obstacle"]
    diff = (central - local).abs()

    fig, axes = plt.subplots(1, 3, figsize=(13, 4.2))
    panels = [
        (axes[0], central, "centralized (batched)", "viridis"),
        (axes[1], local, "decentralized (Mesa, per device)", "viridis"),
        (axes[2], diff, "|difference|", "magma"),
    ]
    for ax, tensor, title, cmap in panels:
        ax.imshow(to_grid(tensor, rows, cols), cmap=cmap, vmin=0.0, vmax=1.0)
        draw_obstacles(ax, obstacle, rows, cols)
        draw_markers(ax, src_pos, dst_pos)
        ax.set_title(title)
        ax.set_xticks([])
        ax.set_yticks([])

    fig.suptitle("Same program, two executions", y=0.99)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=140)
    plt.close(fig)
    print(f"\nFigure written to {path}")


def plot_convergence(central, local, path, label):
    if plt is None:
        return
    final = central[-1]
    curve = [int(mismatches(final, step).sum()) for step in local]
    fig, ax = plt.subplots(figsize=(6.5, 3.6))
    ax.plot(curve, marker="o", markersize=3)
    ax.set_xlabel("round")
    ax.set_ylabel("nodes differing from the\ncentralized fixed point")
    ax.set_title(label)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=140)
    plt.close(fig)
    print(f"Figure written to {path}")


# ----------------------------------------------------------------------
def main():
    args = parse_args()
    torch.manual_seed(args.seed)

    scenario, signals, src_pos, dst_pos, wall_col = build_scenario(args)
    program = make_program(args.tolerance)
    mode = "async" if args.asynchronous else "sync"

    degrees = torch.bincount(scenario.edge_index[1], minlength=scenario.num_nodes)
    print("=== Channel with obstacles: one program, two executions ===")
    print(f"Grid:    {args.rows}x{args.cols}, 8-connected")
    print(f"Source:  {src_pos}   Dest: {dst_pos}   Wall: column {wall_col}")
    print(
        f"Devices: {scenario.num_nodes}   links: {scenario.edge_index.shape[1]}"
        f"   degree: min {int(degrees.min())}, max {int(degrees.max())}"
    )
    print(f"Rounds:  {args.rounds}   activation: {mode}", end="")
    if args.asynchronous:
        print(f" (p={args.activation_prob})")
    else:
        print(" (barrier per round)")

    central = run_centralized(scenario, program, signals, args.rounds)
    result = run_decentralized(
        scenario=scenario,
        program=program,
        signals=signals,
        rounds=args.rounds,
        mode=mode,
        activation_prob=args.activation_prob,
        seed=args.seed,
    )

    bad_rounds = report(central, result.fields)
    print()
    if args.asynchronous:
        final_wrong = int(mismatches(central[-1], result.final).sum())
        print(
            f"{bad_rounds}/{args.rounds} rounds differ in the transient, as expected "
            "without a barrier."
        )
        print(
            f"Final field: {final_wrong} node(s) differ from the centralized fixed point."
        )
        plot_convergence(
            central,
            result.fields,
            f"{args.out_prefix}_async_convergence.png",
            "Asynchronous devices converge to the centralized fixed point",
        )
    elif bad_rounds == 0:
        print(
            f"All {args.rounds} rounds identical, node for node: the decentralized "
            "execution reproduces the batched one exactly."
        )
    else:
        print(f"{bad_rounds}/{args.rounds} rounds differ -- this should not happen in sync mode.")

    if not args.no_viz:
        plot_comparison(
            central[-1],
            result.final,
            signals,
            args,
            src_pos,
            dst_pos,
            f"{args.out_prefix}_{mode}.png",
        )

    return 0 if (args.asynchronous or bad_rounds == 0) else 1


if __name__ == "__main__":
    raise SystemExit(main())
