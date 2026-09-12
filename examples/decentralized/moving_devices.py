#!/usr/bin/env python3
"""Devices that move, and a field that keeps up with them.

Same idea as ``channel_obstacles.py`` -- one program, run centrally and device
by device -- but here the network itself will not hold still.  Devices drift,
links form and break under them, and then the motion decays until they park.

What the animation shows:

* **left** -- the live network. Nodes sit at their current positions, coloured
  by the distance field they are computing; grey means "no route to the source
  yet". Links are redrawn every round, so you can watch the topology churn.
* **right** -- the mean distance over the network, for both executions. The
  centralized curve is drawn thick and the decentralized one dashed on top of
  it; they coincide because the two runs agree node for node. The curve climbs
  and dips while the devices move and flattens once they park -- that flat tail
  is the field re-converging on the geometry it was left with.

Usage::

    uv run --extra cpu --extra decentralized \\
        python examples/decentralized/moving_devices.py
    uv run --extra cpu --extra decentralized \\
        python examples/decentralized/moving_devices.py --format mp4 --async
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "examples"))

import torch
from shared.plotting.common import FuncAnimation, LineCollection, PillowWriter, plt

from diffield import (
    EventSchedule,
    ScheduledEvent,
    SimulationEngine,
    SpatialScenario,
    gradient,
)
from diffield.decentralized import run_decentralized

SOURCE = 0
#: Below this the devices count as parked and the topology stops changing.
STOPPED = 1e-6


def parse_args():
    parser = argparse.ArgumentParser(description="A field tracking moving devices")
    parser.add_argument("--devices", type=int, default=60, help="Number of devices")
    parser.add_argument("--radius", type=float, default=0.32, help="Communication range")
    parser.add_argument("--rounds", type=int, default=80, help="Compute rounds")
    parser.add_argument("--speed", type=float, default=0.010, help="Initial drift speed")
    parser.add_argument(
        "--drift-rounds",
        type=int,
        default=None,
        help="Rounds at full speed before the devices start slowing (default: half)",
    )
    parser.add_argument(
        "--decay",
        type=float,
        default=0.5,
        help="Per-round speed multiplier once slowing starts",
    )
    parser.add_argument(
        "--async",
        dest="asynchronous",
        action="store_true",
        help="Drop the round barrier on the decentralized side",
    )
    parser.add_argument("--activation-prob", type=float, default=0.6)
    parser.add_argument("--seed", type=int, default=11, help="Random seed")
    parser.add_argument("--fps", type=int, default=8, help="Animation frame rate")
    parser.add_argument(
        "--format",
        choices=("gif", "mp4", "both"),
        default="gif",
        help="Animation container; mp4 needs ffmpeg",
    )
    parser.add_argument("--no-anim", action="store_true", help="Report only, no animation")
    parser.add_argument(
        "--out-prefix", type=str, default="generated/decentralized_moving"
    )
    return parser.parse_args()


# ----------------------------------------------------------------------
# scenario and motion
# ----------------------------------------------------------------------
def make_scenario(positions, radius):
    return SpatialScenario(
        positions=positions.clone(), edge_radius=radius, edge_weight_mode="distance"
    )


def program(runtime):
    """The shared closure — identical for both executions."""
    return gradient(runtime.signals["source"], name="dist")


def drift_schedule(
    rounds: int, drift_rounds: int, decay: float, geometry: list | None = None
):
    """Devices move at full speed, then slow to a halt.

    Full speed for ``drift_rounds`` so the topology really churns, then damped
    by ``decay`` each round until the motion is negligible and they park -- at
    which point the topology stops changing and the field can settle.

    Pass ``geometry`` to also record the positions and links each round *uses*,
    which is what the animation draws.
    """

    def move(runtime):
        velocities = runtime.metadata["velocities"]
        if float(velocities.abs().max()) < STOPPED:
            return
        runtime.scenario.step_positions(velocities)
        if runtime.round_idx < drift_rounds:
            return
        damped = velocities * decay
        runtime.metadata["velocities"] = (
            damped if float(damped.abs().max()) >= STOPPED else torch.zeros_like(damped)
        )

    def record(runtime):
        scenario = runtime.scenario
        geometry.append(
            {
                "positions": scenario.positions.detach().cpu().clone(),
                "edge_index": scenario.edge_index.detach().cpu().clone(),
                "speed": float(runtime.metadata["velocities"].abs().max()),
            }
        )

    events = []
    for round_idx in range(rounds):
        events.append(ScheduledEvent(round_idx=round_idx, callback=move, name="move"))
        if geometry is not None:
            events.append(
                ScheduledEvent(round_idx=round_idx, callback=record, name="record")
            )
    return EventSchedule(events)


# ----------------------------------------------------------------------
# runs
# ----------------------------------------------------------------------
def run_centralized(scenario, velocities, rounds, schedule):
    engine = SimulationEngine.from_scenario(scenario)
    runtime = engine.init_runtime(
        signals={"source": scenario.marker(SOURCE)},
        metadata={"velocities": velocities.clone()},
    )
    with torch.no_grad():
        return [
            engine.step(runtime=runtime, program=program, schedule=schedule)
            .detach()
            .cpu()
            .clone()
            for _ in range(rounds)
        ]


def mismatches(left, right):
    agree = (left == right) | (torch.isnan(left) & torch.isnan(right))
    return ~agree


def true_distance_field(scenario, source: int) -> torch.Tensor:
    """Weighted shortest paths on the final layout, computed outside diffield.

    The yardstick for "has the field finished adapting": once the devices park,
    the program should settle on exactly this.
    """
    import networkx as nx

    graph = nx.Graph()
    graph.add_nodes_from(range(scenario.num_nodes))
    edges = scenario.edge_index.cpu().t().tolist()
    weights = scenario.edge_weight.detach().cpu().tolist()
    for (source_node, target_node), weight in zip(edges, weights):
        if source_node != target_node:
            graph.add_edge(int(source_node), int(target_node), weight=weight)

    lengths = nx.single_source_dijkstra_path_length(graph, source, weight="weight")
    field = torch.full((scenario.num_nodes,), float("inf"))
    for node, length in lengths.items():
        field[node] = length
    return field


def max_finite_diff(left, right):
    """Largest disagreement over nodes both runs have a finite value for."""
    comparable = torch.isfinite(left) & torch.isfinite(right)
    if not comparable.any():
        return 0.0
    return float((left[comparable] - right[comparable]).abs().max())


def mean_finite(field):
    finite = torch.isfinite(field)
    return float(field[finite].mean()) if finite.any() else float("nan")


# ----------------------------------------------------------------------
# animation
# ----------------------------------------------------------------------
def undirected_segments(positions, edge_index):
    """Unique line segments for the current links."""
    seen = set()
    segments = []
    for source, target in edge_index.t().tolist():
        if source == target:
            continue
        key = (min(source, target), max(source, target))
        if key in seen:
            continue
        seen.add(key)
        segments.append([positions[key[0]].tolist(), positions[key[1]].tolist()])
    return segments


def animate(geometry, central, local, target_mean, args, paths):
    if plt is None or FuncAnimation is None:
        print("matplotlib animation tools unavailable; skipping animation")
        return

    rounds = len(geometry)
    finite_values = torch.cat([f[torch.isfinite(f)] for f in local if torch.isfinite(f).any()])
    vmax = float(finite_values.max()) if finite_values.numel() else 1.0

    all_positions = torch.stack([frame["positions"] for frame in geometry])
    low = float(all_positions.min()) - 0.05
    high = float(all_positions.max()) + 0.05

    central_means = [mean_finite(f) for f in central]
    local_means = [mean_finite(f) for f in local]
    parked = next(
        (idx for idx, frame in enumerate(geometry) if frame["speed"] < STOPPED), None
    )

    fig, (ax_net, ax_curve) = plt.subplots(1, 2, figsize=(12, 5.2))
    scatter = None

    def draw(round_idx):
        nonlocal scatter
        frame = geometry[round_idx]
        positions = frame["positions"]
        field = local[round_idx]

        ax_net.clear()
        ax_net.add_collection(
            LineCollection(
                undirected_segments(positions, frame["edge_index"]),
                colors="#b9c6d4",
                linewidths=0.7,
                zorder=1,
            )
        )

        finite = torch.isfinite(field)
        if (~finite).any():
            ax_net.scatter(
                positions[~finite, 0], positions[~finite, 1],
                c="#d0d0d0", s=45, zorder=2, edgecolors="white", linewidths=0.6,
            )
        scatter = ax_net.scatter(
            positions[finite, 0], positions[finite, 1],
            c=field[finite], cmap="viridis", vmin=0.0, vmax=vmax,
            s=45, zorder=3, edgecolors="white", linewidths=0.6,
        )
        ax_net.plot(
            positions[SOURCE, 0], positions[SOURCE, 1],
            "*", color="crimson", markersize=18, markeredgecolor="white", zorder=4,
        )

        moving = frame["speed"] >= STOPPED
        worst = max_finite_diff(central[round_idx], field)
        ax_net.set_xlim(low, high)
        ax_net.set_ylim(low, high)
        ax_net.set_xticks([])
        ax_net.set_yticks([])
        ax_net.set_title(
            f"round {round_idx:>3}   links {len(undirected_segments(positions, frame['edge_index']))}"
            f"   devices {'moving' if moving else 'parked'}\n"
            f"max |centralized − decentralized| = {worst:.3g}"
        )

        ax_curve.clear()
        upto = range(round_idx + 1)
        ax_curve.plot(
            list(upto), central_means[: round_idx + 1],
            color="#1f3b5c", linewidth=3.0, label="centralized (batched)",
        )
        ax_curve.plot(
            list(upto), local_means[: round_idx + 1],
            color="#ff9f1c", linewidth=1.6, linestyle="--",
            label="decentralized (Mesa, per device)",
        )
        ax_curve.axhline(
            target_mean, color="#c1121f", linestyle=":", linewidth=1.4,
            label="true metric of the final layout",
        )
        if parked is not None and round_idx >= parked:
            ax_curve.axvline(parked, color="#999999", linestyle=":", linewidth=1.2)
            ax_curve.text(
                parked, ax_curve.get_ylim()[1], " devices parked",
                va="top", fontsize=8, color="#666666",
            )
        ax_curve.set_xlim(0, rounds - 1)
        finite_means = [m for m in central_means if not math.isnan(m)]
        ax_curve.set_ylim(0, max(target_mean, *finite_means) * 1.15)
        ax_curve.set_xlabel("round")
        ax_curve.set_ylabel("mean distance to source")
        ax_curve.set_title("the field tracking a moving network")
        ax_curve.grid(alpha=0.3)
        ax_curve.legend(loc="lower right", fontsize=9)

    draw(0)
    fig.colorbar(scatter, ax=ax_net, fraction=0.046, pad=0.04, label="distance")
    fig.tight_layout()

    animation = FuncAnimation(fig, draw, frames=range(rounds))
    for path in paths:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        if path.endswith(".mp4"):
            from matplotlib.animation import FFMpegWriter

            try:
                animation.save(path, writer=FFMpegWriter(fps=args.fps, bitrate=2400))
            except (FileNotFoundError, RuntimeError) as exc:
                print(f"Could not write {path} ({exc}); is ffmpeg installed?")
                continue
        else:
            animation.save(path, writer=PillowWriter(fps=args.fps))
        print(f"Animation written to {path}")
    plt.close(fig)


# ----------------------------------------------------------------------
def main():
    args = parse_args()
    if args.drift_rounds is None:
        args.drift_rounds = args.rounds // 2
    generator = torch.Generator().manual_seed(args.seed)
    positions = torch.rand(args.devices, 2, generator=generator)
    velocities = (torch.rand(args.devices, 2, generator=generator) - 0.5) * args.speed
    mode = "async" if args.asynchronous else "sync"

    print("=== A field tracking moving devices ===")
    print(f"Devices: {args.devices}   range: {args.radius}   rounds: {args.rounds}")
    print(
        f"Motion:  speed {args.speed} for {args.drift_rounds} rounds, "
        f"then decaying x{args.decay} per round"
    )
    print(f"Decentralized activation: {mode}", end="")
    print(f" (p={args.activation_prob})" if args.asynchronous else " (barrier per round)")

    geometry: list[dict] = []
    central_scenario = make_scenario(positions, args.radius)
    central = run_centralized(
        central_scenario,
        velocities,
        args.rounds,
        drift_schedule(args.rounds, args.drift_rounds, args.decay, geometry),
    )

    local_scenario = make_scenario(positions, args.radius)
    result = run_decentralized(
        scenario=local_scenario,
        program=program,
        signals={"source": local_scenario.marker(SOURCE)},
        rounds=args.rounds,
        mode=mode,
        activation_prob=args.activation_prob,
        seed=args.seed,
        metadata={"velocities": velocities.clone()},
        schedule=drift_schedule(args.rounds, args.drift_rounds, args.decay),
    )

    # Both runs moved the same devices the same way, so one recording serves both.
    assert torch.allclose(central_scenario.positions, local_scenario.positions)
    assert torch.equal(central_scenario.edge_index, local_scenario.edge_index)

    bad = sum(int(mismatches(c, d).any()) for c, d in zip(central, result.fields))
    parked = next(
        (idx for idx, frame in enumerate(geometry) if frame["speed"] < STOPPED), None
    )
    print()
    print(f"Rounds that rewired someone's neighbourhood: {len(result.retopologized_rounds)}"
          f" / {args.rounds}")
    print(f"Devices parked at round: {parked if parked is not None else 'still moving'}")
    if args.asynchronous:
        final_wrong = int(mismatches(central[-1], result.final).sum())
        print(f"{bad}/{args.rounds} rounds differ in the transient, as expected without a barrier.")
        print(f"Final field: {final_wrong} node(s) differ from the centralized one.")
    elif bad == 0:
        print(f"All {args.rounds} rounds identical, node for node — while the graph was moving.")
    else:
        print(f"{bad}/{args.rounds} rounds differ — this should not happen in sync mode.")

    # Has it finished adapting? Measure, do not assume.
    target = true_distance_field(local_scenario, SOURCE)
    reachable = torch.isfinite(target)
    gap = max_finite_diff(target, result.final)
    stranded = int((torch.isfinite(result.final) & ~reachable).sum())
    settled = gap < 1e-5 and stranded == 0

    print(
        f"Distance from the true metric of the final layout: {gap:.3g}"
        f"  ({'settled' if settled else 'still healing'})"
    )
    if gap >= 1e-5:
        print(
            "  Healing is slow by nature: when a link breaks, distances have to *grow*,"
            "\n  and a min-based gradient raises them a step at a time rather than a hop"
            "\n  per round. Give it more --rounds after the devices park."
        )
    if stranded:
        print(
            f"  {stranded} device(s) lost every route to the source but still hold a stale"
            "\n  finite estimate, trading it round after round with each other and climbing"
            "\n  slowly. That is count-to-infinity, a property of a min-based gradient"
            "\n  rather than of decentralization — the batched run does exactly the same,"
            "\n  which is why the two still agree node for node."
        )

    if not args.no_anim:
        suffixes = {"gif": [".gif"], "mp4": [".mp4"], "both": [".gif", ".mp4"]}
        animate(
            geometry, central, result.fields, mean_finite(target), args,
            [f"{args.out_prefix}_{mode}{suffix}" for suffix in suffixes[args.format]],
        )

    return 0 if (args.asynchronous or bad == 0) else 1


if __name__ == "__main__":
    raise SystemExit(main())
