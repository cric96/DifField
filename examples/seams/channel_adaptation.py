"""Runtime adaptation of the self-healing channel: obstacle, clearance and relocation.

The channel program of the paper runs on a static radius graph while a large
obstacle hangs over the middle of the area, so that the devices under it stop
and the channel must detour below it; the obstacle is later cleared (the
devices rejoin with fresh state), and finally the destination moves. Every
round the channel is compared with the channel recomputed from exact shortest
paths on the live graph; a "frozen" plan, the channel as it was just before
the obstacle, is the reference without runtime adaptation.
"""

import time
from pathlib import Path

import numpy as np
import torch
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components, dijkstra

from diffield import SimulationEngine, SnapshotRecorder, SpatialScenario, branch
from diffield.dsl import field
from diffield.sim import EventSchedule, ScheduledEvent

from ..channel.core import CHANNEL_THRESHOLD, channel_body
from .artifacts import json_write, software
from .randomness import rng
from .space_fluid.metrics import recovery

EVENTS = ("obstacle", "clear", "relocate")


def configuration(profile):
    if profile not in ("smoke", "paper-cpu"):
        raise ValueError(profile)
    smoke = profile == "smoke"
    return {
        "profile": profile,
        "data_seed": 20261012,
        "nodes": 200 if smoke else 1000,
        "radius": 0.13 if smoke else 0.06,
        "tolerance": 0.02,
        "rounds": 200 if smoke else 700,
        "events": {"obstacle": 40, "clear": 120, "relocate": 160}
        if smoke
        else {"obstacle": 100, "clear": 400, "relocate": 550},
        "obstacle": [0.3, 0.7, 0.35, 1.0],  # x from, x to, y from, y to
        "source": [0.1, 0.9],
        "dest": [0.9, 0.1],
        "relocated": [0.9, 0.9],
        "seeds": [0, 1] if smoke else list(range(10)),
        "recovery": {"threshold": 0.05, "consecutive": 5},
        "stale_tolerance": 1e-3,
    }


def nearest(positions, point):
    return int((positions - torch.tensor(point)).norm(dim=-1).argmin())


def live_graph(scenario, failed):
    """Weighted adjacency of the edges whose endpoints are both alive."""
    a, b = scenario.edge_index.cpu().numpy()
    weight = scenario.edge_weight.detach().cpu().numpy()
    alive = ~failed.cpu().numpy()
    keep = alive[a] & alive[b] & (a != b)
    n = scenario.num_nodes
    return csr_matrix((weight[keep], (a[keep], b[keep])), shape=(n, n))


def oracle(graph, failed, src, dst, tolerance):
    """Exact distances and channel on the live graph (same rule as ``channel_body``)."""
    distances = dijkstra(graph, directed=True, indices=[src, dst])
    to_src, to_dst = distances
    alive = ~failed.cpu().numpy()
    channel = (
        alive
        & np.isfinite(to_src)
        & np.isfinite(to_dst)
        & (to_src + to_dst <= to_dst[src] + tolerance)
    )
    return channel, to_dst


def connected(graph, channel, src, dst):
    """A path from source to destination using only channel devices."""
    if not (channel[src] and channel[dst]):
        return False
    nodes = np.flatnonzero(channel)
    _, labels = connected_components(graph[nodes][:, nodes], directed=False)
    position = {node: k for k, node in enumerate(nodes.tolist())}
    return bool(labels[position[src]] == labels[position[dst]])


def f1(predicted, reference):
    tp = int((predicted & reference).sum())
    wrong = int((predicted ^ reference).sum())
    return 1.0 if tp + wrong == 0 else 2 * tp / (2 * tp + wrong)


def deployment(config, seed):
    """Uniform devices; the obstacle cuts the straight route and leaves a gap below it."""
    for attempt in range(100):
        generator = rng(config["data_seed"], "channel-adaptation", seed, attempt)
        positions = torch.rand(config["nodes"], 2, generator=generator)
        scenario = SpatialScenario(
            positions=positions, edge_radius=config["radius"], edge_weight_mode="distance"
        )
        src, dst, moved = (nearest(positions, config[k]) for k in ("source", "dest", "relocated"))
        none = torch.zeros(config["nodes"], dtype=torch.bool)
        x0, x1, y0, y1 = config["obstacle"]
        x, y = positions.unbind(-1)
        failed = (x >= x0) & (x <= x1) & (y >= y0) & (y <= y1)
        failed[[src, dst, moved]] = False
        reached = [dijkstra(live_graph(scenario, down), directed=True, indices=src)
                   for down in (none, failed)]  # fmt: skip
        if all(np.isfinite(d[[dst, moved]]).all() for d in reached):
            return {
                "scenario": scenario,
                "source": src,
                "dest": dst,
                "relocated": moved,
                "failed": failed,
                "attempt": attempt,
            }
    raise RuntimeError("No connected deployment found")


def make_program(tolerance):
    def program(runtime):
        signals = runtime.signals
        return branch(
            ~signals["failed"],
            lambda: channel_body(signals["source"], signals["dest"], tolerance),
            lambda: field.of(0.0),
            branch_name="failed",
        )

    return program


def run_seed(config, seed, extra=()):  # noqa: PLR0915 -- one episode loop with its metrics
    """One episode; returns per-round curves and per-event recovery.

    ``extra`` adds rounds to the saved channel snapshots (e.g. for paper figures).
    """
    world = deployment(config, seed)
    scenario, src = world["scenario"], world["source"]
    nodes, events = config["nodes"], config["events"]
    none = torch.zeros(nodes, dtype=torch.bool)
    signals = {
        "failed": none.clone(),
        "source": scenario.marker(src),
        "dest": scenario.marker(world["dest"]),
    }

    def obstacle(runtime):
        runtime.signals["failed"] = world["failed"].clone()

    def clear(runtime):
        runtime.signals["failed"] = none.clone()

    def relocate(runtime):
        runtime.signals["dest"] = scenario.marker(world["relocated"])

    schedule = EventSchedule(
        [ScheduledEvent(events[name], callback, name) for name, callback in
         zip(EVENTS, (obstacle, clear, relocate), strict=True)]
    )  # fmt: skip
    engine = SimulationEngine.from_scenario(scenario)
    runtime = engine.init_runtime(signals=signals)
    program = make_program(config["tolerance"])
    cache = {}
    curves = {k: [] for k in ("quality", "f1", "connected", "frozen_quality", "stale", "size")}
    snapshots, frozen = {}, None
    keep = {events["obstacle"] - 1, events["obstacle"] + 10, events["clear"] - 1,
            events["relocate"] - 1, events["relocate"] + 10, config["rounds"] - 1,
            *extra}  # fmt: skip
    started = time.perf_counter()
    for t in range(config["rounds"]):
        recorder = SnapshotRecorder(state_fields=["dist_dst"], capture_output=True)
        with torch.no_grad():
            output = engine.step(
                runtime=runtime, program=program, recorder=recorder, schedule=schedule
            )
        failed, dst = runtime.signals["failed"], int(runtime.signals["dest"].argmax())
        key = (bytes(failed.numpy()), dst)
        if key not in cache:
            graph = live_graph(scenario, failed)
            cache[key] = (graph, *oracle(graph, failed, src, dst, config["tolerance"]))
        graph, reference, true_dst = cache[key]
        channel = (output.detach().cpu().numpy() > CHANNEL_THRESHOLD) & ~failed.numpy()
        score, linked = f1(channel, reference), connected(graph, channel, src, dst)
        curves["f1"].append(score)
        curves["connected"].append(linked)
        curves["quality"].append(score * linked)
        curves["size"].append(int(channel.sum()))
        if t == events["obstacle"] - 1:
            frozen = channel.copy()
        plan = channel if frozen is None else frozen & ~failed.numpy()
        curves["frozen_quality"].append(f1(plan, reference) * connected(graph, plan, src, dst))
        estimate = recorder.records[t]["dist_dst"].cpu().numpy()
        reachable = np.isfinite(true_dst) & ~failed.numpy()
        error = np.abs(estimate[reachable] - true_dst[reachable])  # inf estimate: inf error
        curves["stale"].append(float((~(error <= config["stale_tolerance"])).mean()))
        if t in keep:
            snapshots[t] = {"channel": torch.from_numpy(channel), "failed": failed.clone()}
    errors = [1 - q for q in curves["quality"]]
    bounds = [*(events[name] for name in EVENTS), config["rounds"]]
    summary = {}
    for k, name in enumerate(EVENTS):
        start, stop = bounds[k], bounds[k + 1]
        summary[name] = {
            "round": start,
            "recovery": recovery(
                errors[:stop], start, config["recovery"]["threshold"],
                config["recovery"]["consecutive"],
            ),
            "quality_min": float(min(curves["quality"][start:stop])),
            "frozen_mean": float(np.mean(curves["frozen_quality"][start:stop])),
            "stale_end": curves["stale"][stop - 1],
        }  # fmt: skip
    degree = scenario.edge_index.shape[1] / nodes
    return {
        "seed": seed,
        "attempt": world["attempt"],
        "mean_degree": degree,
        "failed_devices": int(world["failed"].sum()),
        "seconds": time.perf_counter() - started,
        "events": summary,
        "curves": curves,
    }, {"positions": scenario.positions.clone(), "snapshots": snapshots,
        "source": src, "dest": world["dest"], "relocated": world["relocated"]}  # fmt: skip


def aggregate(runs, config):
    result = {}
    for name in EVENTS:
        entries = [r["events"][name] for r in runs]
        rounds = [e["recovery"]["rounds"] for e in entries if e["recovery"]["rounds"] is not None]
        result[name] = {
            "round": config["events"][name],
            "recovered": len(rounds),
            "censored": len(entries) - len(rounds),
            "median_rounds": float(np.median(rounds)) if rounds else None,
            "min_rounds": min(rounds) if rounds else None,
            "max_rounds": max(rounds) if rounds else None,
            "quality_min_mean": float(np.mean([e["quality_min"] for e in entries])),
            "frozen_mean": float(np.mean([e["frozen_mean"] for e in entries])),
            "stale_end_mean": float(np.mean([e["stale_end"] for e in entries])),
            "stale_end_range": [
                float(min(e["stale_end"] for e in entries)),
                float(max(e["stale_end"] for e in entries)),
            ],
        }
    return result


def write_report(out, config, runs, summary):
    lines = [
        "# Self-healing channel: runtime adaptation",
        "",
        f"Profile **{config['profile']}**; {config['nodes']} devices, radius graph "
        f"{config['radius']} (mean degree {np.mean([r['mean_degree'] for r in runs]):.1f}), "
        f"tolerance {config['tolerance']}, {len(runs)} seeds, {config['rounds']} rounds. "
        "Quality = F1 against the channel recomputed with exact shortest paths on the live "
        "graph, times source-destination connectivity through channel devices. Recovery: "
        f"rounds until quality >= {1 - config['recovery']['threshold']:.2f} for "
        f"{config['recovery']['consecutive']} rounds, cut at the next event. Frozen: the "
        "channel just before the obstacle, never updated.",
        "",
        "| Event | Round | Recovered | Median | Min | Max | Min quality | Frozen quality "
        "| Stale distances at window end |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name in EVENTS:
        s = summary[name]
        lines.append(
            f"| {name} | {s['round']} | {s['recovered']}/{len(runs)} | {s['median_rounds']} "
            f"| {s['min_rounds']} | {s['max_rounds']} | {s['quality_min_mean']:.3f} "
            f"| {s['frozen_mean']:.3f} | {s['stale_end_mean']:.3f} "
            f"[{s['stale_end_range'][0]:.3f}, {s['stale_end_range'][1]:.3f}] |"
        )
    failed = [r["failed_devices"] for r in runs]
    lines += ["", f"Devices under the obstacle per seed: {min(failed)}-{max(failed)}.", ""]
    (out / "REPORT.md").write_text("\n".join(lines))


def campaign(out: Path, profile="paper-cpu"):
    torch.set_num_threads(1)
    config = configuration(profile)
    out.mkdir(parents=True, exist_ok=True)
    json_write(out / "manifest.json", {"config": config, "software": software()})
    runs = []
    for seed in config["seeds"]:
        print(f"Channel adaptation seed={seed}", flush=True)
        run, snapshot = run_seed(config, seed)
        json_write(out / "runs" / f"seed{seed}.json", run)
        if seed == config["seeds"][0]:
            (out / "snapshots").mkdir(parents=True, exist_ok=True)
            torch.save(snapshot, out / "snapshots" / f"seed{seed}.pt")
        runs.append(run)
    summary = aggregate(runs, config)
    json_write(out / "summary.json", {"config": config, "events": summary})
    write_report(out, config, runs, summary)
    print(f"Channel adaptation artifacts: {out}", flush=True)
    return summary
