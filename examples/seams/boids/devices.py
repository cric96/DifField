"""Boids on independent devices: one DeviceRuntime per agent, in step or asynchronous.

The trained program runs unchanged on every agent, which sees only the agents
within sensing range (its star graph) and senses their positions and
velocities. In synchronous rounds the devices must reproduce the batched
rollouts of the adaptation stage exactly. In asynchronous rounds an agent
recomputes its steering with probability ``activation`` and otherwise keeps
applying its last command (sample and hold). The teacher, written as the same
program with its own weights, runs under the same activations as reference.
"""

from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import torch

from diffield import AggregateContext

from ...shared.artifacts import json_write, read_json, software
from ...shared.devices import DeviceNetwork
from ...shared.metrics import recovery_summary, relative_deviation
from ...shared.randomness import seed_for
from .. import SOURCES
from .adaptation import CURVES, EVENTS, episode_metrics, events, initial_states, macrostate
from .adaptation import configuration as adaptation_configuration
from .dynamics import Boids, integrate, radius_edges, teacher
from .learning import load_model

RUNS = (("parametric", "sync"), ("parametric", "async"), ("oracle", "async"))


def configuration(profile):
    config = adaptation_configuration(profile)
    return {**config, "size": 64, "activation": 0.5, "seed": 0}


def active_agents(generator, count, t, probability):
    """The agents that compute a new command in round ``t``: all of them in the first round,
    then each with ``probability``. Every round draws once per agent."""
    draws = torch.rand(count, generator=generator)
    return draws < probability if t > 0 else torch.ones(count, dtype=torch.bool)


@torch.no_grad()
def device_rollout(model, positions, velocities, rounds, schedule, *, activation=1.0, seed=0):
    """Closed loop with one runtime per agent; the first round runs everywhere."""

    def program(runtime):
        return model.acceleration(runtime.signals["position"], runtime.signals["velocity"])

    p, v = positions.clone(), velocities.clone()
    command = torch.zeros_like(p)
    generator = torch.Generator().manual_seed(seed)
    network, curves = None, {k: [] for k in (*CURVES, "activated")}
    for t in range(rounds):
        if t in schedule:
            before = p
            p, v = schedule[t](p, v)
            if len(p) < len(before):  # crashed agents leave; survivors keep their commands
                command = command[torch.cdist(p, before).argmin(1)]
            elif len(p) > len(before):  # newcomers have no command yet
                command = torch.cat((command, command.new_zeros(len(p) - len(before), 2)))
        edges, signals = radius_edges(p), {"position": p, "velocity": v}
        if network is None or len(network.devices) != len(p):
            # Identities change with the population: fresh runtimes, holding the commands.
            network = DeviceNetwork(program, edges, len(p), command, signals=signals)
        else:
            network.move(edges, signals=signals)
        active = active_agents(generator, len(p), t, activation)
        network.run(active.nonzero().flatten().tolist())
        step = integrate(p, v, command)
        p, v = step.positions, step.velocities
        if not (torch.isfinite(p).all() and torch.isfinite(v).all()):
            return "diverged", curves, (p, v)
        for key, value in macrostate(p, v).items():
            curves[key].append(value)
        curves["activated"].append(float(active.float().mean()))
    return "complete", curves, (p, v)


@torch.no_grad()
def batched_rollout(model, positions, velocities, rounds, activation, seed):
    """The positions after every round of :func:`device_rollout` without events, computed on
    the whole swarm at once, so that the GNN baselines can run it too."""
    p, v = positions.clone(), velocities.clone()
    command = torch.zeros_like(p)
    generator = torch.Generator().manual_seed(seed)
    trajectory = []
    for t in range(rounds):
        edges = radius_edges(p)
        active = active_agents(generator, len(p), t, activation)
        if isinstance(model, Boids):
            with AggregateContext(edges, len(p)).round():
                acceleration = model.acceleration(p, v)
        else:
            acceleration = model.net(p, v, edges)
        command[active] = acceleration[active]
        step = integrate(p, v, command)
        p, v = step.positions, step.velocities
        trajectory.append(p.clone())
    return torch.stack(trajectory)


def sync_async_rmse(run):
    """Per-round position RMSE against the teacher on the test episodes of a Boids run, for
    every seed and model, on synchronous and asynchronous agents (no faults)."""
    source = read_json(run / "manifest.json")["config"]
    states = initial_states(run, source, source["nodes"], source["test_episodes"])
    rounds = source["rounds"]
    reference = teacher()

    def rmse(a, b):
        return (a - b).square().mean((-2, -1)).sqrt().numpy()

    references = {}
    for key, pos, vel in states:
        seed = seed_for(source["data_seed"], "boids-devices", key)
        references[key] = (
            batched_rollout(reference, pos, vel, rounds, 1.0, seed),
            batched_rollout(reference, pos, vel, rounds, 0.5, seed),
            seed,
        )
    curves = {
        "diffield_sync": [],
        "diffield_async": [],
        "diffield_async_vs_sync_teacher": [],
        "gnn64_sync": [],
        "gnn64_async": [],
        "gnn32_sync": [],
        "gnn32_async": [],
        "teacher_async": [rmse(t_async, t_sync) for t_sync, t_async, _ in references.values()],
    }
    for variant, prefix in (("parametric", "diffield"), ("gnn64", "gnn64"), ("gnn32", "gnn32")):
        for model_seed in source["seeds"]:
            model = load_model(run, variant, model_seed)
            for key, pos, vel in states:
                t_sync, t_async, seed = references[key]
                p_sync = batched_rollout(model, pos, vel, rounds, 1.0, seed)
                p_async = batched_rollout(model, pos, vel, rounds, 0.5, seed)
                curves[f"{prefix}_sync"].append(rmse(p_sync, t_sync))
                curves[f"{prefix}_async"].append(rmse(p_async, t_async))
                if prefix == "diffield":
                    curves["diffield_async_vs_sync_teacher"].append(rmse(p_async, t_sync))
    return {k: np.asarray(v, dtype=float) for k, v in curves.items()}


def episode(task):
    """All runs of one episode, so that methods share events and activations."""
    run, config, source, index = task
    torch.set_num_threads(1)
    key, positions, velocities = initial_states(run, source, config["size"], index + 1)[index]
    perturb = events(config, source, config["size"], index)
    schedule = {config["events"][name]: perturb[name] for name in EVENTS}
    models = {
        "parametric": load_model(run, "parametric", config["seed"]),
        "oracle": teacher(),
    }
    seed = seed_for(source["data_seed"], "boids-devices", key)
    rows = []
    for method, executor in RUNS:
        activation = 1.0 if executor == "sync" else config["activation"]
        status, curves, _ = device_rollout(models[method], positions, velocities, config["rounds"],
                                           schedule, activation=activation, seed=seed)  # fmt: skip
        rows.append({
            "method": method, "executor": executor, "activation": activation, "episode": key,
            "nodes": config["size"], "status": status, "curves": curves,
            "metrics": episode_metrics(curves, config) if status == "complete" else {},
        })  # fmt: skip
    batched = run / "adaptation/episodes/parametric" / f"seed{config['seed']}"
    batched = read_json(batched / f"N{config['size']}-{key}.json")["curves"]
    synchronous = rows[0]["curves"]
    rows[0]["equal_to_batched"] = all(synchronous[k] == batched[k] for k in CURVES)
    return rows


def mean(values):
    """The mean of ``values``, or ``None`` when there is none: every run diverged."""
    values = list(values)
    return float(np.mean(values)) if values else None


def deviation(rows, reference):
    """The mean over episodes of the spread's relative deviation from the reference's, over the
    episodes whose reference run completed."""
    return mean(
        relative_deviation(r["curves"]["spread"], reference[r["episode"]]["curves"]["spread"])
        for r in rows
        if r["episode"] in reference
    )


def summarize(rows, config):
    done = [r for r in rows if r["status"] == "complete"]
    teacher = {r["episode"]: r for r in done if (r["method"], r["executor"]) == ("oracle", "async")}
    synchronous = {
        r["episode"]: r for r in done if (r["method"], r["executor"]) == ("parametric", "sync")
    }
    summary = {}
    for method, executor in RUNS:
        panel = [r for r in rows if (r["method"], r["executor"]) == (method, executor)]
        complete = [r for r in panel if r["status"] == "complete"]
        entry = {"runs": len(panel), "diverged": len(panel) - len(complete)}
        for name in EVENTS:
            rounds = [r["metrics"][name]["recovery"]["rounds"] for r in complete]
            entry[name] = {
                **recovery_summary(rounds),
                "peak_change": mean(r["metrics"][name]["peak_change"] for r in complete),
            }
        entry["lcc_final"] = mean(r["metrics"]["lcc_final"] for r in complete)
        entry["activated"] = mean(np.mean(r["curves"]["activated"]) for r in complete)
        entry["deviation_from_teacher_async"] = deviation(complete, teacher)
        entry["deviation_from_synchronous"] = deviation(complete, synchronous)
        summary[f"{method}/{executor}"] = entry
    checks = [r["equal_to_batched"] for r in rows if "equal_to_batched" in r]
    summary["synchronous_equal_to_batched"] = {"equal": sum(checks), "episodes": len(checks)}
    return summary


def write_report(out, config, summary):
    checks = summary["synchronous_equal_to_batched"]
    lines = [
        "# Boids on independent devices",
        "",
        f"Profile **{config['profile']}**; {config['episodes']} episodes of {config['size']} "
        f"agents, {config['rounds']} rounds, the events of the adaptation stage. One "
        "DeviceRuntime per agent; asynchronous agents recompute their steering with "
        f"probability {config['activation']} per round and otherwise hold their last command. "
        "Deviations: mean |spread - reference spread| relative to the reference mean.",
        "",
        f"Synchronous devices equal the batched rollouts in {checks['equal']}/"
        f"{checks['episodes']} episodes (every curve, every round).",
        "",
        "| Method | Executor | Runs | Diverged | Split: recovered, median (max) | "
        "Join: recovered, median (max) | Final LCC | Activated | vs teacher async "
        "| vs synchronous |",
        "|---|---|---:|---:|---|---|---:|---:|---:|---:|",
    ]

    def recovered(e):
        return f"{e['recovered']}, {e['median_rounds']} ({e['max_rounds']})"

    def fmt(value):
        return "-" if value is None else f"{value:.3f}"

    for method, executor in RUNS:
        e = summary[f"{method}/{executor}"]
        lines.append(
            f"| {method} | {executor} | {e['runs']} | {e['diverged']} | {recovered(e['split'])} "
            f"| {recovered(e['join'])} | {fmt(e['lcc_final'])} | {fmt(e['activated'])} "
            f"| {fmt(e['deviation_from_teacher_async'])} | {fmt(e['deviation_from_synchronous'])} |"
        )
    (out / "REPORT.md").write_text("\n".join(lines) + "\n")


def campaign(run: Path, profile="paper-cpu", workers=None):
    """``run`` is a Boids campaign with its adaptation stage; results go to ``run/devices``."""
    config = configuration(profile)
    source = read_json(run / "manifest.json")["config"]
    out = run / "devices"
    json_write(out / "manifest.json", {"config": config, "campaign": source,
                                        "software": software(SOURCES)})  # fmt: skip
    tasks = [(run, config, source, index) for index in range(config["episodes"])]
    with ProcessPoolExecutor(max_workers=workers) as pool:
        rows = [row for result in pool.map(episode, tasks) for row in result]
    for row in rows:
        json_write(out / "episodes" / f"{row['method']}-{row['executor']}" /
                   f"N{row['nodes']}-{row['episode']}.json", row)  # fmt: skip
    summary = summarize(rows, config)
    json_write(out / "summary.json", {"config": config, "summary": summary})
    write_report(out, config, summary)
    print(f"Boids on devices: {out}", flush=True)
    return summary
