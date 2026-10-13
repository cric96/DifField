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

from diffield.decentralized.runner import build_runtimes
from diffield.decentralized.topology import all_in_edges

from .artifacts import json_write, read_json, software
from .boids_adaptation import CURVES, EVENTS, episode_metrics, events, initial_states, macrostate
from .boids_adaptation import configuration as adaptation_configuration
from .boids_dynamics import TEACHER, Boids, integrate, radius_edges
from .boids_learning import make_model
from .randomness import seed_for

RUNS = (("parametric", "sync"), ("parametric", "async"), ("oracle", "async"))


def configuration(profile):
    config = adaptation_configuration(profile)
    return {**config, "size": 64, "activation": 0.5, "seed": 0}


def teacher_program():
    """The teacher as a field program: the DifField model with the teacher's weights."""
    model = Boids()
    with torch.no_grad():
        model.log_weights.copy_(torch.tensor(TEACHER).log())
    return model.eval()


def learned_program(run, seed):
    model = make_model("parametric", seed)
    checkpoint = run / "jobs/observed/parametric" / f"seed{seed}" / "best.pt"
    model.load_state_dict(torch.load(checkpoint, weights_only=True)["model"])
    return model.eval()


@torch.no_grad()
def device_rollout(model, positions, velocities, rounds, schedule, *, activation=1.0, seed=0):
    """Closed loop with one runtime per agent; the first round runs everywhere."""

    def program(runtime):
        return model.acceleration(runtime.signals["position"], runtime.signals["velocity"])

    p, v = positions.clone(), velocities.clone()
    command = torch.zeros_like(p)
    generator = torch.Generator().manual_seed(seed)
    devices, curves = {}, {k: [] for k in (*CURVES, "activated")}
    for t in range(rounds):
        if t in schedule:
            before = p
            p, v = schedule[t](p, v)
            if len(p) < len(before):  # crashed agents leave; survivors keep their commands
                command = command[torch.cdist(p, before).argmin(1)]
            elif len(p) > len(before):  # newcomers have no command yet
                command = torch.cat((command, command.new_zeros(len(p) - len(before), 2)))
        edges, signals = radius_edges(p), {"position": p, "velocity": v}
        if len(devices) != len(p):  # identities change with the population: fresh runtimes
            devices = build_runtimes(edge_index=edges, num_nodes=len(p), program=program,
                                     signals=signals)  # fmt: skip
        else:
            for node, links in enumerate(all_in_edges(edges, len(p))):
                devices[node].sync_topology(*links, signals=signals)
        draws = torch.rand(len(p), generator=generator)
        active = draws < activation if t > 0 else torch.ones(len(p), dtype=torch.bool)
        for node in active.nonzero().flatten().tolist():
            command[node], _ = devices[node].step({})
        step = integrate(p, v, command)
        p, v = step.positions, step.velocities
        if not (torch.isfinite(p).all() and torch.isfinite(v).all()):
            return "diverged", curves, (p, v)
        for key, value in macrostate(p, v).items():
            curves[key].append(value)
        curves["activated"].append(float(active.float().mean()))
    return "complete", curves, (p, v)


def episode(task):
    """All runs of one episode, so that methods share events and activations."""
    run, config, source, index = task
    torch.set_num_threads(1)
    key, positions, velocities = initial_states(run, source, config["size"], index + 1)[index]
    perturb = events(config, source, config["size"], index)
    schedule = {config["events"][name]: perturb[name] for name in EVENTS}
    models = {"parametric": learned_program(run, config["seed"]), "oracle": teacher_program()}
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


def deviation(rows, reference):
    """Mean |spread - reference spread| relative to the reference mean, per episode."""
    ratios = []
    for row in rows:
        other = np.array(reference[row["episode"]]["curves"]["spread"])
        ratios.append(np.mean(np.abs(np.array(row["curves"]["spread"]) - other)) / other.mean())
    return float(np.mean(ratios))


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
            rounds = [r["metrics"][name]["recovery"]["rounds"] for r in complete
                      if r["metrics"][name]["recovery"]["rounds"] is not None]  # fmt: skip
            entry[name] = {
                "recovered": len(rounds),
                "median_rounds": float(np.median(rounds)) if rounds else None,
                "max_rounds": max(rounds) if rounds else None,
                "peak_change": float(
                    np.mean([r["metrics"][name]["peak_change"] for r in complete])
                ),
            }
        entry["lcc_final"] = float(np.mean([r["metrics"]["lcc_final"] for r in complete]))
        entry["activated"] = float(np.mean([np.mean(r["curves"]["activated"]) for r in complete]))
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

    for method, executor in RUNS:
        e = summary[f"{method}/{executor}"]
        lines.append(
            f"| {method} | {executor} | {e['runs']} | {e['diverged']} | {recovered(e['split'])} "
            f"| {recovered(e['join'])} | {e['lcc_final']:.3f} | {e['activated']:.3f} "
            f"| {e['deviation_from_teacher_async']:.3f} | {e['deviation_from_synchronous']:.3f} |"
        )
    (out / "REPORT.md").write_text("\n".join(lines) + "\n")


def campaign(run: Path, profile="paper-cpu", workers=None):
    """``run`` is a Boids campaign with its adaptation stage; results go to ``run/devices``."""
    config = configuration(profile)
    source = read_json(run / "manifest.json")["config"]
    out = run / "devices"
    json_write(out / "manifest.json", {"config": config, "campaign": source,
                                        "software": software()})  # fmt: skip
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
