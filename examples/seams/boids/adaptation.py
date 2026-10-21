"""Boids under runtime perturbations never seen in training: split, crash, newcomers.

Reads a finished Boids campaign (``python -m examples.seams boids``) and rolls
every method out in closed loop for far longer than the 24 training rounds, at
the training size and at twice that size. Event draws come from named streams,
so every method meets the same perturbation. A non-finite rollout is recorded
as diverged, never raised.
"""

from pathlib import Path

import numpy as np
import torch
from scipy.sparse.csgraph import connected_components

from diffield.sim import normalize_vectors

from ...shared.artifacts import csv_write, json_write, read_json, software, tensor_write
from ...shared.metrics import (
    event_windows,
    recovery,
    recovery_summary,
    relative_deviation,
)
from ...shared.randomness import rng
from .. import SOURCES
from .dynamics import RADIUS, SEPARATION, SPEED, Boids, dense_step, radius_edges, swarm
from .learning import METHODS, TRACES, VARIANTS, load_model

EVENTS = ("split", "crash", "join")
CURVES = ("spread", "lcc", "speed", "polarization", "crowding", "nodes")


def configuration(profile):
    if profile not in ("smoke", "paper-cpu"):
        raise ValueError(profile)
    smoke = profile == "smoke"
    return {
        "profile": profile,
        "sizes": [32, 64],
        "rounds": 40 if smoke else 400,
        "episodes": 2 if smoke else 20,
        "events": {"split": 10, "crash": 20, "join": 30}
        if smoke
        else {"split": 100, "crash": 200, "join": 300},
        "split_shift": 0.1,
        "crash_fraction": 0.25,
        "join_fraction": 0.25,
        "join_radius": 0.25,
        "recovery": {"threshold": 0.15, "consecutive": 5},
    }


def initial_states(run, campaign, size, count):
    """The campaign's test states at its size; fresh states, same recipe, otherwise."""
    if size == campaign["nodes"]:
        test = torch.load(run / TRACES, weights_only=True)["test"][:count]
        return [(e["key"], e["positions"], e["velocities"]) for e in test]
    states = []
    for i in range(count):
        positions, velocities = swarm(size, rng(campaign["data_seed"], "boids-adapt", size, i))
        states.append((f"boids-adapt-{size}-{i}", positions, velocities))
    return states


def events(config, campaign, size, episode):
    """Perturbations as functions of the current state; draws are method-independent."""
    count = round(config["crash_fraction"] * size)
    joining = round(config["join_fraction"] * size)

    def stream(name):
        return rng(campaign["data_seed"], "boids-adapt-event", name, size, episode)

    def split(p, v):
        side = torch.where(p[:, 0] < p[:, 0].mean(), -1.0, 1.0)
        shifted = p.clone()
        shifted[:, 0] = (p[:, 0] + side * config["split_shift"]).clamp(0, 1)
        return shifted, v

    def crash(p, v):
        keep = torch.randperm(len(p), generator=stream("crash"))[: len(p) - count].sort().values
        return p[keep], v[keep]

    def join(p, v):
        generator = stream("join")
        angle = 2 * torch.pi * torch.rand(joining, generator=generator)
        ring = torch.stack((angle.cos(), angle.sin()), -1) * config["join_radius"]
        heading = normalize_vectors(torch.randn(joining, 2, generator=generator)) * SPEED
        newcomers = (p.mean(0) + ring).clamp(0, 1)
        return torch.cat((p, newcomers)), torch.cat((v, heading))

    return {"split": split, "crash": crash, "join": join}


def macrostate(p, v):
    distance = torch.cdist(p, p)
    adjacency = (distance < RADIUS).numpy()
    _, labels = connected_components(adjacency, directed=False)
    speed = v.norm(dim=-1)
    direction = v / speed.unsqueeze(-1).clamp_min(1e-12)
    distance.fill_diagonal_(float("inf"))
    return {
        "spread": float((p - p.mean(0)).norm(dim=-1).mean()),
        "lcc": float(np.bincount(labels).max() / len(p)),
        "speed": float(speed.mean() / SPEED),
        "polarization": float(direction.mean(0).norm()),
        "crowding": float((distance.min(1).values < SEPARATION / 2).float().mean()),
        "nodes": len(p),
    }


@torch.no_grad()
def perturbed_rollout(model, positions, velocities, rounds, schedule):
    p, v = positions.clone(), velocities.clone()
    curves = {k: [] for k in CURVES}
    for t in range(rounds):
        if t in schedule:
            p, v = schedule[t](p, v)
        step = dense_step(p, v) if model is None else model.step(p, v, radius_edges(p))
        p, v = step.positions, step.velocities
        if not (torch.isfinite(p).all() and torch.isfinite(v).all()):
            return "diverged", curves, (p, v)
        for key, value in macrostate(p, v).items():
            curves[key].append(value)
    return "complete", curves, (p, v)


def episode_metrics(curves, config):
    spread = curves["spread"]
    windows = event_windows([config["events"][name] for name in EVENTS], config["rounds"])
    result = {}
    for name, (start, stop) in zip(EVENTS, windows, strict=True):
        pre = float(np.mean(spread[start - 5 : start]))
        errors = [abs(s - pre) / pre for s in spread[:stop]]
        result[name] = {
            "pre_spread": pre,
            "peak_change": float(max(errors[start:stop])),
            "recovery": recovery(
                errors, start, config["recovery"]["threshold"], config["recovery"]["consecutive"]
            ),
            "lcc_min": float(min(curves["lcc"][start:stop])),
        }
    result["lcc_final"] = float(np.mean(curves["lcc"][-10:]))
    result["speed_final"] = float(np.mean(curves["speed"][-10:]))
    return result


def models(run, campaign):
    yield "oracle", None, None
    yield "fixed", None, Boids()
    for variant in VARIANTS:
        for seed in campaign["seeds"]:
            yield variant, seed, load_model(run, variant, seed)


def summarize(rows, config):
    summary = {}
    for size in config["sizes"]:
        teacher = {
            r["episode"]: r for r in rows
            if r["method"] == "oracle" and r["nodes"] == size and r["status"] == "complete"
        }  # fmt: skip
        for method in METHODS:
            panel = [r for r in rows if r["method"] == method and r["nodes"] == size]
            done = [r for r in panel if r["status"] == "complete"]
            entry = {"runs": len(panel), "diverged": len(panel) - len(done)}
            for name in EVENTS:
                rounds = [r["metrics"][name]["recovery"]["rounds"] for r in done]
                entry[name] = {
                    **recovery_summary(rounds),
                    "peak_change": float(np.mean([r["metrics"][name]["peak_change"] for r in done]))
                    if done
                    else None,
                    "lcc_min": float(np.mean([r["metrics"][name]["lcc_min"] for r in done]))
                    if done
                    else None,
                }
            deviation = [
                relative_deviation(r["curves"]["spread"], teacher[r["episode"]]["curves"]["spread"])
                for r in done
                if r["episode"] in teacher
            ]
            entry["spread_deviation_from_teacher"] = (
                float(np.mean(deviation)) if deviation else None
            )
            entry["lcc_final"] = (
                float(np.mean([r["metrics"]["lcc_final"] for r in done])) if done else None
            )
            entry["spread_final"] = (
                float(np.mean([np.mean(r["curves"]["spread"][-10:]) for r in done]))
                if done
                else None
            )
            summary[f"N{size}/{method}"] = entry
    return summary


def write_report(out, config, summary):
    lines = [
        "# Boids under runtime perturbations",
        "",
        f"Profile **{config['profile']}**; {config['episodes']} episodes per size, "
        f"{config['rounds']} closed-loop rounds (training used 24). Events: split at "
        f"{config['events']['split']} (halves shifted by +-{config['split_shift']} in x), "
        f"crash of {config['crash_fraction']:.0%} at {config['events']['crash']}, "
        f"{config['join_fraction']:.0%} newcomers on a ring at {config['events']['join']}. "
        "Recovery: rounds until the swarm spread is within "
        f"{config['recovery']['threshold']:.0%} of its pre-event level for "
        f"{config['recovery']['consecutive']} rounds. Deviation: mean |spread - teacher spread| "
        "over the episode, relative to the teacher's mean spread.",
        "",
        "| N | Method | Runs | Diverged | Split: recovered, median (max) | Split peak | "
        "Crash peak | Join: recovered, median (max) | Join peak | Final LCC | Deviation |",
        "|---:|---|---:|---:|---|---:|---:|---|---:|---:|---:|",
    ]
    for key, e in summary.items():
        size, method = key.split("/")

        def fmt(value, digits=3):
            return "-" if value is None else f"{value:.{digits}f}"

        def recovered(entry):
            return f"{entry['recovered']}, {entry['median_rounds']} ({entry['max_rounds']})"

        lines.append(
            f"| {size[1:]} | {method} | {e['runs']} | {e['diverged']} | {recovered(e['split'])} | "
            f"{fmt(e['split']['peak_change'])} | {fmt(e['crash']['peak_change'])} | "
            f"{recovered(e['join'])} | "
            f"{fmt(e['join']['peak_change'])} | {fmt(e['lcc_final'])} | "
            f"{fmt(e['spread_deviation_from_teacher'])} |"
        )
    (out / "REPORT.md").write_text("\n".join(lines) + "\n")


def campaign(run: Path, profile="paper-cpu"):
    """``run`` is a finished Boids campaign; results go to ``run/adaptation``."""
    torch.set_num_threads(1)
    config = configuration(profile)
    source = read_json(run / "manifest.json")["config"]
    out = run / "adaptation"
    json_write(out / "manifest.json", {"config": config, "campaign": source,
                                        "software": software(SOURCES)})  # fmt: skip
    rows = []
    for size in config["sizes"]:
        states = initial_states(run, source, size, config["episodes"])
        for method, seed, model in models(run, source):
            print(f"Boids adaptation N={size} {method} seed={seed}", flush=True)
            for index, (key, positions, velocities) in enumerate(states):
                perturb = events(config, source, size, index)
                schedule = {config["events"][name]: perturb[name] for name in EVENTS}
                status, curves, final = perturbed_rollout(
                    model, positions, velocities, config["rounds"], schedule
                )
                row = {
                    "method": method, "seed": seed, "nodes": size, "episode": key,
                    "status": status, "curves": curves,
                    "metrics": episode_metrics(curves, config) if status == "complete" else {},
                }  # fmt: skip
                label = "deterministic" if seed is None else f"seed{seed}"
                json_write(out / "episodes" / method / label / f"N{size}-{key}.json", row)
                if index == 0 and seed in (None, 0):
                    tensor_write(out / "replays" / f"{method}-N{size}.pt",
                                 {"positions": final[0], "velocities": final[1]})  # fmt: skip
                rows.append(row)
    summary = summarize(rows, config)
    json_write(out / "summary.json", {"config": config, "summary": summary})
    table = []
    for r in rows:
        metrics = r["metrics"]
        table.append({key: r[key] for key in ("nodes", "method", "seed", "episode", "status")})
        for name in EVENTS:
            table[-1][f"{name}_rounds"] = metrics[name]["recovery"]["rounds"] if metrics else None
            table[-1][f"{name}_peak"] = metrics[name]["peak_change"] if metrics else None
    csv_write(out / "tables/adaptation.csv", table)
    write_report(out, config, summary)
    print(f"Boids adaptation artifacts: {out}", flush=True)
    return summary
