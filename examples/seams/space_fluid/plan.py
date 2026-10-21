"""The complete, inspectable work inventory of the main study: every job and its file."""

import itertools

import torch

from ...shared.artifacts import json_write, read_json
from .config import CONDITIONS, FIXED, LEARNERS, TEST_FAMILIES
from .data import make_episode


def methods(config):
    return (
        [{"id": f"fixed-{m}", "method": m, "lambda": None} for m in FIXED]
        + [
            {"id": f"{m}-l{value:g}", "method": m, "lambda": value}
            for m, value in itertools.product(LEARNERS, config.lambdas)
            # The GNN has no regions, so the leader penalty cannot trade anything off.
            if m != "gnn" or value == config.main_lambda
        ]
        + [
            {"id": f"kmeans-k{k}", "method": "kmeans", "lambda": None, "k": k}
            for k in config.k_values
        ]
    )


# The GNN has no discrete election whose surrogate could be biased.
UNDIAGNOSED = ("gnn",)


def checkpoint_directory(out, method, seed):
    return out / "checkpoints" / method["id"] / f"seed{seed}"


def episode_specs(config):
    specs = []
    for family in TEST_FAMILIES:
        for index in range(config.test_episodes):
            base = {
                "family": family,
                "index": index,
                "nodes": config.nodes,
                "layout": "jittered",
                "condition": "clean",
                "executor": "batched",
            }
            specs += [{**base, "panel": "main", "condition": c} for c in CONDITIONS]
            specs += [
                {**base, "panel": "topology", "layout": layout}
                for layout in config.topology_layouts
            ]
            specs += [{**base, "panel": "transfer", "nodes": n} for n in config.transfer_nodes]
    families = config.distributed_families
    for nodes, condition, index in itertools.product(
        config.distributed_nodes, config.distributed_conditions, range(config.distributed_episodes)
    ):
        base = {
            "panel": "distributed",
            "family": families[index % len(families)],
            "index": index // len(families),
            "nodes": nodes,
            "layout": "jittered",
            "condition": condition,
        }
        specs.append({**base, "executor": "sync"})
        specs += [{**base, "executor": f"async-{p:g}"} for p in config.activation_probabilities]
    return specs


def episode_from_spec(config, spec):
    return make_episode(
        config, **{k: spec[k] for k in ("family", "index", "nodes", "layout", "condition")}
    )


def result_path(out, spec, method, seed):
    episode_id = (
        f"{spec['family']}-{spec['index']}-n{spec['nodes']}-{spec['layout']}-{spec['condition']}"
    )
    return (
        out
        / "episodes"
        / spec["panel"]
        / spec["executor"]
        / method["id"]
        / f"seed{seed}"
        / f"{episode_id}.json"
    )


def evaluation_jobs(config):
    main = config.main_lambda
    for spec in episode_specs(config):
        pareto = spec["panel"] == "main" and spec["condition"] == "clean"
        for method in methods(config):
            if method["lambda"] not in (None, main) and not pareto:
                continue  # Tradeoff sweep: in-distribution Pareto panel only.
            if (
                spec["panel"] == "distributed"
                and method["id"] != "fixed-combined"
                and (method["method"] not in LEARNERS or method["lambda"] != main)
            ):
                continue
            # Synchronous runs are an equivalence check, not a separate result.
            for seed in config.seeds[:1] if spec["executor"] == "sync" else config.seeds:
                yield spec, method, seed


def save_replay(spec, seed):
    return (
        seed == 0
        and spec["index"] == 0
        and (
            (
                spec["panel"] == "main"
                and spec["condition"] == "clean"
                and spec["family"] in ("gaussian", "ring")
            )
            or spec["panel"] == "distributed"
        )
    )


def progress(out, config):
    training = []
    for method in methods(config):
        if method["method"] not in LEARNERS:
            continue
        for seed in config.seeds:
            directory = checkpoint_directory(out, method, seed)
            path = directory / "training.json"
            done = (
                path.exists()
                and read_json(path).get("complete", False)
                and (directory / "best.pt").exists()
            )
            diagnostic = (
                method["method"] in UNDIAGNOSED or (directory / "sensitivity.json").exists()
            )
            last = directory / "last.pt"
            step = torch.load(last, weights_only=True)["step"] if last.exists() else 0
            training.append(
                {
                    "method": method["id"],
                    "seed": seed,
                    "step": step,
                    "status": "complete" if done and diagnostic else "pending",
                }
            )
    complete, failed, pending = 0, 0, 0
    missing = []
    for spec, method, seed in evaluation_jobs(config):
        path = result_path(out, spec, method, seed)
        replay = out / "replays" / path.relative_to(out / "episodes").with_suffix(".pt")
        status = read_json(path).get("status") if path.exists() else "pending"
        if status == "complete" and (not save_replay(spec, seed) or replay.exists()):
            complete += 1
        elif status == "failed":
            failed += 1
        else:
            pending += 1
            if len(missing) < 10:
                missing.append(str(path.relative_to(out)))
    result = {
        "training": training,
        "evaluation": {
            "complete": complete,
            "failed": failed,
            "pending": pending,
            "expected": complete + failed + pending,
            "pending_examples": missing,
        },
        "status": "complete"
        if not pending and not failed and all(j["status"] == "complete" for j in training)
        else "incomplete",
    }
    json_write(out / "progress.json", result)
    return result
