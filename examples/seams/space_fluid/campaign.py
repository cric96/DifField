"""Resumable protocol with a complete, inspectable work inventory."""

import fcntl
import itertools
import json
import time
from contextlib import contextmanager
from pathlib import Path

import torch

from ..artifacts import Budget, checksum, json_write, read_json, software, tensor_write
from ..randomness import seed_for
from .config import CONDITIONS, FIXED, LEARNERS, TEST_FAMILIES, TRAIN_FAMILIES
from .data import RegionEpisode, combine, make_episode, normalization, training_bank
from .execution import central, decentralized, equivalence, kmeans
from .insights import insights
from .metrics import summarize
from .program import make_program
from .training import load_program, sensitivity, surrogate_objective, train_job

STAGES = ("train", "insights", "evaluate", "report")


@contextmanager
def output_lock(out):
    """Kernel-owned lock survives daemon restarts and releases on process exit.

    A PID file cannot distinguish processes in separate sandbox namespaces.
    flock also prevents report/checkpoint temporary files from racing.
    """
    out.mkdir(parents=True, exist_ok=True)
    with (out / ".run.lock").open("a") as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError(
                "Another Space-Fluid process is using this output directory"
            ) from error
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


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


def initialization(out, config):
    path = out / "manifest.json"
    config_dict = json.loads(json.dumps(config.to_dict()))
    provenance = software()
    if path.exists():
        manifest = read_json(path)
        if manifest.get("experiment") != "space-fluid" or manifest["config"] != config_dict:
            raise ValueError("Configuration mismatch; choose a separate --out directory")
        if manifest["software"]["source_sha256"] != provenance["source_sha256"]:
            raise ValueError("Source mismatch; choose a new --out directory")
        if checksum(out / "data/bank.pt") != manifest["data_sha256"]:
            raise ValueError("Training data checksum mismatch")
        payload = torch.load(out / "data/bank.pt", weights_only=True)
        return {k: [RegionEpisode(**e) for e in v] for k, v in payload.items()}, manifest[
            "normalization"
        ]
    # Do not accidentally place an experiment inside another run or a nonempty directory.
    if out.exists() and any(p.name != ".run.lock" for p in out.iterdir()):
        raise ValueError("New output directory must be empty")
    if any((p / "manifest.json").exists() for p in out.resolve().parents):
        raise ValueError("Experiment output directories must not be nested")
    bank = training_bank(config)
    norm = normalization(bank["train"])
    tensor_write(out / "data/bank.pt", {k: [e.payload() for e in v] for k, v in bank.items()})
    json_write(
        path,
        {
            "experiment": "space-fluid",
            "schema_version": config.schema_version,
            "config": config_dict,
            "software": provenance,
            "normalization": norm,
            "data_sha256": checksum(out / "data/bank.pt"),
            "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "normalization_source": "training simulator truth only",
            "training": "offline; hard validation selects; frozen deployment",
            "radius": 1.0,
            "gradient": (
                "biased straight-through through election (S), cost (G), collect gate (C) "
                "and broadcast; never an exact election derivative"
            ),
            "messages": (
                "ten float32 values per directed transmission for every program, including "
                "the GNN; excludes alignment keys, headers, serialization and retries"
            ),
            "sync": "barrier exchange of published previous states over current links",
            "async": (
                "one initial execution everywhere, then random sequential activations; "
                "only actual sends deliver new observations"
            ),
            "test_sharing": (
                "identical episodes, priorities and schedules across methods and training seeds"
            ),
            "replays": (
                "all methods, seed 0, main/clean Gaussian and ring episode 0; distributed episode 0"
            ),
            "connectivity": "measured on current graph; transient fragmentation is not suppressed",
        },
    )
    specs = episode_specs(config)
    json_write(
        out / "plan.json",
        {
            "methods": methods(config),
            "seeds": config.seeds,
            "episode_specs": specs,
            "training_jobs": sum(m["method"] in LEARNERS for m in methods(config))
            * len(config.seeds),
            "evaluation_jobs": sum(1 for _ in evaluation_jobs(config)),
            "generalization_only": [f for f in TEST_FAMILIES if f not in TRAIN_FAMILIES],
        },
    )
    return bank, norm


def timing(out, config, bank, norm, budget):
    path = out / "timing.json"
    if path.exists():
        return
    budget.check()
    model = make_program("hybrid", **norm)
    episode = combine(bank["train"][: config.batch_size])
    started = time.perf_counter()
    surrogate_objective(episode, model, norm["scale"], config.main_lambda, config).backward()
    training_seconds = time.perf_counter() - started
    pilot = make_episode(config, family="gaussian")
    with torch.no_grad():
        batched = central(pilot, model)
        budget.check()
        distributed = decentralized(pilot, model)
    check = equivalence(batched, distributed)
    if not all(
        check[k] for k in ("identifiers_equal", "values_close", "elections_equal", "traffic_equal")
    ):
        raise AssertionError(f"Timing pilot failed equivalence: {check}")
    training_jobs = (2 * len(config.lambdas) + 1) * len(config.seeds)
    json_write(
        path,
        {
            "batch_forward_backward_seconds": training_seconds,
            "batched_episode_seconds": batched.seconds,
            "sync_episode_seconds": distributed.seconds,
            "nodes": config.nodes,
            "train_rounds": config.train_rounds,
            "eval_rounds": config.eval_rounds,
            "batch_size": config.batch_size,
            "estimated_adam_seconds_excluding_validation_io": training_seconds
            * config.updates
            * training_jobs,
            "note": (
                "single pilot, not a full campaign estimate; validation, "
                "larger networks and DeviceRuntime panels add cost"
            ),
            "equivalence": check,
        },
    )


def train(out, config, bank, norm, budget):
    # Interleave seeds within each method/tradeoff; each checkpoint is independently resumable.
    for method in methods(config):
        if method["method"] not in LEARNERS:
            continue
        for seed in config.seeds:
            directory = checkpoint_directory(out, method, seed)
            complete = directory / "training.json"
            diagnostic = directory / "sensitivity.json"
            if (
                complete.exists()
                and read_json(complete)["complete"]
                and (directory / "best.pt").exists()
            ):
                if method["method"] in UNDIAGNOSED or diagnostic.exists():
                    continue
                model = make_program(method["method"], seed, **norm)
                model.load_state_dict(torch.load(directory / "best.pt", weights_only=True)["model"])
            else:
                print(f"train {method['id']} seed={seed}", flush=True)
                model, _ = train_job(
                    directory, config, bank, norm, method["method"], seed, method["lambda"], budget
                )
            if method["method"] not in UNDIAGNOSED:
                budget.check()
                # First validation sequence of each family, fixed before any outcome.
                episode = combine(bank["validation"][:: config.validation_episodes])
                json_write(
                    diagnostic,
                    {
                        "episode": episode.key,
                        "interpretation": (
                            "training gradient against hard central differences; "
                            "not retraining uncertainty"
                        ),
                        **sensitivity(model, episode, norm, method["lambda"], config),
                    },
                )


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


def evaluate(out, config, norm, budget):  # noqa: PLR0912, PLR0915 -- explicit executor and checkpoint cases
    cache = {}
    current_spec, episode = None, None
    for spec, method, seed in evaluation_jobs(config):
        path = result_path(out, spec, method, seed)
        replay = out / "replays" / path.relative_to(out / "episodes").with_suffix(".pt")
        if (
            path.exists()
            and (not save_replay(spec, seed) or replay.exists())
            and read_json(path).get("status") == "complete"
        ):
            continue
        budget.check()
        if method["method"] not in LEARNERS and seed != config.seeds[0]:
            # Fixed programs and K-means ignore the training seed: reuse the seed-0 result.
            first = read_json(result_path(out, spec, method, config.seeds[0]))
            if first.get("status") == "complete":
                json_write(path, {**first, "seed": seed})
                continue
        if method["method"] in LEARNERS:
            checkpoint = checkpoint_directory(out, method, seed) / "best.pt"
            if not checkpoint.exists():
                continue  # Accounted for explicitly by the complete plan and progress report.
            model_key = (method["id"], seed)
            if model_key not in cache:
                cache[model_key] = load_program(checkpoint)
            model = cache[model_key]
            cp_hash = checksum(checkpoint)
        elif method["method"] != "kmeans":
            model = make_program(method["method"], seed, **norm).eval()
            model.requires_grad_(False)
            cp_hash = None
        else:
            model, cp_hash = None, None
        if current_spec != spec:
            episode, current_spec = episode_from_spec(config, spec), spec
        print(
            f"evaluate {spec['panel']}/{spec['executor']} {episode.key} {method['id']} seed={seed}",
            flush=True,
        )
        try:
            with torch.no_grad():
                check = None
                if model is None:
                    trace = kmeans(episode, method["k"], norm, seed_for(config.data_seed, "kmeans"))
                elif spec["executor"] == "batched":
                    trace = central(episode, model)
                else:
                    probability = (
                        1.0 if spec["executor"] == "sync" else float(spec["executor"].split("-")[1])
                    )
                    trace = decentralized(
                        episode,
                        model,
                        activation_probability=probability,
                        seed=seed_for(config.data_seed, episode.key, "schedule"),
                    )
                    if probability == 1.0:
                        expected = central(episode, model)
                        check = equivalence(expected, trace)
                        if not all(
                            check[k]
                            for k in (
                                "identifiers_equal",
                                "values_close",
                                "elections_equal",
                                "traffic_equal",
                            )
                        ):
                            raise AssertionError(f"Synchronous mismatch: {check}")  # noqa: TRY301 -- record the failed job below
                metrics, curves = summarize(episode, trace, norm["scale"], config)
            if save_replay(spec, seed):
                tensor_write(
                    replay,
                    {
                        "episode": episode.payload(),
                        "trace": trace.payload(),
                        "method": method,
                        "normalization": norm,
                    },
                )
            json_write(
                path,
                {
                    **spec,
                    "method": method["id"],
                    "lambda": method["lambda"],
                    "seed": seed,
                    "episode": episode.key,
                    "status": "complete",
                    "checkpoint_sha256": cp_hash,
                    "metrics": metrics,
                    "curves": curves,
                    "equivalence": check,
                },
            )
        except Exception as error:
            json_write(
                path,
                {
                    **spec,
                    "method": method["id"],
                    "seed": seed,
                    "episode": episode.key,
                    "status": "failed",
                    "reason": f"{type(error).__name__}: {error}",
                    "metrics": {},
                },
            )
            raise


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


def campaign(out: Path, config, stage="all", seconds=14400):
    with output_lock(out):
        return _campaign(out, config, stage, seconds)


def _campaign(out, config, stage, seconds):
    from .report import report  # noqa: PLC0415 -- report reads the campaign inventory

    if seconds <= 0:
        raise ValueError("Budget must be positive")
    if stage not in ("all", *STAGES):
        raise ValueError(stage)
    torch.set_num_threads(config.threads)
    torch.use_deterministic_algorithms(True)
    budget = Budget(seconds)
    bank, norm = initialization(out, config)
    stages = STAGES if stage == "all" else (stage,)
    try:
        for current in stages:
            json_write(out / "stages" / f"{current}.json", {"status": "running"})
            if current == "train":
                timing(out, config, bank, norm, budget)
                train(out, config, bank, norm, budget)
            elif current == "insights":
                insights(out, config, norm, budget)
            elif current == "evaluate":
                evaluate(out, config, norm, budget)
            elif current == "report":
                report(out, config, budget=budget)
            state = progress(out, config)
            complete = (
                all(j["status"] == "complete" for j in state["training"])
                if current == "train"
                else state["evaluation"]["pending"] == state["evaluation"]["failed"] == 0
                if current == "evaluate"
                else (out / "insights.json").exists()
                if current == "insights"
                else True
            )
            json_write(
                out / "stages" / f"{current}.json",
                {"status": "complete" if complete else "incomplete"},
            )
    except TimeoutError as error:
        json_write(
            out / "stages" / f"{current}.json", {"status": "incomplete", "reason": str(error)}
        )
        report(out, config, render=False)
        print(f"Incomplete, resumable campaign: {out}", flush=True)
        return 2
    except Exception as error:
        json_write(
            out / "stages" / f"{current}.json",
            {"status": "failed", "reason": f"{type(error).__name__}: {error}"},
        )
        progress(out, config)
        raise
    print(f"Artifacts: {out}", flush=True)
    return 0 if progress(out, config)["status"] == "complete" or stage in ("train", "report") else 2
