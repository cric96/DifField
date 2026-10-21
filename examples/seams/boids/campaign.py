"""Standalone Boids campaign: observed trajectories, shared test traces, five seeds."""

import time
from pathlib import Path

import torch

from diffield.sim import normalize_vectors

from ...shared.artifacts import checksum, json_write, read_json, software, tensor_write
from ...shared.campaign import (
    Budget,
    check_resume,
    deterministic,
    output_lock,
    require_new,
    resumable,
    status_file,
)
from ...shared.randomness import rng
from .. import SOURCES
from .dynamics import (
    DAMPING,
    FIXED,
    RADIUS,
    SEPARATION,
    SPEED,
    TEACHER,
    Boids,
    dense_step,
    radius_edges,
)
from .learning import (
    RATES,
    REGIME,
    TUNING_SEED,
    VARIANTS,
    combine,
    free_rollout,
    objective,
    train_job,
    transitions,
    weights,
)
from .report import report


def configuration(profile):
    if profile not in ("smoke", "paper-cpu"):
        raise ValueError(profile)
    smoke = profile == "smoke"
    return {
        "version": 2,
        "profile": profile,
        "data_seed": 20260912,
        "methods": ["oracle", "fixed", *VARIANTS],
        "seeds": [0] if smoke else list(range(5)),
        "updates": 2 if smoke else 500,
        "batch_size": 4,
        "validate_every": 1 if smoke else 10,
        "train_episodes": 4 if smoke else 16,
        "validation_episodes": 1 if smoke else 4,
        "test_episodes": 2 if smoke else 20,
        "nodes": 32,
        "rounds": 24,
        "diagnostic_updates": 2 if smoke else 100,
        "variants": list(VARIANTS),
        "regimes": [REGIME],
        "tuning_seed": TUNING_SEED,
        "rates": {k: list(v[:1] if smoke else v) for k, v in RATES.items()},
        "physics": {
            "radius": RADIUS,
            "separation": SEPARATION,
            "speed": SPEED,
            "damping": DAMPING,
            "dt": 1,
            "teacher": TEACHER,
            "fixed": FIXED,
        },
        "device": "cpu",
        "dtype": "float32",
        "threads": 1,
        "checkpoint_selection": "observed one-step validation loss, earliest on ties",
        "loss": "(position MSE + velocity MSE) / SPEED**2",
        "architecture": "edge 5-h-h SiLU; mean; node (h+4)-h-2 SiLU; output * SPEED",
    }


@torch.no_grad()
def traces(config, split, count):
    """Disjoint, deterministic teacher trajectories; no previous experiment required."""
    result = []
    for i in range(count):
        generator = rng(config["data_seed"], "boids", split, i)
        positions = torch.rand(config["nodes"], 2, generator=generator)
        velocities = normalize_vectors(torch.randn(positions.shape, generator=generator)) * SPEED
        target_pos, target_vel = free_rollout(None, positions, velocities, config["rounds"])
        result.append(
            {
                "positions": positions,
                "velocities": velocities,
                "target_pos": target_pos,
                "target_vel": target_vel,
                "key": f"boids-{split}-{i}",
            }
        )
    return result


def initialize(out, config):
    path = out / "manifest.json"
    splits = (
        ("train", "train_episodes"),
        ("validation", "validation_episodes"),
        ("test", "test_episodes"),
    )
    if not path.exists():
        require_new(out)
    out.mkdir(parents=True, exist_ok=True)
    data_path = out / "data/traces.pt"
    if not data_path.exists():
        if path.exists():
            raise ValueError(
                "Saved Boids dataset is missing; restore it or use a new output directory"
            )
        tensor_write(
            data_path, {split: traces(config, split, config[key]) for split, key in splits}
        )
    config = {
        **config,
        "methods": ["oracle", "fixed", *VARIANTS],
        "physics": {**config["physics"], "teacher": list(TEACHER), "fixed": list(FIXED)},
    }
    provenance = software(SOURCES)
    manifest = {
        "config": config,
        "data_sha256": checksum(data_path),
        "software": provenance,
    }
    if path.exists():
        check_resume(read_json(path), manifest)
    data = torch.load(data_path, weights_only=True)
    selected, keys = {}, set()
    for split, count_key in splits:
        count = config[count_key]
        if len(data[split]) < count:
            raise ValueError(f"Not enough {split} traces")
        selected[split] = data[split][:count]
        for episode in selected[split]:
            if episode["key"] in keys:
                raise ValueError("Duplicate episode key across splits")
            keys.add(episode["key"])
            for key in ("positions", "velocities", "target_pos", "target_vel"):
                shape = (config["nodes"], 2)
                if key.startswith("target_"):
                    shape = (config["rounds"], *shape)
                if tuple(episode[key].shape) != shape:
                    raise ValueError(f"Boids trace {key} must have shape {shape}")
            if any(
                not torch.isfinite(episode[k]).all()
                for k in ("positions", "velocities", "target_pos", "target_vel")
            ):
                raise ValueError("Nonfinite source trace")
    if not path.exists():
        json_write(path, manifest)
    return selected


def job_path(out, variant, seed, rate=None):
    if rate is not None:
        return out / "tuning" / REGIME / variant / f"lr{rate}"
    return out / "jobs" / REGIME / variant / f"seed{seed}"


def diagnose(out, data, config, budget):
    """Controlled one-step vs 24-step training; numeric checks use train/validation only."""
    path = out / "diagnostics.json"
    if path.exists():
        return read_json(path)
    oracle = Boids()
    with torch.no_grad():
        oracle.log_weights.copy_(torch.tensor(TEACHER).log())
    saturated, total, errors, curves = 0, 0, [], []
    for split in ("train", "validation"):
        for episode in data[split]:
            budget.check()
            p, v = episode["positions"], episode["velocities"]
            with torch.no_grad():
                for target_p, target_v in zip(
                    episode["target_pos"], episode["target_vel"], strict=True
                ):
                    prediction = dense_step(p, v)
                    saturated += int((prediction.preclip.norm(dim=-1) > SPEED).sum())
                    total += len(p)
                    student = oracle.step(p, v, radius_edges(p))
                    errors.append(
                        {
                            "saved_dense": max(
                                float((prediction.positions - target_p).abs().max()),
                                float((prediction.velocities - target_v).abs().max()),
                            ),
                            "dense_dsl": max(
                                float((prediction.positions - student.positions).abs().max()),
                                float((prediction.velocities - student.velocities).abs().max()),
                            ),
                        }
                    )
                    p, v = target_p, target_v
                for dtype in (torch.float32, torch.float64):
                    dense_p, _ = free_rollout(
                        None,
                        episode["positions"].to(dtype),
                        episode["velocities"].to(dtype),
                        config["rounds"],
                    )
                    student_p, _ = free_rollout(
                        oracle.to(dtype),
                        episode["positions"].to(dtype),
                        episode["velocities"].to(dtype),
                        config["rounds"],
                    )
                    curves.append(
                        {
                            "split": split,
                            "episode": episode["key"],
                            "dtype": str(dtype),
                            "dense_saved_rmse": (dense_p - episode["target_pos"])
                            .square()
                            .mean((1, 2))
                            .sqrt()
                            .tolist(),
                            "dense_dsl_rmse": (dense_p - student_p)
                            .square()
                            .mean((1, 2))
                            .sqrt()
                            .tolist(),
                        }
                    )
                oracle.float()
    ablations = []
    train = [transitions(e) for e in data["train"]]
    for horizon in (1, config["rounds"]):
        model = Boids()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.03)
        last = out / "diagnostic-checkpoints" / f"horizon{horizon}.pt"
        completed, history = 0, []
        if last.exists():
            saved = torch.load(last, weights_only=True)
            model.load_state_dict(saved["model"])
            optimizer.load_state_dict(saved["optimizer"])
            completed, history = saved["completed"], saved["history"]
        for update in range(completed, config["diagnostic_updates"]):
            budget.check()
            idx = torch.randperm(len(train), generator=rng(0, "boids-v2-batch", update))
            idx = idx[: config["batch_size"]].tolist()
            optimizer.zero_grad()
            if horizon == 1:
                loss = objective(model, combine([train[i] for i in idx]))
            else:
                losses = []
                for i in idx:
                    episode = data["train"][i]
                    p, v = free_rollout(model, episode["positions"], episode["velocities"], horizon)
                    losses.append(
                        (
                            (p - episode["target_pos"]).square().mean()
                            + (v - episode["target_vel"]).square().mean()
                        )
                        / SPEED**2
                    )
                loss = torch.stack(losses).mean()
            loss.backward()
            grad = model.log_weights.grad.detach().clone()
            if not torch.isfinite(loss) or not torch.isfinite(grad).all():
                raise FloatingPointError("Nonfinite diagnostic gradient")
            optimizer.step()
            history.append(
                {
                    "update": update + 1,
                    "loss": float(loss.detach()),
                    "gradient": grad.tolist(),
                    "gradient_norm": float(grad.norm()),
                    "weights": weights(model),
                    "batch_indices": idx,
                }
            )
            tensor_write(
                last,
                {
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "completed": update + 1,
                    "history": history,
                },
            )
        with torch.no_grad():
            validation = float(
                objective(model, combine([transitions(e) for e in data["validation"]]))
            )
        ablations.append(
            {
                "horizon": horizon,
                "weights": weights(model),
                "history": history,
                "observed_validation_loss": validation,
            }
        )
    result = {
        "saturated_fraction": saturated / total,
        "transitions_nodes": total,
        "one_step_max_errors": {k: max(e[k] for e in errors) for k in errors[0]},
        "numerical_replays": curves,
        "ablations": ablations,
        "interpretation": "Float64 is a sensitivity control, not a new test dataset or error floor",
    }
    json_write(path, result)
    return result


@torch.no_grad()
def evaluate_job(out, data, config, variant, seed, model, training, budget):
    directory = out / "episodes" / REGIME / variant / f"seed{seed}"
    checkpoint = job_path(out, variant, seed) / "best.pt"
    checkpoint_hash = checksum(checkpoint) if checkpoint.exists() else None
    for episode in data["test"]:
        budget.check()
        path = directory / f"{episode['key']}.json"
        replay_path = out / "replays" / REGIME / variant / f"{episode['key']}.pt"
        needs_replay = seed == 0 and episode["key"] in {e["key"] for e in data["test"][:3]}
        if (
            path.exists()
            and read_json(path).get("status") == "complete"
            and (not needs_replay or replay_path.exists())
        ):
            continue
        started = time.perf_counter()
        p, v = free_rollout(model, episode["positions"], episode["velocities"], config["rounds"])
        seconds = time.perf_counter() - started
        finite = bool(torch.isfinite(p).all() and torch.isfinite(v).all())
        if not finite:
            json_write(
                path,
                {
                    "regime": REGIME,
                    "method": variant,
                    "seed": seed,
                    "episode": episode["key"],
                    "status": "failed",
                    "metrics": {},
                    "reason": "nonfinite rollout",
                },
            )
            raise FloatingPointError(f"Nonfinite Boids rollout: {REGIME}/{variant}/{seed}")
        metrics = {
            "position_rmse": float((p - episode["target_pos"]).square().mean().sqrt())
            if finite
            else None,
            "velocity_rmse": float((v - episode["target_vel"]).square().mean().sqrt())
            if finite
            else None,
            "one_step_loss": float(objective(model, transitions(episode)))
            if model is not None
            else None,
            "inference_seconds": seconds,
        }
        curves = (
            {
                "position_rmse": (p - episode["target_pos"]).square().mean((1, 2)).sqrt().tolist(),
                "velocity_rmse": (v - episode["target_vel"]).square().mean((1, 2)).sqrt().tolist(),
            }
            if finite
            else {}
        )
        parameters = sum(x.numel() for x in model.parameters()) if model is not None else 0
        row = {
            "regime": REGIME,
            "method": variant,
            "seed": seed,
            "episode": episode["key"],
            "status": "complete" if finite else "failed",
            "metrics": metrics,
            "curves": curves,
            "weights": weights(model),
            "parameters": parameters,
            "fitted_parameters": 0 if variant in ("fixed", "oracle") else parameters,
            "checkpoint_sha256": checkpoint_hash,
            "training_seconds": training.get("seconds", 0),
            "selected_update": training.get("selected_update"),
            "input_sha256": read_json(out / "manifest.json")["data_sha256"],
        }
        if needs_replay:
            tensor_write(
                replay_path,
                {"positions": p, "velocities": v},
            )
        json_write(path, row)


def tune(out, config, train, validation, budget):
    """Learning rate of each variant, frozen on the tuning seed before any evaluation."""
    path = out / "tuning-frozen.json"
    if not path.exists():
        frozen = {}
        for variant in VARIANTS:
            scores = []
            for rate in config["rates"]["parametric" if variant == "parametric" else "gnn"]:
                print(f"Boids tuning {REGIME} {variant} lr={rate}", flush=True)
                _, result = train_job(
                    job_path(out, variant, TUNING_SEED, rate),
                    config,
                    train,
                    validation,
                    variant,
                    TUNING_SEED,
                    rate,
                    budget,
                )
                scores.append({"rate": rate, "validation_loss": result["best_validation"]})
            frozen[variant] = {
                "rate": min(scores, key=lambda r: (r["validation_loss"], r["rate"]))["rate"],
                "candidates": scores,
            }
        json_write(path, {REGIME: frozen})
    return read_json(path)[REGIME]


def campaign(out: Path, profile="paper-cpu", seconds=14400):
    with output_lock(out):
        return _campaign(out, profile, seconds)


def _campaign(out, profile, seconds):
    started = time.monotonic()
    deterministic()
    config = configuration(profile)
    data = initialize(out, config)
    config = read_json(out / "manifest.json")["config"]
    # Four hours at most, accumulated across invocations, including artifact generation.
    budget = Budget(min(seconds, 14400), record=out / "budget.json")
    budget.started = started
    progress = out / "progress.json"
    expected = [
        str(job_path(out, variant, seed).relative_to(out))
        for variant in VARIANTS
        for seed in config["seeds"]
    ]
    json_write(
        out / "plans.json",
        {
            "training_jobs": expected,
            "test_episodes": config["test_episodes"],
            "methods": config["methods"],
            "regimes": [REGIME],
            "config": config,
        },
    )

    def run():
        with status_file(progress, stage="diagnostics"):
            diagnose(out, data, config, budget)
            train = [transitions(e) for e in data["train"]]
            validation = combine([transitions(e) for e in data["validation"]])
        with status_file(progress, stage="tuning"):
            frozen = tune(out, config, train, validation, budget)
        stage = "training_and_evaluation"
        with status_file(progress, stage=stage):
            for seed in config["seeds"]:
                for variant in VARIANTS:
                    job = {"stage": stage, "regime": REGIME, "seed": seed, "variant": variant}
                    json_write(progress, {"status": "running", **job})
                    print(f"Boids {REGIME} {variant} seed={seed}", flush=True)
                    model, result = train_job(
                        job_path(out, variant, seed),
                        config,
                        train,
                        validation,
                        variant,
                        seed,
                        frozen[variant]["rate"],
                        budget,
                    )
                    evaluate_job(out, data, config, variant, seed, model, result, budget)
                for variant, model in (("fixed", Boids()), ("oracle", None)):
                    evaluate_job(out, data, config, variant, seed, model, {}, budget)
        with status_file(progress, stage="report"):
            if not report(out, data, config, budget):
                raise RuntimeError("Boids evaluation panel is incomplete")
        json_write(
            progress, {"status": "complete", "stage": "complete", "training_jobs": len(expected)}
        )
        print(f"Boids artifacts: {out}", flush=True)
        return 0

    try:
        return resumable(out, run, partial=lambda: report(out, data, config, None, render=False))
    finally:
        budget.save()
