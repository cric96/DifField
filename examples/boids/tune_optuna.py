#!/usr/bin/env python3
"""Optuna tuner for the boids family."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

from boids.runner import LearnableRunOptions, run_learnable_subprocess
from shared.metrics import is_finite_number, nested_get

try:
    import optuna
    from optuna.importance import get_param_importances
except ImportError as exc:
    raise SystemExit("optuna is required. Install dependencies first (uv sync).") from exc


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Optuna tuning for examples/boids/learnable.py")
    parser.add_argument("--study-name", type=str, default="boids_learnable_optuna")
    parser.add_argument("--storage", type=str, default="")
    parser.add_argument("--out-dir", type=str, default="examples/results/optuna")
    parser.add_argument("--trials", type=int, default=40)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--python", type=str, default=sys.executable)
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--rounds", type=int, default=40)
    parser.add_argument("--num-nodes", type=int, default=40)
    parser.add_argument("--eval-seeds", type=str, default="101,103,107")
    parser.add_argument("--eval-every", type=int, default=10)
    parser.add_argument("--timeout", type=int, default=0, help="Global timeout (seconds), 0 disables")
    parser.add_argument("--sampler", choices=["tpe", "random"], default="tpe")
    parser.add_argument("--skip-viz", action="store_true")
    parser.add_argument("--device", type=str, default="", help="Device (cuda/cpu) [auto if empty]")
    return parser.parse_args()


def _objective(run_options: LearnableRunOptions):
    def objective(trial: optuna.Trial) -> float:
        mode = trial.suggest_categorical("mode", ["weights", "attention", "joint"])
        lr = trial.suggest_float("lr", 1e-3, 4e-2, log=True)
        reg_scale = trial.suggest_float("reg_scale", 0.35, 2.5)
        run_name = f"trial_{trial.number:04d}_{mode}"
        outcome = run_learnable_subprocess(
            options=run_options,
            run_name=run_name,
            mode=mode,
            lr=lr,
            reg_scale=reg_scale,
        )
        if not outcome.ok:
            outcome.run_dir.mkdir(parents=True, exist_ok=True)
            (outcome.run_dir / "failed_trial.log").write_text(
                "STDOUT:\n" + outcome.stdout + "\n\nSTDERR:\n" + outcome.stderr,
                encoding="utf-8",
            )
            raise optuna.TrialPruned("Trial failed before summary generation")

        summary = outcome.summary or {}
        metric = nested_get(summary, "validation.final_val_traj_loss")
        if not is_finite_number(metric):
            metric = nested_get(summary, "training.final_traj_loss")
        if not is_finite_number(metric):
            raise optuna.TrialPruned("No finite objective metric available")

        value = float(metric)
        if bool(nested_get(summary, "training.has_nan", False)):
            value += 5.0
        trial.set_user_attr("run_name", run_name)
        trial.set_user_attr("run_dir", str(outcome.run_dir))
        trial.set_user_attr("final_val_traj_loss", nested_get(summary, "validation.final_val_traj_loss"))
        trial.set_user_attr("final_traj_loss", nested_get(summary, "training.final_traj_loss"))
        trial.set_user_attr("final_center_error", nested_get(summary, "training.final_center_error"))
        trial.set_user_attr("has_nan", nested_get(summary, "training.has_nan", False))
        return value

    return objective


def _write_trials_csv(study: optuna.Study, output_csv: Path) -> None:
    rows = []
    for trial in study.trials:
        row = {
            "trial": trial.number,
            "state": trial.state.name,
            "value": trial.value,
            "run_name": trial.user_attrs.get("run_name", ""),
            "run_dir": trial.user_attrs.get("run_dir", ""),
        }
        for key, value in trial.params.items():
            row[f"param_{key}"] = value
        rows.append(row)

    if not rows:
        return
    fieldnames = sorted({key for row in rows for key in row.keys()})
    with output_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    root = Path(args.out_dir)
    root.mkdir(parents=True, exist_ok=True)
    run_options = LearnableRunOptions(
        python=args.python,
        root=root,
        epochs=args.epochs,
        rounds=args.rounds,
        num_nodes=args.num_nodes,
        eval_seeds=args.eval_seeds,
        eval_every=args.eval_every,
        skip_viz=args.skip_viz,
        device=args.device,
    )

    storage = args.storage.strip()
    if not storage:
        storage = f"sqlite:///{(root / 'study.db').as_posix()}"

    if storage.startswith("sqlite:///"):
        sqlite_path = Path(storage.removeprefix("sqlite:///"))
        sqlite_path.parent.mkdir(parents=True, exist_ok=True)

    if args.sampler == "random":
        sampler: optuna.samplers.BaseSampler = optuna.samplers.RandomSampler(seed=args.seed)
    else:
        sampler = optuna.samplers.TPESampler(seed=args.seed)

    study = optuna.create_study(
        study_name=args.study_name,
        storage=storage,
        direction="minimize",
        load_if_exists=True,
        sampler=sampler,
        pruner=optuna.pruners.MedianPruner(n_startup_trials=8),
    )

    study.optimize(_objective(run_options), n_trials=args.trials, timeout=(None if args.timeout <= 0 else args.timeout))

    best = {
        "study_name": args.study_name,
        "best_value": study.best_value,
        "best_params": study.best_params,
        "best_trial": study.best_trial.number,
        "best_run_name": study.best_trial.user_attrs.get("run_name", ""),
        "best_run_dir": study.best_trial.user_attrs.get("run_dir", ""),
    }
    (root / "best_trial.json").write_text(json.dumps(best, indent=2), encoding="utf-8")

    try:
        importances = get_param_importances(study)
    except (ImportError, ValueError, RuntimeError):
        importances = {}
    (root / "param_importance.json").write_text(json.dumps(importances, indent=2), encoding="utf-8")
    _write_trials_csv(study, root / "trials.csv")

    print(f"best_value={study.best_value:.6f}")
    print(f"best_params={json.dumps(study.best_params)}")
    print(f"saved {root / 'best_trial.json'}")
    print(f"saved {root / 'param_importance.json'}")
    print(f"saved {root / 'trials.csv'}")


if __name__ == "__main__":
    main()