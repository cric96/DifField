#!/usr/bin/env python3
"""Evaluation runner for the learnable boids family."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

from boids.runner import LearnableRunOptions, run_learnable_subprocess
from shared.metrics import is_finite_number, mean, nested_get, std


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate learnable boids across modes and seeds")
    parser.add_argument("--seeds", type=str, default="11,13,17,19,23")
    parser.add_argument("--modes", type=str, default="weights,attention,joint")
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--rounds", type=int, default=40)
    parser.add_argument("--num-nodes", type=int, default=40)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--eval-seeds", type=str, default="101,103,107")
    parser.add_argument("--eval-every", type=int, default=20)
    parser.add_argument("--out-dir", type=str, default="generated/results/evaluation")
    parser.add_argument("--python", type=str, default=sys.executable)
    parser.add_argument("--skip-viz", action="store_true", help="Pass --no-viz --no-gif to each run")
    parser.add_argument("--device", type=str, default="", help="Device (cuda/cpu) [auto if empty]")
    return parser.parse_args()


def _parse_csv(text: str) -> list[str]:
    return [item.strip() for item in text.split(",") if item.strip()]


def main() -> None:
    args = parse_args()
    seeds = _parse_csv(args.seeds)
    modes = _parse_csv(args.modes)
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

    print(f"=== Boids Evaluation: {len(modes)} modes × {len(seeds)} seeds = {len(modes) * len(seeds)} runs ===")
    results: dict[str, list[dict]] = {mode: [] for mode in modes}
    for mode in modes:
        for seed in seeds:
            run_name = f"{mode}_seed{seed}"
            print(f"  running {run_name} ...")
            outcome = run_learnable_subprocess(
                options=run_options,
                run_name=run_name,
                mode=mode,
                seed=int(seed),
                lr=args.lr,
            )
            if not outcome.ok:
                print(f"  FAILED {run_name}: exit={outcome.returncode}")
                if outcome.stderr:
                    for line in outcome.stderr.strip().splitlines()[-5:]:
                        print(f"    {line}")
                continue
            results[mode].append(outcome.summary or {})

    metrics = {
        "final_traj_loss": "training.final_traj_loss",
        "best_traj_loss": "training.best_traj_loss",
        "final_center_error": "training.final_center_error",
    }
    recovery_params = ["w_sep", "w_align", "w_cohesion", "damping", "max_speed"]
    rows = []
    for mode in modes:
        summaries = results[mode]
        if not summaries:
            continue
        row: dict[str, str | float] = {"mode": mode, "runs": len(summaries)}
        for metric_name, metric_path in metrics.items():
            values = [
                float(nested_get(summary, metric_path))
                for summary in summaries
                if is_finite_number(nested_get(summary, metric_path))
            ]
            row[f"{metric_name}_mean"] = mean(values) if values else float("nan")
            row[f"{metric_name}_std"] = std(values) if values else float("nan")
        for name in recovery_params:
            errors = []
            for summary in summaries:
                path = f"parameters.recovery.{name}.rel_error"
                rel_error = nested_get(summary, path)
                if is_finite_number(rel_error):
                    errors.append(float(rel_error))
            row[f"recovery_{name}_rel_mean"] = mean(errors) if errors else float("nan")
        rows.append(row)

    if rows:
        csv_path = root / "evaluation_summary.csv"
        with csv_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
        print(f"\nSaved {csv_path}")

    print("\n### Results\n")
    header = f"| {'mode':^10} | {'runs':>4} | {'traj_loss':>12} | {'best_traj':>12} | {'center_err':>12} |"
    header += "".join(f" {name + '_rel':>12} |" for name in recovery_params)
    print(header)
    print("|" + "-" * 12 + "|" + ("-" * 6 + "|") + (("-" * 14 + "|") * 3) + (("-" * 14 + "|") * len(recovery_params)))

    def _fmt(row: dict[str, str | float], key: str) -> str:
        mean_value = row.get(f"{key}_mean", float("nan"))
        std_value = row.get(f"{key}_std", float("nan"))
        if not is_finite_number(mean_value):
            return "N/A"
        return f"{mean_value:.6f}±{std_value:.6f}" if is_finite_number(std_value) else f"{mean_value:.6f}"

    for row in rows:
        line = f"| {row['mode']:^10} | {row['runs']:>4} | {_fmt(row, 'final_traj_loss'):>12} | {_fmt(row, 'best_traj_loss'):>12} | {_fmt(row, 'final_center_error'):>12} |"
        for name in recovery_params:
            value = row.get(f"recovery_{name}_rel_mean", float("nan"))
            line += f" {value:>11.1%} |" if is_finite_number(value) else "         N/A |"
        print(line)

    with (root / "all_summaries.json").open("w", encoding="utf-8") as handle:
        json.dump({mode: results[mode] for mode in modes}, handle, indent=2)
    print("\nDone.")


if __name__ == "__main__":
    main()