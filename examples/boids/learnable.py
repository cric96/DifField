#!/usr/bin/env python3
"""Thin entry point for learnable aggregate boids experiments."""

from __future__ import annotations

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "examples"))

from boids.cli import parse_learnable_args
from boids.domain.specs import build_learnable_spec
from boids.training import BoidsTrainingLoop
from boids.evaluation import evaluate_seed
from boids.visualization import BoidsRenderer
from boids.reporting import (
    BoidsSummaryBuilder,
    compute_parameter_recovery_metrics,
    extract_teacher_parameters_from_spec,
)
from shared.diagnostics import export_diagnostics, save_history_csv


def main() -> None:
    # 1. Initialization
    args = parse_learnable_args()
    torch.manual_seed(args.seed)

    from datetime import datetime

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = args.run_name.strip() or f"boids_demo_seed{args.seed}_{timestamp}"
    run_dir = Path(args.out_dir) / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    viz_prefix = (
        str(run_dir / "learnable")
        if args.viz_prefix == "generated/boids/learnable"
        else args.viz_prefix
    )

    spec = build_learnable_spec(
        args, run_name=run_name, run_dir=run_dir, viz_prefix=viz_prefix
    )
    renderer = BoidsRenderer(spec)
    trainer = BoidsTrainingLoop(spec)

    # 2. Setup rendering callback
    def on_epoch_end(epoch, model, monitor_results):
        # Optional validation rendering
        renderer.render_validation_checkpoint(model, epoch)

    # 3. Execute training
    model, history = trainer.run(on_epoch_end=on_epoch_end)

    # 4. Final Evaluation and Visualization
    # (Simplified for now, using the results from the trainer's last step for viz)
    initial_conditions = trainer._make_initial_conditions()
    positions0, velocities0 = initial_conditions[0]

    from boids.training.trace import teacher_trace_from_specs

    teacher_trace = teacher_trace_from_specs(
        seed=spec.seed,
        positions0=positions0,
        velocities0=velocities0,
        simulation=spec.simulation,
        teacher=spec.teacher,
        model=spec.model,
    )

    renderer.render_training_suite(
        model,
        teacher_pos_seq=teacher_trace.pos_seq,
        positions0=positions0,
        velocities0=velocities0,
    )

    # 5. Reporting
    teacher_params = extract_teacher_parameters_from_spec(spec)
    learned_params = {
        "w_sep": float(model.w_sep.item()),
        "w_align": float(model.w_align.item()),
        "w_cohesion": float(model.w_cohesion.item()),
        "damping": float(model.damping.item()),
        "max_speed": float(model.max_speed.item()),
    }
    recovery = compute_parameter_recovery_metrics(teacher_params, learned_params)

    summary = BoidsSummaryBuilder(
        run_name=spec.run_name,
        spec=spec,
        history=history,
        model=model,
        effective_curriculum_max_horizon=spec.training.max_horizon,
        recovery=recovery,
    ).build()

    # Save artifacts
    save_history_csv(history, spec.run_dir / "history.csv")
    export_diagnostics(
        history,
        spec.run_dir / "diagnostics",
        title_prefix=f"{spec.run_name} ",
        teacher_params=teacher_params,
        learned_params=learned_params,
        compact=True,
    )

    import json

    with (spec.run_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

    print(f"Artifacts saved to {spec.run_dir}")


if __name__ == "__main__":
    main()
