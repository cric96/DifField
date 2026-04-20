#!/usr/bin/env python3
"""Learnable boids entry point.

This is the single orchestrator for the learnable boids experiment.
The flow is explicit and numbered below.
"""

from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "examples"))

from boids.cli import parse_learnable_args
from boids.domain.specs import build_learnable_spec
from boids.domain.geometry import sample_initial_state
from boids.training.trace import teacher_trace_from_specs
from boids.training.loop import BoidsTrainingLoop
from boids.evaluation.evaluator import evaluate_seed
from boids.visualization.renderer import BoidsRenderer
from boids.reporting.recovery import (
    compute_parameter_recovery_metrics,
    extract_teacher_parameters_from_spec,
)
from boids.reporting.summary import BoidsSummaryBuilder
from shared.diagnostics import export_diagnostics, save_history_csv
from shared.training import MetricHistory, grad_norm


def main() -> None:
    # ── Step 1: Parse arguments and initialise ──────────────────────────────
    args = parse_learnable_args()
    torch.manual_seed(args.seed)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = args.run_name.strip() or f"boids_seed{args.seed}_{timestamp}"
    run_dir = Path(args.out_dir) / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    viz_prefix = (
        str(run_dir / "learnable")
        if args.viz_prefix == "generated/boids/learnable"
        else args.viz_prefix
    )

    spec = build_learnable_spec(args, run_name=run_name, run_dir=run_dir, viz_prefix=viz_prefix)

    # ── Step 2: Build initial conditions and teacher traces ─────────────────
    initial_conditions = [
        sample_initial_state(
            spec.simulation.num_nodes,
            seed=spec.seed + idx * 1337,
            velocity_scale=spec.simulation.init_velocity_scale,
            device=spec.simulation.device,
        )
        for idx in range(spec.training.num_initial_conditions)
    ]

    from boids.training.trace import build_supervision_traces
    supervision_traces = build_supervision_traces(initial_conditions, spec=spec)

    # ── Step 3: Build model, optimizer, renderer ────────────────────────────
    positions0 = supervision_traces[0].positions0
    velocities0 = supervision_traces[0].velocities0

    from boids.model.boids_model import LearnableAggregateBoids
    model = LearnableAggregateBoids.from_specs(
        positions0=positions0,
        simulation=spec.simulation,
        model=spec.model,
    ).to(spec.simulation.device)

    optimizer = torch.optim.Adam(model.trainable_parameters(), lr=spec.training.lr)
    renderer = BoidsRenderer(spec)

    # ── Step 4: Training loop ───────────────────────────────────────────────
    from boids.training.curriculum import curriculum_horizon, scheduled_learning_rate
    from boids.training.supervision import teacher_forced_step_losses
    from boids.cli import HISTORY_KEYS

    history = MetricHistory.from_keys(HISTORY_KEYS)
    teacher_params = extract_teacher_parameters_from_spec(spec)

    for epoch in range(spec.training.epochs):
        # Learning rate schedule
        current_lr = scheduled_learning_rate(
            spec.training.lr, epoch, spec.training.epochs,
            ramp_fraction=spec.training.curriculum_ramp_fraction,
            final_lr_ratio=spec.training.final_lr_ratio,
        )
        for group in optimizer.param_groups:
            group["lr"] = current_lr

        # Curriculum horizon
        horizon = curriculum_horizon(
            epoch, spec.training.epochs,
            spec.training.min_horizon, spec.training.max_horizon,
            ramp_fraction=spec.training.curriculum_ramp_fraction,
        )

        # Forward: compute losses over all traces
        total_loss = torch.zeros((), device=spec.simulation.device)
        total_pos_loss = torch.zeros((), device=spec.simulation.device)
        total_vel_loss = torch.zeros((), device=spec.simulation.device)
        total_sep_focus = torch.zeros((), device=spec.simulation.device)

        for trace in supervision_traces:
            loss, pos_loss, vel_loss, sep_focus = teacher_forced_step_losses(
                model, trace=trace.sliced(horizon),
                velocity_loss_weight=spec.training.velocity_loss_weight,
                sep=spec.simulation.sep,
            )
            total_loss += loss
            total_pos_loss += pos_loss
            total_vel_loss += vel_loss
            total_sep_focus += sep_focus

        n = float(len(supervision_traces))
        avg_total = total_loss / n
        avg_sep_focus = total_sep_focus / n
        objective = avg_total + spec.training.separation_loss_weight * avg_sep_focus

        # Backward and step
        optimizer.zero_grad()
        objective.backward()
        g_norm = grad_norm(model.trainable_parameters())
        torch.nn.utils.clip_grad_norm_(model.trainable_parameters(), 5.0)
        optimizer.step()

        # Monitor: free-run rollout to check center error
        with torch.no_grad():
            _, _, final_pos = model.rollout(
                spec.simulation.rounds,
                positions0=positions0, velocities0=velocities0,
                trunc_window=spec.training.trunc_window,
            )
        teacher_pos_seq = supervision_traces[0].pos_seq
        center_error = (teacher_pos_seq[-1].mean(dim=0) - final_pos.mean(dim=0)).norm().item()

        # Validation evaluation
        eval_metrics = _evaluate_epoch(model, spec, epoch, horizon)

        # Record history
        learned_params = {
            "w_sep": float(model.w_sep.item()),
            "w_align": float(model.w_align.item()),
            "w_cohesion": float(model.w_cohesion.item()),
        }
        recovery = compute_parameter_recovery_metrics(teacher_params, learned_params)

        history.append(
            epoch=float(epoch + 1), horizon=float(horizon),
            total=float(avg_total.item()), per_step_loss=float(avg_total.item()),
            objective_loss=float(objective.item()),
            pos_loss=float((total_pos_loss / n).item()),
            vel_loss=float((total_vel_loss / n).item()),
            sep_focus_loss=float(avg_sep_focus.item()),
            center_error=center_error,
            w_sep=learned_params["w_sep"],
            w_sep_abs_error=float(recovery["w_sep"]["abs_error"]),
            w_sep_rel_error=float(recovery["w_sep"]["rel_error"]),
            w_align=learned_params["w_align"],
            w_align_abs_error=float(recovery["w_align"]["abs_error"]),
            w_align_rel_error=float(recovery["w_align"]["rel_error"]),
            w_cohesion=learned_params["w_cohesion"],
            w_cohesion_abs_error=float(recovery["w_cohesion"]["abs_error"]),
            w_cohesion_rel_error=float(recovery["w_cohesion"]["rel_error"]),
            grad_norm=float(g_norm), lr=float(current_lr),
            cap_fraction=float(model.last_rollout_speed_health.get("mean_cap_fraction", 0.0)),
            pre_clip_speed=float(model.last_rollout_speed_health.get("mean_pre_clip_speed", 0.0)),
            val_curriculum_horizon=float(eval_metrics["curriculum_horizon"]),
            val_curriculum_total_loss=float(eval_metrics["curriculum_total_loss"]),
            val_curriculum_pos_loss=float(eval_metrics["curriculum_pos_loss"]),
            val_curriculum_vel_loss=float(eval_metrics["curriculum_vel_loss"]),
            val_curriculum_per_step_loss=float(eval_metrics["curriculum_per_step_loss"]),
            val_curriculum_center_error=float(eval_metrics["curriculum_center_error"]),
            val_full_horizon=float(eval_metrics["full_horizon"]),
            val_full_total_loss=float(eval_metrics["full_total_loss"]),
            val_full_pos_loss=float(eval_metrics["full_pos_loss"]),
            val_full_vel_loss=float(eval_metrics["full_vel_loss"]),
            val_full_per_step_loss=float(eval_metrics["full_per_step_loss"]),
            val_full_center_error=float(eval_metrics["full_center_error"]),
        )

        # Console logging
        if (epoch + 1) % spec.training.print_every == 0 or epoch == 0:
            msg = (
                f"epoch={epoch + 1:3d} step={avg_total.item():.6f} center={center_error:.6f} "
                f"err_sep={recovery['w_sep']['rel_error']:.1%} "
                f"err_align={recovery['w_align']['rel_error']:.1%} "
                f"err_coh={recovery['w_cohesion']['rel_error']:.1%}"
            )
            if not torch.isnan(torch.tensor(eval_metrics["full_per_step_loss"])):
                msg += (
                    f" val_step={eval_metrics['full_per_step_loss']:.6f}"
                    f" val_center={eval_metrics['full_center_error']:.6f}"
                )
            print(msg)

        # Validation rendering callback
        renderer.render_validation_checkpoint(model, epoch)

    # ── Step 5: Final evaluation and visualization ──────────────────────────
    teacher_trace = teacher_trace_from_specs(
        seed=spec.seed, positions0=positions0, velocities0=velocities0,
        simulation=spec.simulation, teacher=spec.teacher, model=spec.model,
    )

    renderer.render_training_suite(
        model,
        teacher_pos_seq=teacher_trace.pos_seq,
        positions0=positions0, velocities0=velocities0,
    )

    # ── Step 6: Save reporting artifacts ────────────────────────────────────
    learned_params = {
        "w_sep": float(model.w_sep.item()),
        "w_align": float(model.w_align.item()),
        "w_cohesion": float(model.w_cohesion.item()),
    }
    recovery = compute_parameter_recovery_metrics(teacher_params, learned_params)

    summary = BoidsSummaryBuilder(
        run_name=spec.run_name, spec=spec, history=history.to_dict(),
        model=model, effective_curriculum_max_horizon=spec.training.max_horizon,
        recovery=recovery,
    ).build()

    save_history_csv(history.to_dict(), spec.run_dir / "history.csv")
    export_diagnostics(
        history.to_dict(), spec.run_dir / "diagnostics",
        title_prefix=f"{spec.run_name} ",
        teacher_params=teacher_params, learned_params=learned_params,
        compact=True,
    )

    with (spec.run_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

    print(f"Artifacts saved to {spec.run_dir}")


def _evaluate_epoch(model, spec, epoch, horizon) -> dict[str, float]:
    """Evaluate model on held-out seeds at periodic intervals."""
    from boids.evaluation.evaluator import evaluate_seed, mean_eval_metrics

    eval_cfg = spec.evaluation
    do_eval = bool(eval_cfg.seeds) and (
        (epoch + 1) % max(1, eval_cfg.every) == 0
        or epoch == 0
        or epoch == spec.training.epochs - 1
    )
    if not do_eval:
        return {
            key: float("nan")
            for key in [
                "curriculum_horizon", "curriculum_total_loss", "curriculum_pos_loss",
                "curriculum_vel_loss", "curriculum_per_step_loss", "curriculum_center_error",
                "full_horizon", "full_total_loss", "full_pos_loss",
                "full_vel_loss", "full_per_step_loss", "full_center_error",
            ]
        }

    curriculum_pairs = [
        evaluate_seed(
            model, seed=eval_seed, simulation=spec.simulation,
            teacher=spec.teacher, model_spec=spec.model,
            velocity_loss_weight=spec.training.velocity_loss_weight,
            rounds=horizon,
        )
        for eval_seed in eval_cfg.seeds
    ]
    curriculum_metrics = mean_eval_metrics(curriculum_pairs)

    if horizon == spec.simulation.rounds:
        full_metrics = curriculum_metrics
    else:
        full_pairs = [
            evaluate_seed(
                model, seed=eval_seed, simulation=spec.simulation,
                teacher=spec.teacher, model_spec=spec.model,
                velocity_loss_weight=spec.training.velocity_loss_weight,
            )
            for eval_seed in eval_cfg.seeds
        ]
        full_metrics = mean_eval_metrics(full_pairs)

    return {
        "curriculum_horizon": curriculum_metrics["horizon"],
        "curriculum_total_loss": curriculum_metrics["total_loss"],
        "curriculum_pos_loss": curriculum_metrics["pos_loss"],
        "curriculum_vel_loss": curriculum_metrics["vel_loss"],
        "curriculum_per_step_loss": curriculum_metrics["per_step_loss"],
        "curriculum_center_error": curriculum_metrics["center_error"],
        "full_horizon": full_metrics["horizon"],
        "full_total_loss": full_metrics["total_loss"],
        "full_pos_loss": full_metrics["pos_loss"],
        "full_vel_loss": full_metrics["vel_loss"],
        "full_per_step_loss": full_metrics["per_step_loss"],
        "full_center_error": full_metrics["center_error"],
    }


if __name__ == "__main__":
    main()
