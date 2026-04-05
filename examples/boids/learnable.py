#!/usr/bin/env python3
"""Aggregate boids with learnable scalar dynamics weights."""

from __future__ import annotations

import json
import math
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "examples"))

from autofield import build_spatial_graph

try:
    from ..shared.diagnostics import export_diagnostics, save_history_csv, save_summary_csv
    from ..shared.experiment import CheckpointManager, CheckpointPolicy, MovingGraphVisualizationPipeline, VizSpec, flatten_summary_for_csv
    from ..shared.history import MetricHistory
    from ..shared.training import grad_norm
    from .cli import HISTORY_KEYS, parse_learnable_args
    from .config import LearnableBoidsSpec, build_learnable_spec
    from .core import sample_initial_boids_state
    from .evaluation_utils import evaluate_seed
    from .losses import close_pair_distance_loss
    from .model import LearnableAggregateBoids
    from .replay import BoidsTrace, build_supervision_traces
    from .reporting import BoidsSummaryBuilder, compute_parameter_recovery_metrics, extract_learned_parameters, extract_teacher_parameters_from_spec
except ImportError:
    from shared.diagnostics import export_diagnostics, save_history_csv, save_summary_csv
    from shared.experiment import CheckpointManager, CheckpointPolicy, MovingGraphVisualizationPipeline, VizSpec, flatten_summary_for_csv
    from shared.history import MetricHistory
    from shared.training import grad_norm
    from boids.cli import HISTORY_KEYS, parse_learnable_args
    from boids.config import LearnableBoidsSpec, build_learnable_spec
    from boids.core import sample_initial_boids_state
    from boids.evaluation_utils import evaluate_seed
    from boids.losses import close_pair_distance_loss
    from boids.model import LearnableAggregateBoids
    from boids.replay import BoidsTrace, build_supervision_traces
    from boids.reporting import BoidsSummaryBuilder, compute_parameter_recovery_metrics, extract_learned_parameters, extract_teacher_parameters_from_spec


def _mean_eval_metrics(items: list[dict[str, float]]) -> dict[str, float]:
    if not items:
        return {
            "horizon": float("nan"),
            "total_loss": float("nan"),
            "pos_loss": float("nan"),
            "vel_loss": float("nan"),
            "per_step_loss": float("nan"),
            "center_error": float("nan"),
        }
    return {
        key: float(sum(item[key] for item in items) / len(items))
        for key in items[0]
    }


def teacher_forced_step_losses(
    model: LearnableAggregateBoids,
    *,
    trace: BoidsTrace,
    velocity_loss_weight: float,
    sep: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    total_loss = torch.zeros((), device=trace.positions0.device)
    total_pos_loss = torch.zeros((), device=trace.positions0.device)
    total_vel_loss = torch.zeros((), device=trace.positions0.device)
    total_sep_focus_loss = torch.zeros((), device=trace.positions0.device)

    state_pos = trace.positions0
    state_vel = trace.velocities0
    num_steps = int(trace.pos_seq.shape[0])

    for step_idx, (target_pos, target_vel, target_preclip_vel) in enumerate(zip(trace.pos_seq, trace.vel_seq, trace.preclip_vel_seq)):
        pred_pos, pred_vel, pred_preclip_vel = model.step(
            positions=state_pos,
            velocities=state_vel,
            use_init_connectivity=step_idx == 0,
        )
        pos_loss = torch.nn.functional.mse_loss(pred_pos, target_pos)
        vel_loss = torch.nn.functional.mse_loss(pred_preclip_vel, target_preclip_vel)
        loss = pos_loss + velocity_loss_weight * vel_loss
        sep_focus_loss = close_pair_distance_loss(
            pred_pos.unsqueeze(0),
            target_pos.unsqueeze(0),
            sep=sep,
        )
        total_loss = total_loss + loss
        total_pos_loss = total_pos_loss + pos_loss
        total_vel_loss = total_vel_loss + vel_loss
        total_sep_focus_loss = total_sep_focus_loss + sep_focus_loss
        state_pos = target_pos
        state_vel = target_vel

    scale = 1.0 / max(1, num_steps)
    return (
        total_loss * scale,
        total_pos_loss * scale,
        total_vel_loss * scale,
        total_sep_focus_loss * scale,
    )


@dataclass(frozen=True)
class RunContext:
    spec: LearnableBoidsSpec


def curriculum_horizon(epoch: int, total_epochs: int, min_horizon: int, max_horizon: int) -> int:
    return curriculum_horizon_with_fraction(epoch, total_epochs, min_horizon, max_horizon, ramp_fraction=0.85)


def _curriculum_ramp_epochs(total_epochs: int, ramp_fraction: float) -> int:
    if total_epochs <= 1:
        return 1
    return max(1, int(round(total_epochs * ramp_fraction)))


def curriculum_horizon_with_fraction(
    epoch: int,
    total_epochs: int,
    min_horizon: int,
    max_horizon: int,
    *,
    ramp_fraction: float,
) -> int:
    if total_epochs <= 1:
        return max_horizon
    ramp_epochs = _curriculum_ramp_epochs(total_epochs, ramp_fraction)
    progress = min(1.0, epoch / ramp_epochs)
    return int(round(min_horizon + (max_horizon - min_horizon) * progress))


def scheduled_learning_rate(
    base_lr: float,
    epoch: int,
    total_epochs: int,
    *,
    ramp_fraction: float,
    final_lr_ratio: float,
) -> float:
    if total_epochs <= 1:
        return float(base_lr)
    ramp_epochs = _curriculum_ramp_epochs(total_epochs, ramp_fraction)
    if epoch <= ramp_epochs:
        return float(base_lr)
    tail_epochs = max(1, total_epochs - ramp_epochs - 1)
    progress = min(1.0, (epoch - ramp_epochs) / tail_epochs)
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return float(base_lr * (final_lr_ratio + (1.0 - final_lr_ratio) * cosine))


def make_initial_conditions(spec: LearnableBoidsSpec) -> list[tuple[torch.Tensor, torch.Tensor]]:
    return [
        sample_initial_boids_state(
            spec.simulation.num_nodes,
            seed=spec.seed + idx * 1337,
            velocity_scale=spec.simulation.init_velocity_scale,
            device=spec.simulation.device,
        )
        for idx in range(spec.training.num_initial_conditions)
    ]


TRAINED_PARAMETER_NAMES = ("w_sep", "w_align", "w_cohesion", "damping")


class LearnableBoidsWorkflow:
    """Template-method style orchestrator for the learnable boids script."""

    def __init__(self, args: Any):
        self.args = args

    def run(self) -> None:
        torch.manual_seed(self.args.seed)
        ctx = self._build_context()
        model, history_data, teacher_pos_seq, positions0, velocities0, checkpoint_policy, checkpoint_manager = self._train(ctx)
        recovery = self._save_training_outputs(ctx, model, history_data)
        self._render_visualizations(ctx, model, teacher_pos_seq, positions0, velocities0, checkpoint_policy, checkpoint_manager)
        self._save_summary(ctx, model, history_data, recovery)

    def _build_context(self) -> RunContext:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_name = self.args.run_name.strip() or f"boids_demo_seed{self.args.seed}_{timestamp}"
        run_dir = Path(self.args.out_dir) / run_name
        run_dir.mkdir(parents=True, exist_ok=True)
        viz_prefix = str(run_dir / "learnable") if self.args.viz_prefix == "generated/boids/learnable" else self.args.viz_prefix
        return RunContext(spec=build_learnable_spec(self.args, run_name=run_name, run_dir=run_dir, viz_prefix=viz_prefix))

    def _train(
        self,
        ctx: RunContext,
    ) -> tuple[
        LearnableAggregateBoids,
        dict[str, list[float]],
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        CheckpointPolicy,
        CheckpointManager,
    ]:
        spec = ctx.spec
        device = spec.simulation.device
        initial_conditions = make_initial_conditions(spec)
        supervision_traces = build_supervision_traces(initial_conditions, spec=spec)
        positions0 = supervision_traces[0].positions0
        velocities0 = supervision_traces[0].velocities0
        teacher_pos_seq = supervision_traces[0].pos_seq
        model = LearnableAggregateBoids.from_specs(
            positions0=positions0,
            simulation=spec.simulation,
            model=spec.model,
        ).to(device)
        params = model.trainable_parameters()
        optimizer = torch.optim.Adam(params, lr=spec.training.lr)
        history = MetricHistory.from_keys(HISTORY_KEYS)

        checkpoint_policy = CheckpointPolicy(total_epochs=spec.training.epochs, every_epochs=spec.training.checkpoint_every_epochs)
        checkpoint_manager = CheckpointManager(spec.run_dir / "checkpoints", checkpoint_policy)
        teacher_params = extract_teacher_parameters_from_spec(spec)

        validation_seeds = ",".join(str(seed) for seed in spec.evaluation.seeds) if spec.evaluation.seeds else "disabled"
        trace_dir = "disabled" if spec.training.replay_trace_dir is None else str(spec.training.replay_trace_dir)
        print("=== Aggregate Learnable Boids Demo ===")
        print(
            f"target=alignment+cohesion nodes={spec.simulation.num_nodes} "
            f"rounds={spec.simulation.rounds} epochs={spec.training.epochs}"
        )
        print(
            f"teacher: w_sep={spec.teacher.w_sep:.2f} w_align={spec.teacher.w_align:.2f} "
            f"w_cohesion={spec.teacher.w_cohesion:.2f} damping={spec.teacher.damping:.3f} max_speed={spec.teacher.max_speed:.4f}"
        )
        print(
            f"init: w_sep={spec.model.init_w_sep_target:.2f} w_align={spec.model.init_w_align_target:.2f} "
            f"w_cohesion={spec.model.init_w_cohesion_target:.2f} damping={spec.model.init_damping_target:.3f} "
            f"lr={spec.training.lr:.3f} supervision={spec.training.supervision_mode} "
            f"curriculum={spec.training.min_horizon}->{spec.training.max_horizon} validation={validation_seeds} every={spec.evaluation.every}"
        )
        print(f"trace_dir={trace_dir}")
        print(f"run_dir={spec.run_dir}")

        self._render_initial_validation_gif(spec, model)

        for epoch in range(spec.training.epochs):
            current_lr = scheduled_learning_rate(
                spec.training.lr,
                epoch,
                spec.training.epochs,
                ramp_fraction=spec.training.curriculum_ramp_fraction,
                final_lr_ratio=spec.training.final_lr_ratio,
            )
            for group in optimizer.param_groups:
                group["lr"] = current_lr

            horizon = curriculum_horizon_with_fraction(
                epoch,
                spec.training.epochs,
                spec.training.min_horizon,
                spec.training.max_horizon,
                ramp_fraction=spec.training.curriculum_ramp_fraction,
            )
            optimizer.zero_grad()
            total_loss = torch.zeros((), device=device)
            total_pos_loss = torch.zeros((), device=device)
            total_vel_loss = torch.zeros((), device=device)
            total_sep_focus_loss = torch.zeros((), device=device)
            checkpoint_pos_seq = None
            checkpoint_vel_seq = None

            for trace in supervision_traces:
                loss, pos_loss, vel_loss, sep_focus_loss = teacher_forced_step_losses(
                    model,
                    trace=trace.sliced(horizon),
                    velocity_loss_weight=spec.training.velocity_loss_weight,
                    sep=spec.simulation.sep,
                )
                total_loss = total_loss + loss
                total_pos_loss = total_pos_loss + pos_loss
                total_vel_loss = total_vel_loss + vel_loss
                total_sep_focus_loss = total_sep_focus_loss + sep_focus_loss

            num_initial_conditions = float(len(initial_conditions))
            total = total_loss / num_initial_conditions
            pos_loss = total_pos_loss / num_initial_conditions
            vel_loss = total_vel_loss / num_initial_conditions
            sep_focus_loss = total_sep_focus_loss / num_initial_conditions
            per_step_loss = total
            objective_loss = total + spec.training.separation_loss_weight * sep_focus_loss

            objective_loss.backward()
            current_grad_norm = grad_norm(params)
            torch.nn.utils.clip_grad_norm_(params, 5.0)
            optimizer.step()

            monitor_pos_seq, monitor_vel_seq, monitor_final_pos = model.rollout(
                spec.simulation.rounds,
                positions0=positions0,
                velocities0=velocities0,
                trunc_window=spec.training.trunc_window,
            )
            center_error = (
                teacher_pos_seq[-1].mean(dim=0) - monitor_final_pos.mean(dim=0)
            ).norm().item()
            checkpoint_pos_seq = monitor_pos_seq.detach()
            checkpoint_vel_seq = monitor_vel_seq.detach()

            eval_metrics = self._evaluate_epoch(model, spec, epoch, horizon)
            did_eval = eval_metrics["full_total_loss"] == eval_metrics["full_total_loss"]
            if did_eval:
                self._render_validation_gif_for_epoch(spec, model, epoch)

            learned_params = self._current_learned_parameters(model)
            parameter_recovery = compute_parameter_recovery_metrics(teacher_params, learned_params)

            history.append(
                epoch=float(epoch + 1),
                horizon=float(horizon),
                total=float(total.item()),
                per_step_loss=float(per_step_loss.item()),
                objective_loss=float(objective_loss.item()),
                pos_loss=float(pos_loss.item()),
                vel_loss=float(vel_loss.item()),
                sep_focus_loss=float(sep_focus_loss.item()),
                center_error=float(center_error),
                w_sep=learned_params["w_sep"],
                w_sep_abs_error=float(parameter_recovery["w_sep"]["abs_error"]),
                w_sep_rel_error=float(parameter_recovery["w_sep"]["rel_error"]),
                w_align=learned_params["w_align"],
                w_align_abs_error=float(parameter_recovery["w_align"]["abs_error"]),
                w_align_rel_error=float(parameter_recovery["w_align"]["rel_error"]),
                w_cohesion=learned_params["w_cohesion"],
                w_cohesion_abs_error=float(parameter_recovery["w_cohesion"]["abs_error"]),
                w_cohesion_rel_error=float(parameter_recovery["w_cohesion"]["rel_error"]),
                damping=learned_params["damping"],
                damping_abs_error=float(parameter_recovery["damping"]["abs_error"]),
                damping_rel_error=float(parameter_recovery["damping"]["rel_error"]),
                max_speed=learned_params["max_speed"],
                max_speed_abs_error=float(parameter_recovery["max_speed"]["abs_error"]),
                max_speed_rel_error=float(parameter_recovery["max_speed"]["rel_error"]),
                grad_norm=float(current_grad_norm),
                lr=float(current_lr),
                cap_fraction=float(model.last_rollout_speed_health["mean_cap_fraction"]),
                pre_clip_speed=float(model.last_rollout_speed_health["mean_pre_clip_speed"]),
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

            if checkpoint_manager.should_save(epoch) and checkpoint_pos_seq is not None and checkpoint_vel_seq is not None:
                checkpoint_manager.save_rollout(epoch, pred_pos_seq=checkpoint_pos_seq, pred_vel_seq=checkpoint_vel_seq)

            if (epoch + 1) % spec.training.print_every == 0 or epoch == 0:
                message = (
                    f"epoch={epoch + 1:3d} train_step={per_step_loss.item():.6f} center={center_error:.6f} "
                    f"err_sep={parameter_recovery['w_sep']['rel_error']:.1%} "
                    f"err_align={parameter_recovery['w_align']['rel_error']:.1%} "
                    f"err_coh={parameter_recovery['w_cohesion']['rel_error']:.1%} "
                    f"err_damp={parameter_recovery['damping']['rel_error']:.1%}"
                )
                if did_eval:
                    message += (
                        f" val_step={eval_metrics['full_per_step_loss']:.6f}"
                        f" val_center={eval_metrics['full_center_error']:.6f}"
                    )
                print(message)

        print("Training complete.")

        return model, history.to_dict(), teacher_pos_seq, positions0, velocities0, checkpoint_policy, checkpoint_manager

    def _evaluate_epoch(
        self,
        model: LearnableAggregateBoids,
        spec: LearnableBoidsSpec,
        epoch: int,
        horizon: int,
    ) -> dict[str, float]:
        eval_cfg = spec.evaluation
        do_eval = bool(eval_cfg.seeds) and (
            (epoch + 1) % max(1, eval_cfg.every) == 0 or epoch == 0 or epoch == spec.training.epochs - 1
        )
        if not do_eval:
            return {
                "curriculum_horizon": float("nan"),
                "curriculum_total_loss": float("nan"),
                "curriculum_pos_loss": float("nan"),
                "curriculum_vel_loss": float("nan"),
                "curriculum_per_step_loss": float("nan"),
                "curriculum_center_error": float("nan"),
                "full_horizon": float("nan"),
                "full_total_loss": float("nan"),
                "full_pos_loss": float("nan"),
                "full_vel_loss": float("nan"),
                "full_per_step_loss": float("nan"),
                "full_center_error": float("nan"),
            }
        curriculum_pairs = [
            evaluate_seed(
                model,
                seed=eval_seed,
                simulation=spec.simulation,
                teacher=spec.teacher,
                model_spec=spec.model,
                velocity_loss_weight=spec.training.velocity_loss_weight,
                rounds=horizon,
            )
            for eval_seed in eval_cfg.seeds
        ]
        curriculum_metrics = _mean_eval_metrics(curriculum_pairs)

        if horizon == spec.simulation.rounds:
            full_metrics = curriculum_metrics
        else:
            full_pairs = [
                evaluate_seed(
                    model,
                    seed=eval_seed,
                    simulation=spec.simulation,
                    teacher=spec.teacher,
                    model_spec=spec.model,
                    velocity_loss_weight=spec.training.velocity_loss_weight,
                )
                for eval_seed in eval_cfg.seeds
            ]
            full_metrics = _mean_eval_metrics(full_pairs)

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

    def _save_training_outputs(
        self,
        ctx: RunContext,
        model: LearnableAggregateBoids,
        history_data: dict[str, list[float]],
    ) -> dict[str, dict[str, float]]:
        spec = ctx.spec
        teacher_params = extract_teacher_parameters_from_spec(spec)
        learned_params = extract_learned_parameters(history_data)
        recovery = compute_parameter_recovery_metrics(teacher_params, learned_params)

        print("--- Demo Outcome ---")
        print(
            f"  train_step={history_data['per_step_loss'][-1]:.6f} "
            f"val_step={history_data['val_full_per_step_loss'][-1]:.6f} "
            f"val_center={history_data['val_full_center_error'][-1]:.6f}"
        )
        for name in TRAINED_PARAMETER_NAMES:
            item = recovery[name]
            print(
                f"  {name}: target={item['teacher']:.4f} learned={item['learned']:.4f} "
                f"abs_err={item['abs_error']:.4f} rel_err={item['rel_error']:.2%}"
            )

        save_history_csv(history_data, spec.run_dir / "history.csv")
        export_diagnostics(
            history_data,
            spec.run_dir / "diagnostics",
            title_prefix=f"{spec.run_name} ",
            teacher_params=teacher_params,
            learned_params=learned_params,
            compact=True,
        )
        return recovery

    def _build_visualization_runtime(
        self,
        spec: LearnableBoidsSpec,
    ) -> tuple[MovingGraphVisualizationPipeline, VizSpec, int]:
        highlight_idx = int(max(0, min(spec.visualization.highlight_node, spec.simulation.num_nodes - 1)))
        viz_spec = VizSpec(
            enabled=spec.visualization.enabled,
            gif_enabled=spec.visualization.gif_enabled,
            compare_panel_enabled=False,
            show_links=spec.visualization.show_links,
            links_alpha=spec.visualization.links_alpha,
            links_width=spec.visualization.links_width,
            gif_fps=spec.visualization.gif_fps,
        )

        def edge_builder(positions: torch.Tensor) -> torch.Tensor:
            edge_index, _ = build_spatial_graph(
                positions,
                edge_radius=spec.simulation.radius if spec.model.init_connectivity in {"radius", "hybrid"} else None,
                k_neighbors=spec.model.init_k_neighbors if spec.model.init_connectivity == "knn" else None,
            )
            return edge_index

        viz_pipeline = MovingGraphVisualizationPipeline(
            edge_builder=edge_builder,
            rounds=spec.simulation.rounds,
            record_every=spec.visualization.record_every,
        )
        return viz_pipeline, viz_spec, highlight_idx

    def _snapshot_model_for_visualization(
        self,
        spec: LearnableBoidsSpec,
        model: LearnableAggregateBoids,
    ) -> LearnableAggregateBoids:
        snapshot = LearnableAggregateBoids.from_specs(
            positions0=model.positions0.detach().clone(),
            simulation=spec.simulation,
            model=spec.model,
        ).to(spec.simulation.device)
        snapshot.load_state_dict(model.state_dict())
        snapshot.eval()
        return snapshot

    def _current_learned_parameters(
        self,
        model: LearnableAggregateBoids,
    ) -> dict[str, float]:
        return {
            "w_sep": float(model.w_sep.item()),
            "w_align": float(model.w_align.item()),
            "w_cohesion": float(model.w_cohesion.item()),
            "damping": float(model.damping.item()),
            "max_speed": float(model.max_speed.item()),
        }

    def _render_validation_gif_for_epoch(
        self,
        spec: LearnableBoidsSpec,
        model: LearnableAggregateBoids,
        epoch: int,
    ) -> None:
        if not spec.visualization.enabled or not spec.visualization.gif_enabled or not spec.evaluation.seeds:
            return

        validation_seed = spec.evaluation.seeds[0]
        render_model = self._snapshot_model_for_visualization(spec, model)
        viz_pipeline, viz_spec, highlight_idx = self._build_visualization_runtime(spec)
        val_positions0, val_velocities0 = sample_initial_boids_state(
            spec.simulation.num_nodes,
            seed=validation_seed,
            velocity_scale=spec.simulation.init_velocity_scale,
            device=spec.simulation.device,
        )

        with torch.no_grad():
            val_pred_pos_seq, val_pred_vel_seq, _ = render_model.rollout(
                spec.simulation.rounds,
                positions0=val_positions0,
                velocities0=val_velocities0,
            )

        output_dir = spec.run_dir / "validation" / f"epoch_{epoch + 1:04d}"
        output_dir.mkdir(parents=True, exist_ok=True)
        learned_params = self._current_learned_parameters(render_model)
        recovery = compute_parameter_recovery_metrics(
            extract_teacher_parameters_from_spec(spec),
            learned_params,
        )
        with (output_dir / "learned_parameters.json").open("w", encoding="utf-8") as handle:
            json.dump(
                {
                    "target": extract_teacher_parameters_from_spec(spec),
                    "learned": learned_params,
                    "recovery": recovery,
                },
                handle,
                indent=2,
            )
        viz_pipeline.render_gif_only(
            pos_seq=val_pred_pos_seq,
            vel_seq=val_pred_vel_seq,
            highlight_idx=highlight_idx,
            output_path=str(output_dir / f"validation_seed{validation_seed}_pred.gif"),
            title=f"Validation Seed {validation_seed} epoch {epoch + 1}",
            spec=viz_spec,
        )

    def _render_initial_validation_gif(
        self,
        spec: LearnableBoidsSpec,
        model: LearnableAggregateBoids,
    ) -> None:
        self._render_validation_gif_for_epoch(spec, model, -1)

    def _render_visualizations(
        self,
        ctx: RunContext,
        model: LearnableAggregateBoids,
        teacher_pos_seq: torch.Tensor,
        positions0: torch.Tensor,
        velocities0: torch.Tensor,
        checkpoint_policy: CheckpointPolicy,
        checkpoint_manager: CheckpointManager,
    ) -> None:
        spec = ctx.spec
        if not spec.visualization.enabled:
            return

        render_model = self._snapshot_model_for_visualization(spec, model)

        with torch.no_grad():
            pred_pos_seq, pred_vel_seq, _ = render_model.rollout(
                spec.simulation.rounds,
                positions0=positions0,
                velocities0=velocities0,
            )

        viz_pipeline, viz_spec, highlight_idx = self._build_visualization_runtime(spec)
        viz_pipeline.render_standard_suite(
            pos_seq=pred_pos_seq,
            vel_seq=pred_vel_seq,
            teacher_pos_seq=teacher_pos_seq,
            highlight_idx=highlight_idx,
            prefix=spec.visualization.viz_prefix,
            title_prefix="Learnable Aggregate Boids",
            spec=viz_spec,
        )

    def _save_summary(
        self,
        ctx: RunContext,
        model: LearnableAggregateBoids,
        history_data: dict[str, list[float]],
        recovery: dict[str, dict[str, float]],
    ) -> None:
        spec = ctx.spec
        summary = BoidsSummaryBuilder(
            run_name=spec.run_name,
            spec=spec,
            history=history_data,
            model=model,
            effective_curriculum_max_horizon=spec.training.max_horizon,
            recovery=recovery,
        ).build()
        with (spec.run_dir / "summary.json").open("w", encoding="utf-8") as handle:
            json.dump(summary, handle, indent=2)
        save_summary_csv(flatten_summary_for_csv(summary), spec.run_dir / "summary.csv")
        with (spec.run_dir / "run_args.json").open("w", encoding="utf-8") as handle:
            json.dump(vars(self.args), handle, indent=2)


def main() -> None:
    args = parse_learnable_args()
    LearnableBoidsWorkflow(args).run()


if __name__ == "__main__":
    main()