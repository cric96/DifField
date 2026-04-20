"""Orchestration of the boids training loop."""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING

import torch

from ..domain.geometry import sample_initial_state
from ..model.boids_model import LearnableAggregateBoids
from ..reporting.recovery import (
    compute_parameter_recovery_metrics,
    extract_teacher_parameters_from_spec,
)

try:
    from ..shared import MetricHistory, grad_norm
except ImportError:
    from shared import MetricHistory, grad_norm
from .curriculum import curriculum_horizon, scheduled_learning_rate
from .supervision import teacher_forced_step_losses
from .trace import build_supervision_traces

if TYPE_CHECKING:
    from ..domain.specs import LearnableBoidsSpec


class BoidsTrainingLoop:
    """Manages the optimization process for learnable boids parameters."""

    def __init__(self, spec: LearnableBoidsSpec):
        self.spec = spec

    def run(
        self, on_epoch_end=None
    ) -> tuple[LearnableAggregateBoids, dict[str, list[float]]]:
        """Execute the full training process."""
        spec = self.spec
        device = spec.simulation.device

        # 1. Prepare initial conditions and supervision traces
        initial_conditions = self._make_initial_conditions()
        supervision_traces = build_supervision_traces(initial_conditions, spec=spec)

        # We track progress primarily on the first trace
        positions0 = supervision_traces[0].positions0
        velocities0 = supervision_traces[0].velocities0
        teacher_pos_seq = supervision_traces[0].pos_seq

        # 2. Initialize model and optimizer
        model = LearnableAggregateBoids.from_specs(
            positions0=positions0,
            simulation=spec.simulation,
            model=spec.model,
        ).to(device)

        optimizer = torch.optim.Adam(model.trainable_parameters(), lr=spec.training.lr)

        # 3. Setup history and logging
        from ..cli import HISTORY_KEYS

        history = MetricHistory.from_keys(HISTORY_KEYS)
        teacher_params = extract_teacher_parameters_from_spec(spec)

        # 4. Training loop
        for epoch in range(spec.training.epochs):
            # Update learning rate
            current_lr = scheduled_learning_rate(
                spec.training.lr,
                epoch,
                spec.training.epochs,
                ramp_fraction=spec.training.curriculum_ramp_fraction,
                final_lr_ratio=spec.training.final_lr_ratio,
            )
            for group in optimizer.param_groups:
                group["lr"] = current_lr

            # Compute current horizon
            horizon = curriculum_horizon(
                epoch,
                spec.training.epochs,
                spec.training.min_horizon,
                spec.training.max_horizon,
                ramp_fraction=spec.training.curriculum_ramp_fraction,
            )

            # Optimization step
            optimizer.zero_grad()
            epoch_losses = self._compute_epoch_losses(
                model, supervision_traces, horizon
            )

            # Backward and step
            epoch_losses["objective"].backward()
            current_grad_norm = grad_norm(model.trainable_parameters())
            torch.nn.utils.clip_grad_norm_(model.trainable_parameters(), 5.0)
            optimizer.step()

            # Monitor performance (rollout from init conditions)
            monitor_results = self._monitor_epoch(
                model, positions0, velocities0, teacher_pos_seq
            )

            # Evaluation (if needed)
            eval_metrics = self._evaluate_epoch(model, epoch, horizon)

            # Record history
            self._record_history(
                history,
                epoch,
                horizon,
                current_lr,
                current_grad_norm,
                epoch_losses,
                monitor_results,
                eval_metrics,
                model,
                teacher_params,
            )

            # Log progress
            self._log_epoch(
                epoch,
                epoch_losses,
                monitor_results,
                eval_metrics,
                teacher_params,
                model,
            )

            # Optional callback for rendering/checkpointing
            if on_epoch_end:
                on_epoch_end(epoch, model, monitor_results)

        return model, history.to_dict()

    def _make_initial_conditions(self) -> list[tuple[torch.Tensor, torch.Tensor]]:
        spec = self.spec
        return [
            sample_initial_state(
                spec.simulation.num_nodes,
                seed=spec.seed + idx * 1337,
                velocity_scale=spec.simulation.init_velocity_scale,
                device=spec.simulation.device,
            )
            for idx in range(spec.training.num_initial_conditions)
        ]

    def _compute_epoch_losses(
        self, model: LearnableAggregateBoids, traces: list, horizon: int
    ) -> dict[str, torch.Tensor]:
        total_loss = torch.zeros((), device=self.spec.simulation.device)
        total_pos_loss = torch.zeros((), device=self.spec.simulation.device)
        total_vel_loss = torch.zeros((), device=self.spec.simulation.device)
        total_sep_focus_loss = torch.zeros((), device=self.spec.simulation.device)

        for trace in traces:
            loss, pos_loss, vel_loss, sep_focus_loss = teacher_forced_step_losses(
                model,
                trace=trace.sliced(horizon),
                velocity_loss_weight=self.spec.training.velocity_loss_weight,
                sep=self.spec.simulation.sep,
            )
            total_loss = total_loss + loss
            total_pos_loss = total_pos_loss + pos_loss
            total_vel_loss = total_vel_loss + vel_loss
            total_sep_focus_loss = total_sep_focus_loss + sep_focus_loss

        num_traces = float(len(traces))
        avg_total = total_loss / num_traces
        avg_sep = total_sep_focus_loss / num_traces
        objective = avg_total + self.spec.training.separation_loss_weight * avg_sep

        return {
            "total": avg_total,
            "pos": total_pos_loss / num_traces,
            "vel": total_vel_loss / num_traces,
            "sep_focus": avg_sep,
            "objective": objective,
        }

    def _monitor_epoch(
        self, model, pos0, vel0, teacher_pos_seq
    ) -> dict[str, float | torch.Tensor]:
        with torch.no_grad():
            monitor_pos_seq, monitor_vel_seq, monitor_final_pos = model.rollout(
                self.spec.simulation.rounds,
                positions0=pos0,
                velocities0=vel0,
                trunc_window=self.spec.training.trunc_window,
            )
        center_error = (
            (teacher_pos_seq[-1].mean(dim=0) - monitor_final_pos.mean(dim=0))
            .norm()
            .item()
        )
        return {
            "center_error": float(center_error),
            "pos_seq": monitor_pos_seq,
            "vel_seq": monitor_vel_seq,
            "teacher_pos_seq": teacher_pos_seq,
            "positions0": pos0,
            "velocities0": vel0,
        }

    def _evaluate_epoch(self, model, epoch, horizon) -> dict[str, float]:
        from ..evaluation.evaluator import evaluate_seed, mean_eval_metrics

        spec = self.spec
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
                    "curriculum_horizon",
                    "curriculum_total_loss",
                    "curriculum_pos_loss",
                    "curriculum_vel_loss",
                    "curriculum_per_step_loss",
                    "curriculum_center_error",
                    "full_horizon",
                    "full_total_loss",
                    "full_pos_loss",
                    "full_vel_loss",
                    "full_per_step_loss",
                    "full_center_error",
                ]
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
        curriculum_metrics = mean_eval_metrics(curriculum_pairs)

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

    def _record_history(
        self,
        history,
        epoch,
        horizon,
        lr,
        grad_norm_val,
        losses,
        monitor,
        evals,
        model,
        teacher_params,
    ):
        learned_params = {
            "w_sep": float(model.w_sep.item()),
            "w_align": float(model.w_align.item()),
            "w_cohesion": float(model.w_cohesion.item()),
        }
        recovery = compute_parameter_recovery_metrics(teacher_params, learned_params)

        history.append(
            epoch=float(epoch + 1),
            horizon=float(horizon),
            total=float(losses["total"].item()),
            per_step_loss=float(losses["total"].item()),
            objective_loss=float(losses["objective"].item()),
            pos_loss=float(losses["pos"].item()),
            vel_loss=float(losses["vel"].item()),
            sep_focus_loss=float(losses["sep_focus"].item()),
            center_error=monitor["center_error"],
            w_sep=learned_params["w_sep"],
            w_sep_abs_error=float(recovery["w_sep"]["abs_error"]),
            w_sep_rel_error=float(recovery["w_sep"]["rel_error"]),
            w_align=learned_params["w_align"],
            w_align_abs_error=float(recovery["w_align"]["abs_error"]),
            w_align_rel_error=float(recovery["w_align"]["rel_error"]),
            w_cohesion=learned_params["w_cohesion"],
            w_cohesion_abs_error=float(recovery["w_cohesion"]["abs_error"]),
            w_cohesion_rel_error=float(recovery["w_cohesion"]["rel_error"]),
            grad_norm=float(grad_norm_val),
            lr=float(lr),
            cap_fraction=float(
                model.last_rollout_speed_health.get("mean_cap_fraction", 0.0)
            ),
            pre_clip_speed=float(
                model.last_rollout_speed_health.get("mean_pre_clip_speed", 0.0)
            ),
            val_curriculum_horizon=float(evals["curriculum_horizon"]),
            val_curriculum_total_loss=float(evals["curriculum_total_loss"]),
            val_curriculum_pos_loss=float(evals["curriculum_pos_loss"]),
            val_curriculum_vel_loss=float(evals["curriculum_vel_loss"]),
            val_curriculum_per_step_loss=float(evals["curriculum_per_step_loss"]),
            val_curriculum_center_error=float(evals["curriculum_center_error"]),
            val_full_horizon=float(evals["full_horizon"]),
            val_full_total_loss=float(evals["full_total_loss"]),
            val_full_pos_loss=float(evals["full_pos_loss"]),
            val_full_vel_loss=float(evals["full_vel_loss"]),
            val_full_per_step_loss=float(evals["full_per_step_loss"]),
            val_full_center_error=float(evals["full_center_error"]),
        )

    def _log_epoch(self, epoch, losses, monitor, evals, teacher_params, model):
        if (epoch + 1) % self.spec.training.print_every == 0 or epoch == 0:
            learned_params = {
                "w_sep": float(model.w_sep.item()),
                "w_align": float(model.w_align.item()),
                "w_cohesion": float(model.w_cohesion.item()),
            }
            recovery = compute_parameter_recovery_metrics(
                teacher_params, learned_params
            )

            message = (
                f"epoch={epoch + 1:3d} train_step={losses['total'].item():.6f} center={monitor['center_error']:.6f} "
                f"err_sep={recovery['w_sep']['rel_error']:.1%} "
                f"err_align={recovery['w_align']['rel_error']:.1%} "
                f"err_coh={recovery['w_cohesion']['rel_error']:.1%}"
            )
            if not torch.isnan(torch.tensor(evals["full_per_step_loss"])):
                message += (
                    f" val_step={evals['full_per_step_loss']:.6f}"
                    f" val_center={evals['full_center_error']:.6f}"
                )
            print(message)
