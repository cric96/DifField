#!/usr/bin/env python3
"""Aggregate boids with learnable weights and neighbor attention."""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "examples"))

from aggregate_gnn import build_spatial_graph
from boids.cli import HISTORY_KEYS, parse_learnable_args
from boids.config import LearnableBoidsSpec, build_learnable_spec
from boids.evaluation_utils import evaluate_seed
from boids.model import LearnableAggregateBoids, emergent_regularizers, teacher_rollout, teacher_rollout_from_specs
from boids.reporting import BoidsSummaryBuilder, compute_parameter_recovery_metrics, extract_learned_parameters, extract_teacher_parameters
from shared.diagnostics import export_diagnostics, save_history_csv, save_summary_csv
from shared.experiment import CheckpointManager, CheckpointPolicy, MovingGraphVisualizationPipeline, VizSpec, flatten_summary_for_csv
from shared.history import MetricHistory
from shared.training import grad_norm


@dataclass(frozen=True)
class RunContext:
    spec: LearnableBoidsSpec


class LearnableBoidsWorkflow:
    """Template-method style orchestrator for the learnable boids script."""

    def __init__(self, args: Any):
        self.args = args

    def run(self) -> None:
        torch.manual_seed(self.args.seed)
        ctx = self._build_context()
        model, history_data, teacher_pos_seq, checkpoint_policy, checkpoint_manager = self._train(ctx)
        recovery = self._save_training_outputs(ctx, model, history_data)
        self._render_visualizations(ctx, model, teacher_pos_seq, checkpoint_policy, checkpoint_manager)
        self._save_summary(ctx, model, history_data, recovery)

    def _build_context(self) -> RunContext:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_name = self.args.run_name.strip() or f"boids_{self.args.mode}_seed{self.args.seed}_{timestamp}"
        run_dir = Path(self.args.out_dir) / run_name
        run_dir.mkdir(parents=True, exist_ok=True)
        viz_prefix = str(run_dir / "learnable") if self.args.viz_prefix == "generated/boids/learnable" else self.args.viz_prefix
        return RunContext(spec=build_learnable_spec(self.args, run_name=run_name, run_dir=run_dir, viz_prefix=viz_prefix))

    def _train(
        self,
        ctx: RunContext,
    ) -> tuple[LearnableAggregateBoids, dict[str, list[float]], torch.Tensor, CheckpointPolicy, CheckpointManager]:
        spec = ctx.spec
        device = spec.simulation.device
        positions0 = torch.rand(spec.simulation.num_nodes, 2, device=device)
        teacher_pos_seq, _ = teacher_rollout_from_specs(
            positions0=positions0,
            simulation=spec.simulation,
            teacher=spec.teacher,
        )
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

        print("=== Aggregate Learnable Boids ===")
        print(f"mode={spec.model.mode} nodes={spec.simulation.num_nodes} rounds={spec.simulation.rounds}")
        print(
            f"teacher: w_sep={spec.teacher.w_sep:.2f} w_align={spec.teacher.w_align:.2f} "
            f"w_cohesion={spec.teacher.w_cohesion:.2f} damping={spec.teacher.damping:.3f} max_speed={spec.teacher.max_speed:.4f}"
        )
        print(f"run_dir={spec.run_dir}")

        for epoch in range(spec.training.epochs):
            optimizer.zero_grad()
            pred_pos_seq, pred_vel_seq, final_pos = model.rollout(spec.simulation.rounds)
            traj_loss = nn.functional.mse_loss(pred_pos_seq, teacher_pos_seq)
            reg_loss = emergent_regularizers(
                pred_pos_seq,
                pred_vel_seq,
                cohesion_weight=spec.training.cohesion_reg,
                alignment_weight=spec.training.alignment_reg,
                speed_weight=spec.training.speed_reg,
                accel_weight=spec.training.accel_reg,
            )
            total = spec.training.traj_w * traj_loss + reg_loss
            total.backward()
            current_grad_norm = grad_norm(params)
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            optimizer.step()

            center_error = (teacher_pos_seq[-1].mean(dim=0) - final_pos.mean(dim=0)).norm().item()
            val_traj_loss, val_center_error = self._evaluate_epoch(model, spec, epoch)

            history.append(
                epoch=float(epoch + 1),
                total=float(total.item()),
                traj_loss=float(traj_loss.item()),
                reg_loss=float(reg_loss.item()),
                center_error=float(center_error),
                w_sep=float(model.w_sep.item()),
                w_align=float(model.w_align.item()),
                w_cohesion=float(model.w_cohesion.item()),
                damping=float(model.damping.item()),
                max_speed=float(model.max_speed.item()),
                tau_align=float(model.align_aggr.tau.item()),
                tau_cohesion=float(model.cohesion_aggr.tau.item()),
                grad_norm=float(current_grad_norm),
                cap_fraction=float(model.last_rollout_speed_health["mean_cap_fraction"]),
                pre_clip_speed=float(model.last_rollout_speed_health["mean_pre_clip_speed"]),
                val_traj_loss=float(val_traj_loss),
                val_center_error=float(val_center_error),
            )

            if checkpoint_manager.should_save(epoch):
                checkpoint_manager.save_rollout(epoch, pred_pos_seq=pred_pos_seq, pred_vel_seq=pred_vel_seq)

            if (epoch + 1) % spec.training.print_every == 0 or epoch == 0:
                do_eval = val_traj_loss == val_traj_loss
                val_msg = f" val_traj={val_traj_loss:.6f} val_center={val_center_error:.6f}" if do_eval else ""
                print(
                    f"epoch={epoch + 1:3d} total={total.item():.6f} traj={traj_loss.item():.6f} "
                    f"w_sep={model.w_sep.item():.3f} w_align={model.w_align.item():.3f} "
                    f"w_coh={model.w_cohesion.item():.3f} damp={model.damping.item():.3f} "
                    f"tau_align={model.align_aggr.tau.item():.3f} grad={current_grad_norm:.5f}{val_msg}"
                )

        print(
            f"init_connectivity={spec.model.init_connectivity} init_components={int(model.last_init_graph_stats.get('num_components', 0.0))} "
            f"init_min_degree={model.last_init_graph_stats.get('min_degree', float('nan')):.0f} "
            f"init_edges={int(model.last_init_graph_stats.get('num_edges', 0.0))}"
        )
        print(
            f"rollout_mean_edges={model.last_rollout_graph_health.get('mean_num_edges', float('nan')):.2f} "
            f"rollout_mean_min_degree={model.last_rollout_graph_health.get('mean_min_degree', float('nan')):.2f} "
            f"rollout_max_components={model.last_rollout_graph_health.get('max_num_components', float('nan')):.0f}"
        )
        print(
            f"rollout_mean_pre_clip_speed={model.last_rollout_speed_health.get('mean_pre_clip_speed', float('nan')):.5f} "
            f"rollout_mean_cap_fraction={model.last_rollout_speed_health.get('mean_cap_fraction', float('nan')):.5f}"
        )
        print("Training complete.")

        return model, history.to_dict(), teacher_pos_seq, checkpoint_policy, checkpoint_manager

    def _evaluate_epoch(self, model: LearnableAggregateBoids, spec: LearnableBoidsSpec, epoch: int) -> tuple[float, float]:
        eval_cfg = spec.evaluation
        do_eval = bool(eval_cfg.seeds) and (
            (epoch + 1) % max(1, eval_cfg.every) == 0 or epoch == 0 or epoch == spec.training.epochs - 1
        )
        if not do_eval:
            return float("nan"), float("nan")
        val_pairs = [
            evaluate_seed(
                model,
                seed=eval_seed,
                simulation=spec.simulation,
                teacher=spec.teacher,
            )
            for eval_seed in eval_cfg.seeds
        ]
        val_traj_loss = float(sum(item[0] for item in val_pairs) / max(1, len(val_pairs)))
        val_center_error = float(sum(item[1] for item in val_pairs) / max(1, len(val_pairs)))
        return val_traj_loss, val_center_error

    def _save_training_outputs(
        self,
        ctx: RunContext,
        model: LearnableAggregateBoids,
        history_data: dict[str, list[float]],
    ) -> dict[str, dict[str, float]]:
        spec = ctx.spec
        teacher_params = extract_teacher_parameters(self.args)
        learned_params = {
            key: value
            for key, value in extract_learned_parameters(history_data).items()
            if key in teacher_params
        }
        recovery = compute_parameter_recovery_metrics(teacher_params, learned_params)

        print("--- Parameter Recovery ---")
        for name, item in recovery.items():
            print(
                f"  {name}: teacher={item['teacher']:.4f} learned={item['learned']:.4f} "
                f"abs_err={item['abs_error']:.4f} rel_err={item['rel_error']:.2%}"
            )

        save_history_csv(history_data, spec.run_dir / "history.csv")
        export_diagnostics(
            history_data,
            spec.run_dir / "diagnostics",
            title_prefix=f"{spec.run_name} ",
            teacher_params=teacher_params,
            learned_params=learned_params,
        )
        return recovery

    def _render_visualizations(
        self,
        ctx: RunContext,
        model: LearnableAggregateBoids,
        teacher_pos_seq: torch.Tensor,
        checkpoint_policy: CheckpointPolicy,
        checkpoint_manager: CheckpointManager,
    ) -> None:
        spec = ctx.spec
        if not spec.visualization.enabled:
            return

        with torch.no_grad():
            pred_pos_seq, pred_vel_seq, _ = model.rollout(spec.simulation.rounds)

        highlight_idx = int(max(0, min(spec.visualization.highlight_node, spec.simulation.num_nodes - 1)))
        viz_spec = VizSpec(
            enabled=spec.visualization.enabled,
            gif_enabled=spec.visualization.gif_enabled,
            compare_panel_enabled=spec.visualization.compare_panel_enabled,
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
        viz_pipeline.render_standard_suite(
            pos_seq=pred_pos_seq,
            vel_seq=pred_vel_seq,
            teacher_pos_seq=teacher_pos_seq,
            highlight_idx=highlight_idx,
            prefix=spec.visualization.viz_prefix,
            title_prefix="Learnable Aggregate Boids",
            spec=viz_spec,
        )

        for anchor in checkpoint_policy.anchors:
            payload = checkpoint_manager.load_rollout(anchor)
            if payload is None:
                continue
            panel_prefix = str(checkpoint_manager.checkpoint_dir(anchor) / "mid_training")
            viz_pipeline.render_checkpoint_suite(
                pos_seq=payload["pred_pos_seq"],
                vel_seq=payload["pred_vel_seq"],
                teacher_pos_seq=teacher_pos_seq,
                highlight_idx=highlight_idx,
                output_prefix=panel_prefix,
                epoch_number=anchor + 1,
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
            args=self.args,
            history=history_data,
            model=model,
            train_max_speed=spec.model.train_max_speed,
            init_max_speed_target=spec.model.init_max_speed_target,
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