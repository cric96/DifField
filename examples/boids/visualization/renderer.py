"""Rendering utilities for boids experiments: snapshots, trajectories, and GIFs."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

import torch
from autofield import build_spatial_graph

from ..domain.geometry import sample_initial_state
from ..model.boids_model import LearnableAggregateBoids
from ..reporting.recovery import (
    compute_parameter_recovery_metrics,
    extract_teacher_parameters_from_spec,
)

try:
    from ...shared.experiment import MovingGraphVisualizationPipeline, VizSpec
except ImportError:
    from shared.experiment import MovingGraphVisualizationPipeline, VizSpec

if TYPE_CHECKING:
    from ..domain.specs import LearnableBoidsSpec


class BoidsRenderer:
    """Orchestrates visual output generation for boids training and evaluation."""

    def __init__(self, spec: LearnableBoidsSpec):
        self.spec = spec
        self.highlight_idx = int(
            max(
                0, min(spec.visualization.highlight_node, spec.simulation.num_nodes - 1)
            )
        )

        def edge_builder(positions: torch.Tensor) -> torch.Tensor:
            edge_index, _ = build_spatial_graph(
                positions,
                edge_radius=spec.simulation.radius
                if spec.model.init_connectivity in {"radius", "hybrid"}
                else None,
                k_neighbors=spec.model.init_k_neighbors
                if spec.model.init_connectivity == "knn"
                else None,
            )
            return edge_index

        self.viz_pipeline = MovingGraphVisualizationPipeline(
            edge_builder=edge_builder,
            rounds=spec.simulation.rounds,
            record_every=spec.visualization.record_every,
        )
        self.viz_spec = VizSpec(
            enabled=spec.visualization.enabled,
            gif_enabled=spec.visualization.gif_enabled,
            compare_panel_enabled=False,
            show_links=spec.visualization.show_links,
            links_alpha=spec.visualization.links_alpha,
            links_width=spec.visualization.links_width,
            gif_fps=spec.visualization.gif_fps,
        )

    def render_training_suite(
        self,
        model: LearnableAggregateBoids,
        teacher_pos_seq: torch.Tensor,
        positions0: torch.Tensor,
        velocities0: torch.Tensor,
    ) -> None:
        """Generate snapshots, trajectories, and GIFs for a model after training."""
        if not self.viz_spec.enabled:
            return

        render_model = self._snapshot_model(model)
        with torch.no_grad():
            pred_pos_seq, pred_vel_seq, _ = render_model.rollout(
                self.spec.simulation.rounds,
                positions0=positions0,
                velocities0=velocities0,
            )

        self.viz_pipeline.render_standard_suite(
            pos_seq=pred_pos_seq,
            vel_seq=pred_vel_seq,
            teacher_pos_seq=teacher_pos_seq,
            highlight_idx=self.highlight_idx,
            prefix=self.spec.visualization.viz_prefix,
            title_prefix="Learnable Aggregate Boids",
            spec=self.viz_spec,
        )

    def render_validation_checkpoint(
        self, model: LearnableAggregateBoids, epoch: int
    ) -> None:
        """Generate a validation GIF and parameter JSON for a mid-training epoch."""
        if (
            not self.viz_spec.enabled
            or not self.viz_spec.gif_enabled
            or not self.spec.evaluation.seeds
        ):
            return

        validation_seed = self.spec.evaluation.seeds[0]
        render_model = self._snapshot_model(model)
        val_positions0, val_velocities0 = sample_initial_state(
            self.spec.simulation.num_nodes,
            seed=validation_seed,
            velocity_scale=self.spec.simulation.init_velocity_scale,
            device=self.spec.simulation.device,
        )

        with torch.no_grad():
            val_pred_pos_seq, val_pred_vel_seq, _ = render_model.rollout(
                self.spec.simulation.rounds,
                positions0=val_positions0,
                velocities0=val_velocities0,
            )

        output_dir = self.spec.run_dir / "validation" / f"epoch_{epoch + 1:04d}"
        output_dir.mkdir(parents=True, exist_ok=True)

        learned_params = self._extract_params(render_model)
        recovery = compute_parameter_recovery_metrics(
            extract_teacher_parameters_from_spec(self.spec),
            learned_params,
        )

        with (output_dir / "learned_parameters.json").open(
            "w", encoding="utf-8"
        ) as handle:
            json.dump(
                {
                    "target": extract_teacher_parameters_from_spec(self.spec),
                    "learned": learned_params,
                    "recovery": recovery,
                },
                handle,
                indent=2,
            )

        self.viz_pipeline.render_gif_only(
            pos_seq=val_pred_pos_seq,
            vel_seq=val_pred_vel_seq,
            highlight_idx=self.highlight_idx,
            output_path=str(output_dir / f"validation_seed{validation_seed}_pred.gif"),
            title=f"Validation Seed {validation_seed} epoch {epoch + 1}",
            spec=self.viz_spec,
        )

    def _snapshot_model(
        self, model: LearnableAggregateBoids
    ) -> LearnableAggregateBoids:
        snapshot = LearnableAggregateBoids.from_specs(
            positions0=model.positions0.detach().clone(),
            simulation=self.spec.simulation,
            model=self.spec.model,
        ).to(self.spec.simulation.device)
        snapshot.load_state_dict(model.state_dict())
        snapshot.eval()
        return snapshot

    def _extract_params(self, model: LearnableAggregateBoids) -> dict[str, float]:
        return {
            "w_sep": float(model.w_sep.item()),
            "w_align": float(model.w_align.item()),
            "w_cohesion": float(model.w_cohesion.item()),
        }
