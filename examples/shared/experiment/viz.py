"""Orchestration of visualizations for graph-based simulations."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

from ..plotting.comparison import plot_trajectory_comparison
from ..plotting.moving import (
    export_moving_gif,
    plot_moving_snapshots,
    plot_node_trajectories,
)

if TYPE_CHECKING:
    from collections.abc import Callable


@dataclass(frozen=True)
class VizSpec:
    """Shared visual-output switches and style settings."""

    enabled: bool
    gif_enabled: bool
    compare_panel_enabled: bool
    show_links: bool
    links_alpha: float
    links_width: float
    gif_fps: int


class MovingGraphVisualizationPipeline:
    """Reusable orchestration for snapshot/trajectory/comparison/gif outputs."""

    def __init__(
        self,
        *,
        edge_builder: Callable[[torch.Tensor], torch.Tensor],
        rounds: int,
        record_every: int,
    ):
        self.edge_builder = edge_builder
        self.rounds = rounds
        self.record_every = max(1, record_every)

    def _available_rounds(self, *seqs: torch.Tensor) -> int:
        lengths = [int(seq.shape[0]) for seq in seqs if isinstance(seq, torch.Tensor)]
        if not lengths:
            return 0
        return min([self.rounds, *lengths])

    def collect_round_data(
        self,
        *,
        pos_seq: torch.Tensor,
        vel_seq: torch.Tensor,
    ) -> tuple[
        dict[int, torch.Tensor], dict[int, torch.Tensor], dict[int, torch.Tensor]
    ]:
        """Sample trajectory data at regular intervals for plotting."""
        available_rounds = self._available_rounds(pos_seq, vel_seq)
        if available_rounds <= 0:
            return {}, {}, {}

        round_indices = set(range(0, available_rounds, self.record_every)) | {
            available_rounds - 1
        }
        positions_by_round: dict[int, torch.Tensor] = {}
        values_by_round: dict[int, torch.Tensor] = {}
        edge_index_by_round: dict[int, torch.Tensor] = {}
        for round_idx in sorted(round_indices):
            positions_by_round[round_idx] = pos_seq[round_idx].detach().clone()
            values_by_round[round_idx] = vel_seq[round_idx].norm(dim=1).detach().clone()
            edge_index_by_round[round_idx] = self.edge_builder(
                positions_by_round[round_idx]
            )
        return positions_by_round, values_by_round, edge_index_by_round

    def render_standard_suite(
        self,
        *,
        pos_seq: torch.Tensor,
        vel_seq: torch.Tensor,
        teacher_pos_seq: torch.Tensor,
        highlight_idx: int,
        prefix: str,
        title_prefix: str,
        spec: VizSpec,
    ) -> None:
        """Render snapshots, trajectories, comparisons, and GIFs."""
        positions_by_round, values_by_round, edge_index_by_round = (
            self.collect_round_data(pos_seq=pos_seq, vel_seq=vel_seq)
        )
        available_rounds = self._available_rounds(pos_seq, vel_seq, teacher_pos_seq)
        if available_rounds <= 0:
            return

        plot_moving_snapshots(
            positions_by_round=positions_by_round,
            values_by_round=values_by_round,
            source_idx=highlight_idx,
            output_path=f"{prefix}_pred_snapshots.png",
            edge_index_by_round=edge_index_by_round,
            show_links=spec.show_links,
            links_alpha=spec.links_alpha,
            links_width=spec.links_width,
        )
        plot_node_trajectories(
            positions_over_time=[pos_seq[idx] for idx in range(available_rounds)],
            source_idx=highlight_idx,
            output_path=f"{prefix}_pred_trajectories.png",
        )
        plot_node_trajectories(
            positions_over_time=[
                teacher_pos_seq[idx] for idx in range(available_rounds)
            ],
            source_idx=highlight_idx,
            output_path=f"{prefix}_teacher_trajectories.png",
        )
        if spec.compare_panel_enabled:
            plot_trajectory_comparison(
                predicted_positions_over_time=[
                    pos_seq[idx] for idx in range(available_rounds)
                ],
                teacher_positions_over_time=[
                    teacher_pos_seq[idx] for idx in range(available_rounds)
                ],
                source_idx=highlight_idx,
                output_path=f"{prefix}_compare_panel.png",
            )
        if spec.gif_enabled:
            export_moving_gif(
                positions_by_round=positions_by_round,
                values_by_round=values_by_round,
                source_idx=highlight_idx,
                output_path=f"{prefix}_pred.gif",
                title=f"{title_prefix} predicted speed",
                fps=max(1, spec.gif_fps),
                edge_index_by_round=edge_index_by_round,
                show_links=spec.show_links,
                links_alpha=spec.links_alpha,
                links_width=spec.links_width,
            )

    def render_gif_only(
        self,
        *,
        pos_seq: torch.Tensor,
        vel_seq: torch.Tensor,
        highlight_idx: int,
        output_path: str,
        title: str,
        spec: VizSpec,
    ) -> None:
        """Render only a GIF of the simulation."""
        if not spec.gif_enabled:
            return

        positions_by_round, values_by_round, edge_index_by_round = (
            self.collect_round_data(pos_seq=pos_seq, vel_seq=vel_seq)
        )
        if not positions_by_round:
            return

        export_moving_gif(
            positions_by_round=positions_by_round,
            values_by_round=values_by_round,
            source_idx=highlight_idx,
            output_path=output_path,
            title=title,
            fps=max(1, spec.gif_fps),
            edge_index_by_round=edge_index_by_round,
            show_links=spec.show_links,
            links_alpha=spec.links_alpha,
            links_width=spec.links_width,
        )

    def render_checkpoint_suite(
        self,
        *,
        pos_seq: torch.Tensor,
        vel_seq: torch.Tensor,
        teacher_pos_seq: torch.Tensor,
        highlight_idx: int,
        output_prefix: str,
        spec: VizSpec,
    ) -> None:
        """Render a standard visualization suite for a specific checkpoint epoch."""
        positions_by_round, values_by_round, edge_index_by_round = (
            self.collect_round_data(pos_seq=pos_seq, vel_seq=vel_seq)
        )
        available_rounds = self._available_rounds(pos_seq, vel_seq, teacher_pos_seq)
        if available_rounds <= 0:
            return

        plot_moving_snapshots(
            positions_by_round=positions_by_round,
            values_by_round=values_by_round,
            source_idx=highlight_idx,
            output_path=f"{output_prefix}_snapshots.png",
            edge_index_by_round=edge_index_by_round,
            show_links=spec.show_links,
            links_alpha=spec.links_alpha,
            links_width=spec.links_width,
        )
        plot_node_trajectories(
            positions_over_time=[pos_seq[idx] for idx in range(available_rounds)],
            source_idx=highlight_idx,
            output_path=f"{output_prefix}_trajectories.png",
        )
        if spec.compare_panel_enabled:
            plot_trajectory_comparison(
                predicted_positions_over_time=[
                    pos_seq[idx] for idx in range(available_rounds)
                ],
                teacher_positions_over_time=[
                    teacher_pos_seq[idx] for idx in range(available_rounds)
                ],
                source_idx=highlight_idx,
                output_path=f"{output_prefix}_compare.png",
            )
