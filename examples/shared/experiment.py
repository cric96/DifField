"""High-level experiment orchestration helpers for examples."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import torch

from .plotting import export_moving_gif, plot_moving_snapshots, plot_node_trajectories, plot_trajectory_comparison


@dataclass(frozen=True)
class CheckpointPolicy:
    """Declarative epoch selection for checkpoint writes."""

    total_epochs: int
    every_epochs: int

    @property
    def anchors(self) -> list[int]:
        if self.total_epochs <= 0:
            return []
        return sorted({0, max(0, self.total_epochs // 2), self.total_epochs - 1})

    @property
    def epoch_indices(self) -> set[int]:
        if self.total_epochs <= 0:
            return set()
        periodic = set(range(0, self.total_epochs, max(1, self.every_epochs)))
        return periodic | set(self.anchors)


class CheckpointManager:
    """Single-responsibility manager for rollout checkpoint I/O."""

    def __init__(self, root_dir: Path, policy: CheckpointPolicy):
        self.root_dir = root_dir
        self.policy = policy
        self.root_dir.mkdir(parents=True, exist_ok=True)

    def should_save(self, epoch_idx: int) -> bool:
        return epoch_idx in self.policy.epoch_indices

    def checkpoint_dir(self, epoch_idx: int) -> Path:
        return self.root_dir / f"epoch_{epoch_idx + 1:04d}"

    def rollout_path(self, epoch_idx: int) -> Path:
        return self.checkpoint_dir(epoch_idx) / "rollout.pt"

    def save_rollout(self, epoch_idx: int, pred_pos_seq: torch.Tensor, pred_vel_seq: torch.Tensor) -> None:
        ckpt_dir = self.checkpoint_dir(epoch_idx)
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "epoch": epoch_idx + 1,
                "pred_pos_seq": pred_pos_seq.detach().cpu(),
                "pred_vel_seq": pred_vel_seq.detach().cpu(),
            },
            ckpt_dir / "rollout.pt",
        )

    def load_rollout(self, epoch_idx: int) -> dict[str, torch.Tensor] | None:
        path = self.rollout_path(epoch_idx)
        if not path.exists():
            return None
        return torch.load(path, map_location="cpu")


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

    def collect_round_data(
        self,
        *,
        pos_seq: torch.Tensor,
        vel_seq: torch.Tensor,
    ) -> tuple[dict[int, torch.Tensor], dict[int, torch.Tensor], dict[int, torch.Tensor]]:
        round_indices = set(range(0, self.rounds, self.record_every)) | {self.rounds - 1}
        positions_by_round: dict[int, torch.Tensor] = {}
        values_by_round: dict[int, torch.Tensor] = {}
        edge_index_by_round: dict[int, torch.Tensor] = {}
        for round_idx in sorted(round_indices):
            positions_by_round[round_idx] = pos_seq[round_idx].detach().clone()
            values_by_round[round_idx] = vel_seq[round_idx].norm(dim=1).detach().clone()
            edge_index_by_round[round_idx] = self.edge_builder(positions_by_round[round_idx])
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
        positions_by_round, values_by_round, edge_index_by_round = self.collect_round_data(pos_seq=pos_seq, vel_seq=vel_seq)

        plot_moving_snapshots(
            positions_by_round=positions_by_round,
            values_by_round=values_by_round,
            source_idx=highlight_idx,
            output_path=f"{prefix}_pred_snapshots.png",
            title=f"{title_prefix} predicted speed snapshots",
            edge_index_by_round=edge_index_by_round,
            show_links=spec.show_links,
            links_alpha=spec.links_alpha,
            links_width=spec.links_width,
        )
        plot_node_trajectories(
            positions_over_time=[pos_seq[idx] for idx in range(self.rounds)],
            source_idx=highlight_idx,
            output_path=f"{prefix}_pred_trajectories.png",
            title=f"{title_prefix} predicted trajectories",
        )
        plot_node_trajectories(
            positions_over_time=[teacher_pos_seq[idx] for idx in range(self.rounds)],
            source_idx=highlight_idx,
            output_path=f"{prefix}_teacher_trajectories.png",
            title=f"{title_prefix} teacher trajectories",
        )
        if spec.compare_panel_enabled:
            plot_trajectory_comparison(
                predicted_positions_over_time=[pos_seq[idx] for idx in range(self.rounds)],
                teacher_positions_over_time=[teacher_pos_seq[idx] for idx in range(self.rounds)],
                source_idx=highlight_idx,
                output_path=f"{prefix}_compare_panel.png",
                title=f"{title_prefix} predicted vs teacher",
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

    def render_checkpoint_suite(
        self,
        *,
        pos_seq: torch.Tensor,
        vel_seq: torch.Tensor,
        teacher_pos_seq: torch.Tensor,
        highlight_idx: int,
        output_prefix: str,
        epoch_number: int,
        spec: VizSpec,
    ) -> None:
        positions_by_round, values_by_round, edge_index_by_round = self.collect_round_data(pos_seq=pos_seq, vel_seq=vel_seq)

        plot_moving_snapshots(
            positions_by_round=positions_by_round,
            values_by_round=values_by_round,
            source_idx=highlight_idx,
            output_path=f"{output_prefix}_snapshots.png",
            title=f"Mid-training epoch {epoch_number}: speed snapshots",
            edge_index_by_round=edge_index_by_round,
            show_links=spec.show_links,
            links_alpha=spec.links_alpha,
            links_width=spec.links_width,
        )
        plot_node_trajectories(
            positions_over_time=[pos_seq[idx] for idx in range(self.rounds)],
            source_idx=highlight_idx,
            output_path=f"{output_prefix}_trajectories.png",
            title=f"Mid-training epoch {epoch_number}: trajectories",
        )
        if spec.compare_panel_enabled:
            plot_trajectory_comparison(
                predicted_positions_over_time=[pos_seq[idx] for idx in range(self.rounds)],
                teacher_positions_over_time=[teacher_pos_seq[idx] for idx in range(self.rounds)],
                source_idx=highlight_idx,
                output_path=f"{output_prefix}_compare.png",
                title=f"Mid-training epoch {epoch_number}: predicted vs teacher",
            )


def flatten_summary_for_csv(payload: dict[str, object], prefix: str = "") -> dict[str, float | int | str | bool]:
    """Flatten nested summary payloads into scalar CSV columns."""
    flat: dict[str, float | int | str | bool] = {}
    for key, value in payload.items():
        full_key = f"{prefix}.{key}" if prefix else key
        if isinstance(value, dict):
            flat.update(flatten_summary_for_csv(value, full_key))
        elif isinstance(value, (str, bool, int, float)):
            flat[full_key] = value
        else:
            flat[full_key] = json.dumps(value, sort_keys=True)
    return flat