"""Logic for managing experiment checkpoints and rollout persistence."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from pathlib import Path


@dataclass(frozen=True)
class CheckpointPolicy:
    """Declarative epoch selection for checkpoint writes."""

    total_epochs: int
    every_epochs: int

    @property
    def anchors(self) -> list[int]:
        """Epochs that should always be saved (start, middle, end)."""
        if self.total_epochs <= 0:
            return []
        return sorted({0, max(0, self.total_epochs // 2), self.total_epochs - 1})

    @property
    def epoch_indices(self) -> set[int]:
        """Complete set of epoch indices to save based on policy."""
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
        """Check if the given epoch should be saved according to policy."""
        return epoch_idx in self.policy.epoch_indices

    def checkpoint_dir(self, epoch_idx: int) -> Path:
        """Get the directory path for a specific epoch checkpoint."""
        return self.root_dir / f"epoch_{epoch_idx + 1:04d}"

    def rollout_path(self, epoch_idx: int) -> Path:
        """Get the file path for a rollout tensor in a specific epoch."""
        return self.checkpoint_dir(epoch_idx) / "rollout.pt"

    def save_rollout(
        self, epoch_idx: int, pred_pos_seq: torch.Tensor, pred_vel_seq: torch.Tensor
    ) -> None:
        """Save predicted trajectory sequences to disk."""
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
        """Load predicted trajectory sequences from disk."""
        path = self.rollout_path(epoch_idx)
        if not path.exists():
            return None
        return torch.load(path, map_location="cpu")
