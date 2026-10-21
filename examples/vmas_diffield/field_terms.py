"""Local, single-agent steering signals used by the aggregate programs."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from diffield.sim import normalize_vectors


def soft_normalize(x: Tensor, scale: float) -> Tensor:
    """Normalize strong signals while keeping near-zero signals proportional."""
    return x / (x.norm(dim=-1, keepdim=True) + scale)


@dataclass
class FieldTerms:
    """Per-node steering directions and optional scenario diagnostics."""

    separation: Tensor
    alignment: Tensor
    cohesion: Tensor
    goal: Tensor
    has_neigh: Tensor
    explore: Tensor
    disperse: Tensor
    sense: Tensor | None = None
    recruit: Tensor | None = None
    social: Tensor | None = None
    brake: Tensor | None = None
    avoid: Tensor | None = None
    follow: Tensor | None = None
    lead_dir: Tensor | None = None
    leader_mask: Tensor | None = None
    leader_dist: Tensor | None = None


_lidar_angle_cache: dict[tuple[int, str], Tensor] = {}


def _lidar_ray_angles(n_rays: int, device: torch.device) -> Tensor:
    """Return cached full-circle ray angles for a zero-rotation agent."""
    key = (n_rays, str(device))
    if key not in _lidar_angle_cache:
        _lidar_angle_cache[key] = torch.linspace(0, 2 * torch.pi, n_rays + 1, device=device)[
            :n_rays
        ]
    return _lidar_angle_cache[key]


def lidar_sense_term(lidar: Tensor, *, max_range: float) -> tuple[Tensor, Tensor]:
    """Build a proximity-weighted seek direction and detection strength from lidar."""
    angles = _lidar_ray_angles(lidar.shape[-1], lidar.device)
    dirs = torch.stack([angles.cos(), angles.sin()], dim=-1)
    proximity = (max_range - lidar).clamp(min=0.0)
    strength = torch.tanh(proximity.sum(dim=-1, keepdim=True) / max_range)
    direction = torch.einsum("nr,rd->nd", proximity, dirs)
    return soft_normalize(direction, scale=max_range), strength


_SAMPLE_OFFSET_DIRS = torch.tensor(
    [[1, 0], [-1, 0], [0, 1], [0, -1], [-1, -1], [1, -1], [-1, 1], [1, 1]], dtype=torch.float32
)
_SAMPLE_OFFSET_DIRS = _SAMPLE_OFFSET_DIRS / _SAMPLE_OFFSET_DIRS.norm(dim=-1, keepdim=True)


def grid_sense_term(samples: Tensor) -> tuple[Tensor, Tensor]:
    """Estimate local field ascent from the eight neighboring grid samples."""
    dirs = _SAMPLE_OFFSET_DIRS.to(samples.device)
    strength = torch.tanh(samples.sum(dim=-1, keepdim=True))
    direction = torch.einsum("no,od->nd", samples, dirs)
    return soft_normalize(direction, scale=1.0), strength


_EXPLORE_FREQS = torch.tensor([[6.1, -4.3], [-5.7, 3.9], [4.1, 6.7]])
_EXPLORE_PHASES = torch.tensor([0.7, 2.9, 5.2])


def explore_term(pos: Tensor, node_id: Tensor, *, freq_scale: float = 1.0) -> Tensor:
    """Return a smooth, deterministic background search direction."""
    freqs = _EXPLORE_FREQS.to(pos.device) * freq_scale
    phases = _EXPLORE_PHASES.to(pos.device)
    id_phase = node_id.float() * 2.399963
    angle = torch.einsum("nd,kd->nk", pos, freqs) + phases + id_phase.unsqueeze(-1)
    return normalize_vectors(torch.stack([angle.cos(), angle.sin()], dim=-1).mean(dim=1))


def brake_term(vel: Tensor, goal_rel: Tensor, *, arrive_radius: float = 0.12) -> Tensor:
    """Apply velocity-proportional damping that grows near the agent's goal."""
    gd = goal_rel.norm(dim=-1, keepdim=True)
    return -vel * torch.exp(-gd / arrive_radius)
