"""Conditional tensor selection helpers."""

from __future__ import annotations

import torch
from torch import Tensor

from ..constants import CONDITION_THRESHOLD


def soft_where(cond: Tensor, x: Tensor, y: Tensor) -> Tensor:
    r"""Differentiable pointwise conditional selection."""
    cond_float = cond.float()
    if cond_float.dim() < x.dim():
        cond_float = cond_float.unsqueeze(-1)
    return torch.where(cond_float >= CONDITION_THRESHOLD, x, y)