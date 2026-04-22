"""Conditional tensor selection helpers."""

from __future__ import annotations

import torch
from torch import Tensor

from ..constants import CONDITION_THRESHOLD, DEFAULT_TAU_BRANCH
from ..core.mode import get_default_mode


def _hard_where(cond: Tensor, x: Tensor, y: Tensor) -> Tensor:
    """Hard (non-differentiable w.r.t. cond) conditional selection."""
    cond_float = cond.float()
    if cond_float.dim() < x.dim():
        cond_float = cond_float.unsqueeze(-1)
    return torch.where(cond_float >= CONDITION_THRESHOLD, x, y)


def _soft_where(cond: Tensor, x: Tensor, y: Tensor, tau: float) -> Tensor:
    """Soft (differentiable w.r.t. cond) conditional selection via sigmoid."""
    cond_float = cond.float()
    w = torch.sigmoid(tau * (cond_float - CONDITION_THRESHOLD))
    while w.dim() < x.dim():
        w = w.unsqueeze(-1)
    return w * x + (1 - w) * y


def field_where(
    cond: Tensor,
    x: Tensor,
    y: Tensor,
    mode: str | None = None,
    tau: float | None = None,
) -> Tensor:
    r"""Differentiable pointwise conditional selection.

    Args:
        cond: Per-node condition tensor.
        x: Value where condition is active.
        y: Value where condition is inactive.
        mode: ``"hard"`` or ``"soft"``. Defaults to the thread-local setting.
        tau: Temperature for soft mode (larger = sharper). Defaults to
            :data:`~diffield.constants.DEFAULT_TAU_BRANCH`.
    """
    effective_mode = mode if mode is not None else get_default_mode()
    if effective_mode == "soft":
        effective_tau = tau if tau is not None else DEFAULT_TAU_BRANCH
        return _soft_where(cond, x, y, effective_tau)
    return _hard_where(cond, x, y)
