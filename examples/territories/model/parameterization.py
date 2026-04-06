"""Parameterization helpers for territories models."""

from __future__ import annotations

import torch


def softplus_param(raw: torch.Tensor, minimum: float = 1e-3) -> torch.Tensor:
    """Softplus with a small positive bias to ensure strictly positive results."""
    return torch.nn.functional.softplus(raw) + minimum


def inverse_softplus_target(value: float, minimum: float = 1e-3) -> float:
    """Inverse of softplus for initializing raw parameters from target values."""
    adjusted = max(float(value) - minimum, 1e-6)
    return float(
        torch.log(torch.expm1(torch.tensor(adjusted, dtype=torch.float32))).item()
    )
