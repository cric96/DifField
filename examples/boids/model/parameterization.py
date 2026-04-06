"""Parameterization helpers for learnable boids parameters."""

from __future__ import annotations

import torch


def _softplus_param(raw: torch.Tensor, minimum: float = 1e-4) -> torch.Tensor:
    """Softplus with a small positive bias to ensure strictly positive results."""
    return torch.nn.functional.softplus(raw) + minimum


def _bounded_sigmoid(raw: torch.Tensor, low: float, high: float) -> torch.Tensor:
    """Sigmoid scaled to a specific [low, high] range."""
    return low + (high - low) * torch.sigmoid(raw)


def _inverse_sigmoid_target(value: float) -> float:
    """Inverse of sigmoid for initializing raw parameters from target values."""
    clipped = min(max(float(value), 1e-4), 1.0 - 1e-4)
    return float(torch.logit(torch.tensor(clipped, dtype=torch.float32)).item())


def _inverse_softplus_target(value: float, minimum: float = 1e-4) -> float:
    """Inverse of softplus for initializing raw parameters from target values."""
    adjusted = max(float(value) - minimum, 1e-6)
    return float(
        torch.log(torch.expm1(torch.tensor(adjusted, dtype=torch.float32))).item()
    )


def _inverse_bounded_sigmoid_target(value: float, low: float, high: float) -> float:
    """Inverse of bounded sigmoid for initializing raw parameters from target values."""
    if high <= low:
        raise ValueError("max speed bounds must satisfy high > low")
    clipped = min(max(float(value), low + 1e-6), high - 1e-6)
    scaled = (clipped - low) / (high - low)
    return _inverse_sigmoid_target(scaled)
