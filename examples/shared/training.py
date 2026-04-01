"""Generic training utilities shared across examples."""

from __future__ import annotations

import torch
import torch.nn as nn


def parse_int_csv(seed_csv: str) -> list[int]:
    text = seed_csv.strip()
    return [] if not text else [int(item.strip()) for item in text.split(",") if item.strip()]


def grad_norm(params: list[nn.Parameter]) -> float:
    total = 0.0
    for param in params:
        if param.grad is not None:
            total += float(param.grad.detach().pow(2).sum().item())
    return total ** 0.5
