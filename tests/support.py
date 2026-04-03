"""Shared helpers for aggregate_gnn tests."""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor


def triangle_graph() -> tuple[Tensor, int]:
    """Triangle: 0-1, 1-2, 0-2 (undirected)."""
    edge_index = torch.tensor(
        [
            [0, 1, 1, 2, 0, 2],
            [1, 0, 2, 1, 2, 0],
        ],
        dtype=torch.long,
    )
    return edge_index, 3


def line_graph() -> tuple[Tensor, int]:
    """Line: 0-1-2-3 (undirected)."""
    edge_index = torch.tensor(
        [
            [0, 1, 1, 2, 2, 3],
            [1, 0, 2, 1, 3, 2],
        ],
        dtype=torch.long,
    )
    return edge_index, 4


def assert_finite_gradients(params: list[nn.Parameter]) -> None:
    for param in params:
        assert param.grad is not None
        assert torch.isfinite(param.grad).all()