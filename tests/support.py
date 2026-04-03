"""Shared helpers for autofield tests."""

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


def weighted_collect_graph() -> tuple[Tensor, Tensor, int]:
    """Four-node graph with asymmetric edge weights for collect tests."""
    edge_index = torch.tensor(
        [
            [0, 1, 0, 2, 1, 3, 2, 3],
            [1, 0, 2, 0, 3, 1, 3, 2],
        ],
        dtype=torch.long,
    )
    edge_weight = torch.tensor([1.0, 1.0, 2.0, 2.0, 10.0, 10.0, 1.0, 1.0])
    return edge_index, edge_weight, 4


def assert_finite_gradients(params: list[nn.Parameter]) -> None:
    for param in params:
        assert param.grad is not None
        assert torch.isfinite(param.grad).all()