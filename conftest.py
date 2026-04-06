"""Pytest path bootstrap and shared fixtures for the local src-layout project."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
for path in (ROOT, ROOT / "src", ROOT / "examples"):
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)

import pytest
import torch
from torch import Tensor
from autofield import AggregateContext
from autofield.utils import make_grid_graph


@pytest.fixture
def triangle_topology() -> tuple[Tensor, int]:
    """Triangle: 0-1, 1-2, 0-2 (undirected)."""
    edge_index = torch.tensor(
        [
            [0, 1, 1, 2, 0, 2],
            [1, 0, 2, 1, 2, 0],
        ],
        dtype=torch.long,
    )
    return edge_index, 3


@pytest.fixture
def line_topology() -> tuple[Tensor, int]:
    """Line: 0-1-2-3 (undirected)."""
    edge_index = torch.tensor(
        [
            [0, 1, 1, 2, 2, 3],
            [1, 0, 2, 1, 3, 2],
        ],
        dtype=torch.long,
    )
    return edge_index, 4


@pytest.fixture
def weighted_collect_topology() -> tuple[Tensor, Tensor, int]:
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


@pytest.fixture
def triangle_ctx(triangle_topology) -> AggregateContext:
    edge_index, n = triangle_topology
    return AggregateContext(edge_index, n)


@pytest.fixture
def line_ctx(line_topology) -> AggregateContext:
    edge_index, n = line_topology
    return AggregateContext(edge_index, n)


@pytest.fixture
def empty_topology() -> tuple[Tensor, int]:
    """Empty graph."""
    edge_index = torch.zeros((2, 0), dtype=torch.long)
    return edge_index, 0


@pytest.fixture
def isolated_topology() -> tuple[Tensor, int]:
    """Graph with 4 nodes and no edges."""
    edge_index = torch.zeros((2, 0), dtype=torch.long)
    return edge_index, 4


@pytest.fixture
def disconnected_topology() -> tuple[Tensor, int]:
    """Two disconnected lines: 0-1, 2-3."""
    edge_index = torch.tensor(
        [
            [0, 1, 2, 3],
            [1, 0, 3, 2],
        ],
        dtype=torch.long,
    )
    return edge_index, 4


def pytest_collection_modifyitems(config, items):
    for item in items:
        if "examples" in str(item.fspath):
            item.add_marker(pytest.mark.integration)
        else:
            item.add_marker(pytest.mark.unit)
