"""Scenario utilities for aggregate simulations."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Iterable

import torch

from ..utils import make_grid_graph


@dataclass
class GridScenario:
    """Grid-based simulation scenario with helper builders."""

    rows: int
    cols: int
    connectivity: int = 4
    self_loops: bool = False
    device: torch.device | str = "cpu"
    edge_index: torch.Tensor = field(init=False)
    num_nodes: int = field(init=False)

    def __post_init__(self) -> None:
        edge_index, num_nodes = make_grid_graph(
            self.rows,
            self.cols,
            connectivity=self.connectivity,
            include_self_loops=self.self_loops,
        )
        self.device = torch.device(self.device)
        self.edge_index = edge_index.to(self.device)
        self.num_nodes = num_nodes

    def zeros(self, dtype: torch.dtype = torch.float32) -> torch.Tensor:
        return torch.zeros(self.num_nodes, dtype=dtype, device=self.device)

    def full(self, value: float, dtype: torch.dtype = torch.float32) -> torch.Tensor:
        return torch.full((self.num_nodes,), value, dtype=dtype, device=self.device)

    def pos_to_idx(self, row: int, col: int) -> int:
        return row * self.cols + col

    def idx_to_pos(self, idx: int) -> tuple[int, int]:
        return idx // self.cols, idx % self.cols

    def marker(self, row: int, col: int, value: float = 1.0) -> torch.Tensor:
        out = self.zeros()
        out[self.pos_to_idx(row, col)] = value
        return out

    def markers(self, positions: Iterable[tuple[int, int]], value: float = 1.0) -> torch.Tensor:
        out = self.zeros()
        for row, col in positions:
            out[self.pos_to_idx(row, col)] = value
        return out

    def mask_from_positions(self, positions: Iterable[tuple[int, int]]) -> torch.Tensor:
        out = torch.zeros(self.num_nodes, dtype=torch.bool, device=self.device)
        for row, col in positions:
            out[self.pos_to_idx(row, col)] = True
        return out

    def mask_from_predicate(self, predicate: Callable[[int, int], bool]) -> torch.Tensor:
        out = torch.zeros(self.num_nodes, dtype=torch.bool, device=self.device)
        for row in range(self.rows):
            for col in range(self.cols):
                if predicate(row, col):
                    out[self.pos_to_idx(row, col)] = True
        return out
