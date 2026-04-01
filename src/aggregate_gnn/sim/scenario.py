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


@dataclass
class SpatialScenario:
    """2D spatial scenario with dynamic topology rebuilt from node positions.

    Topology is created from the current position field using either:
    - radius graph: connect i->j when ||p_i - p_j|| <= edge_radius
    - k-NN graph: connect i->k nearest neighbors
    """

    positions: torch.Tensor
    edge_radius: float | None = None
    k_neighbors: int | None = None
    edge_weight_mode: str = "unit"
    self_loops: bool = False
    ensure_init_connected: bool = False
    init_min_degree: int = 2
    init_k_neighbors: int = 8
    device: torch.device | str = "cpu"
    edge_index: torch.Tensor = field(init=False)
    edge_weight: torch.Tensor = field(init=False)
    num_nodes: int = field(init=False)
    init_graph_stats: dict[str, float] = field(init=False)
    _init_topology_done: bool = field(init=False, default=False)

    def __post_init__(self) -> None:
        self.device = torch.device(self.device)
        if self.positions.dim() != 2 or self.positions.shape[1] != 2:
            raise ValueError("positions must have shape [num_nodes, 2]")
        self.positions = self.positions.to(self.device, dtype=torch.float32)
        self.num_nodes = int(self.positions.shape[0])
        if self.edge_radius is None and self.k_neighbors is None:
            raise ValueError("Either edge_radius or k_neighbors must be provided")
        self.init_graph_stats = {
            "num_edges": 0.0,
            "min_degree": 0.0,
            "num_components": 0.0,
        }
        self.refresh_topology()

    def zeros(self, dtype: torch.dtype = torch.float32) -> torch.Tensor:
        return torch.zeros(self.num_nodes, dtype=dtype, device=self.device)

    def full(self, value: float, dtype: torch.dtype = torch.float32) -> torch.Tensor:
        return torch.full((self.num_nodes,), value, dtype=dtype, device=self.device)

    def marker(self, node_idx: int, value: float = 1.0) -> torch.Tensor:
        out = self.zeros()
        out[node_idx] = value
        return out

    def update_positions(self, positions: torch.Tensor, refresh_topology: bool = True) -> None:
        if positions.shape != self.positions.shape:
            raise ValueError("positions shape mismatch")
        self.positions = positions.to(self.device, dtype=torch.float32)
        if refresh_topology:
            self.refresh_topology()

    def step_positions(self, velocities: torch.Tensor, dt: float = 1.0, refresh_topology: bool = True) -> None:
        if velocities.shape != self.positions.shape:
            raise ValueError("velocities shape mismatch")
        self.positions = self.positions + velocities.to(self.device, dtype=torch.float32) * dt
        if refresh_topology:
            self.refresh_topology()

    def refresh_topology(self) -> None:
        if self.ensure_init_connected and not self._init_topology_done:
            edge_index, edge_weight, stats = build_well_connected_init_graph(
                self.positions,
                edge_radius=self.edge_radius,
                min_degree=self.init_min_degree,
                start_k_neighbors=self.init_k_neighbors,
                edge_weight_mode=self.edge_weight_mode,
                self_loops=self.self_loops,
            )
            self.edge_index = edge_index
            self.edge_weight = edge_weight
            self.init_graph_stats = stats
            self._init_topology_done = True
            return

        edge_index, edge_weight = build_spatial_graph(
            self.positions,
            edge_radius=self.edge_radius,
            k_neighbors=self.k_neighbors,
            edge_weight_mode=self.edge_weight_mode,
            self_loops=self.self_loops,
        )
        self.edge_index = edge_index
        self.edge_weight = edge_weight

    def sync_context(self, round_ctx) -> None:
        """Push current topology into a round context before a DSL round."""
        round_ctx.edge_index = self.edge_index
        round_ctx.edge_weight = self.edge_weight


def build_spatial_graph(
    positions: torch.Tensor,
    *,
    edge_radius: float | None = None,
    k_neighbors: int | None = None,
    edge_weight_mode: str = "unit",
    self_loops: bool = False,
    eps: float = 1e-6,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build directed edges and inverse-distance weights from node positions."""
    if positions.dim() != 2 or positions.shape[1] != 2:
        raise ValueError("positions must have shape [num_nodes, 2]")

    num_nodes = positions.shape[0]
    if num_nodes == 0:
        empty_edges = torch.zeros((2, 0), dtype=torch.long, device=positions.device)
        empty_weights = torch.zeros((0,), dtype=torch.float32, device=positions.device)
        return empty_edges, empty_weights

    distances = torch.cdist(positions, positions)

    if k_neighbors is not None:
        if k_neighbors <= 0:
            raise ValueError("k_neighbors must be > 0")
        k = min(k_neighbors + (1 if self_loops else 0), num_nodes)
        sorted_idx = torch.argsort(distances, dim=1)
        nbr_idx = sorted_idx[:, :k]
        src = torch.arange(num_nodes, device=positions.device).unsqueeze(1).expand(-1, k)
        mask = torch.ones_like(nbr_idx, dtype=torch.bool)
        if not self_loops:
            mask = nbr_idx != src
        src_nodes = src[mask]
        tgt_nodes = nbr_idx[mask]
    else:
        if edge_radius is None or edge_radius <= 0:
            raise ValueError("edge_radius must be > 0 when k_neighbors is not used")
        mask = distances <= edge_radius
        if not self_loops:
            mask.fill_diagonal_(False)
        src_nodes, tgt_nodes = torch.where(mask)

    if src_nodes.numel() == 0:
        empty_edges = torch.zeros((2, 0), dtype=torch.long, device=positions.device)
        empty_weights = torch.zeros((0,), dtype=torch.float32, device=positions.device)
        return empty_edges, empty_weights

    edge_dist = distances[src_nodes, tgt_nodes]
    if edge_weight_mode == "unit":
        edge_weight = torch.ones_like(edge_dist)
    elif edge_weight_mode == "inverse_distance":
        edge_weight = 1.0 / (edge_dist + eps)
    else:
        raise ValueError("edge_weight_mode must be one of: unit, inverse_distance")
    edge_index = torch.stack([src_nodes, tgt_nodes], dim=0)
    return edge_index.long(), edge_weight.float()


def _graph_stats(edge_index: torch.Tensor, num_nodes: int) -> dict[str, float]:
    if num_nodes <= 0:
        return {"num_edges": 0.0, "min_degree": 0.0, "num_components": 0.0}

    src = edge_index[0]
    tgt = edge_index[1]
    neighbors = [set() for _ in range(num_nodes)]
    for s, t in zip(src.tolist(), tgt.tolist()):
        neighbors[s].add(t)
        neighbors[t].add(s)

    degree = torch.tensor([len(ns) for ns in neighbors], dtype=torch.float32)
    visited = [False] * num_nodes
    components = 0
    for node in range(num_nodes):
        if visited[node]:
            continue
        components += 1
        stack = [node]
        visited[node] = True
        while stack:
            cur = stack.pop()
            for nxt in neighbors[cur]:
                if not visited[nxt]:
                    visited[nxt] = True
                    stack.append(nxt)

    return {
        "num_edges": float(edge_index.shape[1]),
        "min_degree": float(degree.min().item()) if degree.numel() > 0 else 0.0,
        "num_components": float(components),
    }


def _merge_edge_sets(radius_edges: torch.Tensor, knn_edges: torch.Tensor, num_nodes: int, device: torch.device) -> torch.Tensor:
    if radius_edges.numel() == 0 and knn_edges.numel() == 0:
        return torch.zeros((2, 0), dtype=torch.long, device=device)

    seen: set[tuple[int, int]] = set()
    merged: list[tuple[int, int]] = []
    for edges in (radius_edges, knn_edges):
        if edges.numel() == 0:
            continue
        for s, t in edges.t().tolist():
            key = (int(s), int(t))
            if key not in seen:
                seen.add(key)
                merged.append(key)

    if not merged:
        return torch.zeros((2, 0), dtype=torch.long, device=device)
    merged_tensor = torch.tensor(merged, dtype=torch.long, device=device)
    return merged_tensor.t().contiguous()


def build_well_connected_init_graph(
    positions: torch.Tensor,
    *,
    edge_radius: float | None,
    min_degree: int,
    start_k_neighbors: int,
    edge_weight_mode: str = "unit",
    self_loops: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, float]]:
    """Build a radius graph and augment it with k-NN edges until connected/non-sparse."""
    if positions.dim() != 2 or positions.shape[1] != 2:
        raise ValueError("positions must have shape [num_nodes, 2]")
    num_nodes = int(positions.shape[0])
    if num_nodes == 0:
        empty_edges = torch.zeros((2, 0), dtype=torch.long, device=positions.device)
        empty_weights = torch.zeros((0,), dtype=torch.float32, device=positions.device)
        return empty_edges, empty_weights, {"num_edges": 0.0, "min_degree": 0.0, "num_components": 0.0}

    radius_edges, _ = build_spatial_graph(
        positions,
        edge_radius=edge_radius,
        k_neighbors=None,
        edge_weight_mode=edge_weight_mode,
        self_loops=self_loops,
    )
    merged_edges = radius_edges
    stats = _graph_stats(merged_edges, num_nodes)

    k = max(1, int(start_k_neighbors))
    required_degree = max(1, int(min_degree))
    while stats["num_components"] > 1.0 or stats["min_degree"] < float(required_degree):
        knn_edges, _ = build_spatial_graph(
            positions,
            edge_radius=None,
            k_neighbors=k,
            edge_weight_mode=edge_weight_mode,
            self_loops=self_loops,
        )
        merged_edges = _merge_edge_sets(radius_edges, knn_edges, num_nodes, positions.device)
        stats = _graph_stats(merged_edges, num_nodes)
        if k >= num_nodes - 1:
            break
        k += 1

    if merged_edges.numel() == 0:
        empty_weights = torch.zeros((0,), dtype=torch.float32, device=positions.device)
        return merged_edges, empty_weights, stats

    distances = torch.cdist(positions, positions)
    src_nodes = merged_edges[0]
    tgt_nodes = merged_edges[1]
    edge_dist = distances[src_nodes, tgt_nodes]
    if edge_weight_mode == "unit":
        edge_weight = torch.ones_like(edge_dist)
    elif edge_weight_mode == "inverse_distance":
        edge_weight = 1.0 / (edge_dist + 1e-6)
    else:
        raise ValueError("edge_weight_mode must be one of: unit, inverse_distance")

    return merged_edges.long(), edge_weight.float(), stats
