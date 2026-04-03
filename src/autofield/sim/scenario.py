"""Scenario utilities for aggregate simulations."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Iterable

import torch

from ..pyg_backend import (
    build_fully_connected_edge_index,
    build_spatial_edge_index,
    maybe_make_data,
)
from ..utils import make_grid_graph


def _resolve_static_edge_weight(
    edge_weight: torch.Tensor | float | None,
    *,
    num_edges: int,
    device: torch.device,
) -> torch.Tensor:
    if edge_weight is None:
        return torch.ones(num_edges, device=device, dtype=torch.float32)

    if isinstance(edge_weight, torch.Tensor):
        resolved = edge_weight.to(device=device, dtype=torch.float32)
        if resolved.numel() == 1:
            return resolved.reshape(()).expand(num_edges)
        resolved = resolved.reshape(-1)
        if resolved.shape[0] != num_edges:
            raise ValueError(f"edge_weight must have {num_edges} elements, got {resolved.shape[0]}")
        return resolved

    return torch.full((num_edges,), float(edge_weight), dtype=torch.float32, device=device)


def _resolve_scalar_metric(
    value: torch.Tensor | float,
    *,
    device: torch.device,
    dtype: torch.dtype,
    name: str,
) -> torch.Tensor:
    if isinstance(value, torch.Tensor):
        resolved = value.to(device=device, dtype=dtype)
    else:
        resolved = torch.tensor(float(value), device=device, dtype=dtype)
    if resolved.numel() != 1:
        raise ValueError(f"{name} must be scalar")
    return resolved.reshape(())


def _edge_distance_to_weight(
    edge_dist: torch.Tensor,
    edge_weight_mode: str,
    *,
    eps: float,
) -> torch.Tensor:
    if edge_weight_mode == "unit":
        return torch.ones_like(edge_dist)
    if edge_weight_mode == "distance":
        return edge_dist.clamp_min(eps)
    if edge_weight_mode == "inverse_distance":
        return 1.0 / (edge_dist + eps)
    raise ValueError("edge_weight_mode must be one of: unit, distance, inverse_distance")


@dataclass
class GridScenario:
    """Grid-based simulation scenario with helper builders."""

    rows: int
    cols: int
    connectivity: int = 4
    self_loops: bool = False
    device: torch.device | str = "cpu"
    edge_weight: torch.Tensor | float | None = None
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
        self.edge_weight = _resolve_static_edge_weight(
            self.edge_weight,
            num_edges=self.edge_index.shape[1],
            device=self.device,
        )

    def set_edge_weight(self, edge_weight: torch.Tensor | float | None) -> torch.Tensor:
        self.edge_weight = _resolve_static_edge_weight(
            edge_weight,
            num_edges=self.edge_index.shape[1],
            device=self.device,
        )
        return self.edge_weight

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

    def sync_context(self, round_ctx) -> None:
        """Push current topology into a round context before a DSL round."""
        round_ctx.edge_index = self.edge_index
        round_ctx.edge_weight = self.edge_weight
        round_ctx.data = maybe_make_data(self.edge_index, self.num_nodes, self.edge_weight)


@dataclass
class FullyConnectedScenario:
    """A fully connected topology where every node is connected to every other.

    This scenario ignores spatial positions and builds a complete graph.
    Useful for small-to-medium scale simulations where all-to-all communication
    is required.
    """

    num_nodes: int
    self_loops: bool = False
    device: torch.device | str = "cpu"
    edge_index: torch.Tensor = field(init=False)
    edge_weight: torch.Tensor | float | None = None

    def __post_init__(self) -> None:
        self.device = torch.device(self.device)
        self.edge_index = build_fully_connected_edge_index(
            self.num_nodes,
            self_loops=self.self_loops,
            device=self.device,
        )
        self.edge_weight = _resolve_static_edge_weight(
            self.edge_weight,
            num_edges=self.edge_index.shape[1],
            device=self.device,
        )

    def set_edge_weight(self, edge_weight: torch.Tensor | float | None) -> torch.Tensor:
        self.edge_weight = _resolve_static_edge_weight(
            edge_weight,
            num_edges=self.edge_index.shape[1],
            device=self.device,
        )
        return self.edge_weight

    def zeros(self, dtype: torch.dtype = torch.float32) -> torch.Tensor:
        return torch.zeros(self.num_nodes, dtype=dtype, device=self.device)

    def full(self, value: float, dtype: torch.dtype = torch.float32) -> torch.Tensor:
        return torch.full((self.num_nodes,), value, dtype=dtype, device=self.device)

    def marker(self, node_idx: int, value: float = 1.0) -> torch.Tensor:
        out = self.zeros()
        out[node_idx] = value
        return out

    def sync_context(self, round_ctx) -> None:
        """Push current topology into a round context before a DSL round."""
        round_ctx.edge_index = self.edge_index
        round_ctx.edge_weight = self.edge_weight
        round_ctx.data = maybe_make_data(self.edge_index, self.num_nodes, self.edge_weight)


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
    fully_connected: bool = False
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
        if self.edge_radius is None and self.k_neighbors is None and not self.fully_connected:
            raise ValueError("Either edge_radius, k_neighbors, or fully_connected must be provided")
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
            fully_connected=self.fully_connected,
            edge_weight_mode=self.edge_weight_mode,
            self_loops=self.self_loops,
        )
        self.edge_index = edge_index
        self.edge_weight = edge_weight

    def sync_context(self, round_ctx) -> None:
        """Push current topology into a round context before a DSL round."""
        round_ctx.edge_index = self.edge_index
        round_ctx.edge_weight = self.edge_weight
        round_ctx.data = maybe_make_data(self.edge_index, self.num_nodes, self.edge_weight)


def build_spatial_graph(
    positions: torch.Tensor,
    *,
    edge_radius: float | None = None,
    k_neighbors: int | None = None,
    fully_connected: bool = False,
    edge_weight_mode: str = "unit",
    self_loops: bool = False,
    eps: float = 1e-6,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build directed edges and edge weights from node positions."""
    if positions.dim() != 2 or positions.shape[1] != 2:
        raise ValueError("positions must have shape [num_nodes, 2]")

    num_nodes = positions.shape[0]
    if num_nodes == 0:
        empty_edges = torch.zeros((2, 0), dtype=torch.long, device=positions.device)
        empty_weights = torch.zeros((0,), dtype=torch.float32, device=positions.device)
        return empty_edges, empty_weights

    if fully_connected:
        edge_index = build_fully_connected_edge_index(
            num_nodes,
            self_loops=self_loops,
            device=positions.device,
        )
    else:
        edge_index = build_spatial_edge_index(
            positions,
            edge_radius=edge_radius,
            k_neighbors=k_neighbors,
            self_loops=self_loops,
        )

    if edge_index.shape[1] == 0:
        empty_edges = torch.zeros((2, 0), dtype=torch.long, device=positions.device)
        empty_weights = torch.zeros((0,), dtype=torch.float32, device=positions.device)
        return empty_edges, empty_weights

    src_nodes, tgt_nodes = edge_index[0], edge_index[1]
    edge_dist = (positions[src_nodes] - positions[tgt_nodes]).norm(dim=-1)
    edge_weight = _edge_distance_to_weight(edge_dist, edge_weight_mode, eps=eps)
    return edge_index.long(), edge_weight.float()


def build_relaxed_radius_graph(
    positions: torch.Tensor,
    *,
    edge_radius: torch.Tensor | float,
    edge_weight_mode: str = "distance",
    self_loops: bool = False,
    relaxation_tau: float = 0.05,
    penalty_strength: float = 10.0,
    eps: float = 1e-6,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build a dense candidate graph with a smooth radius penalty.

    Unlike :func:`build_spatial_graph`, this helper keeps the candidate edge
    set fixed (fully connected, optionally with self loops) and encodes the
    radius preference as an additive cost penalty. This preserves a gradient
    path to positions and to a learnable radius parameter while avoiding
    discrete edge birth/death in the backward pass.
    """
    if positions.dim() != 2 or positions.shape[1] != 2:
        raise ValueError("positions must have shape [num_nodes, 2]")
    if relaxation_tau <= 0.0:
        raise ValueError("relaxation_tau must be > 0")
    if penalty_strength < 0.0:
        raise ValueError("penalty_strength must be >= 0")

    num_nodes = int(positions.shape[0])
    if num_nodes == 0:
        empty_edges = torch.zeros((2, 0), dtype=torch.long, device=positions.device)
        empty_weights = torch.zeros((0,), dtype=torch.float32, device=positions.device)
        return empty_edges, empty_weights

    edge_index = build_fully_connected_edge_index(
        num_nodes,
        self_loops=self_loops,
        device=positions.device,
    )
    if edge_index.shape[1] == 0:
        empty_weights = torch.zeros((0,), dtype=torch.float32, device=positions.device)
        return edge_index.long(), empty_weights

    src_nodes, tgt_nodes = edge_index[0], edge_index[1]
    edge_dist = (positions[src_nodes] - positions[tgt_nodes]).norm(dim=-1)
    base_weight = _edge_distance_to_weight(edge_dist, edge_weight_mode, eps=eps)
    radius = _resolve_scalar_metric(
        edge_radius,
        device=positions.device,
        dtype=positions.dtype,
        name="edge_radius",
    )
    smooth_excess = relaxation_tau * torch.nn.functional.softplus((edge_dist - radius) / relaxation_tau)
    edge_weight = base_weight + penalty_strength * smooth_excess
    return edge_index.long(), edge_weight.float()


@dataclass
class RelaxedRadiusScenario:
    """Spatial scenario with smooth radius-aware path costs.

    The candidate graph stays fully connected, while edges longer than the
    preferred radius incur a smooth additive penalty. This is intended for
    differentiable connectivity learning with ``nbr_range()`` and weighted
    shortest-path style programs.
    """

    positions: torch.Tensor
    edge_radius: torch.Tensor | float
    edge_weight_mode: str = "distance"
    relaxation_tau: float = 0.05
    penalty_strength: float = 10.0
    self_loops: bool = False
    device: torch.device | str = "cpu"
    edge_index: torch.Tensor = field(init=False)
    edge_weight: torch.Tensor = field(init=False)
    num_nodes: int = field(init=False)

    def __post_init__(self) -> None:
        self.device = torch.device(self.device)
        if self.positions.dim() != 2 or self.positions.shape[1] != 2:
            raise ValueError("positions must have shape [num_nodes, 2]")
        self.positions = self.positions.to(self.device, dtype=torch.float32)
        self.edge_radius = _resolve_scalar_metric(
            self.edge_radius,
            device=self.device,
            dtype=torch.float32,
            name="edge_radius",
        )
        self.num_nodes = int(self.positions.shape[0])
        self.refresh_topology()

    def zeros(self, dtype: torch.dtype = torch.float32) -> torch.Tensor:
        return torch.zeros(self.num_nodes, dtype=dtype, device=self.device)

    def full(self, value: float, dtype: torch.dtype = torch.float32) -> torch.Tensor:
        return torch.full((self.num_nodes,), value, dtype=dtype, device=self.device)

    def marker(self, node_idx: int, value: float = 1.0) -> torch.Tensor:
        out = self.zeros()
        out[node_idx] = value
        return out

    def set_edge_radius(self, edge_radius: torch.Tensor | float) -> torch.Tensor:
        self.edge_radius = _resolve_scalar_metric(
            edge_radius,
            device=self.device,
            dtype=torch.float32,
            name="edge_radius",
        )
        return self.edge_radius

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
        self.edge_index, self.edge_weight = build_relaxed_radius_graph(
            self.positions,
            edge_radius=self.edge_radius,
            edge_weight_mode=self.edge_weight_mode,
            self_loops=self.self_loops,
            relaxation_tau=self.relaxation_tau,
            penalty_strength=self.penalty_strength,
        )

    def sync_context(self, round_ctx) -> None:
        """Push current topology into a round context before a DSL round."""
        round_ctx.edge_index = self.edge_index
        round_ctx.edge_weight = self.edge_weight
        round_ctx.data = maybe_make_data(self.edge_index, self.num_nodes, self.edge_weight)


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
