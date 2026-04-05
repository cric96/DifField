"""Lower-level boids geometry, initialization, and scenario helpers.

This module intentionally stays below the aggregate DSL level: no rep/nbr.
"""

from __future__ import annotations

import torch
from torch_geometric.utils import degree as pyg_degree

from autofield import SpatialScenario, normalize_vectors


def edge_connectivity_stats(edge_index: torch.Tensor, num_nodes: int) -> tuple[float, float, float]:
    if num_nodes <= 0:
        return 0.0, 0.0, 0.0
    if edge_index.numel() == 0:
        return 0.0, 0.0, float(num_nodes)

    src = edge_index[0]
    tgt = edge_index[1]
    min_degree = float(pyg_degree(src, num_nodes=num_nodes).min().item())

    src_list = src.tolist()
    tgt_list = tgt.tolist()
    neighbors = [set() for _ in range(num_nodes)]
    for src_node, tgt_node in zip(src_list, tgt_list):
        neighbors[src_node].add(tgt_node)
        neighbors[tgt_node].add(src_node)
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
    return float(edge_index.shape[1]), min_degree, float(components)


def graph_health_summary(samples: list[tuple[float, float, float]]) -> dict[str, float]:
    if not samples:
        return {
            "mean_num_edges": float("nan"),
            "mean_min_degree": float("nan"),
            "max_num_components": float("nan"),
        }
    edge_counts = [item[0] for item in samples]
    min_degrees = [item[1] for item in samples]
    components = [item[2] for item in samples]
    return {
        "mean_num_edges": float(sum(edge_counts) / len(edge_counts)),
        "mean_min_degree": float(sum(min_degrees) / len(min_degrees)),
        "max_num_components": float(max(components)),
    }


def hard_separation_force(positions: torch.Tensor, sep: float) -> torch.Tensor:
    dx = positions[:, 0].unsqueeze(1) - positions[:, 0].unsqueeze(0)
    dy = positions[:, 1].unsqueeze(1) - positions[:, 1].unsqueeze(0)
    dist = torch.sqrt(dx.pow(2) + dy.pow(2) + 1e-9)
    sep_mask = (dist <= sep) & (dist > 0)
    sep_force = torch.stack([(dx * sep_mask).sum(dim=1), (dy * sep_mask).sum(dim=1)], dim=1)
    return normalize_vectors(sep_force)


def build_boids_scenario(
    positions: torch.Tensor,
    *,
    radius: float,
    init_connectivity: str,
    init_k_neighbors: int,
    init_min_degree: int,
    ensure_init_connected: bool,
) -> SpatialScenario:
    return SpatialScenario(
        positions=positions,
        edge_radius=radius if init_connectivity in {"radius", "hybrid"} else None,
        k_neighbors=init_k_neighbors if init_connectivity == "knn" else None,
        ensure_init_connected=ensure_init_connected and init_connectivity == "hybrid",
        init_min_degree=init_min_degree,
        init_k_neighbors=init_k_neighbors,
        device=positions.device,
    )


def sample_initial_boids_state(
    num_nodes: int,
    *,
    seed: int,
    velocity_scale: float,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    generator = torch.Generator()
    generator.manual_seed(seed)
    positions0 = torch.rand(num_nodes, 2, generator=generator, dtype=torch.float32).to(device)
    if velocity_scale <= 0.0:
        velocities0 = torch.zeros_like(positions0)
    else:
        directions = (torch.rand(num_nodes, 2, generator=generator, dtype=torch.float32) * 2.0 - 1.0).to(device)
        velocities0 = normalize_vectors(directions) * velocity_scale
    return positions0, velocities0