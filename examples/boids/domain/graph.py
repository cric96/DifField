"""Graph health and connectivity analysis for boids simulations."""

from __future__ import annotations

import torch
from torch_geometric.utils import degree as pyg_degree


def edge_connectivity_stats(
    edge_index: torch.Tensor, num_nodes: int
) -> tuple[float, float, float]:
    """Compute basic graph health metrics: num_edges, min_degree, num_components."""
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
    """Aggregate graph health metrics over a sequence of snapshots."""
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
