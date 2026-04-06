"""Spatial scenario construction for boids."""

from __future__ import annotations

import torch
from autofield import SpatialScenario


def build_scenario(
    positions: torch.Tensor,
    *,
    radius: float,
    init_connectivity: str,
    init_k_neighbors: int,
    init_min_degree: int,
    ensure_init_connected: bool,
) -> SpatialScenario:
    """Factory for SpatialScenario with boids-specific defaults."""
    return SpatialScenario(
        positions=positions,
        edge_radius=radius if init_connectivity in {"radius", "hybrid"} else None,
        k_neighbors=init_k_neighbors if init_connectivity == "knn" else None,
        ensure_init_connected=ensure_init_connected and init_connectivity == "hybrid",
        init_min_degree=init_min_degree,
        init_k_neighbors=init_k_neighbors,
        device=positions.device,
    )
