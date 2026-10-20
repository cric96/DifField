"""Programs shared by the decentralized tests."""

from __future__ import annotations

import torch

from diffield.dsl import broadcast, field, gather_min, iterate, mux, scatter, scatter_range


def _distance_to(source: torch.Tensor, noise: torch.Tensor, name: str) -> torch.Tensor:
    return iterate(
        field.inf(),
        lambda distance: mux(
            source, field.of(0.0), gather_min(scatter(distance) + scatter_range() + noise)
        ),
        name=name,
    )


def channel_body(
    source: torch.Tensor,
    dest: torch.Tensor,
    tolerance: float,
    noise: torch.Tensor | None = None,
) -> torch.Tensor:
    """The nodes on a path from *source* to *dest* at most *tolerance* longer than the shortest.

    Returns:
        A field of 1.0 on the channel and 0.0 elsewhere.
    """
    edge_noise = torch.zeros_like(source) if noise is None else noise
    dist_src = _distance_to(source, edge_noise, "dist_src")
    dist_dst = _distance_to(dest, edge_noise, "dist_dst")
    dist_sd = broadcast(source > 0.5, dist_dst, name="dist_channel")
    on_path = dist_src + dist_dst <= dist_sd + tolerance
    finite = torch.isfinite(dist_src) & torch.isfinite(dist_dst) & torch.isfinite(dist_sd)
    return (on_path & finite).float()
