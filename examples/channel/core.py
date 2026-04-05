"""Core reusable logic for channel examples."""

from __future__ import annotations

import torch

from autofield import broadcast, mux, nbr, rep
from autofield.dsl import field

CHANNEL_THRESHOLD = 0.5


def channel_body(source: torch.Tensor, dest: torch.Tensor, tolerance: float, noise: torch.Tensor | None = None) -> torch.Tensor:
    hop_noise = torch.zeros_like(source) if noise is None else noise
    dist_src = rep(
        "dist_src",
        float("inf"),
        lambda dist_old: mux(source, field.of(0.0), nbr(dist_old + 1.0 + hop_noise, aggr="min")),
    )
    dist_dst = rep(
        "dist_dst",
        float("inf"),
        lambda dist_old: mux(dest, field.of(0.0), nbr(dist_old + 1.0 + hop_noise, aggr="min")),
    )
    dist_sd = broadcast(source > 0.5, dist_dst, name="dist_channel")

    on_path = dist_src + dist_dst <= dist_sd + tolerance
    finite = torch.isfinite(dist_src) & torch.isfinite(dist_dst) & torch.isfinite(dist_sd)
    return (on_path & finite).float()


def build_snapshot_payloads(
    records: dict[int, dict[str, torch.Tensor]],
    *,
    num_nodes: int,
) -> dict[int, dict[str, torch.Tensor]]:
    snapshots: dict[int, dict[str, torch.Tensor]] = {}
    for round_idx, payload in records.items():
        dist_src = payload.get("dist_src", torch.full((num_nodes,), float("inf")))
        dist_dst = payload.get("dist_dst", torch.full((num_nodes,), float("inf")))
        dist_sd_state = payload.get("_gc_dist_channel")
        if dist_sd_state is None:
            dist_sd = torch.full((num_nodes,), float("inf"))
        elif dist_sd_state.dim() > 1:
            dist_sd = dist_sd_state[:, 1]
        else:
            dist_sd = dist_sd_state
        snapshots[round_idx] = {
            "dist_src": dist_src,
            "dist_dst": dist_dst,
            "sum": dist_src + dist_dst,
            "dist_sd": dist_sd,
            "channel": payload["output"],
        }
    return snapshots


def distance_src_to_dst(final: dict[str, torch.Tensor], dst_pos: tuple[int, int], cols: int) -> float:
    dst_id = dst_pos[0] * cols + dst_pos[1]
    return float(final["dist_src"][dst_id].item())


def count_channel_nodes(final: dict[str, torch.Tensor], threshold: float = CHANNEL_THRESHOLD) -> int:
    return int((final["channel"] > threshold).sum().item())
