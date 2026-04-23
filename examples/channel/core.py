"""Core reusable logic for channel examples."""

from __future__ import annotations

import torch

from diffield.dsl import broadcast, field, gather_min, iterate, mux, scatter, scatter_range

CHANNEL_THRESHOLD = 0.5


def inf_field(num_nodes: int, device: torch.device | None = None) -> torch.Tensor:
    return torch.full((num_nodes,), float("inf"), device=device)


def channel_body(
    source: torch.Tensor,
    dest: torch.Tensor,
    tolerance: float,
    noise: torch.Tensor | None = None,
) -> torch.Tensor:
    edge_noise = torch.zeros_like(source) if noise is None else noise
    dist_src = iterate(
        field.inf(),
        lambda dist_old: mux(
            source,
            field.of(0.0),
            gather_min(scatter(dist_old) + scatter_range() + edge_noise),
        ),
        name="dist_src",
    )
    # materialize scatter_range please as tensor matrix

    dist_dst = iterate(
        field.inf(),
        lambda dist_old: mux(
            dest,
            field.of(0.0),
            gather_min(scatter(dist_old) + scatter_range() + edge_noise),
        ),
        name="dist_dst",
    )
    dist_sd = broadcast(source > 0.5, dist_dst, name="dist_channel")

    on_path = dist_src + dist_dst <= dist_sd + tolerance
    finite = (
        torch.isfinite(dist_src) & torch.isfinite(dist_dst) & torch.isfinite(dist_sd)
    )
    return (on_path & finite).float()


def build_snapshot_payloads(
    records: dict[int, dict[str, torch.Tensor]],
    *,
    num_nodes: int,
) -> dict[int, dict[str, torch.Tensor]]:
    snapshots: dict[int, dict[str, torch.Tensor]] = {}
    for round_idx, payload in records.items():
        default_device = payload["output"].device if "output" in payload else None
        default_inf = inf_field(num_nodes, device=default_device)
        dist_src = payload.get("dist_src", default_inf)
        dist_dst = payload.get("dist_dst", default_inf)
        dist_sd_state = payload.get("_gc_dist_channel")
        if dist_sd_state is None:
            dist_sd = default_inf
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


def distance_src_to_dst(
    final: dict[str, torch.Tensor], dst_pos: tuple[int, int], cols: int
) -> float:
    dst_id = dst_pos[0] * cols + dst_pos[1]
    return float(final["dist_src"][dst_id].item())


def count_channel_nodes(
    final: dict[str, torch.Tensor], threshold: float = CHANNEL_THRESHOLD
) -> int:
    return int((final["channel"] > threshold).sum().item())
