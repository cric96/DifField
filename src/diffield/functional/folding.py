"""Bucket-wise folding and row selection helpers."""

from __future__ import annotations

import operator
from collections.abc import Callable

import torch
from torch import Tensor

from ..constants import FILL_VALUE_MIN, LOG_EPSILON
from ..pyg_backend import scatter_hard
from .aggregation import scatter_aggr


def scatter_min_by_first(
    src: Tensor,
    index: Tensor,
    num_nodes: int,
    *,
    mode: str = "hard",
    tau: float | Tensor = 1.0,
    fill_row: Tensor | None = None,
) -> Tensor:
    """Select the full row whose first component is minimal in each bucket."""
    if src.dim() != 2:
        raise ValueError("scatter_min_by_first expects src shaped [E, D]")
    if src.shape[1] < 1:
        raise ValueError("scatter_min_by_first requires at least one column")

    fill = _expand_bucket_init(fill_row, num_nodes, src.shape[1:], src.device, src.dtype)
    if src.shape[0] == 0:
        return fill

    if mode == "hard":
        cost = src[:, 0]
        best_cost = scatter_aggr(
            cost,
            index,
            num_nodes,
            aggr="min",
            mode="hard",
            fill_value=float(fill[0, 0].item()) if fill.shape[0] > 0 else FILL_VALUE_MIN,
        )
        msg_ids = torch.arange(src.shape[0], device=index.device, dtype=torch.long)
        best_mask = cost == best_cost[index]
        chosen_msg = scatter_hard(
            torch.where(best_mask, msg_ids, msg_ids.new_full(msg_ids.shape, -1)),
            index,
            num_nodes,
            aggr="max",
            fill_value=-1,
        )
        out = fill.clone()
        valid = chosen_msg >= 0
        out[valid] = src[chosen_msg[valid]]
        return out

    if mode != "soft":
        raise ValueError(f"Unknown mode: {mode}")

    if isinstance(tau, Tensor):
        effective_tau = tau.to(device=src.device, dtype=src.dtype).clamp_min(LOG_EPSILON)
    else:
        effective_tau = src.new_tensor(max(float(tau), LOG_EPSILON))
    cost = src[:, 0]
    cost_out = scatter_aggr(
        cost,
        index,
        num_nodes,
        aggr="min",
        mode="soft",
        tau=effective_tau,
        fill_value=float(fill[0, 0].item()) if fill.shape[0] > 0 else FILL_VALUE_MIN,
    )

    finite = torch.isfinite(cost)
    safe_cost = torch.where(finite, cost, torch.zeros_like(cost))
    logits_raw = -safe_cost / effective_tau
    logits = torch.where(finite, logits_raw, torch.full_like(cost, float("-inf")))
    bucket_max = scatter_hard(logits, index, num_nodes, aggr="max", fill_value=float("-inf"))
    safe_bucket_max = torch.where(finite, bucket_max[index], torch.zeros_like(cost))
    shifted = torch.where(finite, logits_raw - safe_bucket_max, torch.full_like(cost, float("-inf")))
    score = torch.where(finite, shifted.exp(), torch.zeros_like(cost))
    score_sum = scatter_hard(score, index, num_nodes, aggr="sum", fill_value=0.0)

    out = fill.clone()
    has_messages = score_sum > 0
    out[has_messages, 0] = cost_out[has_messages]

    if src.shape[1] == 1:
        return out

    payload = src[:, 1:]
    weighted_payload = payload * score.unsqueeze(-1)
    payload_sum = scatter_hard(weighted_payload, index, num_nodes, aggr="sum", fill_value=0.0)
    payload_out = payload_sum / score_sum.clamp(min=LOG_EPSILON).unsqueeze(-1)
    out[has_messages, 1:] = payload_out[has_messages]
    return out


def scatter_binary_fold(
    src: Tensor,
    index: Tensor,
    num_nodes: int,
    accumulation: Callable[[Tensor, Tensor], Tensor],
    init: Tensor | float,
) -> Tensor:
    """Fold bucketed values with a binary accumulation function."""
    feature_shape = src.shape[1:] if src.dim() > 1 else ()
    init_tensor = _expand_bucket_init(init, num_nodes, feature_shape, src.device, src.dtype)
    fast_out = _scatter_binary_fold_fast_path(src, index, num_nodes, accumulation, init_tensor)
    if fast_out is not None:
        return fast_out

    buckets = [init_tensor[node_idx] for node_idx in range(num_nodes)]
    for msg_idx in range(src.shape[0]):
        bucket = int(index[msg_idx].item())
        buckets[bucket] = accumulation(buckets[bucket], src[msg_idx])
    return torch.stack(buckets, dim=0)


def _scatter_binary_fold_fast_path(
    src: Tensor,
    index: Tensor,
    num_nodes: int,
    accumulation: Callable[[Tensor, Tensor], Tensor],
    init_tensor: Tensor,
) -> Tensor | None:
    if accumulation in (torch.add, operator.add):
        aggregated = scatter_aggr(src, index, num_nodes, aggr="sum")
        return init_tensor + aggregated

    if accumulation is torch.logical_or:
        aggregated = scatter_hard(
            src.bool().to(dtype=torch.float32),
            index,
            num_nodes,
            aggr="max",
            fill_value=0.0,
        ) > 0
        return torch.logical_or(init_tensor.bool(), aggregated)

    if accumulation in (torch.maximum, torch.max):
        if not src.is_floating_point():
            return None
        aggregated = scatter_aggr(
            src,
            index,
            num_nodes,
            aggr="max",
            fill_value=torch.finfo(src.dtype).min,
        )
        return torch.maximum(init_tensor, aggregated)

    return None


def _expand_bucket_init(
    value: Tensor | float | None,
    num_nodes: int,
    feature_shape: tuple[int, ...],
    device: torch.device,
    dtype: torch.dtype,
) -> Tensor:
    full_shape = (num_nodes,) + feature_shape

    if value is None:
        fill = torch.zeros(full_shape, device=device, dtype=dtype)
        if feature_shape:
            fill[..., 0] = FILL_VALUE_MIN
        return fill

    if isinstance(value, Tensor):
        tensor = value.to(device=device, dtype=dtype)
        if tensor.shape == full_shape:
            return tensor.clone()
        if tensor.shape == feature_shape:
            return tensor.expand(full_shape).clone()
        if tensor.dim() == 0:
            return tensor.expand(full_shape).clone()
        raise ValueError(
            f"Cannot broadcast init with shape {tuple(tensor.shape)} to {full_shape}",
        )

    return torch.full(full_shape, float(value), device=device, dtype=dtype)
