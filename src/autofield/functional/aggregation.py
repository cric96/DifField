"""Scatter-based aggregation primitives."""

from __future__ import annotations

import torch
from torch import Tensor

from ..constants import FILL_VALUE_DEFAULT, LOG_EPSILON
from ..pyg_backend import scatter_hard


def scatter_aggr(
    src: Tensor,
    index: Tensor,
    num_nodes: int,
    aggr: str = "sum",
    mode: str = "hard",
    tau: float = 1.0,
    fill_value: float = FILL_VALUE_DEFAULT,
) -> Tensor:
    """Aggregate *src* values into *num_nodes* buckets given by *index*."""
    if callable(aggr) and not isinstance(aggr, str):
        return aggr(src, index, num_nodes)
    if aggr in ("sum", "mean"):
        return _scatter_sum_mean(src, index, num_nodes, aggr)
    if aggr == "min":
        return _scatter_min(src, index, num_nodes, mode, tau, fill_value)
    if aggr == "max":
        return _scatter_max(src, index, num_nodes, mode, tau, fill_value)
    raise ValueError(f"Unknown aggregation: {aggr}")


def _scatter_sum_mean(src: Tensor, index: Tensor, num_nodes: int, aggr: str) -> Tensor:
    return scatter_hard(src, index, num_nodes, aggr=aggr, fill_value=FILL_VALUE_DEFAULT)


def _scatter_min(
    src: Tensor,
    index: Tensor,
    num_nodes: int,
    mode: str,
    tau: float,
    fill_value: float,
) -> Tensor:
    if mode == "hard":
        return scatter_hard(src, index, num_nodes, aggr="min", fill_value=fill_value)
    return _scatter_softmin(src, index, num_nodes, tau, fill_value)


def _scatter_max(
    src: Tensor,
    index: Tensor,
    num_nodes: int,
    mode: str,
    tau: float,
    fill_value: float,
) -> Tensor:
    if mode == "hard":
        return scatter_hard(src, index, num_nodes, aggr="max", fill_value=fill_value)
    return _scatter_softmax(src, index, num_nodes, tau, fill_value)


def _scatter_softmin(src: Tensor, index: Tensor, num_nodes: int, tau: float, fill_value: float) -> Tensor:
    r"""Differentiable soft-min per bucket using the logsumexp trick."""
    neg_src_scaled = -src / tau
    shape = (num_nodes,) + src.shape[1:]

    bucket_max = scatter_hard(
        neg_src_scaled,
        index,
        num_nodes,
        aggr="max",
        fill_value=float("-inf"),
    )
    shifted = neg_src_scaled - bucket_max[index]
    exp_shifted = shifted.exp()
    sum_exp = scatter_hard(exp_shifted, index, num_nodes, aggr="sum", fill_value=0.0)

    has_messages = sum_exp > 0
    log_sum_exp = torch.where(
        has_messages,
        bucket_max + sum_exp.clamp(min=LOG_EPSILON).log(),
        src.new_full(shape, -fill_value / tau),
    )
    result = -tau * log_sum_exp

    msg_count = src.new_zeros(num_nodes)
    msg_count.scatter_add_(0, index, src.new_ones(index.shape[0]))
    is_isolated = msg_count == 0
    if result.dim() > 1:
        is_isolated = is_isolated.unsqueeze(-1)
    return torch.where(is_isolated, src.new_full(shape, fill_value), result)


def _scatter_softmax(src: Tensor, index: Tensor, num_nodes: int, tau: float, fill_value: float) -> Tensor:
    r"""Differentiable soft-max per bucket."""
    return -_scatter_softmin(-src, index, num_nodes, tau, -fill_value)