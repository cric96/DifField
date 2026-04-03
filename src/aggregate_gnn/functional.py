"""Low-level differentiable operations for aggregate-GNN equivalence.

Provides scatter-based aggregation, edge masking, and smooth approximations
to non-differentiable operations (min, max, conditional selection).
"""

from __future__ import annotations

import operator
from typing import Callable

import torch
from torch import Tensor

from .constants import (
    CONDITION_THRESHOLD,
    DEFAULT_TAU_BRANCH,
    FILL_VALUE_DEFAULT,
    FILL_VALUE_MAX,
    FILL_VALUE_MIN,
    LOG_EPSILON,
)
from .pyg_backend import scatter_hard, subgraph_for_nodes


# ---------------------------------------------------------------------------
# Scatter-based aggregation
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Scatter-based aggregation
# ---------------------------------------------------------------------------

def scatter_aggr(
    src: Tensor,
    index: Tensor,
    num_nodes: int,
    aggr: str = "sum",
    mode: str = "hard",
    tau: float = 1.0,
    fill_value: float = FILL_VALUE_DEFAULT,
) -> Tensor:
    """Aggregate *src* values into *num_nodes* buckets given by *index*.

    Equivalent to:  ``out[i] = ⊕_{k : index[k]=i} src[k]``

    Parameters
    ----------
    src : Tensor [E, *F]
        Source values (one per edge / message).
    index : Tensor [E]  (long)
        Target node index for each source value.
    num_nodes : int
        Total number of nodes (output size along dim-0).
    aggr : str or callable
        Built-in: ``"sum"``, ``"mean"``, ``"min"``, ``"max"``.
        Custom: a callable ``aggr(msg, index, num_nodes) -> Tensor [N, *F]``
        that reduces *msg* [E, *F] by *index* [E] into *num_nodes* buckets.
        When a callable is given, *mode*, *tau*, and *fill_value* are ignored.
    mode : str
        ``"hard"`` – exact aggregation (min/max use ``scatter_reduce_``).
        ``"soft"`` – differentiable smooth approximation for min/max.
    tau : float
        Temperature for soft min/max (lower → closer to hard).
    fill_value : float
        Initial fill value for the output.  Important for min (use ``+∞``)
        and max (use ``-∞``) so that isolated nodes get a sensible default.
    """
    if callable(aggr) and not isinstance(aggr, str):
        return aggr(src, index, num_nodes)
    if aggr in ("sum", "mean"):
        return _scatter_sum_mean(src, index, num_nodes, aggr)
    if aggr == "min":
        return _scatter_min(src, index, num_nodes, mode, tau, fill_value)
    if aggr == "max":
        return _scatter_max(src, index, num_nodes, mode, tau, fill_value)
    raise ValueError(f"Unknown aggregation: {aggr}")


def _scatter_sum_mean(
    src: Tensor, index: Tensor, num_nodes: int, aggr: str,
) -> Tensor:
    return scatter_hard(src, index, num_nodes, aggr=aggr, fill_value=FILL_VALUE_DEFAULT)


def _scatter_min(
    src: Tensor, index: Tensor, num_nodes: int,
    mode: str, tau: float, fill_value: float,
) -> Tensor:
    if mode == "hard":
        return scatter_hard(src, index, num_nodes, aggr="min", fill_value=fill_value)
    return _scatter_softmin(src, index, num_nodes, tau, fill_value)


def _scatter_max(
    src: Tensor, index: Tensor, num_nodes: int,
    mode: str, tau: float, fill_value: float,
) -> Tensor:
    if mode == "hard":
        return scatter_hard(src, index, num_nodes, aggr="max", fill_value=fill_value)
    return _scatter_softmax(src, index, num_nodes, tau, fill_value)


def _scatter_softmin(
    src: Tensor, index: Tensor, num_nodes: int, tau: float, fill_value: float,
) -> Tensor:
    r"""Differentiable soft-min per bucket using the logsumexp trick.

    .. math::

        \text{softmin}_{\tau}(\mathbf{x})_i
            = -\tau \;\ln \sum_{j \in \mathcal{B}(i)}
              \exp\!\bigl(-x_j / \tau\bigr)

    For numerical stability we apply the *log-sum-exp* identity:

    .. math::

        \ln \sum_j e^{z_j}
            = m + \ln \sum_j e^{z_j - m},
        \qquad m = \max_j z_j

    Steps
    -----
    1. Negate & scale:   ``z_j = -x_j / τ``
    2. Per-bucket max:   ``m_i = max_{j ∈ B(i)} z_j``
    3. Shift:            ``z'_j = z_j - m_{index[j]}``
    4. Exponentiate:     ``e_j = exp(z'_j)``
    5. Scatter-sum:      ``S_i = Σ_{j ∈ B(i)} e_j``
    6. Log + unshift:    ``logsumexp_i = m_i + log(S_i)``
    7. Final:            ``result_i = -τ · logsumexp_i``
    """
    # Step 1: negate & scale
    neg_src_scaled = -src / tau  # [E, *F]

    shape = (num_nodes,) + src.shape[1:]

    # Step 2: per-bucket max for numerical stability
    bucket_max = scatter_hard(
        neg_src_scaled,
        index,
        num_nodes,
        aggr="max",
        fill_value=float("-inf"),
    )

    # Steps 3-4: shift by bucket max, then exponentiate
    shifted = neg_src_scaled - bucket_max[index]
    exp_shifted = shifted.exp()

    # Step 5: scatter-sum of exponentials per bucket
    sum_exp = scatter_hard(
        exp_shifted,
        index,
        num_nodes,
        aggr="sum",
        fill_value=0.0,
    )

    # Step 6: log-sum-exp = max + log(sum_exp)
    has_messages = sum_exp > 0
    log_sum_exp = torch.where(
        has_messages,
        bucket_max + sum_exp.clamp(min=LOG_EPSILON).log(),
        src.new_full(shape, -fill_value / tau),
    )

    # Step 7: multiply by -τ to recover the soft-min
    result = -tau * log_sum_exp

    # Isolated nodes (no incoming messages) retain *fill_value*
    msg_count = src.new_zeros(num_nodes)
    msg_count.scatter_add_(0, index, src.new_ones(index.shape[0]))
    is_isolated = msg_count == 0
    if result.dim() > 1:
        is_isolated = is_isolated.unsqueeze(-1)
    return torch.where(is_isolated, src.new_full(shape, fill_value), result)


def _scatter_softmax(
    src: Tensor, index: Tensor, num_nodes: int, tau: float, fill_value: float,
) -> Tensor:
    r"""Differentiable soft-max per bucket.

    Uses the identity:  ``softmax_τ(x) = −softmin_τ(−x)``.
    """
    return -_scatter_softmin(-src, index, num_nodes, tau, -fill_value)


def scatter_min_by_first(
    src: Tensor,
    index: Tensor,
    num_nodes: int,
    *,
    mode: str = "hard",
    tau: float = 1.0,
    fill_row: Tensor | None = None,
) -> Tensor:
    """Select the full row whose first component is minimal in each bucket.

    This is the tensor analogue of the Scala helper ``minByFirst`` used by
    building blocks such as gradient-cast, where the bucket key is a scalar
    distance and the remaining columns carry payload data.
    """
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

    effective_tau = max(float(tau), LOG_EPSILON)
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

    logits = -cost / effective_tau
    bucket_max = scatter_hard(logits, index, num_nodes, aggr="max", fill_value=float("-inf"))
    score = (logits - bucket_max[index]).exp()
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
    """Fold bucketed values with a binary accumulation function.

    The function assumes the reducer is associative enough for the chosen edge
    ordering to be meaningful, matching the standard aggregate-computing use
    case for collection building blocks.
    """
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


# ---------------------------------------------------------------------------
# Edge masking for branch
# ---------------------------------------------------------------------------

def mask_edges(
    edge_index: Tensor,
    cond: Tensor,
    mode: str = "hard",
    tau: float = DEFAULT_TAU_BRANCH,
) -> tuple[Tensor, Tensor]:
    r"""Keep only intra-partition edges based on a per-node boolean/float condition.

    Hard mode keeps edge (i,j) iff ``cond[i] == cond[j]``.

    Soft mode computes a continuous edge weight:

    .. math::

        P(\text{same}) = c_i \, c_j + (1 - c_i)(1 - c_j)

        w_{ij} = \sigma\!\bigl(\tau \cdot (P(\text{same}) - 0.5)\bigr)

    Parameters
    ----------
    edge_index : Tensor [2, E]
    cond : Tensor [N]  (boolean for hard, float in [0,1] for soft)
    mode : ``"hard"`` | ``"soft"``
    tau : temperature for soft mode

    Returns
    -------
    (edge_index_out, edge_weight) where edge_weight is 1.0 for kept edges (hard)
    or a continuous weight in [0,1] (soft).
    """
    src, tgt = edge_index[0], edge_index[1]
    c_src = cond[src].float()
    c_tgt = cond[tgt].float()

    if mode == "hard":
        same = (cond[src] == cond[tgt])
        kept = same.nonzero(as_tuple=True)[0]
        return edge_index[:, kept], edge_index.new_ones(kept.shape[0], dtype=torch.float32)

    # P(same partition) = c_s·c_t + (1−c_s)(1−c_t)
    p_same = c_src * c_tgt + (1 - c_src) * (1 - c_tgt)
    weight = torch.sigmoid(tau * (p_same - CONDITION_THRESHOLD))
    return edge_index, weight


def mask_edges_for_partition(
    edge_index: Tensor,
    cond: Tensor,
    partition: bool,
    edge_weight: Tensor | None = None,
    message_weight: Tensor | None = None,
    mode: str = "hard",
    tau: float = DEFAULT_TAU_BRANCH,
) -> tuple[Tensor, Tensor | None, Tensor | None]:
    r"""Keep only edges where BOTH endpoints belong to *partition*.

    Soft mode computes:

    .. math::

        P(\text{both in } k) = P_s \cdot P_t, \qquad
        w_{ij} = \sigma\!\bigl(\tau (P - 0.5)\bigr)

    where :math:`P_s = c_s` if *partition* is True, else :math:`1 - c_s`.

    Parameters
    ----------
    edge_index : Tensor [2, E]
    cond : Tensor [N]
    partition : bool (True for 'if_true' branch, False for 'if_false')
    edge_weight : Tensor [E], optional
        Edge metric / cost kept on the subgraph.
    message_weight : Tensor [E], optional
        Optional multiplicative transport weight to keep / compose.
    """
    src, tgt = edge_index[0], edge_index[1]

    if mode == "hard":
        node_mask = cond.bool() if partition else ~cond.bool()
        ei_out, ew_out = subgraph_for_nodes(node_mask, edge_index, edge_weight=edge_weight)
        mw_out = None
        if message_weight is not None:
            _, mw_out = subgraph_for_nodes(node_mask, edge_index, edge_weight=message_weight)
        return ei_out, ew_out, mw_out

    # Soft mode: P(both in partition) = P(src ∈ k) · P(tgt ∈ k)
    p_src = cond[src].float() if partition else (1.0 - cond[src].float())
    p_tgt = cond[tgt].float() if partition else (1.0 - cond[tgt].float())
    p_both = p_src * p_tgt

    mask_weight = torch.sigmoid(tau * (p_both - CONDITION_THRESHOLD))
    if message_weight is not None:
        mask_weight = message_weight * mask_weight
    return edge_index, edge_weight, mask_weight


# ---------------------------------------------------------------------------
# Conditional selection (mux)
# ---------------------------------------------------------------------------

def soft_where(cond: Tensor, x: Tensor, y: Tensor) -> Tensor:
    r"""Differentiable pointwise conditional selection.

    Selects ``x`` where ``cond >= CONDITION_THRESHOLD`` (0.5), else ``y``.

    Uses ``torch.where`` which is safe with ``inf``/``nan`` (avoids the
    ``0 * inf = nan`` pitfall of the linear interpolation ``c*x + (1-c)*y``).
    Differentiable w.r.t. both *x* and *y*.
    """
    cond_float = cond.float()
    if cond_float.dim() < x.dim():
        cond_float = cond_float.unsqueeze(-1)
    return torch.where(cond_float >= CONDITION_THRESHOLD, x, y)
