"""Edge masking helpers used by conditional constructs."""

from __future__ import annotations

import torch
from torch import Tensor

from ..constants import CONDITION_THRESHOLD, DEFAULT_TAU_BRANCH
from ..pyg_backend import subgraph_for_nodes


def mask_edges(
    edge_index: Tensor,
    cond: Tensor,
    mode: str = "hard",
    tau: float = DEFAULT_TAU_BRANCH,
) -> tuple[Tensor, Tensor]:
    r"""Keep only intra-partition edges based on a per-node boolean/float condition."""
    src, tgt = edge_index[0], edge_index[1]
    c_src = cond[src].float()
    c_tgt = cond[tgt].float()

    if mode == "hard":
        same = cond[src] == cond[tgt]
        kept = same.nonzero(as_tuple=True)[0]
        return edge_index[:, kept], edge_index.new_ones(kept.shape[0], dtype=torch.float32)

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
    r"""Keep only edges where BOTH endpoints belong to *partition*."""
    src, tgt = edge_index[0], edge_index[1]

    if mode == "hard":
        node_mask = cond.bool() if partition else ~cond.bool()
        edge_index_out, edge_weight_out = subgraph_for_nodes(
            node_mask, edge_index, edge_weight=edge_weight
        )
        message_weight_out = None
        if message_weight is not None:
            _, message_weight_out = subgraph_for_nodes(
                node_mask, edge_index, edge_weight=message_weight
            )
        return edge_index_out, edge_weight_out, message_weight_out

    p_src = cond[src].float() if partition else (1.0 - cond[src].float())
    p_tgt = cond[tgt].float() if partition else (1.0 - cond[tgt].float())
    p_both = p_src * p_tgt

    mask_weight = torch.sigmoid(tau * (p_both - CONDITION_THRESHOLD))
    if message_weight is not None:
        mask_weight = message_weight * mask_weight
    return edge_index, edge_weight, mask_weight
