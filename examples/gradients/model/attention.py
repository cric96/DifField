"""Attention-based minimum aggregator for gradients."""

from __future__ import annotations

import torch
from torch import nn
from torch_geometric.utils import scatter as pyg_scatter
from torch_geometric.utils import softmax as pyg_softmax

LEAKY_RELU_SLOPE = 0.2
MASKED_LOGIT = -1e9


class AttentionMinAggr(nn.Module):
    """Learnable soft-min aggregator using an attention mechanism."""

    def __init__(self, tau_init: float = 1.0):
        super().__init__()
        self.a = nn.Parameter(torch.tensor(1.0))
        self.b = nn.Parameter(torch.tensor(0.0))
        self._log_tau = nn.Parameter(torch.tensor(float(tau_init)).log())

    @property
    def tau(self) -> torch.Tensor:
        """Temperature parameter for softmax."""
        return self._log_tau.exp()

    def forward(
        self, msg: torch.Tensor, index: torch.Tensor, num_nodes: int
    ) -> torch.Tensor:
        """Aggregate messages using attention weights."""
        tau = self.tau
        finite = msg.isfinite()
        safe_msg = torch.where(finite, msg, torch.zeros_like(msg))

        score = torch.nn.functional.leaky_relu(
            self.a * safe_msg + self.b, LEAKY_RELU_SLOPE
        )
        neg_score = -score / tau
        neg_score = torch.where(
            finite, neg_score, torch.full_like(neg_score, MASKED_LOGIT)
        )

        alpha = pyg_softmax(neg_score, index=index, num_nodes=num_nodes)
        out = pyg_scatter(
            alpha * safe_msg, index=index, dim=0, dim_size=num_nodes, reduce="sum"
        )

        has_finite = pyg_scatter(
            finite.float(), index=index, dim=0, dim_size=num_nodes, reduce="sum"
        )
        return torch.where(
            has_finite > 0, out, torch.tensor(float("inf"), device=msg.device)
        )
