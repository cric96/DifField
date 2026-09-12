"""Learnable models for distance gradient estimation."""

from __future__ import annotations

import torch
from torch import nn

from ..domain.program import run_gradient_program
from .attention import AttentionMinAggr


class GradientModel(nn.Module):
    """Simple model with a learnable hop weight."""

    def __init__(self, scenario, rounds: int, init_w: float = 3.0):
        super().__init__()
        self.scenario = scenario
        self.rounds = rounds
        self.w = nn.Parameter(torch.tensor(init_w))

    def forward(self, source: torch.Tensor) -> torch.Tensor:
        """Estimate distances from the source."""
        output, _ = run_gradient_program(
            self.scenario, source, rounds=self.rounds, weight=self.w
        )
        return output


class AttentionGradientModel(nn.Module):
    """Model with learnable hop weight and learnable attention aggregator."""

    def __init__(self, scenario, rounds: int):
        super().__init__()
        self.scenario = scenario
        self.rounds = rounds
        self.w = nn.Parameter(torch.tensor(1.5))
        self.attn_aggr = AttentionMinAggr(tau_init=1.0)

    def forward(self, source: torch.Tensor) -> torch.Tensor:
        """Estimate distances using attention aggregation."""
        output, _ = run_gradient_program(
            self.scenario,
            source,
            rounds=self.rounds,
            weight=self.w,
            aggr=self.attn_aggr,
        )
        return output
