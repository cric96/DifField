"""Shared test utilities for aggregate-program tests."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from collections.abc import Callable
from torch import Tensor, nn

# ---------------------------------------------------------------------------
# Tensor helpers
# ---------------------------------------------------------------------------

def values(*items: float) -> Tensor:
    """Create a float32 tensor from positional arguments."""
    return torch.tensor(items, dtype=torch.float32)


def flags(*items: bool) -> Tensor:
    """Create a bool tensor from positional arguments."""
    return torch.tensor(items, dtype=torch.bool)


def make_source_mask(n: int, *root_indices: int) -> Tensor:
    """Create a boolean source mask with True at the given root indices."""
    mask = torch.zeros(n, dtype=torch.bool)
    for idx in root_indices:
        mask[idx] = True
    return mask


# ---------------------------------------------------------------------------
# Round helpers
# ---------------------------------------------------------------------------

def collect_round_outputs(ctx, rounds: int, program: Callable[[], Tensor]) -> list[Tensor]:
    """Run a program for *rounds* rounds and collect cloned outputs."""
    outputs: list[Tensor] = []
    for _ in range(rounds):
        with ctx.round():
            output = program()
        outputs.append(output.detach().clone())
    return outputs


# ---------------------------------------------------------------------------
# Common constants
# ---------------------------------------------------------------------------

ROUNDS = 3
GRADIENT_ROUNDS = 4
PROPAGATION_ROUNDS = 4  # Rounds needed for full convergence on line topology
SOFT_BRANCH_TAU = 0.05
SOFT_BRANCH_MAX_ERROR = 26.0
SOFT_MATCH_TAU = 0.05
SOFT_MATCH_ATOL = 1e-3

# Source override specs for line topology (node 0 as root)
LINE_SOURCE = ((0, 1.0),)

# Source override specs for weighted-collect topology (nodes 0 and 3 as roots)
WEIGHTED_SOURCES = ((0, 1.0), (3, 1.0))


# ---------------------------------------------------------------------------
# Reusable nn.Module subclasses
# ---------------------------------------------------------------------------

class AddConstant(nn.Module):
    """Simple module that adds a scalar constant to its input."""

    def __init__(self, value: float):
        super().__init__()
        self.value = value

    def forward(self, x: Tensor, ctx=None) -> Tensor:
        return x + self.value
