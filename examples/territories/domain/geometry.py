"""Geometric primitives for grid-based territories."""

from __future__ import annotations

import torch


def meshgrid(
    rows: int, cols: int, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor]:
    """Create a 2D coordinate meshgrid normalized to [0, 1]."""
    row_coords = torch.linspace(0.0, 1.0, rows, device=device)
    col_coords = torch.linspace(0.0, 1.0, cols, device=device)
    grid_row, grid_col = torch.meshgrid(row_coords, col_coords, indexing="ij")
    return grid_row, grid_col


def squared_distance(a: tuple[int, int], b: tuple[int, int]) -> int:
    """Compute squared Euclidean distance between two grid points."""
    return (a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2


def interior_index(size: int, ratio: float) -> int:
    """Map a [0, 1] ratio to a grid index, staying away from boundaries if possible."""
    if size <= 2:
        return max(0, min(size - 1, int(round(ratio * max(size - 1, 0)))))
    lower = 1
    upper = size - 2
    raw = int(round(ratio * upper))
    return max(lower, min(upper, raw))


def grid_position(
    rows: int, cols: int, row_ratio: float, col_ratio: float
) -> tuple[int, int]:
    """Map normalized coordinates to an interior grid position."""
    return (
        interior_index(rows, row_ratio),
        interior_index(cols, col_ratio),
    )


def seeded_value(seed: int, index: int, scale: float) -> float:
    """Deterministic random value for field generation."""
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed) * 997 + index * 131)
    return float((torch.rand((), generator=generator).item() * 2.0 - 1.0) * scale)
