"""Shared graph-construction utilities."""

from __future__ import annotations

import torch
from torch import Tensor


def make_grid_graph(
    rows: int,
    cols: int,
    *,
    connectivity: int = 4,
    include_self_loops: bool = True,
) -> tuple[Tensor, int]:
    """Build a 4-connected or 8-connected grid graph.

    Parameters
    ----------
    rows, cols : int
        Grid dimensions.
    connectivity : int
        Grid connectivity (4 for Manhattan, 8 for Chebyshev/chessboard).
    include_self_loops : bool
        If ``True`` (default), every node has an edge to itself.

    Returns
    -------
    edge_index : Tensor [2, E]
    num_nodes : int
    """
    if connectivity not in (4, 8):
        raise ValueError("connectivity must be 4 or 8")

    edges: list[tuple[int, int]] = []
    for r in range(rows):
        for c in range(cols):
            node = r * cols + c
            
            # Horizontal / Vertical (4-connected)
            if c + 1 < cols:
                right = r * cols + (c + 1)
                edges.append((node, right))
                edges.append((right, node))
            if r + 1 < rows:
                below = (r + 1) * cols + c
                edges.append((node, below))
                edges.append((below, node))
                
            # Diagonals (8-connected)
            if connectivity == 8:
                if c + 1 < cols and r + 1 < rows:
                    bottom_right = (r + 1) * cols + (c + 1)
                    edges.append((node, bottom_right))
                    edges.append((bottom_right, node))
                if c - 1 >= 0 and r + 1 < rows:
                    bottom_left = (r + 1) * cols + (c - 1)
                    edges.append((node, bottom_left))
                    edges.append((bottom_left, node))

    if include_self_loops:
        for node in range(rows * cols):
            edges.append((node, node))
    src, tgt = zip(*edges)
    edge_index = torch.tensor([src, tgt], dtype=torch.long)
    return edge_index, rows * cols


def get_grid_distances(
    rows: int,
    cols: int,
    src_r: int,
    src_c: int,
    connectivity: int = 4,
) -> Tensor:
    """Compute exact discrete distances on a grid from a given source.

    Parameters
    ----------
    rows, cols : int
        Grid dimensions.
    src_r, src_c : int
        Source node coordinates.
    connectivity : int
        4 for Manhattan distance, 8 for Chebyshev distance.

    Returns
    -------
    distances : Tensor [rows * cols]
        Flattened 1D tensor of exact distances from the source.
    """
    target = torch.zeros(rows * cols, dtype=torch.float32)
    for r in range(rows):
        for c in range(cols):
            dr = abs(r - src_r)
            dc = abs(c - src_c)
            if connectivity == 4:
                dist = dr + dc  # Manhattan
            else:
                dist = max(dr, dc)  # Chebyshev
            target[r * cols + c] = float(dist)
    return target
