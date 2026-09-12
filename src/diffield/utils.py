"""Shared graph-construction utilities."""

from __future__ import annotations

import torch
from torch import Tensor

from .pyg_backend import build_grid_edge_index


def make_grid_graph(
    rows: int,
    cols: int,
    *,
    connectivity: int = 4,
    include_self_loops: bool = False,
) -> tuple[Tensor, int]:
    """Build a 4-connected or 8-connected grid graph.

    Parameters
    ----------
    rows, cols : int
        Grid dimensions.
    connectivity : int
        Grid connectivity (4 for Manhattan, 8 for Chebyshev/chessboard).
    include_self_loops : bool
        If ``True``, every node has an edge to itself.  Off by default: a
        graph self-loop would carry the default edge weight of 1.0, which
        misreports a device as being at range 1 from itself.  To fold a node's
        own value into a reduction, pass ``include_self=True`` to
        :func:`~diffield.dsl.primitives.gather`, which uses range 0 for it.

    Returns
    -------
    edge_index : Tensor [2, E]
    num_nodes : int
    """
    edge_index = build_grid_edge_index(
        rows,
        cols,
        connectivity=connectivity,
        include_self_loops=include_self_loops,
    )
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


def get_device(device_str: str = "") -> torch.device:
    """Get the appropriate torch device based on a string or availability.

    Parameters
    ----------
    device_str : str
        The device string (e.g., "cuda", "cpu"). If empty, chooses "cuda" if
        available, else "cpu".

    Returns
    -------
    device : torch.device
    """
    if device_str:
        return torch.device(device_str)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")
