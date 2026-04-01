"""PyG backend helpers.

PyTorch Geometric is a required runtime dependency for this project.
"""

from __future__ import annotations

import torch
from torch import Tensor

try:
    from torch_geometric.nn import knn_graph, radius_graph
    from torch_geometric.data import Data
    from torch_geometric.utils import grid as pyg_grid
    from torch_geometric.utils import subgraph as pyg_subgraph
    from torch_geometric.utils import scatter as pyg_scatter
except Exception as exc:  # pragma: no cover - import-time guard
    raise RuntimeError(
        "PyTorch Geometric is required. Install `torch-geometric` in the active environment."
    ) from exc

HAS_PYG = True


def pyg_supported() -> bool:
    """PyG support flag (always True in the PyG-only runtime)."""
    return True


def maybe_make_data(
    edge_index: Tensor,
    num_nodes: int,
    edge_weight: Tensor | None = None,
    x: Tensor | None = None,
    pos: Tensor | None = None,
    batch: Tensor | None = None,
    **attrs,
):
    """Create a PyG ``Data`` object with optional node-level metadata."""
    kwargs = {
        "edge_index": edge_index,
        "num_nodes": num_nodes,
    }
    if x is not None:
        kwargs["x"] = x
    if pos is not None:
        kwargs["pos"] = pos
    if batch is not None:
        kwargs["batch"] = batch
    if edge_weight is not None:
        kwargs["edge_attr"] = edge_weight
    kwargs.update(attrs)
    return Data(**kwargs)


def build_grid_edge_index(
    rows: int,
    cols: int,
    *,
    connectivity: int = 4,
    include_self_loops: bool = True,
    device: torch.device | None = None,
) -> Tensor:
    """Build grid edges using PyG utilities and connectivity filtering."""
    if connectivity not in (4, 8):
        raise ValueError("connectivity must be 4 or 8")

    edge_index, pos = pyg_grid(rows, cols, device=device)
    src, tgt = edge_index[0], edge_index[1]
    delta = (pos[src] - pos[tgt]).abs()
    manhattan = delta.sum(dim=-1)
    chebyshev = delta.max(dim=-1).values
    is_self = manhattan == 0

    if connectivity == 4:
        keep = manhattan == 1
    else:
        keep = chebyshev == 1

    if include_self_loops:
        keep = keep | is_self

    return edge_index[:, keep].long()


def build_spatial_edge_index(
    positions: Tensor,
    *,
    edge_radius: float | None = None,
    k_neighbors: int | None = None,
    self_loops: bool = False,
) -> Tensor:
    """Build radius or k-NN directed edges using PyG graph builders.

    When optional torch-cluster kernels are unavailable, fall back to
    deterministic dense-tensor implementations.
    """
    num_nodes = int(positions.shape[0])
    if num_nodes == 0:
        return torch.zeros((2, 0), dtype=torch.long, device=positions.device)

    if k_neighbors is not None:
        if k_neighbors <= 0:
            raise ValueError("k_neighbors must be > 0")
        k = min(int(k_neighbors), max(1, num_nodes - 1 if not self_loops else num_nodes))
        try:
            return knn_graph(positions, k=k, loop=self_loops).long()
        except ImportError:
            dist = torch.cdist(positions, positions)
            if not self_loops:
                dist.fill_diagonal_(float("inf"))
            nn_idx = torch.topk(dist, k=k, dim=1, largest=False).indices
            src = torch.arange(num_nodes, device=positions.device).unsqueeze(1).expand(-1, k)
            return torch.stack([src.reshape(-1), nn_idx.reshape(-1)], dim=0).long()

    if edge_radius is None or edge_radius <= 0:
        raise ValueError("edge_radius must be > 0 when k_neighbors is not used")
    max_nbrs = max(1, num_nodes)
    try:
        return radius_graph(positions, r=float(edge_radius), loop=self_loops, max_num_neighbors=max_nbrs).long()
    except ImportError:
        dist = torch.cdist(positions, positions)
        mask = dist <= float(edge_radius)
        if not self_loops:
            mask.fill_diagonal_(False)
        src, tgt = torch.where(mask)
        return torch.stack([src, tgt], dim=0).long()


def subgraph_for_nodes(
    node_mask: Tensor,
    edge_index: Tensor,
    edge_weight: Tensor | None = None,
) -> tuple[Tensor, Tensor | None]:
    """Return induced subgraph for a node mask without relabeling node ids."""
    out = pyg_subgraph(
        node_mask,
        edge_index,
        edge_attr=edge_weight,
        relabel_nodes=False,
        num_nodes=int(node_mask.numel()),
    )
    return out


def scatter_hard(
    src: Tensor,
    index: Tensor,
    num_nodes: int,
    aggr: str,
    fill_value: float,
) -> Tensor:
    """Hard aggregation via PyG ``scatter``.

    Keeps existing behavior for isolated nodes in min/max with explicit
    ``fill_value`` handling.
    """
    reduce = {
        "sum": "sum",
        "mean": "mean",
        "min": "min",
        "max": "max",
    }.get(aggr)
    if reduce is None:
        msg = f"Unsupported hard aggregation for PyG backend: {aggr}"
        raise ValueError(msg)

    out = pyg_scatter(src, index, dim=0, dim_size=num_nodes, reduce=reduce)
    if aggr in ("min", "max"):
        msg_count = src.new_zeros(num_nodes)
        msg_count.scatter_add_(0, index, src.new_ones(index.shape[0]))
        isolated = msg_count == 0
        if out.dim() > 1:
            isolated = isolated.unsqueeze(-1)
        out = torch.where(isolated, src.new_full(out.shape, fill_value), out)
    return out
