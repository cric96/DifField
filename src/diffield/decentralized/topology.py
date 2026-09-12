"""Topology helpers shared by the decentralised backends.

A decentralised run needs, for every device, the list of devices whose messages
reach it and the range of each of those links.  Centrally that information is a
flat ``edge_index``/``edge_weight`` pair; locally it is a neighbour list.
"""

from __future__ import annotations

import torch
from torch import Tensor


def in_edges_of(
    node: int,
    edge_index: Tensor,
    edge_weight: Tensor | None = None,
) -> tuple[list[int], list[float]]:
    """Neighbours whose messages reach *node*, and the range of each link.

    Self-loops are dropped: :class:`~diffield.core.device.DeviceContext` models
    them separately through ``self_loop``.  Neighbours come back sorted by id so
    that the local star graph has a deterministic, reproducible layout.
    """
    mask = (edge_index[1] == node) & (edge_index[0] != node)
    positions = mask.nonzero(as_tuple=True)[0]
    sources = edge_index[0, positions]
    order = torch.argsort(sources)
    positions = positions[order]

    neighbor_ids = edge_index[0, positions].tolist()
    if edge_weight is None:
        ranges = [1.0] * len(neighbor_ids)
    else:
        ranges = edge_weight[positions].tolist()
    return neighbor_ids, ranges


def all_in_edges(
    edge_index: Tensor,
    num_nodes: int,
    edge_weight: Tensor | None = None,
) -> list[tuple[list[int], list[float]]]:
    """:func:`in_edges_of` for every node at once, in one pass.

    Moving devices means recomputing the whole neighbourhood table every round,
    which is why this exists rather than a loop over :func:`in_edges_of`.
    """
    sources, targets = edge_index[0], edge_index[1]
    keep = sources != targets
    sources, targets = sources[keep], targets[keep]
    weights = (
        edge_weight[keep]
        if edge_weight is not None
        else torch.ones(sources.shape[0], dtype=torch.float32)
    )

    # Sort by target, then by source: the same order in_edges_of produces.
    order = torch.argsort(targets * num_nodes + sources)
    sources, targets, weights = sources[order], targets[order], weights[order]

    table: list[tuple[list[int], list[float]]] = [([], []) for _ in range(num_nodes)]
    for source, target, weight in zip(
        sources.tolist(), targets.tolist(), weights.tolist()
    ):
        neighbor_ids, ranges = table[target]
        neighbor_ids.append(source)
        ranges.append(weight)
    return table


def require_symmetric(edge_index: Tensor) -> None:
    """Raise unless every link is bidirectional.

    The decentralised backends hand the topology to a simulator that only knows
    undirected adjacency, so a one-way link would silently become two-way.
    Every graph builder in :mod:`diffield.pyg_backend` emits both directions.
    """
    forward = {(int(s), int(t)) for s, t in edge_index.t().tolist()}
    missing = [(s, t) for s, t in forward if s != t and (t, s) not in forward]
    if missing:
        raise ValueError(
            "decentralised execution needs a symmetric topology; "
            f"{len(missing)} one-way link(s), e.g. {missing[:3]}"
        )


def edge_index_to_networkx(edge_index: Tensor, num_nodes: int):
    """Build the undirected graph a simulator can navigate from *edge_index*."""
    import networkx as nx

    require_symmetric(edge_index)
    graph = nx.Graph()
    graph.add_nodes_from(range(num_nodes))
    for source, target in edge_index.t().tolist():
        if source != target:
            graph.add_edge(int(source), int(target))
    return graph
