"""Top-level execution contexts for aggregate programs."""

from __future__ import annotations

from contextlib import contextmanager

import torch
from torch import Tensor

from ..pyg_backend import Data, maybe_make_data
from .context import RoundContext
from .stack import with_context


class AggregateContext:
    """Top-level context wrapping a graph and execution state."""

    def __init__(
        self,
        edge_index: Tensor | Data,
        num_nodes: int | None = None,
        edge_weight: Tensor | None = None,
    ) -> None:
        if isinstance(edge_index, Data):
            data = edge_index
            if data.num_nodes is None:
                raise ValueError("PyG Data must define num_nodes for AggregateContext")
            data_edge_weight = edge_weight
            if data_edge_weight is None and hasattr(data, "edge_attr"):
                data_edge_weight = data.edge_attr
            self._ctx = RoundContext(
                data.edge_index, int(data.num_nodes), edge_weight=data_edge_weight
            )
            self._ctx.data = data
            return

        if num_nodes is None:
            raise ValueError(
                "num_nodes is required when constructing AggregateContext from edge_index"
            )
        self._ctx = RoundContext(edge_index, num_nodes, edge_weight=edge_weight)

    @contextmanager
    def round(self):
        """Execute one round of the aggregate program."""
        with with_context(self._ctx), self._ctx.round():
            yield self._ctx

    def reset(self) -> None:
        self._ctx.reset()

    def update_topology(
        self,
        edge_index: Tensor,
        *,
        positions: Tensor | None = None,
        edge_weight: Tensor | None = None,
    ) -> None:
        """Swap the graph in place, keeping the recurrent field state.

        This is the mobile-network execution model: the node set (and its
        :class:`StateManager`) persists while links appear and disappear, so
        stateful fields (``gradient``, ``elect``, ``iterate``) self-heal
        across rounds instead of restarting. When ``positions`` is given and
        ``edge_weight`` is not, metric edge lengths are derived from it
        (detached — routing costs are structure, not a gradient path).
        """
        if edge_weight is None and positions is not None:
            if positions.shape[0] != self._ctx.num_nodes:
                raise ValueError(
                    f"positions has {positions.shape[0]} rows, "
                    f"expected num_nodes={self._ctx.num_nodes}"
                )
            with torch.no_grad():
                src, tgt = edge_index[0], edge_index[1]
                edge_weight = (positions[src] - positions[tgt]).norm(dim=-1)
        if edge_weight is None:
            edge_weight = torch.ones(edge_index.shape[1], device=edge_index.device)
        self._ctx.edge_index = edge_index
        self._ctx.edge_weight = edge_weight
        self._ctx.data = maybe_make_data(edge_index, self._ctx.num_nodes, edge_weight)

    @property
    def round_num(self) -> int:
        return self._ctx.round_num

    @property
    def num_nodes(self) -> int:
        return self._ctx.num_nodes

    @property
    def state(self):
        return self._ctx.state

    def get_state(self, name: str):
        return self._ctx.state.get_state(name=name)
