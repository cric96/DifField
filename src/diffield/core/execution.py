"""Top-level execution contexts for aggregate programs."""

from __future__ import annotations

from contextlib import contextmanager

from torch import Tensor

from ..pyg_backend import Data
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
        with with_context(self._ctx):
            with self._ctx.round():
                yield self._ctx

    def reset(self) -> None:
        self._ctx.reset()

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
