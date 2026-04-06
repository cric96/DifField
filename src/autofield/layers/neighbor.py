"""Neighborhood message-passing layers."""

from __future__ import annotations

from typing import Callable, Optional

import torch
import torch.nn as nn
from torch import Tensor
from torch_geometric.nn import MessagePassing

from ..constants import DEFAULT_TAU_SOFT_AGGR, FILL_VALUE_DEFAULT, FILL_VALUE_MAX, FILL_VALUE_MIN
from ..core import RoundContext, resolve_context
from ..functional import scatter_aggr
from ..dsl.helpers import edge_sources_targets, scale_messages
from .common import register_callable


class _PyGNbrMessagePassing(MessagePassing):
    """PyG MessagePassing wrapper that preserves aggregate semantics."""

    def __init__(self, owner: "NbrLayer") -> None:
        use_builtin_aggr = (
            isinstance(owner.aggr, str)
            and owner.mode == "hard"
            and owner.aggr in {"sum", "mean"}
        )
        super().__init__(
            aggr=owner.aggr if use_builtin_aggr else None,
            flow="source_to_target",
            node_dim=0,
        )
        self.owner = owner
        self._use_builtin_aggr = use_builtin_aggr

    def forward(
        self,
        x: Tensor,
        edge_index: Tensor,
        message_weight: Tensor | None,
        num_nodes: int,
    ) -> Tensor:
        return self.propagate(
            edge_index,
            x=x,
            message_weight=message_weight,
            size=(num_nodes, num_nodes),
        )

    def message(self, x_j: Tensor, message_weight: Tensor | None = None) -> Tensor:
        msg = x_j
        if self.owner.transform_fn is not None:
            msg = self.owner.transform_fn(msg)
        if message_weight is not None:
            weight = message_weight.unsqueeze(-1) if msg.dim() > 1 else message_weight
            msg = msg * weight
        return msg

    def aggregate(
        self,
        inputs: Tensor,
        index: Tensor,
        ptr: Tensor | None = None,
        dim_size: int | None = None,
    ) -> Tensor:
        if self._use_builtin_aggr:
            return super().aggregate(inputs, index, ptr=ptr, dim_size=dim_size)
        if dim_size is None:
            raise ValueError("PyG propagate did not provide dim_size for aggregation")
        return scatter_aggr(
            inputs,
            index,
            dim_size,
            aggr=self.owner.aggr,
            mode=self.owner.mode,
            tau=self.owner.tau,
            fill_value=self.owner.fill_value,
        )


class NbrLayer(nn.Module):
    r"""Neighborhood aggregation: ``m_i = ⊕_{j∈N(i)} φ(x_j)``."""

    def __init__(
        self,
        aggr: str | Callable = "sum",
        transform_fn: Optional[Callable[[Tensor], Tensor] | nn.Module] = None,
        mode: str = "hard",
        tau: float = DEFAULT_TAU_SOFT_AGGR,
        fill_value: float | None = None,
        tag: str | None = None,
        include_self: bool | None = None,
    ) -> None:
        super().__init__()
        self.aggr = aggr
        self.mode = mode
        self.tau = tau
        self.tag = tag
        self.include_self = include_self
        if fill_value is None:
            if isinstance(aggr, str):
                self.fill_value = (
                    FILL_VALUE_MIN if aggr == "min"
                    else FILL_VALUE_MAX if aggr == "max"
                    else FILL_VALUE_DEFAULT
                )
            else:
                self.fill_value = FILL_VALUE_DEFAULT
        else:
            self.fill_value = fill_value

        if transform_fn is not None:
            register_callable(self, "transform_fn", transform_fn, "_transform_fn")
        else:
            self.transform_fn = None
        self._mp = _PyGNbrMessagePassing(self)

    def forward(
        self,
        x: Tensor,
        ctx: RoundContext | None = None,
        edge_index: Tensor | None = None,
        edge_weight: Tensor | None = None,
        tag: str | None = None,
    ) -> Tensor:
        """Aggregate neighbor features."""
        ctx = resolve_context(ctx)
        effective_tag = tag if tag is not None else self.tag

        from ..dsl.neighbor import NeighborExpr

        if isinstance(x, NeighborExpr) and effective_tag is not None:
            raise ValueError("Tagged nbr is not supported for edge-wise neighbor expressions")

        if effective_tag is not None:
            ctx.exports[effective_tag] = x

        src = x
        if effective_tag is not None and effective_tag in ctx._neighbor_message_overrides:
            src = ctx._neighbor_message_overrides[effective_tag]

        edge_idx = edge_index if edge_index is not None else ctx.edge_index
        message_weight = edge_weight if edge_weight is not None else ctx.message_weight

        if self.include_self is None and not isinstance(src, NeighborExpr):
            return self._mp(src, edge_idx, message_weight, ctx.num_nodes)

        messages, target_index = self._build_messages(
            src,
            ctx=ctx,
            edge_idx=edge_idx,
            message_weight=message_weight,
        )
        return self._aggregate_messages(messages, target_index, ctx.num_nodes)

    def _build_messages(
        self,
        src: Tensor | "NeighborExpr",
        *,
        ctx: RoundContext,
        edge_idx: Tensor,
        message_weight: Tensor | None,
    ) -> tuple[Tensor, Tensor]:
        from ..dsl.neighbor import NeighborExpr

        source_index, target_index = edge_sources_targets(edge_idx)
        include_self = self.include_self

        if include_self is None:
            if isinstance(src, NeighborExpr):
                messages = src.evaluate(
                    ctx=ctx,
                    edge_index=edge_idx,
                    edge_weight=ctx.edge_weight,
                )
                if message_weight is not None:
                    messages = scale_messages(messages, message_weight)
                return messages, target_index

            messages = src[source_index]
            if self.transform_fn is not None:
                messages = self.transform_fn(messages)
            if message_weight is not None:
                messages = scale_messages(messages, message_weight)
            return messages, target_index

        keep_mask = source_index != target_index
        kept_edge_idx = edge_idx[:, keep_mask]
        kept_target_index = target_index[keep_mask]
        kept_message_weight = message_weight[keep_mask] if message_weight is not None else None

        if isinstance(src, NeighborExpr):
            messages = src.evaluate(
                ctx=ctx,
                edge_index=kept_edge_idx,
                edge_weight=ctx.edge_weight[keep_mask] if ctx.edge_weight is not None else None,
            )
            if kept_message_weight is not None:
                messages = scale_messages(messages, kept_message_weight)
        else:
            kept_source_index = kept_edge_idx[0]
            messages = src[kept_source_index]
            if self.transform_fn is not None:
                messages = self.transform_fn(messages)
            if kept_message_weight is not None:
                messages = scale_messages(messages, kept_message_weight)

        if include_self is True:
            self_messages = self._self_messages(src, ctx)
            messages = torch.cat((messages, self_messages), dim=0)
            self_targets = torch.arange(ctx.num_nodes, device=target_index.device, dtype=target_index.dtype)
            kept_target_index = torch.cat((kept_target_index, self_targets), dim=0)

        return messages, kept_target_index

    def _self_messages(self, src: Tensor | "NeighborExpr", ctx: RoundContext) -> Tensor:
        from ..dsl.neighbor import NeighborExpr

        if isinstance(src, NeighborExpr):
            self_index = torch.arange(ctx.num_nodes, device=ctx.edge_index.device, dtype=torch.long)
            self_edge_index = torch.stack((self_index, self_index), dim=0)
            self_edge_weight = torch.zeros(
                ctx.num_nodes,
                device=ctx.edge_index.device,
                dtype=ctx.edge_weight.dtype if ctx.edge_weight is not None else torch.float32,
            )
            return src.evaluate(ctx=ctx, edge_index=self_edge_index, edge_weight=self_edge_weight)

        self_messages = src
        if self.transform_fn is not None:
            self_messages = self.transform_fn(self_messages)
        return self_messages

    def _aggregate_messages(self, messages: Tensor, target_index: Tensor, num_nodes: int) -> Tensor:
        return scatter_aggr(
            messages,
            target_index,
            num_nodes,
            aggr=self.aggr,
            mode=self.mode,
            tau=self.tau,
            fill_value=self.fill_value,
        )