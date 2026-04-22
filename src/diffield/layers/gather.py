"""Neighborhood aggregation layers."""

from __future__ import annotations

from typing import Callable, Optional, TYPE_CHECKING

import torch
import torch.nn as nn
from torch import Tensor
from torch_geometric.nn import MessagePassing

from ..constants import (
    DEFAULT_TAU_SOFT_AGGR,
    FILL_VALUE_DEFAULT,
    FILL_VALUE_MAX,
    FILL_VALUE_MIN,
)
from ..core import RoundContext, resolve_context
from ..functional import scatter_aggr
from ..dsl.helpers import edge_sources_targets

if TYPE_CHECKING:
    from ..dsl.scattering import LinkField


class _PyGMessagePassing(MessagePassing):
    """PyG MessagePassing wrapper with built-in and custom aggregation paths."""

    def __init__(self, owner: "GatherLayer") -> None:
        use_builtin = (
            isinstance(owner.aggr, str)
            and owner.mode == "hard"
            and owner.aggr in {"sum", "mean"}
        )
        super().__init__(
            aggr=owner.aggr if use_builtin else None,
            flow="source_to_target",
            node_dim=0,
        )
        self.owner = owner
        self._use_builtin = use_builtin

    def forward(
        self,
        x: Tensor,
        edge_index: Tensor,
        num_nodes: int,
    ) -> Tensor:
        return self.propagate(
            edge_index,
            x=x,
            size=(num_nodes, num_nodes),
        )

    def message(self, x_j: Tensor) -> Tensor:
        return x_j

    def aggregate(
        self,
        inputs: Tensor,
        index: Tensor,
        ptr: Tensor | None = None,
        dim_size: int | None = None,
    ) -> Tensor:
        if self._use_builtin:
            return super().aggregate(inputs, index, ptr=ptr, dim_size=dim_size)
        if dim_size is None:
            raise ValueError("PyG propagate did not provide dim_size")
        return scatter_aggr(
            inputs,
            index,
            dim_size,
            aggr=self.owner.aggr,
            mode=self.owner.mode,
            tau=self.owner.tau,
            fill_value=self.owner.fill_value,
        )


class GatherLayer(nn.Module):
    r"""Gather neighbours and fold into a single field.

    ``m_i = ⊕_{j∈N(i)} msg_{j→i}`` where *msg* comes from a
    :class:`~diffield.dsl.scattering.LinkField` or a plain tensor.
    """

    def __init__(
        self,
        aggr: str | Callable | nn.Module = "sum",
        mode: str = "hard",
        tau: float = DEFAULT_TAU_SOFT_AGGR,
        fill_value: float | None = None,
        include_self: bool | None = None,
    ) -> None:
        super().__init__()
        self.aggr = aggr
        self.mode = mode
        self.tau = tau
        self.include_self = include_self

        if fill_value is None:
            if isinstance(aggr, str):
                self.fill_value = (
                    FILL_VALUE_MIN
                    if aggr == "min"
                    else FILL_VALUE_MAX
                    if aggr == "max"
                    else FILL_VALUE_DEFAULT
                )
            else:
                self.fill_value = FILL_VALUE_DEFAULT
        else:
            self.fill_value = fill_value

        if isinstance(aggr, nn.Module):
            self._aggr_module = aggr
        else:
            self._aggr_module = None

        self._mp = _PyGMessagePassing(self)

    # ------------------------------------------------------------------
    # forward
    # ------------------------------------------------------------------
    def forward(
        self,
        x: Tensor | LinkField,
        ctx: RoundContext | None = None,
        edge_index: Tensor | None = None,
        tag: str | None = None,
    ) -> Tensor:
        ctx = resolve_context(ctx)
        effective_tag = tag if tag is not None else getattr(x, "tag", None)

        from ..dsl.scattering import LinkField

        if effective_tag is not None:
            # We can only export if we have a source field (i.e. not a complex expression)
            source_field = getattr(x, "source_field", None)
            if source_field is not None:
                ctx.exports[effective_tag] = source_field
            elif not isinstance(x, LinkField):
                ctx.exports[effective_tag] = x

        src = x
        if effective_tag is not None:
            override = ctx.get_message_override(effective_tag)
            if override is not None:
                src = override

        edge_idx = edge_index if edge_index is not None else ctx.edge_index

        # Fast path: plain tensor, include_self not set, simple aggr, AND no message weight
        if (
            self.include_self is None 
            and not isinstance(src, LinkField) 
            and ctx.message_weight is None
        ):
            return self._mp(src, edge_idx, ctx.num_nodes)

        messages, target_index = self._build_messages(
            src,
            ctx=ctx,
            edge_idx=edge_idx,
        )
        return self._aggregate_messages(messages, target_index, ctx.num_nodes)

    # ------------------------------------------------------------------
    # message building
    # ------------------------------------------------------------------
    def _build_messages(
        self,
        src: Tensor | LinkField,
        *,
        ctx: RoundContext,
        edge_idx: Tensor,
    ) -> tuple[Tensor, Tensor]:
        from ..dsl.scattering import LinkField
        from ..dsl.helpers import scale_messages

        source_index, target_index = edge_sources_targets(edge_idx)
        include_self = self.include_self

        if include_self is None:
            if isinstance(src, LinkField):
                messages = src.evaluate(
                    ctx=ctx,
                    edge_index=edge_idx,
                    edge_weight=ctx.edge_weight,
                )
            else:
                messages = src[source_index]
            
            if ctx.message_weight is not None:
                messages = scale_messages(messages, ctx.message_weight)
            
            return messages, target_index

        # Explicit include_self / exclude_self
        keep_mask = source_index != target_index
        kept_edge_idx = edge_idx[:, keep_mask]
        kept_target_index = target_index[keep_mask]

        if isinstance(src, LinkField):
            messages = src.evaluate(
                ctx=ctx,
                edge_index=kept_edge_idx,
                edge_weight=ctx.edge_weight[keep_mask]
                if ctx.edge_weight is not None
                else None,
            )
        else:
            kept_source_index = kept_edge_idx[0]
            messages = src[kept_source_index]
        
        if ctx.message_weight is not None:
            messages = scale_messages(messages, ctx.message_weight[keep_mask])

        if include_self is True:
            self_messages = self._self_messages(src, ctx)
            messages = torch.cat((messages, self_messages), dim=0)
            self_targets = torch.arange(
                ctx.num_nodes, device=target_index.device, dtype=target_index.dtype
            )
            kept_target_index = torch.cat((kept_target_index, self_targets), dim=0)

        return messages, kept_target_index

    def _self_messages(self, src: Tensor | LinkField, ctx: RoundContext) -> Tensor:
        from ..dsl.scattering import LinkField

        if isinstance(src, LinkField):
            self_index = torch.arange(
                ctx.num_nodes, device=ctx.edge_index.device, dtype=torch.long
            )
            self_edge_index = torch.stack((self_index, self_index), dim=0)
            self_edge_weight = torch.zeros(
                ctx.num_nodes,
                device=ctx.edge_index.device,
                dtype=ctx.edge_weight.dtype
                if ctx.edge_weight is not None
                else torch.float32,
            )
            return src.evaluate(
                ctx=ctx, edge_index=self_edge_index, edge_weight=self_edge_weight
            )

        return src

    # ------------------------------------------------------------------
    # aggregation
    # ------------------------------------------------------------------
    def _aggregate_messages(
        self, messages: Tensor, target_index: Tensor, num_nodes: int
    ) -> Tensor:
        if self._aggr_module is not None:
            return self._aggr_module(messages, target_index, num_nodes)
        return scatter_aggr(
            messages,
            target_index,
            num_nodes,
            aggr=self.aggr,
            mode=self.mode,
            tau=self.tau,
            fill_value=self.fill_value,
        )
