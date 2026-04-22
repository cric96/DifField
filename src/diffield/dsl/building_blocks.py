"""Derived aggregate building blocks built from core DSL primitives."""

from __future__ import annotations

from typing import Callable

import torch
from torch import Tensor

from ..constants import BROADCAST_NEAR_ZERO, DEFAULT_TAU_SOFT_AGGR
from ..core import RoundContext
from ..core.mode import get_default_mode
from ..functional import field_where, scatter_binary_fold, scatter_min_by_first
from .helpers import (
    broadcast_like,
    edge_sources_targets,
    ensure_field,
    hard_parent_ids_with_edge_cost,
    pack_cast_state,
    require_scalar_field,
    resolve_edge_cost,
    scale_messages,
    soft_parent_weights_with_edge_cost,
    unpack_cast_state,
    validate_cast_mode,
)
from .scattering import LinkField, scatter, scatter_range
from .primitives import field, mux, iterate
from .gathering import gather_min


def gradient(
    source: Tensor,
    weight: LinkField | None = None,
    *,
    name: str = "gradient",
    mode: str | None = None,
    tau: float | None = None,
    fill_value: float = float("inf"),
) -> Tensor:
    r"""Compute a minimum-cost distance field from scalar source nodes.

    ``source`` is a scalar node field. ``weight`` is an optional edge-wise
    :class:`LinkField`; when omitted, :func:`scatter_range` is used.
    """
    effective_mode = mode if mode is not None else get_default_mode()
    effective_tau = tau if tau is not None else DEFAULT_TAU_SOFT_AGGR
    validate_cast_mode(effective_mode)
    source_field = require_scalar_field(source, name="source")
    step = scatter_range() if weight is None else weight

    return iterate(
        field.of(fill_value),
        lambda dist: mux(
            source_field,
            field.of(0.0),
            gather_min(
                scatter(dist) + step,
                mode=effective_mode,
                tau=effective_tau,
                fill_value=fill_value,
            ),
        ),
        name=f"_grad_{name}",
    )


def gradient_cast(
    source: Tensor,
    center: Tensor,
    accumulation: Callable[[Tensor], Tensor],
    *,
    weight: LinkField | None = None,
    name: str = "gradient_cast",
    mode: str | None = None,
    tau: float | None = None,
) -> Tensor:
    r"""Propagate payloads outward along a minimum-potential gradient.

    ``source`` is a scalar node field, ``center`` is a node field payload, and
    ``weight`` is an optional edge-wise :class:`LinkField`.
    """
    effective_mode = mode if mode is not None else get_default_mode()
    effective_tau = tau if tau is not None else DEFAULT_TAU_SOFT_AGGR
    validate_cast_mode(effective_mode)
    source_field = require_scalar_field(source, name="source")
    center_field = ensure_field(center)
    payload_shape = center_field.shape[1:]
    init_state = pack_cast_state(field.inf(), center_field)
    source_state = pack_cast_state(field.zeros(), center_field)

    def update(state: Tensor, _x: Tensor, ctx: RoundContext) -> Tensor:
        old_distance, old_payload = unpack_cast_state(state, payload_shape)
        src, tgt = edge_sources_targets(ctx.edge_index)
        edge_cost = resolve_edge_cost(weight, ctx)
        messages = pack_cast_state(
            old_distance[src] + edge_cost,
            accumulation(old_payload[src]),
        )
        propagated = scatter_min_by_first(
            messages,
            tgt,
            ctx.num_nodes,
            mode=effective_mode,
            tau=effective_tau,
            fill_row=init_state,
        )
        return torch.where(source_field.unsqueeze(-1) >= 0.5, source_state, propagated)

    state = iterate(init_state, update, name=f"_gc_{name}")
    return unpack_cast_state(state, payload_shape)[1]


def broadcast(
    mask: Tensor,
    value: Tensor,
    *,
    name: str = "bc_cc",
    weight: LinkField | None = None,
    mode: str | None = None,
    tau: float | None = None,
) -> Tensor:
    r"""Propagate a node field from root nodes through the network.

    ``mask`` marks roots, ``value`` is the payload field, and ``weight`` is an
    optional edge-wise :class:`LinkField`.
    """
    cond = mask if mask.dtype == torch.bool else (mask <= BROADCAST_NEAR_ZERO)
    effective_mode = mode if mode is not None else get_default_mode()
    effective_tau = tau if tau is not None else DEFAULT_TAU_SOFT_AGGR
    return gradient_cast(
        source=cond,
        center=value,
        accumulation=lambda x: x,
        weight=weight,
        name=name,
        mode=effective_mode,
        tau=effective_tau,
    )


def collect_cast(
    potential: Tensor,
    local: Tensor,
    null: Tensor,
    accumulation: Callable[[Tensor, Tensor], Tensor],
    *,
    weight: LinkField | None = None,
    name: str = "collect_cast",
    mode: str | None = None,
    tau: float | None = None,
) -> Tensor:
    r"""Collect payloads from children toward local minima of a potential field.

    ``potential`` is a scalar node field, ``local`` and ``null`` are payload
    fields, and ``weight`` is an optional edge-wise :class:`LinkField`.
    """
    effective_mode = mode if mode is not None else get_default_mode()
    effective_tau = tau if tau is not None else DEFAULT_TAU_SOFT_AGGR
    validate_cast_mode(effective_mode)
    potential_field = require_scalar_field(potential, name="potential")
    local_field = ensure_field(local)
    null_field = broadcast_like(null, local_field)

    def update(collected: Tensor, _x: Tensor, ctx: RoundContext) -> Tensor:
        src, tgt = edge_sources_targets(ctx.edge_index)
        edge_cost = resolve_edge_cost(weight, ctx)
        if effective_mode == "hard":
            parent_ids = hard_parent_ids_with_edge_cost(
                potential_field, edge_cost=edge_cost, ctx=ctx
            )
            keep = parent_ids[src] == tgt
            child_values = scatter_binary_fold(
                collected[src[keep]],
                tgt[keep],
                ctx.num_nodes,
                accumulation,
                null_field,
            )
        else:
            parent_weight = soft_parent_weights_with_edge_cost(
                potential_field, effective_tau, edge_cost=edge_cost, ctx=ctx
            )
            child_values = scatter_binary_fold(
                scale_messages(collected[src], parent_weight),
                tgt,
                ctx.num_nodes,
                accumulation,
                null_field,
            )
        return accumulation(local_field, child_values)

    return iterate(local_field, update, name=f"_cc_{name}")
