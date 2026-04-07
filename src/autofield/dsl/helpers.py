"""Internal helpers shared across DSL primitives and building blocks."""

from __future__ import annotations

import torch
from torch import Tensor
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .neighbor import NeighborExpr

from ..constants import DEFAULT_TAU_SOFT_AGGR, LOG_EPSILON
from ..core import RoundContext, resolve_context
from ..functional import scatter_aggr, scatter_min_by_first


def ensure_field(value: float | Tensor, ctx: RoundContext | None = None) -> Tensor:
    """Normalize scalars and tensors to a node field in *ctx*."""
    ctx = resolve_context(ctx)
    if isinstance(value, Tensor):
        tensor = value.to(ctx.edge_index.device)
        if tensor.dim() == 0:
            return tensor.expand(ctx.num_nodes)
        if tensor.shape[0] != ctx.num_nodes:
            raise ValueError(
                f"Expected first dimension {ctx.num_nodes}, got {tuple(tensor.shape)}",
            )
        return tensor
    return torch.full(
        (ctx.num_nodes,),
        float(value),
        dtype=torch.float32,
        device=ctx.edge_index.device,
    )


def broadcast_like(value: float | Tensor, template: Tensor) -> Tensor:
    """Broadcast *value* to match the shape of *template*."""
    if isinstance(value, Tensor):
        tensor = value.to(device=template.device, dtype=template.dtype)
        if tensor.dim() == 0:
            return tensor.expand_as(template)
        if tensor.shape == template.shape:
            return tensor
        if tensor.shape == template.shape[1:]:
            return tensor.unsqueeze(0).expand_as(template)
        raise ValueError(
            f"Cannot broadcast shape {tuple(tensor.shape)} to {tuple(template.shape)}",
        )
    return torch.full_like(template, float(value))


def require_scalar_field(
    name: str, value: float | Tensor, ctx: RoundContext | None = None
) -> Tensor:
    """Require *value* to normalize to a scalar node field."""
    field_value = ensure_field(value, ctx)
    if field_value.dim() != 1:
        raise ValueError(f"{name} must be a scalar field shaped [num_nodes]")
    return field_value


def pack_cast_state(distance: Tensor, payload: Tensor) -> Tensor:
    """Pack scalar distance and payload into a single state tensor."""
    payload_flat = (
        payload.unsqueeze(-1)
        if payload.dim() == 1
        else payload.reshape(payload.shape[0], -1)
    )
    return torch.cat((distance.unsqueeze(-1), payload_flat), dim=-1)


def unpack_cast_state(
    packed: Tensor, payload_shape: tuple[int, ...]
) -> tuple[Tensor, Tensor]:
    """Unpack a cast state created by :func:`pack_cast_state`."""
    distance = packed[:, 0]
    payload_flat = packed[:, 1:]
    if not payload_shape:
        return distance, payload_flat[:, 0]
    return distance, payload_flat.reshape((packed.shape[0],) + payload_shape)


def validate_cast_mode(mode: str) -> None:
    """Validate supported cast modes."""
    if mode not in {"hard", "soft"}:
        raise ValueError(f"Unknown mode: {mode}")


def resolve_edge_cost(
    weight: float | Tensor | "NeighborExpr" | None,
    ctx: RoundContext,
) -> Tensor:
    """Resolve an edge cost tensor from various input types."""
    from .neighbor import NeighborExpr

    if weight is None:
        if ctx.edge_weight is not None:
            return ctx.edge_weight
        return torch.ones(
            ctx.edge_index.shape[1], device=ctx.edge_index.device, dtype=torch.float32
        )
    if isinstance(weight, NeighborExpr):
        return weight.evaluate(
            ctx=ctx, edge_index=ctx.edge_index, edge_weight=ctx.edge_weight
        )

    src, _ = edge_sources_targets(ctx.edge_index)
    field_value = ensure_field(weight, ctx)
    return field_value[src]


def edge_sources_targets(edge_index: Tensor) -> tuple[Tensor, Tensor]:
    """Return source and target node ids from an edge index."""
    return edge_index[0], edge_index[1]


def hard_parent_ids(potential: Tensor, ctx: RoundContext | None = None) -> Tensor:
    """Select one admissible parent per node according to the potential field."""
    return hard_parent_ids_with_edge_cost(potential, edge_cost=None, ctx=ctx)


def hard_parent_ids_with_edge_cost(
    potential: Tensor,
    edge_cost: Tensor | None,
    ctx: RoundContext | None = None,
) -> Tensor:
    """Select one admissible parent per node with an optional explicit edge cost."""
    ctx = resolve_context(ctx)
    src, tgt = edge_sources_targets(ctx.edge_index)
    parent_potential = potential[src]
    child_potential = potential[tgt]
    effective_edge_cost = ctx.edge_weight if edge_cost is None else edge_cost
    path_cost = parent_potential + effective_edge_cost
    tolerance = 1e-6 + 1e-5 * torch.maximum(path_cost.abs(), child_potential.abs())
    admissible = (parent_potential < child_potential) & (
        path_cost <= child_potential + tolerance
    )

    parent_candidates = torch.stack(
        (
            torch.where(
                admissible,
                parent_potential,
                torch.full_like(parent_potential, float("inf")),
            ),
            src.to(dtype=parent_potential.dtype),
        ),
        dim=-1,
    )
    fill_row = torch.tensor(
        [float("inf"), -1.0],
        device=parent_candidates.device,
        dtype=parent_candidates.dtype,
    )
    best_parent = scatter_min_by_first(
        parent_candidates,
        tgt,
        ctx.num_nodes,
        mode="hard",
        tau=DEFAULT_TAU_SOFT_AGGR,
        fill_row=fill_row,
    )
    candidate_potential = best_parent[:, 0]
    candidate_parent = best_parent[:, 1].round().long()
    sentinel = torch.full_like(candidate_parent, -1)
    return torch.where(torch.isfinite(candidate_potential), candidate_parent, sentinel)


def soft_parent_weights(
    potential: Tensor, tau: float | Tensor, ctx: RoundContext | None = None
) -> Tensor:
    """Compute relaxed parent-selection weights for soft collect semantics."""
    return soft_parent_weights_with_edge_cost(potential, tau, edge_cost=None, ctx=ctx)


def _compute_soft_gates(
    safe_child_potential: Tensor,
    safe_parent_potential: Tensor,
    safe_path_cost: Tensor,
    effective_tau: Tensor,
    finite: Tensor,
) -> tuple[Tensor, Tensor]:
    """Compute the lower and path gates for soft parent selection."""
    lower_gate = torch.where(
        finite,
        torch.sigmoid((safe_child_potential - safe_parent_potential) / effective_tau),
        torch.zeros_like(safe_parent_potential),
    )
    path_gate = torch.where(
        finite,
        torch.sigmoid((safe_child_potential - safe_path_cost) / effective_tau),
        torch.zeros_like(safe_parent_potential),
    )
    return lower_gate, path_gate


def _compute_stable_logits_and_stabilize(
    safe_parent_potential: Tensor,
    effective_tau: Tensor,
    finite: Tensor,
    lower_gate: Tensor,
    path_gate: Tensor,
    src: Tensor,
    num_nodes: int,
) -> Tensor:
    """Compute stabilized exp(logits) gated by the lower and path gates."""
    logits_raw = -safe_parent_potential / effective_tau
    logits = torch.where(
        finite,
        logits_raw,
        torch.full_like(safe_parent_potential, float("-inf")),
    )
    bucket_max = scatter_aggr(
        logits, src, num_nodes, aggr="max", fill_value=float("-inf")
    )
    safe_bucket_max = torch.where(
        finite, bucket_max[src], torch.zeros_like(safe_parent_potential)
    )
    shifted = torch.where(
        finite,
        logits_raw - safe_bucket_max,
        torch.full_like(safe_parent_potential, float("-inf")),
    )
    return torch.where(
        finite,
        shifted.exp() * lower_gate * path_gate,
        torch.zeros_like(safe_parent_potential),
    )


def soft_parent_weights_with_edge_cost(
    potential: Tensor,
    tau: float | Tensor,
    edge_cost: Tensor | None,
    ctx: RoundContext | None = None,
) -> Tensor:
    """Compute relaxed parent-selection weights with an optional explicit edge cost."""
    ctx = resolve_context(ctx)
    src, tgt = edge_sources_targets(ctx.edge_index)
    if isinstance(tau, Tensor):
        effective_tau = tau.to(
            device=potential.device, dtype=potential.dtype
        ).clamp_min(LOG_EPSILON)
    else:
        effective_tau = potential.new_tensor(max(float(tau), LOG_EPSILON))
    parent_potential = potential[tgt]
    child_potential = potential[src]
    effective_edge_cost = ctx.edge_weight if edge_cost is None else edge_cost
    path_cost = parent_potential + effective_edge_cost
    finite = (
        torch.isfinite(parent_potential)
        & torch.isfinite(child_potential)
        & torch.isfinite(path_cost)
    )
    safe_parent_potential = torch.where(
        finite, parent_potential, torch.zeros_like(parent_potential)
    )
    safe_child_potential = torch.where(
        finite, child_potential, torch.zeros_like(child_potential)
    )
    safe_path_cost = torch.where(finite, path_cost, torch.zeros_like(path_cost))

    lower_gate, path_gate = _compute_soft_gates(
        safe_child_potential,
        safe_parent_potential,
        safe_path_cost,
        effective_tau,
        finite,
    )

    stabilized = _compute_stable_logits_and_stabilize(
        safe_parent_potential,
        effective_tau,
        finite,
        lower_gate,
        path_gate,
        src,
        ctx.num_nodes,
    )

    denom = scatter_aggr(stabilized, src, ctx.num_nodes, aggr="sum")
    return torch.where(
        denom[src] > 0,
        stabilized / denom[src].clamp(min=LOG_EPSILON),
        torch.zeros_like(stabilized),
    )


def scale_messages(messages: Tensor, weights: Tensor) -> Tensor:
    """Broadcast scalar edge weights across message feature dimensions."""
    scaled_weights = weights
    while scaled_weights.dim() < messages.dim():
        scaled_weights = scaled_weights.unsqueeze(-1)
    return messages * scaled_weights
