"""Derived aggregate building blocks built from core DSL primitives."""

from __future__ import annotations

from collections.abc import Callable

import torch
from torch import Tensor

from ..constants import (
    CONDITION_THRESHOLD,
    DEFAULT_TAU_SOFT_AGGR,
    ELECTION_NONE,
    LOG_EPSILON,
)
from ..core import RoundContext, aggregate
from ..core.mode import get_default_mode
from ..functional import field_where, scatter_binary_fold, scatter_min_by_first
from .gathering import gather_max, gather_min, gather_sum
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
from .primitives import field, gather, iterate, mux
from .scattering import LinkField, link_cat, scatter, scatter_range


@aggregate
def gradient(
    source: Tensor,
    weight: LinkField | None = None,
    *,
    name: str | None = None,
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
    )


@aggregate
def gradient_cast(
    source: Tensor,
    center: Tensor,
    accumulation: Callable[[Tensor], Tensor],
    *,
    weight: LinkField | None = None,
    name: str | None = None,
    mode: str | None = None,
    tau: float | None = None,
) -> Tensor:
    r"""Propagate payloads outward along a minimum-potential gradient.

    ``source`` is a scalar node field, ``center`` is a node field payload, and
    ``weight`` is an optional edge-wise :class:`LinkField`.

    Written entirely in terms of :func:`iterate`, :func:`scatter` and
    :func:`gather`, so it inherits branch edge masking, message overrides and
    exports from the core primitives rather than reimplementing them.
    """
    effective_mode = mode if mode is not None else get_default_mode()
    effective_tau = tau if tau is not None else DEFAULT_TAU_SOFT_AGGR
    validate_cast_mode(effective_mode)
    source_field = require_scalar_field(source, name="source")
    center_field = ensure_field(center)
    payload_shape = center_field.shape[1:]
    init_state = pack_cast_state(field.inf(), center_field)
    source_state = pack_cast_state(field.zeros(), center_field)
    step = scatter_range() if weight is None else weight

    def min_by_first(messages: Tensor, index: Tensor, num_nodes: int) -> Tensor:
        """Keep the whole message whose leading cost column is minimal."""
        return scatter_min_by_first(
            messages,
            index,
            num_nodes,
            mode=effective_mode,
            tau=effective_tau,
            fill_row=init_state,
        )

    def update(state: Tensor) -> Tensor:
        old_distance, old_payload = unpack_cast_state(state, payload_shape)
        message = link_cat(
            [
                scatter(old_distance) + step,
                scatter(old_payload).map(accumulation, label="accumulation"),
            ]
        )
        propagated = gather(message, aggr=min_by_first)
        # The source mask is a 0/1 indicator, not a learned condition, so this
        # selection stays hard in both modes. Blending it would also mix units:
        # `tau` here is an aggregation temperature (lower = sharper) while
        # field_where's soft tau is a sigmoid sharpness (higher = sharper).
        return field_where(source_field, source_state, propagated, mode="hard")

    state = iterate(init_state, update)
    return unpack_cast_state(state, payload_shape)[1]


@aggregate
def broadcast(
    mask: Tensor,
    value: Tensor,
    *,
    name: str | None = None,
    weight: LinkField | None = None,
    mode: str | None = None,
    tau: float | None = None,
) -> Tensor:
    r"""Propagate a node field from root nodes through the network.

    ``mask`` marks roots, ``value`` is the payload field, and ``weight`` is an
    optional edge-wise :class:`LinkField`.

    A float mask marks roots where it is at or above
    :data:`~diffield.constants.CONDITION_THRESHOLD`, the same convention as
    :func:`gradient` and :func:`mux`.  (It previously used the opposite test,
    treating near-zero values as roots, which inverted float masks.)
    """
    cond = mask if mask.dtype == torch.bool else (mask >= CONDITION_THRESHOLD)
    effective_mode = mode if mode is not None else get_default_mode()
    effective_tau = tau if tau is not None else DEFAULT_TAU_SOFT_AGGR
    return gradient_cast(
        source=cond,
        center=value,
        accumulation=lambda x: x,
        weight=weight,
        mode=effective_mode,
        tau=effective_tau,
    )


@aggregate
def collect_cast(
    potential: Tensor,
    local: Tensor,
    null: Tensor,
    accumulation: Callable[[Tensor, Tensor], Tensor],
    *,
    weight: LinkField | None = None,
    name: str | None = None,
    mode: str | None = None,
    tau: float | None = None,
) -> Tensor:
    r"""Collect payloads from children toward local minima of a potential field.

    ``potential`` is a scalar node field, ``local`` and ``null`` are payload
    fields, and ``weight`` is an optional edge-wise :class:`LinkField`.

    Like :func:`gradient_cast`, the payload travels through :func:`scatter` and
    :func:`gather`, so branch edge masking, message overrides and exports apply
    without being reimplemented here.  Parent selection is carried as a gate
    column alongside the payload: a link field derived from the potential and
    the edge metric, hard (a 0/1 indicator) or soft (a normalised weight).
    """
    effective_mode = mode if mode is not None else get_default_mode()
    effective_tau = tau if tau is not None else DEFAULT_TAU_SOFT_AGGR
    validate_cast_mode(effective_mode)
    potential_field = require_scalar_field(potential, name="potential")
    local_field = ensure_field(local)
    null_field = broadcast_like(null, local_field)
    payload_shape = local_field.shape[1:]

    def parent_gate_expr() -> LinkField:
        """The child -> parent gate, as a link field over the active topology."""

        def evaluate(
            ctx: RoundContext, edge_index: Tensor, edge_weight: Tensor | None
        ) -> Tensor:
            edge_cost = (
                edge_weight
                if weight is None and edge_weight is not None
                else resolve_edge_cost(weight, ctx)
            )
            if effective_mode == "hard":
                parent_ids = hard_parent_ids_with_edge_cost(
                    potential_field,
                    edge_cost=edge_cost,
                    ctx=ctx,
                    edge_index=edge_index,
                )
                child, parent = edge_sources_targets(edge_index)
                return (parent_ids[child] == parent).to(potential_field.dtype)
            return soft_parent_weights_with_edge_cost(
                potential_field,
                effective_tau,
                edge_cost=edge_cost,
                ctx=ctx,
                edge_index=edge_index,
            )

        return LinkField(evaluate, _repr="parent_gate()")

    def collect_aggr(messages: Tensor, index: Tensor, num_nodes: int) -> Tensor:
        """Fold gated child payloads into each parent."""
        gate = messages[:, 0]
        payload_flat = messages[:, 1:]
        payload = (
            payload_flat[:, 0]
            if not payload_shape
            else payload_flat.reshape((messages.shape[0],) + payload_shape)
        )
        if effective_mode == "hard":
            keep = gate >= CONDITION_THRESHOLD
            return scatter_binary_fold(
                payload[keep], index[keep], num_nodes, accumulation, null_field
            )
        return scatter_binary_fold(
            scale_messages(payload, gate),
            index,
            num_nodes,
            accumulation,
            null_field,
        )

    def update(collected: Tensor) -> Tensor:
        message = link_cat([parent_gate_expr(), scatter(collected)])
        child_values = gather(message, aggr=collect_aggr)
        return accumulation(local_field, child_values)

    return iterate(local_field, update)


@aggregate
def elect(
    key: Tensor,
    eligible: Tensor | None = None,
    *,
    grain: float = float("inf"),
    weight: LinkField | None = None,
    name: str | None = None,
    mode: str | None = None,
    tau: float | None = None,
) -> tuple[Tensor, Tensor]:
    r"""Distributed leader election — the *S* (sparse-choice) block.

    A faithful transcription of ScaFi's ``BlockS`` (``breakUsingUids`` +
    ``distanceCompetition``), composed from the DSL's own operators:
    ``share`` → :func:`iterate`, ``G``/``distanceTo`` → :func:`gradient`,
    ``minHood(mux(nbr(...)))`` → :func:`gather_min` over a masked
    :class:`LinkField`, and the competition itself is a :func:`mux` cascade.

    Every node whose ``eligible`` flag is set (all nodes when ``None``) starts
    as a candidate carrying a scalar ``key`` — lower wins, and keys must be
    unique per component, so compose them with :func:`mid` (e.g.
    ``quality * num_nodes + mid()``: quality competes, the id tie-breaks).
    Keys should be **stable over time** (the literature's ``randomUid`` is
    drawn once and held): when a winning key changes, the old value lingers
    as an unsourced ghost until its distance field rises past ``grain``, so a
    volatile key churns leadership for ~``grain`` rounds per change.
    Each round a node measures its **gradient** distance ``d`` to the leader
    it currently believes in, then competes:

    * ``d > grain`` — no leader within reach: candidate itself again;
    * ``0.5·grain ≤ d ≤ grain`` — buffer zone: abdicate (keeps regions apart);
    * ``d < 0.5·grain`` — adopt the lowest key among itself and the
      neighbours whose own distance-plus-hop stays under ``0.5·grain``.

    Self-stabilisation comes from the gradient: when a leader vanishes its
    distance field rises each round, nodes fall through the buffer zone and
    re-candidate, and the next-lowest key wins — no ad-hoc TTLs. Leaders are
    spaced ≥ ``grain`` apart (with ``grain=inf``: one leader per component,
    but vanished leaders are then never forgotten).

    Returns ``(leader, lead)``: the boolean leader mask and each node's
    currently-adopted key (:data:`~diffield.constants.ELECTION_NONE` where no
    leader is within reach). Follow-up fields (e.g. the distance used to
    *follow* the leader) are the caller's next block: ``gradient(leader)``.
    """
    effective_mode = mode if mode is not None else get_default_mode()
    effective_tau = tau if tau is not None else DEFAULT_TAU_SOFT_AGGR
    validate_cast_mode(effective_mode)
    key_field = require_scalar_field(key, name="key")
    none_field = field.of(ELECTION_NONE)
    own = (
        key_field
        if eligible is None
        else mux(eligible, key_field, none_field, mode=effective_mode, tau=effective_tau)
    )
    half_grain = 0.5 * grain
    step = scatter_range() if weight is None else weight

    def compete(lead: Tensor) -> Tensor:
        held = ((lead - own).abs() < 0.5) & (own < ELECTION_NONE)
        d = gradient(
            held.float(),
            weight=weight,
            mode=effective_mode,
            tau=effective_tau,
        )
        # minHood over neighbours still inside the half-grain disc: an offer
        # from a neighbour whose distance-plus-hop leaves the disc is pushed
        # past ELECTION_NONE, so gather_min ignores it (ScaFi's nbr-mux).
        in_disc = (scatter(d) + step) < half_grain
        offer = scatter(lead) + (1.0 - in_disc) * ELECTION_NONE
        nbr_best = gather_min(
            offer,
            fill_value=ELECTION_NONE,
            mode=effective_mode,
            tau=effective_tau,
        )
        best = mux(
            (d < half_grain) & (lead <= nbr_best),
            lead,
            nbr_best,
            mode=effective_mode,
            tau=effective_tau,
        )
        return mux(
            d > grain,
            own,
            mux(d >= half_grain, none_field, best, mode=effective_mode, tau=effective_tau),
            mode=effective_mode,
            tau=effective_tau,
        )

    lead = iterate(own, compete)
    leader = ((lead - own).abs() < 0.5) & (own < ELECTION_NONE)
    return leader, lead


@aggregate
def descend(
    potential: Tensor,
    toward: Tensor,
    *,
    tau: float | None = None,
) -> Tensor:
    r"""Soft steepest-descent direction on a potential field.

    For each node, neighbours are weighted by a softmax over
    ``-potential_j / tau`` and the result is the weighted mean of
    ``toward_j - toward_i`` — with ``toward`` = positions, the movement
    direction toward the neighbourhood's lowest-potential node (the read-out
    used to *follow* a :func:`gradient` field). Composed entirely from
    neighbourhood folds: :func:`gather_max` provides the numerically-stable
    softmax shift, the exponential runs edge-wise on the
    :class:`LinkField`, and :func:`gather_sum` folds weights and weighted
    offsets. Non-finite potentials get zero weight (``exp(-inf) = 0``), so
    unreached regions contribute nothing and isolated nodes return zero.

    The result is differentiable through ``toward``; pass a detached
    ``potential`` to keep the routing field out of the gradient path (the
    usual routing/steering autograd split).
    """
    effective_tau = tau if tau is not None else DEFAULT_TAU_SOFT_AGGR
    pot = require_scalar_field(potential, name="potential")
    toward_field = ensure_field(toward)
    logits = -pot / effective_tau
    peak = gather_max(scatter(logits), fill_value=float("-inf"))
    peak_safe = torch.where(torch.isfinite(peak), peak, torch.zeros_like(peak))
    excite = (scatter(logits) - peak_safe).exp()
    total = gather_sum(excite, fill_value=0.0)
    pull = gather_sum(
        excite.pointwise() * (scatter(toward_field) - toward_field),
        fill_value=0.0,
    )
    return pull / (total + LOG_EPSILON).unsqueeze(-1)
