"""Derived aggregate building blocks built from core DSL primitives."""

from __future__ import annotations

import operator
from collections.abc import Callable
from typing import NamedTuple

import torch
from torch import Tensor

from ..constants import (
    CONDITION_THRESHOLD,
    DEFAULT_TAU_SOFT_AGGR,
    ELECTION_NONE,
    LOG_EPSILON,
)
from ..core import RoundContext, aggregate
from ..core.mode import get_default_mode, get_default_tau
from ..functional import field_where, scatter_binary_fold, scatter_min_by_first
from .gathering import gather_max, gather_min, gather_sum
from .helpers import (
    broadcast_like,
    edge_sources_targets,
    ensure_field,
    hard_parent_ids_with_edge_cost,
    lexicographic_select,
    pack_cast_state,
    quantize,
    require_scalar_field,
    resolve_edge_cost,
    scale_messages,
    soft_parent_weights_with_edge_cost,
    surrogate_probabilities,
    unpack_cast_state,
    validate_cast_mode,
)
from .primitives import field, gather, iterate, mid, mux, nbr
from .scattering import LinkField, link_cat, link_map, membership, scatter, scatter_range


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
    effective_tau = tau if tau is not None else get_default_tau(DEFAULT_TAU_SOFT_AGGR)
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
            mode="hard",  # a 0/1 source indicator, never a learned condition
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
    effective_tau = tau if tau is not None else get_default_tau(DEFAULT_TAU_SOFT_AGGR)
    validate_cast_mode(effective_mode)
    source_field = require_scalar_field(source, name="source")
    center_field = ensure_field(center)
    payload_shape = center_field.shape[1:]
    init_state = pack_cast_state(field.inf(), center_field)
    source_state = pack_cast_state(field.zeros(), center_field)
    step = scatter_range() if weight is None else weight

    def min_by_first(messages: Tensor, index: Tensor, num_nodes: int) -> Tensor:
        """Keep the whole message of the cheapest sender (ties: lowest sender id)."""
        if effective_mode == "soft":
            return scatter_min_by_first(
                messages[:, :-1],
                index,
                num_nodes,
                mode="soft",
                tau=effective_tau,
                fill_row=init_state,
            )
        cost, sender = messages[:, 0], messages[:, -1]
        rows = lexicographic_select(
            (quantize(cost), sender), torch.isfinite(cost), index, num_nodes
        )
        chosen = rows < len(messages)
        out = init_state.clone()
        out[chosen] = messages[rows[chosen], :-1]
        return out

    def update(state: Tensor) -> Tensor:
        old_distance, old_payload = unpack_cast_state(state, payload_shape)
        message = link_cat(
            [
                scatter(old_distance) + step,
                scatter(old_payload).map(accumulation, label="accumulation"),
                scatter(mid()),
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
    value: Tensor | tuple[Tensor, ...],
    weight: LinkField | None = None,
    *,
    name: str | None = None,
    mode: str | None = None,
    tau: float | None = None,
) -> Tensor | tuple[Tensor, ...]:
    r"""Propagate a node field from root nodes through the network.

    ``mask`` marks roots, ``value`` is the payload field (or a tuple of fields,
    returned as a tuple), and ``weight`` is an optional edge-wise
    :class:`LinkField` metric.

    A float mask marks roots where it is at or above
    :data:`~diffield.constants.CONDITION_THRESHOLD`, the same convention as
    :func:`gradient` and :func:`mux`.  (It previously used the opposite test,
    treating near-zero values as roots, which inverted float masks.)
    """
    cond = mask if mask.dtype == torch.bool else (mask >= CONDITION_THRESHOLD)
    effective_mode = mode if mode is not None else get_default_mode()
    effective_tau = tau if tau is not None else get_default_tau(DEFAULT_TAU_SOFT_AGGR)
    packed = isinstance(value, (tuple, list))
    payload = torch.stack([ensure_field(v) for v in value], -1) if packed else value
    out = gradient_cast(
        source=cond,
        center=payload,
        accumulation=lambda x: x,
        weight=weight,
        mode=effective_mode,
        tau=effective_tau,
    )
    return tuple(out.unbind(-1)) if packed else out


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
    causal: bool = False,
) -> Tensor:
    r"""Collect payloads from children toward local minima of a potential field.

    ``potential`` is a scalar node field, ``local`` and ``null`` are payload
    fields, and ``weight`` is an optional edge-wise :class:`LinkField`.

    Like :func:`gradient_cast`, the payload travels through :func:`scatter` and
    :func:`gather`, so branch edge masking, message overrides and exports apply
    without being reimplemented here.  Parent selection is carried as a gate
    column alongside the payload: a link field derived from the potential and
    the edge metric, hard (a 0/1 indicator) or soft (a normalised weight).

    Set ``causal=True`` for execution by independent devices on a symmetric
    network (currently only ``mode="hard"``). Each device publishes its
    potential, chosen parent's stable ``mid()`` and collected payload. It
    selects its parent from previous messages, breaking equal path costs by
    the smallest ID, and accepts children that previously nominated it.
    Initially no parent or remote payload is assumed. This adds a round to
    publish potentials and a round to publish parent choices before remote
    collection starts. Payloads subsequently travel one hop per round.
    Topology changes can temporarily duplicate or omit contributions; a stable
    strictly descending routing tree conserves them after convergence.

    The default retains the legacy global parent gate for compatibility. That
    gate requires the whole topology and is **not** device-execution equivalent.
    Discrete parent choices are not differentiable in either hard variant;
    the causal payload recurrence remains differentiable.
    """
    effective_mode = mode if mode is not None else get_default_mode()
    effective_tau = tau if tau is not None else get_default_tau(DEFAULT_TAU_SOFT_AGGR)
    validate_cast_mode(effective_mode)
    if causal:
        return converge_cast(
            potential, local, accumulation, weight, null=null, mode=effective_mode,
            tau=effective_tau,
        )
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
    effective_tau = tau if tau is not None else get_default_tau(DEFAULT_TAU_SOFT_AGGR)
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
        in_disc = (nbr(d) + step) < half_grain
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
    effective_tau = tau if tau is not None else get_default_tau(DEFAULT_TAU_SOFT_AGGR)
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


distance_to = gradient
"""``distance_to(source, metric)``: alias of :func:`gradient` (ScaFi ``G``,
Collektive ``distanceTo``)."""


@aggregate
def converge_cast(
    potential: Tensor,
    local: Tensor | tuple[Tensor, ...],
    accumulate: Callable[[Tensor, Tensor], Tensor] = operator.add,
    metric: LinkField | None = None,
    *,
    null: float | Tensor = 0.0,
    name: str | None = None,
    mode: str | None = None,
    tau: float | None = None,
) -> Tensor | tuple[Tensor, ...]:
    r"""Accumulate *local* values toward the minima of *potential* (ScaFi ``C``,
    FCPP ``sp_collection``, Collektive ``convergeCast``).

    Each device picks as parent its neighbour with lower published potential
    minimising ``potential + metric`` (ties: lowest id), publishes it, and
    folds the payloads of the children that nominated it in the previous
    round.  Only published state is read, so it runs unchanged on independent
    devices.  In soft mode a child sends its payload scaled by the softmax
    probability of its chosen parent (and by :func:`membership` inside a soft
    ``aligned_on``), so gradients reach the metric and the potential.  A tuple
    *local* (e.g. ``(value, 1.0)``) is collected jointly and returned as a tuple.
    """
    effective_mode = mode if mode is not None else get_default_mode()
    effective_tau = tau if tau is not None else get_default_tau(DEFAULT_TAU_SOFT_AGGR)
    validate_cast_mode(effective_mode)
    potential = require_scalar_field(potential, name="potential")
    packed = isinstance(local, (tuple, list))
    if packed:
        local = [v if isinstance(v, Tensor) else torch.full_like(potential, v) for v in local]
        values = torch.stack([ensure_field(v) for v in local], -1)
    else:
        values = ensure_field(local)
    null_field = broadcast_like(torch.as_tensor(null), values)
    shape = values.shape[1:]
    ids = mid().to(potential.dtype)
    step = scatter_range() if metric is None else metric
    inf = float("inf")
    initial = torch.cat(
        (
            torch.full_like(potential, inf)[:, None],  # published potential
            torch.full_like(potential, -1)[:, None],  # published parent
            torch.ones_like(potential)[:, None],  # published gate
            null_field.reshape(len(potential), -1),  # published subtree payload
        ),
        -1,
    )

    def fold(messages: Tensor, index: Tensor, num_nodes: int) -> Tensor:
        keep = messages[:, 0] >= CONDITION_THRESHOLD
        payload = messages[:, 1:].reshape((messages.shape[0], *shape))
        return scatter_binary_fold(payload[keep], index[keep], num_nodes, accumulate, null_field)

    def update(old: Tensor) -> Tensor:
        old_potential, old_parent, old_gate = old[:, 0], old[:, 1], old[:, 2]
        old_payload = old[:, 3:]
        cost = link_map(
            lambda p_j, s, p_i: torch.where(
                (quantize(p_j) < quantize(p_i)) & torch.isfinite(p_j + s), p_j + s, inf
            ),
            scatter(old_potential),
            step,
            potential,
        )
        best = gather_min(cost, mode="hard", fill_value=inf)
        tied = link_map(
            lambda c, j, b: torch.where(
                torch.isfinite(c) & (quantize(c) <= quantize(b)), j, inf
            ),
            cost,
            scatter(ids),
            best,
        )
        parent = gather_min(tied, mode="hard", fill_value=inf)
        has_parent = torch.isfinite(best)
        parent = torch.where(has_parent, parent, torch.full_like(parent, -1))
        if effective_mode == "soft":
            reference = torch.where(has_parent, best, torch.zeros_like(best))
            odds = link_map(
                lambda c, b: torch.where(
                    torch.isfinite(c), torch.exp((b - c) / effective_tau), 0.0
                ),
                cost,
                reference,
            )
            chosen = 1 / gather_sum(odds, fill_value=1.0)
            gate = torch.where(has_parent, chosen, torch.ones_like(best))
        else:
            gate = torch.ones_like(best)
        nominated = (scatter(old_parent) - ids).abs() < 0.5
        sent = scatter(old_gate[:, None] * old_payload) * membership().pointwise()
        received = gather(link_cat([nominated, sent]), aggr=fold)
        payload = accumulate(values, received)
        published = (potential, parent, gate)
        flat = payload.reshape(len(potential), -1)
        return torch.cat([*(x[:, None] for x in published), flat], -1)

    state = iterate(initial, update)
    collected = state[:, 3:].reshape(values.shape)
    return tuple(collected.unbind(-1)) if packed else collected


class Election(NamedTuple):
    """Result of :func:`bounded_election`."""

    leader: Tensor
    """Elected leader id (hard, never relaxed)."""
    distance: Tensor
    """Accumulated metric to the leader."""
    elected: Tensor
    """Whether the device leads (its own candidacy wins; a probability when soft)."""
    confidence: Tensor
    """Probability of following the chosen leader (1 when hard): soft region membership."""
    support: LinkField | None = None
    """Soft mode only: per link ``j -> i``, the probability that ``i`` follows the
    candidacy relayed by ``j`` when it names another leader (read by :func:`follow`)."""


@aggregate
def bounded_election(
    strength: Tensor,
    radius: float = 1.0,
    metric: LinkField | None = None,
    *,
    name: str | None = None,
    mode: str | None = None,
    tau: float | None = None,
) -> Election:
    r"""Bounded priority election (Pianini et al., ACSOS 2022, Alg. 1; the
    election of Space-Fluid, Casadei et al., LMCS 2023, Fig. 6).

    Every device proposes ``(-strength, 0, id)`` and relays the best candidacy
    it knows after adding the link *metric*.  Candidacies whose accumulated
    distance reaches *radius*, and echoes of the device's own candidacy, are
    dropped.  The lexicographic minimum of ``(-strength, distance, leader id)``
    wins, so the stronger leader prevails within its radius and regions are
    connected.  In soft mode the candidacy is chosen by the relaxed "first
    admissible in order" probabilities (:func:`surrogate_probabilities`):
    numeric outputs become expectations while the leader id stays hard.
    """
    effective_mode = mode if mode is not None else get_default_mode()
    effective_tau = tau if tau is not None else get_default_tau(DEFAULT_TAU_SOFT_AGGR)
    validate_cast_mode(effective_mode)
    strength = require_scalar_field(strength, name="strength")
    ids = mid().to(strength.dtype)
    step = scatter_range() if metric is None else metric
    zero = torch.zeros_like(strength)
    own = torch.stack((-strength, zero, ids, ids), -1)  # key, distance, leader, sender
    # key, distance, leader (-1: not published), elected, confidence
    initial = torch.stack((zero, zero, zero - 1, zero, zero + 1), -1)

    def choose(messages: Tensor, index: Tensor, num_nodes: int) -> Tensor:
        offers = torch.cat((messages[:, :4], own))
        target = torch.cat((index, torch.arange(num_nodes, device=index.device)))
        is_own = torch.arange(len(offers), device=index.device) >= len(messages)
        receiver = torch.cat((messages[:, 4], ids))
        key, distance, leader, sender = offers.unbind(-1)
        eligible = is_own | ((leader >= 0) & (leader != receiver))
        admissible = eligible & ((distance.detach() < radius) | is_own)
        rows = lexicographic_select(
            (key, quantize(distance), leader, sender), admissible, target, num_nodes
        )
        if effective_mode == "hard":
            return torch.stack(
                (key[rows], distance[rows], leader[rows], is_own[rows].to(key), zero + 1), -1
            )
        chance = surrogate_probabilities(
            key, distance, eligible, is_own, target, num_nodes, effective_tau, radius
        )

        def expected(column: Tensor) -> Tensor:
            return column.new_zeros(num_nodes).index_add(0, target, chance * column)

        winner = leader[rows].detach()
        same_leader = (leader == winner[target]).to(key)  # any route to the chosen leader
        relayed["support"] = (chance * (1 - same_leader))[: len(messages)]
        return torch.stack(
            (expected(key), expected(distance), winner, expected(is_own.to(key)),
             expected(same_leader)),
            -1,
        )  # fmt: skip

    def compete(old: Tensor) -> Tensor:
        offers = link_cat(
            [scatter(old[:, 0]), scatter(old[:, 1]) + step, scatter(old[:, 2]), scatter(ids), ids]
        )
        return gather(offers, aggr=choose, mode="hard")

    def support(_ctx: RoundContext, edge_index: Tensor, _weight: Tensor | None) -> Tensor:
        value = relayed["support"]
        if value.shape[0] != edge_index.shape[1]:
            raise ValueError("election.support is defined on the election's own links")
        return value

    relayed: dict[str, Tensor] = {}
    state = iterate(initial, compete)
    soft = effective_mode == "soft"
    return Election(
        state[:, 2],
        state[:, 1],
        state[:, 3],
        state[:, 4],
        LinkField(support, _repr="election.support") if soft else None,
    )


@aggregate
def follow(election: Election, value: Tensor, *, name: str | None = None) -> Tensor:
    r"""Read *value* as a member of the region a device would follow.

    Hard: the identity.  Soft: the confidence-weighted own *value* mixed with the
    neighbours' *value* (from :func:`nbr`) on links whose relayed candidacy names
    another leader, weighted by ``election.support``.  This is what lets the
    gradient see "this device would be better off in the next region".
    """
    if election.support is None:
        return value
    neighbours = nbr(value)
    elsewhere = gather_sum(election.support * neighbours, fill_value=0.0)
    weight = election.confidence + gather_sum(election.support, fill_value=0.0)
    return (election.confidence * value + elsewhere) / weight
