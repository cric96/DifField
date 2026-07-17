"""Per-scenario aggregate programs, written **only** with the DIFFIELD DSL.

Every coordination mechanism below is composed from the library's own
field-calculus operators (``diffield.dsl``) — no raw tensor message passing,
no ``edge_index`` indexing, no scatter primitives. The building blocks:

  * ``elect`` — the *S* (sparse-choice) block: leader election by distance
    competition through a gradient with a key tie-breaker, transcribed from
    the aggregate-computing literature (ScaFi's ``BlockS``);
  * ``gradient`` — self-healing minimum-cost distance fields (the *G* block);
  * ``broadcast`` / ``collect_cast`` — payload dissemination down a gradient
    and accumulation toward potential minima (*G* + *C*);
  * ``descend`` — soft steepest-descent read-out on a potential field;
  * ``iterate`` + ``gather_*``/``scatter`` — bespoke recurrent fields (the
    decaying stigmergy gossip, the regional-centroid relaxation), spelled as
    neighbourhood folds exactly like the library's own blocks;
  * pointwise torch math on node fields (``Field = Tensor``) is the *local*
    part of field calculus and stays plain Python.

Execution model — authentic aggregate-computing deployment semantics: a
``FieldProgram`` owns one persistent ``AggregateContext`` per rollout; every
physics step refreshes the topology in place (``update_topology``, metric
edge lengths from live positions) and runs **one** aggregate round, so the
stateful fields converge over the first ~diameter steps and then self-heal
continuously while agents move, links drop, or sources change. ``reset()``
drops the context and the next rollout starts from scratch.

Autograd split (the pattern proven in earlier iterations): *routing* state
(distance fields, election, collected counts, stigmergy) advances under
``torch.no_grad`` — discrete-ish structure, not a quantity to differentiate —
while the *steering read-outs* (``descend`` directions, gates) are recomputed
from live positions/observations, so SHAC's analytic gradient reaches every
learnable term weight through the physics.

Per-scenario programs (see ``vmas_env.SCENARIO_SPEC`` for the term lists):

  * ``flocking`` / ``flocking_beacon`` — the classic S-then-G composition:
    ``elect`` among goal-informed agents (key = goal-distance bin, id
    tie-breaker), a ``gradient`` from the elected leader, ``follow`` =
    ``descend`` on it, ``lead_dir`` = ``broadcast`` of the leader's goal
    direction. In ``flocking_beacon`` only ``n_knowers`` agent(s) observe the
    goal, so the broadcast is the *only* way the rest of the swarm can steer —
    a 1-hop feed-forward GNN structurally cannot relay it.
  * ``navigation`` — a local PD arrival controller (goal P-gain, brake
    D-gain, closing-speed ``avoid`` as a LinkField fold): aggregate computing
    subsumes plain local control; no multi-hop state is needed, none is used.
  * ``discovery`` — the full G+C loop: agents whose lidar fires ("callers")
    source a ``gradient`` (hop-floored costs, so the field keeps rising once
    a caller vanishes and far recruits release themselves); ``collect_cast``
    counts committed recruits toward each caller and ``broadcast`` returns
    the count, releasing surplus recruits back to exploration.
  * ``sampling`` — computational stigmergy: a decaying max-gossip
    (``iterate`` + ``gather_max``) advertises the best locally-sensed density
    with a per-hop/per-round penalty, and agents climb the advertised-value
    surface itself (``descend`` on its negation) — consumed sources stop
    being refreshed and fade out on their own.
"""

from __future__ import annotations

import torch
from torch import Tensor
from vmas_diffield.vmas_env import (
    DISCOVERY_LIDAR_RANGE,
    SAMPLING_AGENT_LIDAR_RAYS,
    SCENARIO_SPEC,
    FieldTerms,
    Perception,
    brake_term,
    explore_term,
    grid_sense_term,
    lidar_sense_term,
    soft_normalize,
)

from diffield.dsl import (
    ELECTION_NONE,
    AggregateContext,
    as_scatter_expr,
    broadcast,
    collect_cast,
    descend,
    elect,
    gather_avg,
    gather_max,
    gather_sum,
    gradient,
    has_neighbors,
    iterate,
    mid,
    mux,
    scatter,
    scatter_range,
)
from diffield.sim import normalize_vectors

GRAD_BIG = 1e3   # finite unreached-distance sentinel (keeps downstream features NaN-free)
HOP_COST = 0.25  # per-hop floor on metric edge costs: bounds how long a stale
                 # gradient can keep pointing at a vanished source (rises
                 # >= HOP_COST per round, so forgetting takes <= reach/HOP_COST rounds)

# discovery quota: covering needs agents_per_target(=2) agents, i.e. 1 recruit
# per caller; release recruits once ~2 are committed, and only the far ones.
RECRUIT_REACH = 3.0    # recruiting radius on the hop-floored metric (~4-6 hops)
QUOTA_RELEASE_AT = 1.5
QUOTA_FAR_DIST = 1.1

# sampling patrol smoothness: reward is paid only on *fresh* cells, so swept
# trail per step is the budget; a slower-turning explore flow lets velocity
# build along a heading instead of curling back over the agent's own trail
# (discovery keeps the default 1.0 — its patrol only has to stumble into
# lidar range, and its recruit dynamics are tuned around the tighter meander).
SAMPLING_EXPLORE_FREQ_SCALE = 0.5


class FieldProgram:
    """A scenario's aggregate program with persistent per-rollout field state."""

    def __init__(
        self,
        scenario: str,
        *,
        sep: float = 0.2,
        elect_grain: float = 16.0,
        desc_tau: float = 0.15,
        caller_thresh: float = 0.25,
        stig_hop_penalty: float = 0.05,
        nav_arrive_radius: float = 0.12,
    ) -> None:
        self.scenario = scenario
        self.spec = SCENARIO_SPEC[scenario]
        self.sep = sep
        self.nav_arrive_radius = nav_arrive_radius
        # S-block grain on the hop metric: agents follow a leader within
        # grain/2 hops, and a vanished leader is forgotten in ~grain rounds.
        self.elect_grain = elect_grain
        self.desc_tau = desc_tau
        self.caller_thresh = caller_thresh
        self.stig_hop_penalty = stig_hop_penalty
        self._ac: AggregateContext | None = None

    def reset(self) -> None:
        """Drop all field state; the next ``step`` starts a fresh execution."""
        self._ac = None

    def warmup(self, p: Perception, rounds: int) -> None:
        """Run a few no-grad rounds on the initial graph to pre-converge fields."""
        with torch.no_grad():
            for _ in range(rounds):
                self.step(p)

    def _context(self, p: Perception) -> AggregateContext:
        """Refresh the topology in place, keeping the recurrent field state."""
        ac = self._ac
        if ac is None or ac.num_nodes != p.num_nodes:
            ac = AggregateContext(p.edge_index, p.num_nodes)
            self._ac = ac
        ac.update_topology(p.edge_index, positions=p.pos)
        return ac

    def step(self, p: Perception) -> FieldTerms:
        """One aggregate round on the current graph -> this step's field terms."""
        ac = self._context(p)
        with ac.round():
            if self.scenario in ("flocking", "flocking_beacon"):
                return self._flock_terms(p)
            if self.scenario == "navigation":
                return self._nav_terms(p)
            if self.scenario == "discovery":
                return self._disc_terms(p)
            return self._samp_terms(p)

    # ── shared folds (differentiable, DSL gather/scatter) ─────────────────────

    def _boids(self, p: Perception) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        pos, vel = p.pos, p.vel
        neigh_vel = gather_avg(scatter(vel))
        neigh_pos = gather_avg(scatter(pos))
        delta = pos - scatter(pos)
        dist = delta.norm(dim=-1)
        sep_mask = ((dist <= self.sep) & (dist > 0)).pointwise()
        sep_vec = gather_sum(delta * sep_mask)
        has_neigh = has_neighbors().to(pos.dtype).unsqueeze(-1)
        # scale=0.05: shrink near-cancelling neighbourhood signals toward zero
        # instead of blowing them up to unit noise (see vmas_env.soft_normalize).
        separation = soft_normalize(sep_vec, scale=0.05) * has_neigh
        alignment = soft_normalize(neigh_vel - vel, scale=0.05) * has_neigh
        cohesion = soft_normalize(neigh_pos - pos, scale=0.05) * has_neigh
        return separation, alignment, cohesion, has_neigh

    def _avoid(self, pos: Tensor, vel: Tensor) -> Tensor:
        """Anticipatory collision avoidance: repulsion weighted by closing speed.

        Plain ``separation`` pushes on distance alone, so it also shoves apart
        agents already moving apart. Weighting each neighbour's repulsion by
        how fast the pair is closing over how near it already is (~1/time-to-
        collision) keeps it silent for diverging pairs — the same ``nbr``-style
        fold as alignment, applied to relative position *and* velocity.
        """
        eps = 1e-6
        rel = scatter(pos) - pos
        rvel = scatter(vel) - vel
        d = rel.norm(dim=-1)
        closing = (-(rel * rvel).sum(dim=-1) / (d + eps)).relu()
        urgency = (closing / (d + 0.05)).pointwise()
        away = -rel / (d + eps).pointwise()
        return soft_normalize(gather_sum(urgency * away, fill_value=0.0), scale=0.5)

    def _centroid(self, pos: Tensor, *, name: str) -> Tensor:
        """Leaky diffusion average of position: a multi-hop regional centroid.

        ``avg <- 0.5*own_pos + 0.5*mean(neighbours' avg)`` per round — converges
        to a neighbourhood-weighted centroid and tracks it as the swarm moves.
        Isolated nodes hold their own position (nothing to disperse from).
        """
        anchor = pos.detach()

        def relax(state: Tensor) -> Tensor:
            nb = gather_avg(scatter(state), fill_value=0.0)
            return mux(has_neighbors(), 0.5 * anchor + 0.5 * nb, anchor)

        with torch.no_grad():
            return iterate(anchor, relax, name=name)

    def _stigmergy(self, own_val: Tensor, *, name: str) -> Tensor:
        """Decaying max-gossip: the best advertised value within the horizon.

        Each round every node advertises the best (hop/round-penalized) value
        it knows of; a consumed source stops refreshing its peak, which then
        decays by the penalty every round and fades out — self-stabilizing
        forgetting with no explicit TTL.
        """

        def relax(state: Tensor) -> Tensor:
            heard = gather_max(scatter(state), fill_value=float("-inf"))
            decayed = heard - self.stig_hop_penalty
            return torch.maximum(
                own_val, torch.where(torch.isfinite(decayed), decayed, own_val)
            )

        return iterate(own_val, relax, name=name)

    # ── flocking / flocking_beacon: S-then-G (elect + gradient + broadcast) ───

    def _flock_terms(self, p: Perception) -> FieldTerms:
        if p.goal_rel is None or p.knows is None:
            raise ValueError("flocking programs need goal_rel/knows in the perception")
        pos = p.pos
        separation, alignment, cohesion, has_neigh = self._boids(p)
        goal = normalize_vectors(-p.goal_rel)  # zero rows for masked non-knowers

        n = p.n_agents
        hop = as_scatter_expr(1.0)  # hop-count metric for the election
        with torch.no_grad():
            # Election key: the static agent id (ScaFi's breakUsingUids draws
            # its uid once and holds it) — a volatile quality key would break
            # the leader's own candidacy every time it changes, orphaning the
            # competition gradient for ~grain rounds (measured: ghost-leader
            # cycles). Among knowers, any one of them serves the program
            # equally: what matters is that a goal-informed agent leads.
            key = mid() % n
            leader_b, lead = elect(
                key, p.knows, grain=self.elect_grain, weight=hop, name="flock_leader"
            )
            d_lead = gradient(leader_b.float(), name="flock_dlead", fill_value=GRAD_BIG)
            reach = ((lead < ELECTION_NONE) & ~leader_b).to(pos.dtype).unsqueeze(-1)
            gdir = normalize_vectors(-p.goal_rel)
            lead_dir_field = broadcast(leader_b, gdir, name="flock_leaddir")
        follow = soft_normalize(descend(d_lead, pos, tau=self.desc_tau), scale=0.05) * reach
        lead_dir = lead_dir_field * reach
        leader_f = leader_b.to(pos.dtype).unsqueeze(-1)
        zeros = torch.zeros_like(pos)
        return FieldTerms(
            separation=separation, alignment=alignment, cohesion=cohesion, goal=goal,
            has_neigh=has_neigh, explore=zeros, disperse=zeros, follow=follow,
            lead_dir=lead_dir, leader_mask=leader_f, leader_dist=d_lead,
        )

    # ── navigation: local PD arrival + anticipatory avoidance ────────────────

    def _nav_terms(self, p: Perception) -> FieldTerms:
        if p.goal_rel is None:
            raise ValueError("navigation needs goal_rel in the perception")
        pos, vel = p.pos, p.vel
        separation, alignment, cohesion, has_neigh = self._boids(p)
        # True P gain: proportional inside ~the on-goal disc so the D term can
        # park the agent (constant-magnitude pull would ping-pong through it).
        goal = soft_normalize(-p.goal_rel, scale=0.08)
        brake = brake_term(vel, p.goal_rel, arrive_radius=self.nav_arrive_radius)
        avoid = self._avoid(pos, vel)
        zeros = torch.zeros_like(pos)
        return FieldTerms(
            separation=separation, alignment=alignment, cohesion=cohesion, goal=goal,
            has_neigh=has_neigh, explore=zeros, disperse=zeros, brake=brake, avoid=avoid,
        )

    # ── discovery: recruit gradient + quota collect/broadcast ────────────────

    def _disc_terms(self, p: Perception) -> FieldTerms:
        pos = p.pos
        separation, alignment, cohesion, has_neigh = self._boids(p)
        sense, sense_strength = lidar_sense_term(p.extra, max_range=DISCOVERY_LIDAR_RANGE)

        with torch.no_grad():
            callers = sense_strength.squeeze(-1) > self.caller_thresh
            # Hop-floored costs: when a caller vanishes the field rises by at
            # least HOP_COST per round, past RECRUIT_REACH recruits let go —
            # reachability and forgetting both come from the gradient itself.
            d_rec = gradient(
                callers.float(), weight=scatter_range() + HOP_COST,
                name="disc_drecruit", fill_value=GRAD_BIG,
            )
            reach_b = (d_rec < RECRUIT_REACH) & ~callers
            # Quota: count committed recruits toward each caller (C block),
            # send the count back out (G block), release far surplus recruits.
            cnt = collect_cast(
                d_rec, reach_b.float(), torch.tensor(0.0, device=pos.device),
                torch.add, name="disc_count",
            )
            cnt_seen = broadcast(callers, cnt, name="disc_quota")
            release = (
                torch.sigmoid((cnt_seen - QUOTA_RELEASE_AT) / 0.25)
                * torch.sigmoid((d_rec - QUOTA_FAR_DIST) / 0.1)
            )
            active = (reach_b.float() * (1.0 - release)).unsqueeze(-1)

        recruit = (
            soft_normalize(descend(d_rec, pos, tau=self.desc_tau), scale=0.05)
            * active * (1.0 - sense_strength)
        )
        blind = 1.0 - sense_strength
        explore = explore_term(pos, mid() % p.n_agents) * blind * (1.0 - active)
        disperse_dir = pos - self._centroid(pos, name="disc_centroid")
        disperse = soft_normalize(disperse_dir, scale=0.1) * has_neigh
        return FieldTerms(
            separation=separation, alignment=alignment, cohesion=cohesion,
            goal=torch.zeros_like(pos), has_neigh=has_neigh, explore=explore,
            disperse=disperse, sense=sense, recruit=recruit,
        )

    # ── sampling: decaying-gossip stigmergy ──────────────────────────────────

    def _samp_terms(self, p: Perception) -> FieldTerms:
        pos = p.pos
        separation, alignment, cohesion, has_neigh = self._boids(p)
        samples = p.extra[:, SAMPLING_AGENT_LIDAR_RAYS:]
        sense, sense_strength = grid_sense_term(samples)
        own_val = samples.mean(dim=-1)

        with torch.no_grad():
            best = self._stigmergy(own_val.detach(), name="samp_stig")
        adv = torch.relu(best - own_val)  # differentiable through own sensing
        strength = torch.tanh(adv / 0.15).unsqueeze(-1)
        # climb the advertised-value surface itself, hop by hop
        social = soft_normalize(descend(-best, pos, tau=self.desc_tau), scale=0.1) * strength

        blind = 1.0 - sense_strength
        explore = (
            explore_term(pos, mid() % p.n_agents, freq_scale=SAMPLING_EXPLORE_FREQ_SCALE)
            * blind * (1.0 - strength.detach())
        )
        disperse_dir = pos - self._centroid(pos, name="samp_centroid")
        disperse = soft_normalize(disperse_dir, scale=0.1) * has_neigh
        return FieldTerms(
            separation=separation, alignment=alignment, cohesion=cohesion,
            goal=torch.zeros_like(pos), has_neigh=has_neigh, explore=explore,
            disperse=disperse, sense=sense, social=social,
        )
