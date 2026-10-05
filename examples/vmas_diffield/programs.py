"""Per-scenario aggregate programs built only with the DIFFIELD DSL.

Coordination uses field-calculus operators (election, gradients, broadcasts,
collection, descent, and recurrent neighbourhood folds), plus pointwise torch
math for local computations. Each rollout owns a persistent
``AggregateContext``: every physics step refreshes topology and runs one round.
Routing state advances under ``torch.no_grad``; steering read-outs remain
differentiable. ``reset()`` starts the next rollout with a fresh context.

Scenario compositions (see ``vmas_env.SCENARIO_SPEC`` for term lists):
* ``flocking`` / ``flocking_beacon``: elect a goal-informed leader, follow its
    gradient, and broadcast its goal direction.
* ``navigation``: local PD arrival control with link-based collision avoidance.
* ``discovery``: caller gradients, recruit collection, and count broadcasts.
* ``sampling``: decaying max-gossip advertises sensed density for agents to
    follow; consumed sources fade when no longer refreshed.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from torch import Tensor
from vmas_diffield.field_terms import (
    FieldTerms,
    brake_term,
    explore_term,
    grid_sense_term,
    lidar_sense_term,
    soft_normalize,
)
from vmas_diffield.scenarios import (
    DISCOVERY_LIDAR_RANGE,
    SAMPLING_AGENT_LIDAR_RAYS,
    SCENARIO_SPEC,
)

if TYPE_CHECKING:
    from vmas_diffield.vmas_env import Perception

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

GRAD_BIG = 1e3   # Finite sentinel for unreached distances.
HOP_COST = 0.25  # Per-hop floor that decays stale gradients.

# Discovery quota: one recruit per caller; release distant surplus recruits.
RECRUIT_REACH = 3.0    # Hop-floored recruiting radius (about 4-6 hops).
QUOTA_RELEASE_AT = 1.5
QUOTA_FAR_DIST = 1.1

# Sampling turns more slowly to cover fresh cells; discovery uses the default.
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
        # Election grain controls leader range and forgetting time.
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
        """Refresh topology while preserving recurrent field state."""
        ac = self._ac
        if ac is None or ac.num_nodes != p.num_nodes:
            ac = AggregateContext(p.edge_index, p.num_nodes)
            self._ac = ac
        ac.update_topology(p.edge_index, positions=p.pos)
        return ac

    def step(self, p: Perception) -> FieldTerms:
        """Run one aggregate round and return this step's field terms."""
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
        # Shrink near-cancelling signals instead of amplifying noise.
        separation = soft_normalize(sep_vec, scale=0.05) * has_neigh
        alignment = soft_normalize(neigh_vel - vel, scale=0.05) * has_neigh
        cohesion = soft_normalize(neigh_pos - pos, scale=0.05) * has_neigh
        return separation, alignment, cohesion, has_neigh

    def _avoid(self, pos: Tensor, vel: Tensor) -> Tensor:
        """Weight repulsion by closing speed to avoid diverging pairs."""
        eps = 1e-6
        rel = scatter(pos) - pos
        rvel = scatter(vel) - vel
        d = rel.norm(dim=-1)
        closing = (-(rel * rvel).sum(dim=-1) / (d + eps)).relu()
        urgency = (closing / (d + 0.05)).pointwise()
        away = -rel / (d + eps).pointwise()
        return soft_normalize(gather_sum(urgency * away, fill_value=0.0), scale=0.5)

    def _centroid(self, pos: Tensor, *, name: str) -> Tensor:
        """Compute a leaky multi-hop centroid; isolated nodes hold position."""
        anchor = pos.detach()

        def relax(state: Tensor) -> Tensor:
            nb = gather_avg(scatter(state), fill_value=0.0)
            return mux(has_neighbors(), 0.5 * anchor + 0.5 * nb, anchor)

        with torch.no_grad():
            return iterate(anchor, relax, name=name)

    def _stigmergy(self, own_val: Tensor, *, name: str) -> Tensor:
        """Propagate the best value with decay so stale sources fade."""

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
            # Stable IDs prevent changing quality from orphaning candidacy.
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
        # Proportional pull near the goal lets the D term park the agent.
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
            # Hop-floored costs make stale gradients rise until recruits release.
            d_rec = gradient(
                callers.float(), weight=scatter_range() + HOP_COST,
                name="disc_drecruit", fill_value=GRAD_BIG,
            )
            reach_b = (d_rec < RECRUIT_REACH) & ~callers
            # Count recruits per caller, broadcast quotas, and release surplus.
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
        # Follow the advertised-value surface hop by hop.
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
