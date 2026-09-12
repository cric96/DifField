"""VMAS integration: differentiable env adapter + perception + local field terms.

Bridges the VMAS vectorized differentiable simulator (proroklab) to DIFFIELD:
  * builds a differentiable env (``grad_enabled=True``),
  * reads the per-agent observation. Every supported scenario's observation
    starts with ``[pos(2), vel(2), ...]``; the remainder is scenario-specific
    and kept as ``extra`` (goal-relative + lidar for flocking/navigation,
    lidar for discovery, lidar + local field samples for sampling),
  * builds a *batched* radius graph over agents (one graph per parallel env,
    no cross-env edges) so the aggregate programs run on the flattened
    ``[num_envs * n_agents, F]`` node view,
  * hosts the *local* (single-node / one-fold) field-term functions:
    ``lidar_sense_term`` / ``grid_sense_term`` (task perception), ``explore``
    (background wander), ``brake`` / ``avoid`` (arrival control). The
    *multi-hop aggregate programs* — election, gradients, broadcast,
    collect, gossip — live in ``programs.FieldProgram``, written with the
    DIFFIELD DSL and executed one aggregate round per physics step with
    persistent field state (see that module's docstring),
  * packs forces back into the VMAS per-agent action list.

``flocking_beacon`` is a partial-observability variant of ``flocking`` built
entirely in this adapter: only ``spec.n_knowers`` agent(s) per env receive the
goal-relative vector (everyone else sees zeros) and every agent's ``extra``
gains a trailing "knows the goal" bit. ALL policies get the same masked
observation; ``Perception.true_goal_rel`` keeps the unmasked vector for
rewards/metrics only. The aggregate program disseminates the goal direction
from the knower via election + gradient + broadcast — a 1-hop feed-forward
GNN has no mechanism to relay it beyond its receptive field.

Observation prefix ``[pos(2), vel(2)]`` verified for vmas 1.5.2 scenarios
flocking, navigation, discovery, sampling.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import vmas
from torch import Tensor

from diffield.sim import normalize_vectors


def soft_normalize(x: Tensor, scale: float) -> Tensor:
    """Like ``normalize_vectors`` but scales toward zero (instead of blowing up
    to a unit vector) when ``x``'s magnitude is small relative to ``scale``.

    ``normalize_vectors``'s ``eps`` is tiny (1e-8), so it treats *any* nonzero
    raw vector as a confident, full-strength unit direction -- including ones
    whose direction is essentially measurement/geometry noise (e.g. a nearly
    symmetric neighbourhood's net relative position, which should net to
    ~nothing, not a full-strength push in whatever direction the noise
    happens to point). That blow-up was empirically confirmed to make
    ``separation``/``alignment``/``cohesion``/``sense``/``disperse`` flip
    direction almost 180° between consecutive steps (measured cosine
    similarity as low as -0.999) whenever their raw signal passed near zero,
    which cancels out net displacement over an episode even though
    instantaneous force looks fine -- the literal cause of the "agents barely
    move" symptom. Dividing by ``norm + scale`` instead makes small raw
    vectors shrink smoothly toward zero (proportional response) while still
    saturating to a unit vector for a strong signal.
    """
    return x / (x.norm(dim=-1, keepdim=True) + scale)

_POS = slice(0, 2)
_VEL = slice(2, 4)
_GOAL_REL = slice(4, 6)  # only meaningful when spec.has_goal (pos - goal)


@dataclass(frozen=True)
class ScenarioSpec:
    has_goal: bool          # goal-relative vector available at obs[:, 4:6]
    field_terms: tuple[str, ...]
    primary_metric: str     # metric plotted as the headline learning curve
    n_agents: int
    smooth_collision: bool  # add a differentiable soft-collision term to SHAC loss
    sense_kind: str | None = None  # "lidar" | "grid" -> how to build the `sense` field term
    vmas_name: str | None = None   # underlying VMAS scenario (defaults to the key)
    n_knowers: int | None = None   # partial obs: how many agents see the goal (None = all)


# discovery's target lidar / sampling's local grid readings, both at VMAS defaults
# (this repo never overrides them — see make_diff_env). Hardcoded here the same way
# the observation-prefix slices above are: these are fixed scenario wire formats,
# not runtime-configurable in this pipeline.
DISCOVERY_LIDAR_RANGE = 0.35
SAMPLING_AGENT_LIDAR_RAYS = 12  # leading obs block sampling doesn't use for `sense`

SCENARIO_SPEC: dict[str, ScenarioSpec] = {
    # cohesive group tracks a common moving target: boids + goal pursuit, plus
    # the S+G aggregate block — distributed leader election, distance gradient
    # from the leader (`follow`), broadcast of the leader's goal direction
    # (`lead_dir`); SHAC learns how much to rely on the leader vs own pursuit
    "flocking": ScenarioSpec(
        True,
        ("separation", "alignment", "cohesion", "goal", "follow", "lead_dir"),
        "order", 6, True,
    ),
    # flocking with partial observability: only n_knowers agent(s) observe the
    # goal (masked in obs_to_perception for EVERY policy), so the program's
    # election + gradient + broadcast is the only multi-hop route the goal
    # direction has to the rest of the swarm; primary metric is goal
    # proximity computed from true_goal_rel (evaluation may use privileged
    # state even though policies cannot) — the scenario's question is "can
    # the swarm track a target most of it cannot see"
    "flocking_beacon": ScenarioSpec(
        True,
        ("separation", "alignment", "cohesion", "goal", "follow", "lead_dir"),
        "goal_prox", 6, True, vmas_name="flocking", n_knowers=1,
    ),
    # each agent reaches its OWN goal -> leader-follower does not fit, and
    # flock terms (alignment/cohesion) are counterproductive; the fitting
    # program is a PD arrival controller (goal = P gain, brake = D gain near
    # the goal, so agents park instead of orbiting) + `avoid`, anticipatory
    # closing-speed collision avoidance
    "navigation": ScenarioSpec(
        True, ("separation", "goal", "brake", "avoid"), "on_goal_frac", 6, True
    ),
    # cooperative search: a target counts as covered only when >=2 agents are
    # within covering range simultaneously -> `sense` (own-lidar seek) +
    # `recruit` (multi-hop distance-gradient convergence on a teammate that
    # currently senses a target, with a collect/broadcast quota loop that
    # releases surplus recruits); `explore`/`disperse` are the patrol strategy
    # while nothing is sensed and nobody calls
    "discovery": ScenarioSpec(
        False,
        ("separation", "sense", "recruit", "explore", "disperse"),
        "coverage", 6, True, sense_kind="lidar",
    ),
    # collective sampling of a value field (sampled cells read 0 in the obs):
    # `sense` climbs the local unsampled-density stencil, `social` climbs the
    # best value advertised by TTL gossip over the comm graph (multi-hop
    # stigmergy), `disperse` keeps the swarm partitioned over the arena
    "sampling": ScenarioSpec(
        False,
        ("separation", "sense", "social", "explore", "disperse"),
        "coverage", 4, True, sense_kind="grid",
    ),
}


def make_diff_env(scenario, *, num_envs, n_agents, device, max_steps):
    """Create a differentiable VMAS environment."""
    if scenario not in SCENARIO_SPEC:
        raise ValueError(f"scenario must be one of {tuple(SCENARIO_SPEC)}, got {scenario}")
    vmas_name = SCENARIO_SPEC[scenario].vmas_name or scenario
    return vmas.make_env(
        scenario=vmas_name, num_envs=num_envs, device=str(device),
        continuous_actions=True, grad_enabled=True, n_agents=n_agents, max_steps=max_steps,
    )


def detach_env(env) -> None:
    """Detach all persistent VMAS tensors so each update's graph is independent.

    VMAS resets state in-place and keeps reward-shaping buffers across resets;
    without detaching, a second ``loss.backward()`` hits freed saved tensors.
    """
    world = env.world
    entities = list(world.agents) + list(getattr(world, "landmarks", []))
    for e in entities:
        st = e.state
        for attr in ("pos", "vel", "rot", "ang_vel", "force", "torque"):
            t = getattr(st, attr, None)
            if isinstance(t, Tensor):
                setattr(st, attr, t.detach())
        for k, v in list(vars(e).items()):
            if isinstance(v, Tensor):
                setattr(e, k, v.detach())
    for k, v in list(vars(env.scenario).items()):
        if isinstance(v, Tensor):
            setattr(env.scenario, k, v.detach())


@dataclass
class Perception:
    """Flattened per-node (``[B*N, *]``) view of a VMAS observation.

    ``goal_rel``/``extra`` are what policies may consume (masked in
    ``n_knowers`` partial-obs scenarios); ``true_goal_rel`` keeps the unmasked
    vector for rewards/metrics only; ``knows`` marks the goal-informed agents
    (all-True for fully observable goal scenarios, ``None`` without a goal).
    """

    pos: Tensor
    vel: Tensor
    extra: Tensor            # obs tail (masked goal_rel + lidar + ... [+ knows bit])
    goal_rel: Tensor | None  # pos - goal, zeroed for non-knowers
    true_goal_rel: Tensor | None  # unmasked (rewards/metrics only, never policy input)
    edge_index: Tensor
    num_nodes: int
    batch_size: int
    n_agents: int
    knows: Tensor | None = None  # bool [B*N]


def obs_to_perception(
    obs_list: list[Tensor], *, scenario: str, radius: float, knower_shift: int = 0
) -> Perception:
    """Stack a VMAS obs list into a flattened node view + batched radius graph.

    ``knower_shift`` rotates which agents are the goal "knowers" in partial-obs
    scenarios (knower iff ``(agent_id - shift) mod N < n_knowers``) — used by
    the self-healing evaluation to move the beacon mid-episode and watch the
    program re-elect and re-converge.
    """
    spec = SCENARIO_SPEC[scenario]
    obs = torch.stack(obs_list, dim=1)  # [B, N, obs_dim]
    b, n, _ = obs.shape
    flat = obs.reshape(b * n, -1)  # env-major: node = env*N + agent
    pos = flat[:, _POS]
    extra = flat[:, 4:]
    true_goal_rel = flat[:, _GOAL_REL] if spec.has_goal else None
    goal_rel = true_goal_rel
    knows = None
    if spec.has_goal:
        knows = torch.ones(b * n, dtype=torch.bool, device=pos.device)
    if spec.n_knowers is not None and true_goal_rel is not None:
        agent_id = torch.arange(b * n, device=pos.device) % n
        knows = ((agent_id - knower_shift) % n) < spec.n_knowers
        mask = knows.to(flat.dtype).unsqueeze(-1)
        goal_rel = true_goal_rel * mask
        # extra begins with the goal_rel slice for has_goal scenarios: mask it
        # there too (the GNN baseline reads extra) and append the knows bit so
        # every policy can tell informed agents apart.
        extra = torch.cat([extra[:, :2] * mask, extra[:, 2:], mask], dim=-1)
    return Perception(
        pos=pos, vel=flat[:, _VEL], extra=extra, goal_rel=goal_rel,
        true_goal_rel=true_goal_rel,
        edge_index=_batched_radius_graph(pos.reshape(b, n, 2), radius),
        num_nodes=b * n, batch_size=b, n_agents=n, knows=knows,
    )


def _batched_radius_graph(pos: Tensor, radius: float) -> Tensor:
    """Per-env radius graph over agents via dense distances (no torch-cluster).

    Edges are ``src=neighbour j -> tgt=self i`` in flattened ids (``env*N+agent``),
    matching DIFFIELD's convention where ``gather`` aggregates at the target.
    """
    _, n, _ = pos.shape
    dist = torch.cdist(pos, pos)
    adj = (dist <= radius) & (dist > 0.0)
    env_idx, i_idx, j_idx = adj.nonzero(as_tuple=True)
    return torch.stack([env_idx * n + j_idx, env_idx * n + i_idx], dim=0).to(pos.device)


@dataclass
class FieldTerms:
    """Steering directions per node (``[B*N, 2]``) produced by the scenario's
    aggregate program (see ``programs.FieldProgram``), plus masks/diagnostics."""

    separation: Tensor
    alignment: Tensor
    cohesion: Tensor
    goal: Tensor
    has_neigh: Tensor
    explore: Tensor                    # background wander, active only when "blind"
    disperse: Tensor                   # push away from the gossip regional centroid
    sense: Tensor | None = None        # seek direction from local sensing (lidar/grid)
    recruit: Tensor | None = None      # descend the caller distance-gradient (discovery)
    social: Tensor | None = None       # climb the gossip-advertised best value (sampling)
    brake: Tensor | None = None        # arrival damping: -vel gated near own goal (navigation)
    avoid: Tensor | None = None        # anticipatory closing-speed repulsion (navigation)
    follow: Tensor | None = None       # descent on the elected-leader distance gradient
    lead_dir: Tensor | None = None     # leader's goal direction, broadcast down the gradient
    leader_mask: Tensor | None = None  # [B*N, 1] elected leaders
    leader_dist: Tensor | None = None  # [B*N] distance-to-leader field (diagnostic)


_lidar_angle_cache: dict[tuple[int, str], Tensor] = {}


def _lidar_ray_angles(n_rays: int, device: torch.device) -> Tensor:
    """World-frame ray angles for a full-circle ``n_rays``-ray Lidar (agent rot == 0
    for every scenario this pipeline uses -- see module docstring)."""
    key = (n_rays, str(device))
    if key not in _lidar_angle_cache:
        _lidar_angle_cache[key] = torch.linspace(0, 2 * torch.pi, n_rays + 1, device=device)[
            :n_rays
        ]
    return _lidar_angle_cache[key]


def lidar_sense_term(lidar: Tensor, *, max_range: float) -> tuple[Tensor, Tensor]:
    """Seek direction toward the nearest sensed entity: a potential-field gradient
    built from raw range readings, one unit ray direction per reading weighted by
    proximity (``max_range - dist``, zero when nothing was hit). Also returns a
    ``[*, 1]`` detection strength (0 when nothing is in range) used to gate `explore`."""
    angles = _lidar_ray_angles(lidar.shape[-1], lidar.device)
    dirs = torch.stack([angles.cos(), angles.sin()], dim=-1)  # [rays, 2]
    proximity = (max_range - lidar).clamp(min=0.0)  # [*, rays]
    strength = torch.tanh(proximity.sum(dim=-1, keepdim=True) / max_range)
    return soft_normalize(torch.einsum("nr,rd->nd", proximity, dirs), scale=max_range), strength


_SAMPLE_OFFSET_DIRS = torch.tensor(
    [[1, 0], [-1, 0], [0, 1], [0, -1], [-1, -1], [1, -1], [-1, 1], [1, 1]], dtype=torch.float32
)
_SAMPLE_OFFSET_DIRS = _SAMPLE_OFFSET_DIRS / _SAMPLE_OFFSET_DIRS.norm(dim=-1, keepdim=True)


def grid_sense_term(samples: Tensor) -> tuple[Tensor, Tensor]:
    """Ascent direction on the locally-sampled value field: a finite-difference
    gradient built from the 8 grid-offset density readings sampling's own
    observation already carries (see ``vmas.scenarios.sampling.observation``).
    Also returns a ``[*, 1]`` detection strength (0 when the local field is flat)."""
    dirs = _SAMPLE_OFFSET_DIRS.to(samples.device)  # [8, 2]
    strength = torch.tanh(samples.sum(dim=-1, keepdim=True))
    return soft_normalize(torch.einsum("no,od->nd", samples, dirs), scale=1.0), strength


# A handful of fixed, irrational-ish spatial frequencies/phases: `explore_term`
# sums a few travelling sinusoids of the agent's own position into a smooth,
# deterministic "flow field" over the arena, offset per-agent so nearby agents
# don't wander in lockstep. Fixed (not learned) -- it only needs to be a
# space-filling background drift, not a fitted function.
_EXPLORE_FREQS = torch.tensor([[6.1, -4.3], [-5.7, 3.9], [4.1, 6.7]])
_EXPLORE_PHASES = torch.tensor([0.7, 2.9, 5.2])


def explore_term(pos: Tensor, node_id: Tensor, *, freq_scale: float = 1.0) -> Tensor:
    """Background wander field: fixed smooth pseudo-random flow over the arena.

    Purely a function of position (+ a per-agent phase offset from ``node_id``),
    so it is always defined and never literally zero -- it exists to give any
    agent that isn't currently sensing a target a baseline search drift, so the
    swarm keeps sweeping the arena instead of settling at a static
    separation/cohesion/disperse equilibrium (and, as a special case, so an
    isolated agent never freezes in place under zero force -- VMAS decelerates
    an agent to rest under zero force, and a frozen agent can never re-enter
    anyone's sensing range to recover). Gated by sensing strength alone (not
    neighbour count) in the scenario programs (``programs.FieldProgram``) so it
    fades out smoothly as a target comes into range, but doesn't otherwise care
    whether other agents are nearby.

    ``freq_scale`` scales the spatial frequencies: <1 makes the flow turn less
    per unit travelled, i.e. longer straight sweep runs. Under VMAS drag the
    swept trail per step is what an agent's velocity can integrate along a
    steady heading, so a program whose task pays for fresh ground (sampling)
    wants a smoother field than the default meander (measured on sampling:
    freq_scale 0.5 ≈ +7% covered cells and +5% collected value vs 1.0).
    """
    freqs = _EXPLORE_FREQS.to(pos.device) * freq_scale
    phases = _EXPLORE_PHASES.to(pos.device)
    id_phase = node_id.float() * 2.399963  # irrational multiplier decorrelates neighbours' ids
    angle = torch.einsum("nd,kd->nk", pos, freqs) + phases + id_phase.unsqueeze(-1)
    return normalize_vectors(torch.stack([angle.cos(), angle.sin()], dim=-1).mean(dim=1))


def brake_term(vel: Tensor, goal_rel: Tensor, *, arrive_radius: float = 0.12) -> Tensor:
    """Arrival damping: ``-vel`` gated by proximity to the agent's own goal.

    Together with ``goal`` (a unit pull) this forms a PD controller: P drives
    the approach, D bleeds velocity off as the goal nears. Without it a
    goal-seeking agent at terminal speed crosses navigation's 0.1 "on goal"
    disc in a couple of steps and orbits back and forth through it, which is
    exactly what the per-step ``on_goal_frac`` metric punishes. Deliberately
    *not* normalized: damping is proportional to actual velocity by
    definition, and the gate ``exp(-|goal_rel|/arrive_radius)`` fades it out
    smoothly away from the goal so it never fights the cruise phase.

    ``arrive_radius`` sets how early braking engages: at ``gd = arrive_radius``
    the damping is at ``1/e`` strength. It is deliberately tight (0.12, ~the
    on-goal radius) so braking is spent stopping the agent *on* the disc rather
    than slowing the cruise — though the approach deceleration is dominated by
    the ``goal`` P-gain's ``soft_normalize`` shrinking as ``gd`` shrinks, so the
    exact value is not sensitive (swept on the expert: 0.06-0.20 all within
    ±0.002 on_goal_frac). It is exposed mainly so the hybrid's gate has a
    meaningful per-agent brake knob to modulate (down while cruising, up on
    arrival).
    """
    gd = goal_rel.norm(dim=-1, keepdim=True)
    return -vel * torch.exp(-gd / arrive_radius)


# Every neighbourhood term (separation/alignment/cohesion/avoid and the
# multi-hop recruit, social/stigmergy, follow/lead_dir, disperse) is built by
# ``programs.FieldProgram`` from the DIFFIELD DSL — this module only provides
# the environment plumbing and the *local* (single-agent) terms above.


def pack_actions(force_flat, *, batch_size, n_agents, u_range):
    """Map a flattened ``[B*N, 2]`` force to the VMAS per-agent action list.

    Squashes with ``tanh`` to the action range (smooth, keeps gradients).
    """
    action = (u_range * torch.tanh(force_flat / u_range)).reshape(batch_size, n_agents, 2)
    return [action[:, i, :] for i in range(n_agents)]


def env_action_range(env) -> float:
    return float(env.agents[0].u_range)


def env0_positions(obs_list: list[Tensor]) -> Tensor:
    """Agent positions of parallel env 0 as ``[N, 2]`` (for trajectory/GIF)."""
    return torch.stack(obs_list, dim=1)[0, :, _POS].detach().cpu()
