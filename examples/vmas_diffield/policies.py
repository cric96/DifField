"""Controllers for the VMAS experiments.

All policies output a raw per-node force ``[B*N, 2]`` (the trainer squashes it
to the VMAS action range via ``vmas_env.pack_actions``). They share a common
``forward(perception, field_terms)`` interface so the trainers are
policy-agnostic; the field terms come from the scenario's aggregate program
(``programs.FieldProgram``).

The policy spectrum, from pure program to pure network:

  * ``expert``     — the aggregate program with hand-set weights (no training):
                     the zero-parameter baseline.
  * ``parametric`` — the program with *learned static* weights (SHAC): the
                     interpretable parameter-optimization object.
  * ``hybrid``     — ``ModulatedFieldPolicy``: the program whose weights are
                     *controlled* per-agent per-step by a neural gate network
                     ("aggregate + MLP/GNN" — the controller reads local
                     features that include 1-hop neighbour means, so it is a
                     1-hop message-passing net). Gates are zero-init to 1:
                     at initialization the hybrid IS the parametric program.
  * ``hybrid_res`` — the older residual hybrid (program + free force
                     correction), kept as an opt-in ablation: structured
                     control vs unstructured correction.
  * ``neural``     — black-box baseline (1-hop message passing + MLP).
  * ``neural_dK``  — depth-K message-passing student (imitation/expressivity
                     study: how deep must a feed-forward GNN be to imitate the
                     recurrent multi-hop program?).
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, cast

import torch
from torch import Tensor, nn
from torch_geometric.utils import scatter as pyg_scatter
from vmas_diffield.vmas_env import SCENARIO_SPEC

if TYPE_CHECKING:
    from vmas_diffield.vmas_env import FieldTerms, Perception, ScenarioSpec

# Hand-set program weights: the `expert` baseline and the imitation teacher.
# Keys must mirror SCENARIO_SPEC[scenario].field_terms exactly. Magnitude
# rationale (measured, session-5/6 notes): "drive" terms (goal/sense/recruit/
# social/explore/disperse/follow/lead_dir) sit well above "formation" glue
# (separation/avoid) because a comparably-weighted mix partially self-cancels,
# and under VMAS drag (v* ~ 0.4*F) that leaves a barely-perceptible terminal
# speed. ParametricFieldPolicy.forward is linear in the weights.
#
# The weight SCALE matters, not just the ratios: the actuator squashes the
# summed force with tanh (vmas_env.pack_actions), so a resultant of ~1 cruises
# at ~74% throttle at best. For tasks that pay by swept ground per step
# (sampling: reward only on never-sampled cells) the drive terms must push the
# resultant well past tanh saturation — that "full throttle" is exactly the
# regime a trained GNN's unbounded output head lives in, and under-driving it
# was measured to halve the program's speed (0.23 vs 0.44) and give the GNN a
# +50% covered-cells lead despite the program picking better cells.
EXPERT_WEIGHTS: dict[str, dict[str, float]] = {
    # follow/lead_dir modest under full observability: every agent already
    # sees the goal, the leader field mainly adds cohesion of intent
    "flocking": {
        "w_separation": 0.5, "w_alignment": 1.0, "w_cohesion": 0.6, "w_goal": 0.5,
        "w_follow": 0.5, "w_lead_dir": 0.4,
    },
    # partial obs: dissemination IS the drive — non-knowers can only steer
    # from follow (descend toward the knower) and lead_dir (broadcast goal
    # direction); goal only acts on the knower itself. A/B (64 envs, 60
    # steps): zeroing lead_dir costs -26% goal_prox, zeroing follow costs ~0
    # (the broadcast direction suffices; converging on the messenger mostly
    # drags the formation) — hence lead_dir high, follow modest.
    "flocking_beacon": {
        "w_separation": 0.5, "w_alignment": 0.8, "w_cohesion": 0.8, "w_goal": 1.5,
        "w_follow": 0.6, "w_lead_dir": 1.5,
    },
    # PD arrival: goal is the P gain, brake the D gain (parks agents on their
    # goals); avoid only fires on genuine collision courses
    "navigation": {"w_separation": 0.5, "w_goal": 2.5, "w_brake": 1.5, "w_avoid": 0.5},
    # sense > recruit > explore mirrors the program's priority cascade: chase
    # what you see, else join a teammate who sees something, else patrol
    "discovery": {
        "w_separation": 1.0, "w_sense": 2.5, "w_recruit": 2.0,
        "w_explore": 1.2, "w_disperse": 0.9,
    },
    # saturation scale (see block comment): resultant ~3-4x past tanh's knee,
    # priority cascade preserved (sense > social > explore/separation >
    # disperse). Swept: x1 of this set covers 0.036 of the grid at speed 0.23;
    # this set covers 0.050 at 0.36 and out-collects the trained GNN 1.01 vs
    # 0.81 value/step (96 envs x 3 eval seeds).
    "sampling": {
        "w_separation": 4.0, "w_sense": 12.0, "w_social": 6.0,
        "w_explore": 4.0, "w_disperse": 3.0,
    },
}


def _mlp(in_dim: int, hidden: int, out_dim: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(in_dim, hidden), nn.SiLU(),
        nn.Linear(hidden, hidden), nn.SiLU(),
        nn.Linear(hidden, out_dim),
    )


def neighbour_means(p: Perception) -> Tensor:
    """Mean relative neighbour position & velocity per node -> ``[B*N, 4]``."""
    src, tgt = p.edge_index[0], p.edge_index[1]
    rel_pos = p.pos[src] - p.pos[tgt]
    rel_vel = p.vel[src] - p.vel[tgt]
    agg_pos = pyg_scatter(rel_pos, tgt, dim=0, dim_size=p.num_nodes, reduce="mean")
    agg_vel = pyg_scatter(rel_vel, tgt, dim=0, dim_size=p.num_nodes, reduce="mean")
    return torch.cat([agg_pos, agg_vel], dim=-1)


def _local_features(p: Perception) -> Tensor:
    """Per-node black-box features: own vel + scenario obs tail + neighbour means."""
    return torch.cat([p.vel, p.extra, neighbour_means(p)], dim=-1)


def local_feature_dim(p: Perception) -> int:
    return 2 + p.extra.shape[-1] + 4


def _init_raw_weights(n_terms: int, init: float) -> nn.Parameter:
    raw = math.log(math.expm1(init))
    return nn.Parameter(torch.full((n_terms,), raw))


# Per-scenario softplus-weight initialisation for the LEARNED field policies
# (parametric/hybrid). The default 0.3 is a deliberately *wrong* near-zero
# start: every reported learning curve then shows the weights being learned
# through the physics, not a hand prior being kept (navigation recovers its
# whole PD structure from here — w_goal rises first, then the brake once agents
# start arriving; see the phase-uniform-window note in train.py). Sampling is
# the exception: its productive regime is the tanh-saturating high-drive scale
# (see EXPERT_WEIGHTS[sampling]); from 0.3 the field LR needs far more than the
# 150-update budget to crawl there (measured: the reward curve is still rising
# monotonically at update 149, weights only reach ~1.3 of the ~6-12 optimum),
# so the learned policy is left under-driven and loses coverage to the GNN. A
# scalar warm start in the productive band lets the budget refine instead of
# crawl.
SCENARIO_INIT: dict[str, float] = {"sampling": 3.0}
DEFAULT_INIT = 0.3


def init_for_scenario(scenario: str) -> float:
    return SCENARIO_INIT.get(scenario, DEFAULT_INIT)


# ── Interpretable parametric field policy ─────────────────────────────────────


class ParametricFieldPolicy(nn.Module):
    """Force = sum_k softplus(w_k) * field_term_k over the scenario's active terms."""

    def __init__(self, terms: tuple[str, ...], *, init: float = 0.3) -> None:
        super().__init__()
        self.terms = terms
        self.raw = _init_raw_weights(len(terms), init)

    @property
    def weights(self) -> dict[str, float]:
        w = torch.nn.functional.softplus(self.raw)
        return {f"w_{t}": float(w[i].item()) for i, t in enumerate(self.terms)}

    def forward(self, p: Perception, ft: FieldTerms) -> Tensor:
        w = torch.nn.functional.softplus(self.raw)
        terms = torch.stack([getattr(ft, t) for t in self.terms], dim=0)  # [K, B*N, 2]
        return (w.view(-1, 1, 1) * terms).sum(dim=0)


def make_expert_policy(scenario: str) -> ParametricFieldPolicy:
    """The scenario program with hand-set weights, frozen (no training)."""
    spec = SCENARIO_SPEC[scenario]
    expert = ParametricFieldPolicy(spec.field_terms)
    target = EXPERT_WEIGHTS[scenario]
    with torch.no_grad():
        for i, term in enumerate(spec.field_terms):
            expert.raw[i] = math.log(math.expm1(target[f"w_{term}"]))
    for prm in expert.parameters():
        prm.requires_grad_(False)
    return expert


# ── Hybrid: the program's weights, controlled by a neural gate network ────────


class ModulatedFieldPolicy(nn.Module):
    """Aggregate program + neural controller of its parameters (the "hybrid").

    ``force_i = sum_k softplus(w_k) * gate_k(i) * term_k(i)`` with
    ``gate(i) = 2*sigmoid(head(body(features_i))) in (0, 2)`` and a zero-init
    head, so gates start exactly at 1 and the policy *is* the parametric
    program at initialization. The controller reads the same local features as
    the neural baseline (own vel + obs tail + 1-hop neighbour means), i.e. it
    is a small 1-hop message-passing network whose entire output space is the
    program's interpretable knobs — you can plot gate_k(i, t) and read the
    adaptation strategy.
    """

    def __init__(
        self, terms: tuple[str, ...], p_dim: int, *, hidden: int = 64, init: float = 0.3
    ) -> None:
        super().__init__()
        self.terms = terms
        self.raw = _init_raw_weights(len(terms), init)
        self.body: nn.Sequential = nn.Sequential(
            nn.Linear(p_dim, hidden), nn.SiLU(), nn.Linear(hidden, hidden), nn.SiLU()
        )
        self.gate_head: nn.Linear = nn.Linear(hidden, len(terms))
        nn.init.zeros_(self.gate_head.weight)
        nn.init.zeros_(self.gate_head.bias)

    @property
    def weights(self) -> dict[str, float]:
        """Base (static) weights; the gates modulate around these."""
        w = torch.nn.functional.softplus(self.raw)
        return {f"w_{t}": float(w[i].item()) for i, t in enumerate(self.terms)}

    def gates(self, p: Perception) -> Tensor:
        """Per-agent, per-step multiplicative gates in (0, 2) -> ``[B*N, K]``."""
        return 2.0 * torch.sigmoid(self.gate_head(self.body(_local_features(p))))

    def forward(self, p: Perception, ft: FieldTerms) -> Tensor:
        w = torch.nn.functional.softplus(self.raw)          # [K]
        g = self.gates(p)                                   # [B*N, K]
        terms = torch.stack([getattr(ft, t) for t in self.terms], dim=0)  # [K, B*N, 2]
        w_eff = (w.view(-1, 1) * g.transpose(0, 1)).unsqueeze(-1)         # [K, B*N, 1]
        return (w_eff * terms).sum(dim=0)


# ── Residual hybrid (opt-in ablation): program + free force correction ────────


class HybridFieldPolicy(nn.Module):
    """Parametric field program + a zero-initialised neural residual force.

    Kept as the ``hybrid_res`` ablation against ``ModulatedFieldPolicy``:
    does *controlling the program* beat *correcting its output*?
    """

    def __init__(self, terms: tuple[str, ...], p_dim: int, *, hidden: int = 64) -> None:
        super().__init__()
        self.field: ParametricFieldPolicy = ParametricFieldPolicy(terms)
        self.body: nn.Sequential = nn.Sequential(
            nn.Linear(p_dim, hidden), nn.SiLU(), nn.Linear(hidden, hidden), nn.SiLU()
        )
        self.head: nn.Linear = nn.Linear(hidden, 2)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    @property
    def weights(self) -> dict[str, float]:
        return self.field.weights

    def residual(self, p: Perception) -> Tensor:
        """The neural correction term alone (for regularisation)."""
        return self.head(self.body(_local_features(p)))

    def forward(self, p: Perception, ft: FieldTerms) -> Tensor:
        return self.field(p, ft) + self.residual(p)


# ── Black-box neural baselines ────────────────────────────────────────────────


class NeuralGNNPolicy(nn.Module):
    """1-hop message passing over the agent graph + per-agent MLP head."""

    def __init__(self, p_dim: int, *, hidden: int = 64) -> None:
        super().__init__()
        self.net = _mlp(p_dim, hidden, 2)

    def forward(self, p: Perception, ft: FieldTerms) -> Tensor:
        return self.net(_local_features(p))


class MessagePassingPolicy(nn.Module):
    """Depth-``k`` message passing: the receptive field is exactly ``k`` hops.

    ``h0 = enc(own obs)``; k rounds of ``h <- mlp([h, mean_j h_j])``; force =
    ``head(h_k)``. The imitation/expressivity study varies ``k`` against the
    aggregate program teacher, whose recurrent fields have an unbounded
    (episode-long) effective receptive field.
    """

    def __init__(self, own_dim: int, *, hidden: int = 64, depth: int = 2) -> None:
        super().__init__()
        self.depth = depth
        self.enc = nn.Sequential(nn.Linear(own_dim, hidden), nn.SiLU())
        self.layers = nn.ModuleList(
            nn.Sequential(nn.Linear(2 * hidden, hidden), nn.SiLU()) for _ in range(depth)
        )
        self.head = nn.Linear(hidden, 2)

    def forward(self, p: Perception, ft: FieldTerms) -> Tensor:
        src, tgt = p.edge_index[0], p.edge_index[1]
        h = self.enc(torch.cat([p.vel, p.extra], dim=-1))
        for layer in self.layers:
            m = pyg_scatter(h[src], tgt, dim=0, dim_size=p.num_nodes, reduce="mean")
            h = layer(torch.cat([h, m], dim=-1))
        return self.head(h)


# ── Critic ────────────────────────────────────────────────────────────────────


class Critic(nn.Module):
    """Per-agent value function on local features."""

    def __init__(self, p_dim: int, *, hidden: int = 64) -> None:
        super().__init__()
        self.net = _mlp(p_dim, hidden, 1)

    def forward(self, p: Perception) -> Tensor:
        return self.net(_local_features(p)).squeeze(-1)


def build_policy(
    kind: str, p: Perception, spec: ScenarioSpec, scenario: str, *, hidden: int = 64
) -> nn.Module:
    p_dim = local_feature_dim(p)
    init = init_for_scenario(scenario)
    if kind == "expert":
        return make_expert_policy(scenario)
    if kind == "parametric":
        return ParametricFieldPolicy(spec.field_terms, init=init)
    if kind == "hybrid":
        return ModulatedFieldPolicy(spec.field_terms, p_dim, hidden=hidden, init=init)
    if kind == "hybrid_res":
        return HybridFieldPolicy(spec.field_terms, p_dim, hidden=hidden)
    if kind == "neural":
        return NeuralGNNPolicy(p_dim, hidden=hidden)
    if kind.startswith("neural_d"):
        depth = int(kind.removeprefix("neural_d"))
        return MessagePassingPolicy(2 + p.extra.shape[-1], hidden=hidden, depth=depth)
    raise ValueError(f"unknown policy kind: {kind}")


def policy_weights(policy: nn.Module) -> dict[str, float] | None:
    """``policy.weights`` for the interpretable policies, ``None`` for neural ones.

    ``nn.Module``'s dynamic ``__getattr__`` hides the property's return type
    from static analysis; ``cast`` documents the runtime contract.
    """
    w = getattr(policy, "weights", None)
    return cast("dict[str, float] | None", w)
