"""VMAS controllers returning raw per-node forces ``[B*N, 2]``.

The trainer squashes forces to the action range. All policies share a
``forward(perception, field_terms)`` interface. Kinds: ``expert`` (fixed
program), ``parametric`` (learned static weights), ``hybrid`` (neural gates
modulate program weights), ``hybrid_res`` (program plus residual force),
``neural`` (1-hop baseline), and ``neural_dK`` (depth-K message-passing net).
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, cast

import torch
from torch import Tensor, nn
from torch_geometric.utils import scatter as pyg_scatter
from vmas_diffield.scenarios import SCENARIO_SPEC

if TYPE_CHECKING:
    from vmas_diffield.field_terms import FieldTerms
    from vmas_diffield.scenarios import ScenarioSpec
    from vmas_diffield.vmas_env import Perception

# Fixed expert/teacher weights. Keys match each scenario's active terms.
# Drive terms outweigh formation terms; sampling uses larger values to
# compensate for tanh action squashing and sustain coverage speed.
EXPERT_WEIGHTS: dict[str, dict[str, float]] = {
    # Under full observability, follow/lead_dir mainly reinforce shared intent.
    "flocking": {
        "w_separation": 0.5, "w_alignment": 1.0, "w_cohesion": 0.6, "w_goal": 0.5,
        "w_follow": 0.5, "w_lead_dir": 0.4,
    },
    # With partial observability, lead_dir broadcasts goal direction to agents
    # that cannot see it; follow provides a weaker attraction to informed agents.
    "flocking_beacon": {
        "w_separation": 0.5, "w_alignment": 0.8, "w_cohesion": 0.8, "w_goal": 1.5,
        "w_follow": 0.6, "w_lead_dir": 1.5,
    },
    # PD arrival: goal attracts, brake damps at the target, avoid prevents collisions.
    "navigation": {"w_separation": 0.5, "w_goal": 2.5, "w_brake": 1.5, "w_avoid": 0.5},
    # Priority: pursue sensed targets, recruit informed teammates, then explore.
    "discovery": {
        "w_separation": 1.0, "w_sense": 2.5, "w_recruit": 2.0,
        "w_explore": 1.2, "w_disperse": 0.9,
    },
    # High drive compensates for tanh squashing; preserve the priority cascade.
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


# Learned field-policy initialization. The small default lets training learn
# weights from physics; sampling starts higher to reach its saturation regime.
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
