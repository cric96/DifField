#!/usr/bin/env python3
"""DIFFIELD as a reinforcement-learning problem, solved with SHAC.

This experiment answers the reviewer concern that the learning evaluation relies
on teacher-generated trajectories and teacher forcing (R1), and that the paper
should show learning of expressive collective programs rather than fitting a
hand-designed controller. Here there is **no teacher**: a swarm policy is trained
purely from a reward signal, using *Short-Horizon Actor-Critic* (SHAC, Xu et al.,
2022) -- a method that is only applicable because DIFFIELD is a *differentiable*
simulator.

Why SHAC fits DIFFIELD
----------------------
SHAC backpropagates **analytic gradients** through a short horizon ``h`` of the
differentiable dynamics to obtain a low-variance policy gradient, and uses a
learned critic to bootstrap the return beyond the horizon (avoiding the
exploding/vanishing gradients of full-episode BPTT). DIFFIELD provides exactly
the differentiable, GPU-parallel collective dynamics SHAC needs.

What is compared
----------------
Two policies are trained under the *identical* SHAC objective and reward:
  * ``NeuralActor``  -- a small MLP on DIFFIELD-aggregated local observations
    (own velocity + gather/scatter alignment / cohesion / separation features);
    learns an expressive collective program.
  * ``BoidsActor``   -- the same 3 interpretable weights (w_sep, w_align,
    w_cohesion) as the imitation experiment, but now optimised by RL from reward
    instead of imitation. Keeps full interpretability.

Reward (emergent flocking, no teacher; all terms differentiable in the state):
  + velocity alignment with neighbours      (move together)
  + cohesion (closeness to local centroid)  (stay grouped)
  + cruising speed                          (keep moving, avoid the trivial stop)
  - separation violations                   (avoid collisions)
  - wall penalty                            (stay in the arena)

Outputs (under ``generated/boids-shac/`` by default)
----------------------------------------------------
  - ``shac_summary.json`` / ``shac_summary.csv``        -- final metrics (mean/CI)
  - ``shac_reward_curve.png``                           -- return vs. episode
  - ``shac_flocking_metrics.png``                       -- order param + cohesion
  - ``shac_final_metrics.png``                          -- final behaviour bars
  - ``shac_neural_flocking.gif`` / ``shac_boids_flocking.gif`` -- learned swarms

Run:
    uv run python examples/boids-evaluation/shac_rl.py
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import torch
import torch.nn.functional as F  # noqa: N812
from torch import nn

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "examples"))

from shared.metrics import aggregate, aggregate_curve, mean  # noqa: E402
from shared.plotting import band, color_of, export_moving_gif, panel_label  # noqa: E402

from diffield.dsl import AggregateContext, gather_avg, gather_sum, scatter  # noqa: E402
from diffield.sim import SpatialScenario, limit_speed, normalize_vectors  # noqa: E402

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from shared.plotting import apply_paper_style

    apply_paper_style()
except ImportError:
    plt = None


# Display names for figures: the interpretable boids field-program policy is the
# DIFFIELD policy; the MLP is the neural baseline. POLICY_TAG maps each policy
# onto the paper-wide role palette (style.py) so colors match every other figure.
POLICY_DISPLAY = {"neural": "neural", "boids": "DIFFIELD"}
POLICY_TAG = {"neural": "neural", "boids": "diffield"}
OBS_DIM = 9  # [vel(2), align(2), cohesion(2), sep_force(2), has_neigh(1)]


# ═════════════════════════════════════════════════════════════════════════════
# DIFFERENTIABLE FLOCKING ENVIRONMENT (DIFFIELD-based)
# ═════════════════════════════════════════════════════════════════════════════


@dataclass
class RewardWeights:
    align: float = 1.0
    cohesion: float = 0.5
    speed: float = 0.4
    separation: float = 1.0
    wall: float = 0.5


@dataclass
class State:
    pos: torch.Tensor
    vel: torch.Tensor


def _bounce_diff(
    pos: torch.Tensor, vel: torch.Tensor, low: float = 0.0, high: float = 1.0
) -> tuple[torch.Tensor, torch.Tensor]:
    """Autograd-safe box reflection (no in-place ops).

    Flips velocity components where the position leaves the box and clamps the
    position back inside. The out-of-bounds mask is detached (discrete), but the
    gradient flows through the velocity scaling and the position clamp.
    """
    out = ((pos < low) | (pos > high)).to(pos.dtype)
    new_vel = vel * (1.0 - 2.0 * out)
    new_pos = pos.clamp(low, high)
    return new_pos, new_vel


class FlockingEnv:
    """Differentiable boids arena with DIFFIELD-based local perception.

    The per-agent observation is computed with DIFFIELD primitives
    (``scatter`` / ``gather_avg`` / ``gather_sum``) so the policy perceives the
    swarm through a field program; the transition and reward are differentiable
    tensor ops, so SHAC can backpropagate through the rollout.
    """

    def __init__(
        self,
        *,
        num_nodes: int,
        radius: float,
        sep: float,
        dt: float,
        damping: float,
        max_speed: float,
        velocity_scale: float,
        wall_margin: float,
        cohesion_k: float,
        cohesion_temp: float,
        reward_weights: RewardWeights,
        device: torch.device,
    ) -> None:
        self.num_nodes = num_nodes
        self.radius = radius
        self.sep = sep
        self.dt = dt
        self.damping = damping
        self.max_speed = max_speed
        self.velocity_scale = velocity_scale
        self.wall_margin = wall_margin
        self.cohesion_k = cohesion_k
        self.cohesion_temp = cohesion_temp
        self.rw = reward_weights
        self.device = device

    def reset(self, seed: int) -> State:
        gen = torch.Generator().manual_seed(seed)
        pos = torch.rand(self.num_nodes, 2, generator=gen, dtype=torch.float32)
        directions = torch.rand(self.num_nodes, 2, generator=gen, dtype=torch.float32) * 2 - 1
        vel = normalize_vectors(directions) * self.velocity_scale
        return State(pos.to(self.device), vel.to(self.device))

    def observe(self, state: State) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Per-agent local features via DIFFIELD gather/scatter (differentiable)."""
        pos, vel = state.pos, state.vel
        scenario = SpatialScenario(positions=pos, edge_radius=self.radius, device=self.device)
        ctx = AggregateContext(
            scenario.edge_index, scenario.num_nodes, edge_weight=scenario.edge_weight
        )
        scenario.sync_context(ctx._ctx)

        with ctx.round():
            neigh_vel = gather_avg(scatter(vel))
            neigh_pos = gather_avg(scatter(pos))
            delta = pos - scatter(pos)
            dist = delta.norm(dim=-1)
            sep_mask = ((dist <= self.sep) & (dist > 0)).pointwise()
            sep_vec = gather_sum(delta * sep_mask)

        align = neigh_vel - vel
        cohesion = neigh_pos - pos
        sep_dir = normalize_vectors(sep_vec)
        align_dir = normalize_vectors(align)
        cohesion_dir = normalize_vectors(cohesion)

        deg = torch.bincount(scenario.edge_index[1], minlength=self.num_nodes).to(self.device)
        has_neigh = (deg > 0).to(pos.dtype).unsqueeze(-1)

        # Observation components are normalised to ~O(1) so the policy/critic see
        # well-scaled inputs (velocity in units of max_speed, cohesion in units of
        # the interaction radius). Neighbour-derived terms zero out for isolates.
        obs = torch.cat(
            [
                vel / self.max_speed,
                (align / self.max_speed) * has_neigh,
                (cohesion / self.radius) * has_neigh,
                sep_dir * has_neigh,
                has_neigh,
            ],
            dim=-1,
        )
        feats = {
            # Unit steering directions and magnitude-preserving scaled forces
            # (alignment/cohesion weaken as they are achieved, like classic boids),
            # both available to the interpretable policy.
            "align_dir": align_dir,
            "cohesion_dir": cohesion_dir,
            "sep_dir": sep_dir,
            "align_scaled": (align / self.max_speed) * has_neigh,
            "cohesion_scaled": (cohesion / self.radius) * has_neigh,
            "has_neigh": has_neigh,
        }
        return obs, feats

    def step(self, state: State, acc: torch.Tensor) -> State:
        """Differentiable integration: damping + acceleration, clip, move, bounce."""
        pre_clip = self.damping * state.vel + self.dt * acc
        pre_clip = torch.nan_to_num(pre_clip, nan=0.0, posinf=0.0, neginf=0.0)
        clipped = limit_speed(pre_clip, self.max_speed)
        new_pos = state.pos + self.dt * clipped
        new_pos, new_vel = _bounce_diff(new_pos, clipped)
        return State(new_pos, new_vel)

    def reward(self, state: State) -> torch.Tensor:
        """Per-agent emergent-flocking reward (differentiable in the state)."""
        pos, vel = state.pos, state.vel
        dmat = torch.cdist(pos, pos)
        neigh = ((dmat <= self.radius) & (dmat > 0)).to(pos.dtype)
        deg = neigh.sum(dim=1)
        has = (deg > 0).to(pos.dtype)
        deg_c = deg.clamp(min=1).unsqueeze(-1)

        nv_mean = (neigh @ vel) / deg_c
        align_cos = F.cosine_similarity(vel, nv_mean, dim=-1, eps=1e-6) * has

        # Soft connectivity cohesion: reward having up to ``cohesion_k`` neighbours
        # (differentiable soft degree via a sigmoid of the radius gap). It rewards
        # staying in a group, saturates so there is no incentive to collapse, and
        # does not penalise the spatial extent of a moving flock (which a
        # distance-to-centroid reward would, fighting alignment).
        soft_adj = torch.sigmoid((self.radius - dmat) / self.cohesion_temp)
        soft_adj = soft_adj * (1.0 - torch.eye(pos.shape[0], device=pos.device))
        soft_deg = soft_adj.sum(dim=1)
        coh = (soft_deg / self.cohesion_k).clamp(max=1.0)

        close = ((dmat < self.sep) & (dmat > 0)).to(pos.dtype)
        sep_pen = (torch.relu((self.sep - dmat) / self.sep) * close).sum(dim=1)

        speed = vel.norm(dim=-1)
        speed_rew = (speed.clamp(max=self.max_speed) / self.max_speed)

        m = self.wall_margin
        wall = (torch.relu(m - pos) + torch.relu(pos - (1.0 - m))).sum(dim=1) / m

        return (
            self.rw.align * align_cos
            + self.rw.cohesion * coh
            + self.rw.speed * speed_rew
            - self.rw.separation * sep_pen
            - self.rw.wall * wall
        )

    @torch.no_grad()
    def flocking_metrics(self, state: State) -> dict[str, float]:
        """Standard swarm descriptors: polarization, cohesion radius, collisions."""
        pos, vel = state.pos, state.vel
        dirs = vel / (vel.norm(dim=-1, keepdim=True) + 1e-8)
        order = dirs.mean(dim=0).norm().item()
        cohesion_radius = (pos - pos.mean(dim=0)).norm(dim=-1).mean().item()
        dmat = torch.cdist(pos, pos)
        n = pos.shape[0]
        pairs = (dmat < self.sep).sum().item() - n  # exclude diagonal
        collisions = pairs / max(1, n * (n - 1))
        return {"order": order, "cohesion_radius": cohesion_radius, "collisions": collisions}


# ═════════════════════════════════════════════════════════════════════════════
# POLICIES (actor) AND CRITIC
# ═════════════════════════════════════════════════════════════════════════════


def _mlp(in_dim: int, hidden: int, out_dim: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(in_dim, hidden), nn.SiLU(),
        nn.Linear(hidden, hidden), nn.SiLU(),
        nn.Linear(hidden, out_dim),
    )


class NeuralActor(nn.Module):
    """MLP on DIFFIELD-aggregated local observations -> bounded acceleration.

    The final layer is zero-initialised so the policy starts as a near no-op
    (agents keep their initial headings, giving a low initial order parameter);
    this makes subsequent learning clearly visible rather than masked by a random
    bias that trivially aligns the swarm.
    """

    def __init__(self, *, hidden: int, act_scale: float) -> None:
        super().__init__()
        self.body = nn.Sequential(
            nn.Linear(OBS_DIM, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.SiLU(),
        )
        self.head = nn.Linear(hidden, 2)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)
        self.act_scale = act_scale

    def act(self, obs: torch.Tensor, feats: dict[str, torch.Tensor]) -> torch.Tensor:
        return self.act_scale * torch.tanh(self.head(self.body(obs)))


class BoidsActor(nn.Module):
    """The 3 interpretable boids weights, optimised by RL (softplus-positive).

    Combines normalised steering directions (separation / alignment / cohesion)
    so all three forces share the ``act_scale`` magnitude and the learned weights
    are well-conditioned, directly interpretable relative importances.
    """

    def __init__(self, *, act_scale: float, init: float = 0.5) -> None:
        super().__init__()
        raw = math.log(math.expm1(init))  # inverse softplus
        self.w_sep_raw = nn.Parameter(torch.tensor(raw))
        self.w_align_raw = nn.Parameter(torch.tensor(raw))
        self.w_cohesion_raw = nn.Parameter(torch.tensor(raw))
        self.act_scale = act_scale

    @property
    def weights(self) -> dict[str, float]:
        return {
            "w_sep": F.softplus(self.w_sep_raw).item(),
            "w_align": F.softplus(self.w_align_raw).item(),
            "w_cohesion": F.softplus(self.w_cohesion_raw).item(),
        }

    def act(self, obs: torch.Tensor, feats: dict[str, torch.Tensor]) -> torch.Tensor:
        hn = feats["has_neigh"]
        acc = (
            F.softplus(self.w_sep_raw) * feats["sep_dir"]
            + F.softplus(self.w_align_raw) * feats["align_scaled"]
            + F.softplus(self.w_cohesion_raw) * feats["cohesion_scaled"]
        )
        return self.act_scale * acc * hn


class Critic(nn.Module):
    """Per-agent value function V(obs)."""

    def __init__(self, *, hidden: int) -> None:
        super().__init__()
        self.net = _mlp(OBS_DIM, hidden, 1)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.net(obs).squeeze(-1)


def build_actor(kind: str, args: argparse.Namespace, device: torch.device) -> nn.Module:
    if kind == "neural":
        return NeuralActor(hidden=args.hidden, act_scale=args.act_scale).to(device)
    return BoidsActor(act_scale=args.act_scale, init=0.5).to(device)


# ═════════════════════════════════════════════════════════════════════════════
# SHAC TRAINER
# ═════════════════════════════════════════════════════════════════════════════


def td_lambda_targets(
    rewards: list[torch.Tensor],
    values: list[torch.Tensor],
    last_value: torch.Tensor,
    gamma: float,
    lam: float,
) -> list[torch.Tensor]:
    """TD(lambda) / lambda-return targets for each visited state (detached)."""
    targets: list[torch.Tensor] = [torch.zeros_like(last_value)] * len(rewards)
    running = last_value
    for t in reversed(range(len(rewards))):
        next_v = values[t + 1] if t + 1 < len(values) else last_value
        running = rewards[t] + gamma * ((1.0 - lam) * next_v + lam * running)
        targets[t] = running
    return targets


class SHAC:
    """Short-Horizon Actor-Critic over the differentiable flocking environment."""

    def __init__(
        self,
        env: FlockingEnv,
        actor: nn.Module,
        *,
        hidden: int,
        horizon: int,
        gamma: float,
        lam: float,
        tau: float,
        actor_lr: float,
        critic_lr: float,
        critic_epochs: int,
        grad_clip: float,
        device: torch.device,
    ) -> None:
        self.env = env
        self.actor = actor
        self.critic = Critic(hidden=hidden).to(device)
        self.target_critic = Critic(hidden=hidden).to(device)
        self.target_critic.load_state_dict(self.critic.state_dict())
        self.opt_actor = torch.optim.Adam(actor.parameters(), lr=actor_lr)
        self.opt_critic = torch.optim.Adam(self.critic.parameters(), lr=critic_lr)
        self.h = horizon
        self.gamma = gamma
        self.lam = lam
        self.tau = tau
        self.critic_epochs = critic_epochs
        self.grad_clip = grad_clip
        # Exploration: stddev of additive action noise during training rollouts
        # (set per episode with annealing; 0 = deterministic, used at eval).
        self.explore_std = 0.0

    def train_chunk(self, state: State) -> tuple[State, float]:
        """One SHAC iteration: short-horizon actor update + critic regression."""
        # --- Short-horizon differentiable rollout (actor loss) ---
        obs_seq: list[torch.Tensor] = []
        rew_seq: list[torch.Tensor] = []
        actor_loss = torch.zeros((), device=state.pos.device)
        discount = 1.0
        cur = state
        for _ in range(self.h):
            obs, feats = self.env.observe(cur)
            acc = self.actor.act(obs, feats)
            if self.explore_std > 0.0:
                # Reparameterized exploration: noise perturbs the rollout (and
                # thus the return) while the analytic gradient flows through the
                # deterministic mean action.
                acc = acc + self.explore_std * torch.randn_like(acc)
            cur = self.env.step(cur, acc)
            r = self.env.reward(cur)
            actor_loss = actor_loss - discount * r.mean()
            obs_seq.append(obs)
            rew_seq.append(r)
            discount *= self.gamma

        obs_terminal, _ = self.env.observe(cur)
        terminal_value = self.target_critic(obs_terminal)
        actor_loss = actor_loss - discount * terminal_value.mean()

        self.opt_actor.zero_grad()
        actor_loss.backward()
        nn.utils.clip_grad_norm_(self.actor.parameters(), self.grad_clip)
        self.opt_actor.step()

        # --- Critic regression on TD(lambda) targets (detached data) ---
        with torch.no_grad():
            det_obs = [o.detach() for o in obs_seq]
            det_obs_terminal = obs_terminal.detach()
            values = [self.target_critic(o) for o in det_obs]
            last_value = self.target_critic(det_obs_terminal)
            targets = td_lambda_targets(rew_seq, values, last_value, self.gamma, self.lam)
            targets = [t.detach() for t in targets]

        for _ in range(self.critic_epochs):
            loss_c = torch.zeros((), device=state.pos.device)
            for o, tgt in zip(det_obs, targets, strict=True):
                loss_c = loss_c + F.mse_loss(self.critic(o), tgt)
            loss_c = loss_c / self.h
            self.opt_critic.zero_grad()
            loss_c.backward()
            self.opt_critic.step()

        self._polyak_update()

        mean_reward = float(torch.stack(rew_seq).mean().item())
        next_state = State(cur.pos.detach(), cur.vel.detach())
        return next_state, mean_reward

    def _polyak_update(self) -> None:
        for tp, p in zip(
            self.target_critic.parameters(), self.critic.parameters(), strict=True
        ):
            tp.data.mul_(1.0 - self.tau).add_(self.tau * p.data)

    @torch.no_grad()
    def evaluate(self, *, eval_seed: int, rounds: int) -> dict:
        """Greedy free-running rollout; returns mean reward + flocking metrics + trace."""
        state = self.env.reset(eval_seed)
        pos_seq, vel_seq, edge_seq = [], [], []
        rewards, metrics = [], []
        for _ in range(rounds):
            scenario = SpatialScenario(
                positions=state.pos, edge_radius=self.env.radius, device=self.env.device
            )
            edge_seq.append(scenario.edge_index.clone())
            obs, feats = self.env.observe(state)
            acc = self.actor.act(obs, feats)
            state = self.env.step(state, acc)
            rewards.append(float(self.env.reward(state).mean().item()))
            metrics.append(self.env.flocking_metrics(state))
            pos_seq.append(state.pos.clone())
            vel_seq.append(state.vel.clone())
        return {
            "reward": sum(rewards) / len(rewards),
            "order": sum(m["order"] for m in metrics) / len(metrics),
            "cohesion_radius": sum(m["cohesion_radius"] for m in metrics) / len(metrics),
            "collisions": sum(m["collisions"] for m in metrics) / len(metrics),
            "pos_seq": torch.stack(pos_seq),
            "vel_seq": torch.stack(vel_seq),
            "edge_seq": edge_seq,
        }


# ═════════════════════════════════════════════════════════════════════════════
# RUN BOOKKEEPING
# ═════════════════════════════════════════════════════════════════════════════


@dataclass
class RunResult:
    seed: int
    eval_episodes: list[int]
    eval_reward: list[float]
    eval_order: list[float]
    eval_cohesion: list[float]
    eval_collisions: list[float]
    final: dict[str, float]
    weights: dict[str, float] | None
    train_time_s: float


@dataclass
class PolicyReport:
    kind: str
    runs: list[RunResult] = field(default_factory=list)
    best_eval: dict | None = None
    best_reward: float = -math.inf


def train_one(
    kind: str,
    *,
    seed: int,
    args: argparse.Namespace,
    device: torch.device,
) -> tuple[RunResult, dict]:
    torch.manual_seed(seed)
    env = FlockingEnv(
        num_nodes=args.num_nodes, radius=args.radius, sep=args.sep, dt=args.dt,
        damping=args.damping, max_speed=args.max_speed, velocity_scale=args.velocity_scale,
        wall_margin=args.wall_margin,
        cohesion_k=args.cohesion_k, cohesion_temp=args.cohesion_temp,
        reward_weights=RewardWeights(
            align=args.w_align, cohesion=args.w_cohesion, speed=args.w_speed,
            separation=args.w_separation, wall=args.w_wall,
        ),
        device=device,
    )
    actor = build_actor(kind, args, device)
    actor_lr = args.actor_lr_neural if kind == "neural" else args.actor_lr_boids
    shac = SHAC(
        env, actor, hidden=args.hidden, horizon=args.horizon, gamma=args.gamma,
        lam=args.lam, tau=args.tau, actor_lr=actor_lr, critic_lr=args.critic_lr,
        critic_epochs=args.critic_epochs, grad_clip=args.grad_clip, device=device,
    )

    chunks_per_ep = max(1, args.episode_len // args.horizon)
    eval_episodes, eval_reward, eval_order = [], [], []
    eval_cohesion, eval_collisions = [], []

    t0 = time.perf_counter()
    for ep in range(args.episodes):
        # Anneal exploration noise (fraction of the action scale) over training.
        frac = ep / max(1, args.episodes - 1)
        explore_frac = args.explore_frac_start + frac * (
            args.explore_frac_end - args.explore_frac_start
        )
        shac.explore_std = args.act_scale * explore_frac

        state = env.reset(1000 * seed + ep)  # fresh swarm config each episode
        for _ in range(chunks_per_ep):
            state, _ = shac.train_chunk(state)

        if ep % args.eval_every == 0 or ep == args.episodes - 1:
            ev = shac.evaluate(eval_seed=args.eval_seed, rounds=args.episode_len)
            eval_episodes.append(ep)
            eval_reward.append(ev["reward"])
            eval_order.append(ev["order"])
            eval_cohesion.append(ev["cohesion_radius"])
            eval_collisions.append(ev["collisions"])
            print(
                f"  [{kind:>6}] seed={seed} ep={ep:>3}/{args.episodes}  "
                f"reward={ev['reward']:.3f}  order={ev['order']:.3f}  "
                f"cohesion={ev['cohesion_radius']:.3f}  coll={ev['collisions']:.3f}"
            )
    train_time = time.perf_counter() - t0

    # Final performance = mean over the last few eval checkpoints (robust to the
    # single-rollout noise of greedy evaluation), plus a final rollout for the GIF.
    final_eval = shac.evaluate(eval_seed=args.eval_seed, rounds=args.episode_len)
    k = max(1, min(args.final_window, len(eval_reward)))
    weights = actor.weights if isinstance(actor, BoidsActor) else None
    run = RunResult(
        seed=seed,
        eval_episodes=eval_episodes,
        eval_reward=eval_reward,
        eval_order=eval_order,
        eval_cohesion=eval_cohesion,
        eval_collisions=eval_collisions,
        final={
            "reward": mean(eval_reward[-k:]),
            "order": mean(eval_order[-k:]),
            "cohesion_radius": mean(eval_cohesion[-k:]),
            "collisions": mean(eval_collisions[-k:]),
        },
        weights=weights,
        train_time_s=train_time,
    )
    return run, final_eval


# ═════════════════════════════════════════════════════════════════════════════
# PLOTTING
# ═════════════════════════════════════════════════════════════════════════════


def plot_reward_curve(reports: list[PolicyReport], out_path: Path) -> None:
    if plt is None:
        return
    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    for rep in reports:
        means, cis = aggregate_curve([r.eval_reward for r in rep.runs])
        x = rep.runs[0].eval_episodes[: len(means)]
        band(ax, x, means, cis, role=POLICY_TAG[rep.kind], label=POLICY_DISPLAY[rep.kind])
    ax.set_xlabel("episode")
    ax.set_ylabel("mean per-step reward (greedy eval)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved {out_path}")


def plot_flocking_metrics(reports: list[PolicyReport], out_path: Path) -> None:
    if plt is None:
        return
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.4))
    for rep in reports:
        x = rep.runs[0].eval_episodes
        om, oc = aggregate_curve([r.eval_order for r in rep.runs])
        cm, cc = aggregate_curve([r.eval_cohesion for r in rep.runs])
        label = POLICY_DISPLAY[rep.kind]
        band(axes[0], x[: len(om)], om, oc, role=POLICY_TAG[rep.kind], label=label)
        band(axes[1], x[: len(cm)], cm, cc, role=POLICY_TAG[rep.kind], label=label)
    axes[0].set_xlabel("episode")
    axes[0].set_ylabel("order parameter (↑)")
    axes[1].set_xlabel("episode")
    axes[1].set_ylabel("mean distance to centroid (↓)")
    for ax, tag in zip(axes, "ab", strict=True):
        ax.legend()
        ax.grid(True, alpha=0.3)
        panel_label(ax, tag)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved {out_path}")


def plot_final_metrics(reports: list[PolicyReport], out_path: Path) -> None:
    if plt is None:
        return
    metrics = [("reward", "reward (↑)"), ("order", "order (↑)"),
               ("collisions", "collisions (↓)")]
    fig, axes = plt.subplots(1, len(metrics), figsize=(4.2 * len(metrics), 4.2))
    for ax, (key, ylabel), tag in zip(axes, metrics, "abc", strict=True):
        labels = [POLICY_DISPLAY[rep.kind] for rep in reports]
        means = [aggregate([r.final[key] for r in rep.runs])["mean"] for rep in reports]
        cis = [aggregate([r.final[key] for r in rep.runs])["ci"] for rep in reports]
        ax.bar(
            labels, means, yerr=cis, capsize=5,
            color=[color_of(POLICY_TAG[rep.kind]) for rep in reports],
        )
        ax.set_ylabel(ylabel)
        ax.grid(True, axis="y", alpha=0.3)
        panel_label(ax, tag)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved {out_path}")


def render_flocking_gif(
    report: PolicyReport, *, rounds: int, out_dir: Path, fps: int,
    show_links: bool, links_alpha: float, links_width: float,
) -> None:
    if report.best_eval is None:
        return
    ev = report.best_eval
    pos_seq, vel_seq, edge_seq = ev["pos_seq"], ev["vel_seq"], ev["edge_seq"]
    export_moving_gif(
        positions_by_round={r: pos_seq[r] for r in range(rounds)},
        values_by_round={r: vel_seq[r].norm(dim=1) for r in range(rounds)},
        edge_index_by_round={r: edge_seq[r] for r in range(rounds)},
        source_idx=0,
        output_path=str(out_dir / f"shac_{POLICY_TAG[report.kind]}_flocking.gif"),
        title=f"SHAC-learned flocking ({POLICY_DISPLAY[report.kind]} policy, reward only)",
        fps=fps, show_links=show_links, links_alpha=links_alpha,
        links_width=links_width, show_source=False,
    )


# ═════════════════════════════════════════════════════════════════════════════
# PERSISTENCE
# ═════════════════════════════════════════════════════════════════════════════


def build_summary(reports: list[PolicyReport], args: argparse.Namespace) -> dict:
    out: dict = {
        "config": {
            "seeds": args.seed_list, "eval_seed": args.eval_seed,
            "episodes": args.episodes, "episode_len": args.episode_len,
            "horizon": args.horizon, "num_nodes": args.num_nodes,
            "gamma": args.gamma, "lam": args.lam,
            "reward_weights": {
                "align": args.w_align, "cohesion": args.w_cohesion, "speed": args.w_speed,
                "separation": args.w_separation, "wall": args.w_wall,
            },
        },
        "results": [],
    }
    for rep in reports:
        entry = {
            "policy": rep.kind,
            "train_time_s": aggregate([r.train_time_s for r in rep.runs]),
            "final_reward": aggregate([r.final["reward"] for r in rep.runs]),
            "final_order": aggregate([r.final["order"] for r in rep.runs]),
            "final_cohesion_radius": aggregate([r.final["cohesion_radius"] for r in rep.runs]),
            "final_collisions": aggregate([r.final["collisions"] for r in rep.runs]),
            "per_seed": [
                {"seed": r.seed, **r.final, "weights": r.weights} for r in rep.runs
            ],
        }
        out["results"].append(entry)
    return out


def write_csv(summary: dict, csv_path: Path) -> None:
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "policy", "final_reward_mean", "final_reward_ci",
            "final_order_mean", "final_order_ci",
            "final_cohesion_mean", "final_cohesion_ci",
            "final_collisions_mean", "final_collisions_ci",
        ])
        for r in summary["results"]:
            coh = r["final_cohesion_radius"]
            writer.writerow([
                r["policy"],
                f"{r['final_reward']['mean']:.4f}", f"{r['final_reward']['ci']:.4f}",
                f"{r['final_order']['mean']:.4f}", f"{r['final_order']['ci']:.4f}",
                f"{coh['mean']:.4f}", f"{coh['ci']:.4f}",
                f"{r['final_collisions']['mean']:.4f}", f"{r['final_collisions']['ci']:.4f}",
            ])


# ═════════════════════════════════════════════════════════════════════════════
# MAIN
# ═════════════════════════════════════════════════════════════════════════════


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="DIFFIELD-as-RL via SHAC (emergent flocking)")
    p.add_argument("--policies", type=str, default="neural,boids")
    p.add_argument("--seeds", type=str, default="0,1,2,3,4")
    p.add_argument("--eval-seed", type=int, default=777)
    p.add_argument("--episodes", type=int, default=80)
    p.add_argument("--episode-len", type=int, default=48)
    p.add_argument("--horizon", type=int, default=16)
    p.add_argument("--eval-every", type=int, default=5)
    p.add_argument("--final-window", type=int, default=3,
                   help="Average the last N eval checkpoints for the final metric")
    p.add_argument("--num-nodes", type=int, default=50)
    # SHAC hyper-parameters
    p.add_argument("--hidden", type=int, default=64)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--lam", type=float, default=0.95)
    p.add_argument("--tau", type=float, default=0.2)
    p.add_argument("--actor-lr-neural", type=float, default=3e-3)
    p.add_argument("--actor-lr-boids", type=float, default=2e-2)
    p.add_argument("--critic-lr", type=float, default=1e-3)
    p.add_argument("--critic-epochs", type=int, default=8)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--act-scale", type=float, default=0.03)
    # Exploration (annealed action-noise stddev as a fraction of act-scale).
    p.add_argument("--explore-frac-start", type=float, default=0.5)
    p.add_argument("--explore-frac-end", type=float, default=0.05)
    # Physics
    p.add_argument("--radius", type=float, default=0.23)
    p.add_argument("--sep", type=float, default=0.06)
    p.add_argument("--dt", type=float, default=1.0)
    p.add_argument("--damping", type=float, default=0.94)
    p.add_argument("--max-speed", type=float, default=0.014)
    p.add_argument("--velocity-scale", type=float, default=0.014)
    p.add_argument("--wall-margin", type=float, default=0.05)
    # Reward weights (tuned so both policies learn coordinated flocking).
    p.add_argument("--w-align", type=float, default=4.0)
    p.add_argument("--w-cohesion", type=float, default=0.4)
    p.add_argument("--w-speed", type=float, default=1.2)
    p.add_argument("--w-separation", type=float, default=0.6)
    p.add_argument("--w-wall", type=float, default=0.4)
    p.add_argument("--cohesion-k", type=float, default=6.0,
                   help="Target neighbour count for the soft connectivity reward")
    p.add_argument("--cohesion-temp", type=float, default=0.02,
                   help="Softness of the connectivity sigmoid (in distance units)")
    # Rendering / output
    p.add_argument("--gif-fps", type=int, default=8)
    p.add_argument("--hide-links", action="store_true")
    p.add_argument("--links-alpha", type=float, default=0.15)
    p.add_argument("--links-width", type=float, default=0.6)
    p.add_argument("--no-render", action="store_true")
    p.add_argument("--out-dir", type=str, default="generated/boids-shac")
    p.add_argument("--device", type=str, default="")
    return p.parse_args()


def _device(s: str) -> torch.device:
    if s:
        return torch.device(s)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def parse_csv(text: str) -> list[str]:
    return [t.strip() for t in text.split(",") if t.strip()]


def main() -> None:
    args = parse_args()
    device = _device(args.device)
    seeds = [int(s) for s in parse_csv(args.seeds)]
    policies = parse_csv(args.policies)
    args.seed_list = seeds

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=== DIFFIELD-as-RL via SHAC (emergent flocking, no teacher) ===")
    print(f"  device={device}  policies={policies}  seeds={seeds}")
    print(f"  episodes={args.episodes} ep_len={args.episode_len} horizon={args.horizon} "
          f"nodes={args.num_nodes}")

    reports: list[PolicyReport] = []
    for kind in policies:
        print(f"\n--- Training {kind} policy ---")
        rep = PolicyReport(kind=kind)
        for seed in seeds:
            run, final_eval = train_one(kind, seed=seed, args=args, device=device)
            rep.runs.append(run)
            if run.final["reward"] > rep.best_reward:
                rep.best_reward = run.final["reward"]
                rep.best_eval = final_eval
        reports.append(rep)

    summary = build_summary(reports, args)
    with (out_dir / "shac_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    write_csv(summary, out_dir / "shac_summary.csv")

    plot_reward_curve(reports, out_dir / "shac_reward_curve.png")
    plot_flocking_metrics(reports, out_dir / "shac_flocking_metrics.png")
    plot_final_metrics(reports, out_dir / "shac_final_metrics.png")

    if not args.no_render:
        for rep in reports:
            render_flocking_gif(
                rep, rounds=args.episode_len, out_dir=out_dir, fps=args.gif_fps,
                show_links=not args.hide_links, links_alpha=args.links_alpha,
                links_width=args.links_width,
            )

    print("\n=== Final behaviour (mean +/- 95% CI over seeds) ===")
    header = f"{'policy':>8} | {'reward':>16} | {'order':>16} | {'collisions':>16}"
    print(header)
    print("-" * len(header))
    for r in summary["results"]:
        rw, od, co = r["final_reward"], r["final_order"], r["final_collisions"]
        print(
            f"{r['policy']:>8} | {rw['mean']:>7.3f} +/- {rw['ci']:<5.3f} | "
            f"{od['mean']:>7.3f} +/- {od['ci']:<5.3f} | "
            f"{co['mean']:>7.4f} +/- {co['ci']:<5.4f}"
        )
    for r in summary["results"]:
        if r["policy"] == "boids" and r["per_seed"] and r["per_seed"][0]["weights"]:
            ws = [s["weights"] for s in r["per_seed"]]
            print("\nboids policy learned weights (RL from reward, no teacher):")
            for name in ("w_sep", "w_align", "w_cohesion"):
                st = aggregate([w[name] for w in ws])
                print(f"  {name:>12} = {st['mean']:.4f} +/- {st['ci']:.4f}")
    print(f"\nArtifacts saved to {out_dir}")


if __name__ == "__main__":
    main()
