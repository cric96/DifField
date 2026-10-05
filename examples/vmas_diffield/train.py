"""Shared VMAS rewards, evaluation helpers, and trainer entry points."""

from __future__ import annotations

import dataclasses
import time
from dataclasses import dataclass

import torch
import torch.nn.functional as F  # noqa: N812
from torch import Tensor, nn
from vmas_diffield.field_terms import FieldTerms
from vmas_diffield.policies import (
    HybridFieldPolicy,
    ModulatedFieldPolicy,
    policy_weights,
)
from vmas_diffield.programs import FieldProgram
from vmas_diffield.scenarios import SCENARIO_SPEC
from vmas_diffield.vmas_env import (
    Perception,
    detach_env,
    env0_positions,
    obs_to_perception,
    pack_actions,
)


def make_program(scenario: str, args) -> FieldProgram:
    return FieldProgram(
        scenario,
        sep=args.sep,
        elect_grain=args.elect_grain,
        desc_tau=args.desc_tau,
        nav_arrive_radius=args.nav_arrive_radius,
    )


def _node_reward(rews: list[Tensor]) -> Tensor:
    return torch.stack(rews, dim=1).reshape(-1)


def detach_terms(ft: FieldTerms) -> FieldTerms:
    """Detached copy of every tensor field (for imitation datasets)."""
    vals = {
        f.name: (v.detach() if isinstance(v := getattr(ft, f.name), Tensor) else v)
        for f in dataclasses.fields(ft)
    }
    return FieldTerms(**vals)


def smooth_collision_penalty(
    pos_flat: Tensor, *, batch_size: int, n_agents: int, min_dist: float
) -> Tensor:
    """Differentiable mean soft-overlap penalty (substitute for VMAS's hard term)."""
    pos = pos_flat.reshape(batch_size, n_agents, 2)
    dmat = torch.cdist(pos, pos)
    eye = torch.eye(n_agents, device=pos.device).unsqueeze(0)
    overlap = torch.relu(min_dist - dmat) * (1.0 - eye)
    return (overlap**2).sum(dim=(1, 2)).mean()


def flocking_shaped_reward(
    p: Perception,
    *,
    radius: float,
    sep: float,
    cohesion_k: float,
    cohesion_temp: float,
    w_align: float,
    w_cohesion: float,
    w_separation: float,
    w_goal: float,
) -> Tensor:
    """Reward alignment, bounded neighbor cohesion, goal proximity, and separation."""
    b, n = p.batch_size, p.n_agents
    pos = p.pos.reshape(b, n, 2)
    vel = p.vel.reshape(b, n, 2)
    dmat = torch.cdist(pos, pos)
    eye = torch.eye(n, device=pos.device).unsqueeze(0)

    neigh = ((dmat <= radius) & (dmat > 0)).to(pos.dtype)
    deg = neigh.sum(dim=-1)
    has = (deg > 0).to(pos.dtype)
    deg_c = deg.clamp(min=1).unsqueeze(-1)
    nv_mean = torch.bmm(neigh, vel) / deg_c
    align_cos = F.cosine_similarity(vel, nv_mean, dim=-1, eps=1e-6) * has

    # Saturation avoids rewarding arbitrarily dense flocks.
    soft_adj = torch.sigmoid((radius - dmat) / cohesion_temp) * (1.0 - eye)
    coh = (soft_adj.sum(dim=-1) / cohesion_k).clamp(max=1.0)

    close = ((dmat < sep) & (dmat > 0)).to(pos.dtype)
    sep_pen = (torch.relu((sep - dmat) / sep) * close).sum(dim=-1)

    goal_rew = torch.zeros(b, n, device=pos.device)
    if p.true_goal_rel is not None:
        gd = p.true_goal_rel.reshape(b, n, 2).norm(dim=-1)
        goal_rew = 1.0 - gd.clamp(max=1.0)

    r = w_align * align_cos + w_cohesion * coh - w_separation * sep_pen + w_goal * goal_rew
    return r.reshape(b * n)


def discovery_shaped_reward(
    env, p_next: Perception, rews: list[Tensor], *, w_approach: float
) -> Tensor:
    """Add a dense nearest-uncovered-target signal to VMAS's sparse reward."""
    base = _node_reward(rews)
    sc = env.scenario
    b, n = p_next.batch_size, p_next.n_agents
    agents_pos = p_next.pos.reshape(b, n, 2)
    dists = torch.cdist(agents_pos, sc.targets_pos)  # [B, n, n_targets]
    dists = dists.masked_fill(sc.covered_targets.unsqueeze(1), float("inf"))
    nearest = torch.nan_to_num(dists.min(dim=-1).values, posinf=0.0)  # none left -> no penalty
    return base - w_approach * nearest.reshape(b * n)


def navigation_shaped_reward(
    p_next: Perception,
    *,
    w_approach: float,
    approach_scale: float,
    w_hold: float,
    hold_scale: float,
) -> Tensor:
    """Combine broad goal approach and sharp on-goal rewards to favor arrival and holding."""
    if p_next.true_goal_rel is None:
        raise ValueError("navigation reward needs true_goal_rel in the perception")
    gd = p_next.true_goal_rel.norm(dim=-1)
    approach = torch.exp(-gd / approach_scale)
    hold = torch.exp(-gd / hold_scale)
    return w_approach * approach + w_hold * hold


def shac_reward(scenario: str, args, env, p_next: Perception, rews: list[Tensor]) -> Tensor:
    """Return the configured per-node reward used in training and evaluation."""
    if scenario == "navigation":
        return navigation_shaped_reward(
            p_next,
            w_approach=args.nav_w_approach,
            approach_scale=args.nav_approach_scale,
            w_hold=args.nav_w_hold,
            hold_scale=args.nav_hold_scale,
        )
    if scenario in ("flocking", "flocking_beacon"):
        w_goal = args.beacon_w_goal if scenario == "flocking_beacon" else args.flock_w_goal
        return flocking_shaped_reward(
            p_next,
            radius=args.radius,
            sep=args.sep,
            cohesion_k=args.flock_cohesion_k,
            cohesion_temp=args.flock_cohesion_temp,
            w_align=args.flock_w_align,
            w_cohesion=args.flock_w_cohesion,
            w_separation=args.flock_w_separation,
            w_goal=w_goal,
        )
    if scenario == "discovery":
        return discovery_shaped_reward(env, p_next, rews, w_approach=args.disc_w_approach)
    return _node_reward(rews)


@torch.no_grad()
def scenario_metrics(p: Perception, scenario: str) -> dict[str, float]:
    spec = SCENARIO_SPEC[scenario]
    vel = p.vel.reshape(p.batch_size, p.n_agents, 2)
    dirs = vel / (vel.norm(dim=-1, keepdim=True) + 1e-8)
    pos = p.pos.reshape(p.batch_size, p.n_agents, 2)
    dmat = torch.cdist(pos, pos)
    n = p.n_agents
    spread = (dmat.sum(dim=(1, 2)) / max(1, n * (n - 1))).mean()
    out = {
        "order": float(dirs.mean(dim=1).norm(dim=-1).mean().item()),
        "spread": float(spread.item()),
    }
    if spec.has_goal and p.true_goal_rel is not None:
        gd = p.true_goal_rel.norm(dim=-1)
        out["goal_dist"] = float(gd.mean().item())
        out["on_goal_frac"] = float((gd < 0.1).float().mean().item())
        # Bounded tracking score: 1 at the target and 0 beyond unit distance.
        out["goal_prox"] = float((1.0 - gd.clamp(max=1.0)).mean().item())
    return out


@torch.no_grad()
def task_coverage(env, scenario: str) -> float | None:
    """Ground-truth per-step task coverage (not a geometric proxy)."""
    sc = env.scenario
    if scenario == "discovery":
        return float(sc.covered_targets.float().mean().item())
    if scenario == "sampling":
        return float(sc.sampled.float().mean().item())
    return None


@dataclass
class RunResult:
    seed: int
    eval_steps: list[int]
    eval_reward: list[float]
    eval_metric: list[float]
    weight_traj: list[dict[str, float] | None]
    final: dict[str, float]
    weights: dict[str, float] | None
    pos_trace: list[Tensor]
    train_time_s: float
    policy_state: dict[str, Tensor]
    n_params: int
    gate_trace: list[Tensor] = dataclasses.field(default_factory=list)
    actor_loss_curve: list[float] = dataclasses.field(default_factory=list)
    critic_loss_curve: list[float] = dataclasses.field(default_factory=list)
    grad_norm_field: list[float] = dataclasses.field(default_factory=list)
    grad_norm_neural: list[float] = dataclasses.field(default_factory=list)


def _td_lambda(rewards, values, last_value, gamma, lam):
    targets = [torch.zeros_like(last_value)] * len(rewards)
    running = last_value
    for t in reversed(range(len(rewards))):
        nxt = values[t + 1] if t + 1 < len(values) else last_value
        running = rewards[t] + gamma * ((1.0 - lam) * nxt + lam * running)
        targets[t] = running
    return targets


def _reset_rollout(env, program: FieldProgram, *, scenario: str, args):
    """Reset env + program together; warm the fields up on the initial graph."""
    obs = env.reset()
    detach_env(env)
    program.reset()
    if args.program_warmup > 0:
        p0 = obs_to_perception(obs, scenario=scenario, radius=args.radius)
        program.warmup(p0, args.program_warmup)
    return obs


@torch.no_grad()
def evaluate(
    env,
    policy,
    program: FieldProgram,
    *,
    scenario: str,
    args,
    u_range,
    steps,
    record_trace: bool = False,
    record_gates: bool = False,
):
    spec = SCENARIO_SPEC[scenario]
    obs = _reset_rollout(env, program, scenario=scenario, args=args)
    rewards, metric_acc, coverage_acc, trace, gate_trace = [], [], [], [], []
    for _ in range(steps):
        p = obs_to_perception(obs, scenario=scenario, radius=args.radius)
        ft = program.step(p)
        force = policy(p, ft)
        if record_gates and isinstance(policy, ModulatedFieldPolicy):
            g = policy.gates(p).reshape(p.batch_size, p.n_agents, -1)
            gate_trace.append(g[0].detach().cpu())
        acts = pack_actions(force, batch_size=p.batch_size, n_agents=p.n_agents, u_range=u_range)
        if record_trace:
            trace.append(env0_positions(obs))
        obs, rews, _, _ = env.step(acts)
        p_next = obs_to_perception(obs, scenario=scenario, radius=args.radius)
        rewards.append(float(shac_reward(scenario, args, env, p_next, rews).mean().item()))
        metric_acc.append(scenario_metrics(p_next, scenario))
        cov = task_coverage(env, scenario)
        if cov is not None:
            coverage_acc.append(cov)
    avg = {k: sum(m[k] for m in metric_acc) / len(metric_acc) for k in metric_acc[0]}
    avg["reward"] = sum(rewards) / len(rewards)
    if coverage_acc:
        avg["coverage"] = sum(coverage_acc) / len(coverage_acc)
    avg["primary"] = avg[spec.primary_metric]
    return avg, trace, gate_trace


@torch.no_grad()
def evaluate_perturbed(
    env,
    policy,
    program: FieldProgram,
    *,
    scenario: str,
    args,
    u_range,
    steps,
    drop_rate: float = 0.0,
    swap_knower: bool = False,
    seed: int = 0,
):
    """Return per-step metrics after mid-episode communication or role perturbations."""
    spec = SCENARIO_SPEC[scenario]
    gen = torch.Generator(device="cpu").manual_seed(seed)
    obs = _reset_rollout(env, program, scenario=scenario, args=args)
    onset = steps // 2
    series: dict[str, list[float]] = {"reward": [], "primary": []}
    for t in range(steps):
        shift = 1 if (swap_knower and t >= onset) else 0
        p = obs_to_perception(obs, scenario=scenario, radius=args.radius, knower_shift=shift)
        if drop_rate > 0.0 and t >= onset:
            keep = torch.rand(p.edge_index.shape[1], generator=gen) >= drop_rate
            p = dataclasses.replace(p, edge_index=p.edge_index[:, keep.to(p.edge_index.device)])
        ft = program.step(p)
        force = policy(p, ft)
        acts = pack_actions(force, batch_size=p.batch_size, n_agents=p.n_agents, u_range=u_range)
        obs, rews, _, _ = env.step(acts)
        p_next = obs_to_perception(obs, scenario=scenario, radius=args.radius)
        reward = float(shac_reward(scenario, args, env, p_next, rews).mean().item())
        metrics = scenario_metrics(p_next, scenario)
        cov = task_coverage(env, scenario)
        if cov is not None:
            metrics["coverage"] = cov
        metrics["reward"] = reward
        series["reward"].append(reward)
        series["primary"].append(metrics[spec.primary_metric])
    return series


def _expert_run(env, policy, program, *, scenario, args, u_range, seed) -> RunResult:
    """Evaluate the frozen hand-weighted program (no training)."""
    t0 = time.perf_counter()
    ev, trace, _ = evaluate(
        env,
        policy,
        program,
        scenario=scenario,
        args=args,
        u_range=u_range,
        steps=args.episode_len,
        record_trace=True,
    )
    w = policy_weights(policy)
    return RunResult(
        seed=seed,
        eval_steps=[0, args.updates - 1],
        eval_reward=[ev["reward"]] * 2,
        eval_metric=[ev["primary"]] * 2,
        weight_traj=[w, w],
        final={
            "reward": ev["reward"],
            "primary": ev["primary"],
            **{m: v for m, v in ev.items() if m not in ("reward", "primary")},
        },
        weights=w,
        pos_trace=trace,
        train_time_s=time.perf_counter() - t0,
        policy_state={k: v.detach().cpu().clone() for k, v in policy.state_dict().items()},
        n_params=0,
    )


def _param_groups(policy: nn.Module) -> tuple[list[nn.Parameter], list[nn.Parameter]]:
    """Split policy parameters into field and neural groups."""
    if isinstance(policy, ModulatedFieldPolicy):
        return [policy.raw], list(policy.body.parameters()) + list(policy.gate_head.parameters())
    if isinstance(policy, HybridFieldPolicy):
        return (
            list(policy.field.parameters()),
            list(policy.body.parameters()) + list(policy.head.parameters()),
        )
    if hasattr(policy, "weights"):  # ParametricFieldPolicy
        return list(policy.parameters()), []
    return [], list(policy.parameters())


def _grad_norm(params: list[nn.Parameter]) -> float:
    """L2 norm over the (pre-clip) gradients of ``params``; 0.0 when empty."""
    sq = sum(float(prm.grad.pow(2).sum().item()) for prm in params if prm.grad is not None)
    return sq**0.5


def _make_actor_optimizer(policy: nn.Module, args) -> torch.optim.Adam:
    """Build the actor optimizer with separate field and neural learning rates."""
    field_params, neural_params = _param_groups(policy)
    groups = []
    if field_params:
        groups.append({"params": field_params, "lr": args.actor_lr_field})
    if neural_params:
        groups.append({"params": neural_params, "lr": args.actor_lr_neural})
    return torch.optim.Adam(groups)


def train_shac(scenario: str, kind: str, *, seed: int, args, env) -> RunResult:
    from vmas_diffield.shac import train_shac as train  # noqa: PLC0415

    return train(scenario, kind, seed=seed, args=args, env=env)


@dataclass
class ImitationResult:
    kind: str
    seed: int
    loss_curve: list[float]
    recovered: dict[str, float] | None
    expert: dict[str, float]
    final_loss: float
    pos_trace: list[Tensor]


def train_imitation(scenario: str, kind: str, *, seed: int, args, env) -> ImitationResult:
    from vmas_diffield.imitation import train_imitation as train  # noqa: PLC0415

    return train(scenario, kind, seed=seed, args=args, env=env)
