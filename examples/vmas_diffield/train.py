"""Trainers for the VMAS experiments: SHAC (analytic-gradient RL) and imitation.

SHAC uses a per-update reset short-horizon scheme (each update resets all parallel
envs, rolls out ``horizon`` differentiable steps, backprops the discounted VMAS
reward + a critic bootstrap into the policy, then fits the critic on TD(lambda)
targets). Per-update reset + ``detach_env`` keep each update's autograd graph
independent. For collision-heavy scenarios a differentiable soft-collision
penalty is added to the actor loss (VMAS's own collision reward is a hard
threshold → zero gradient; this restores a usable analytic gradient).

Phase-uniform windows: after each reset the env is advanced by ``k ~
U{0..t_max}`` *no-grad* steps under the current (exploring) policy before the
differentiable window starts, so across updates the gradient sees every phase
of the evaluated episode — travel, arrival, hold — not just the first
``horizon`` steps from spawn. Without this, any term whose payoff lives late
in the episode is invisible to the analytic gradient: measured on navigation
from the flat init, the ``brake`` D-gain's |grad| is ~0.0003 (noise) when
training only ever covers steps 0-12 from spawn, and SHAC deletes the brake
while inflating ``separation`` as a degenerate stand-in (w_brake 0.3→0.03,
w_separation 0.3→1.2); with phase-uniform windows the brake gradient turns on
(~0.007→0.07 as agents start reaching goals) and the true PD structure is
recovered from scratch (w_goal≈2, w_brake≈2.7, formation glue small).

The phase cap ``t_max`` ramps 0 → ``episode_len - horizon`` over the first
``phase_ramp_frac`` of updates (then stays full). A from-scratch policy needs
the spawn-anchored distribution early: under a full-range prefix from update 0
the still-random policy scatters the envs and the black-box GNN's take-off
became bimodal across seeds (measured: 2/5 seeds reach ~1.43 — higher than the
old spawn-only scheme ever did — while 3/5 stall near 0.2-0.6 for most of the
run; reward CI blew up 0.03 → 0.66). The ramp restores the old scheme's
reliable early bootstrap for every policy, then extends coverage to the
late-episode phases the program terms need — one identical scheme for all
policies, so the baseline comparison stays fair.

Per-update diagnostics (actor loss, critic loss, pre-clip grad norms split into
field-weight vs neural parameters) are recorded on the ``RunResult`` so every
run can show *that* and *how* it learns (``shac_loss_curve.png``).

Every rollout carries the scenario's aggregate program (``programs.FieldProgram``)
alongside the env: the program is reset with the env, optionally pre-converged
for a few warm-up rounds, and stepped once per physics step (one aggregate
round per control step — the deployment semantics of field calculus).

Imitation behaviour-clones the hand-weighted aggregate program (the ``expert``
policy). Because the program is stateful, the dataset stores the *field terms*
computed during the expert rollout, so every student sees exactly the fields
the expert acted on; depth-K GNN students (``neural_dK``) probe how much
feed-forward receptive field it takes to match the recurrent program.
"""

from __future__ import annotations

import dataclasses
import time
from dataclasses import dataclass

import torch
import torch.nn.functional as F  # noqa: N812
from torch import Tensor, nn
from vmas_diffield.policies import (
    EXPERT_WEIGHTS,
    Critic,
    HybridFieldPolicy,
    ModulatedFieldPolicy,
    _local_features,
    build_policy,
    make_expert_policy,
    policy_weights,
)
from vmas_diffield.programs import FieldProgram
from vmas_diffield.vmas_env import (
    SCENARIO_SPEC,
    FieldTerms,
    Perception,
    detach_env,
    env0_positions,
    env_action_range,
    obs_to_perception,
    pack_actions,
)


def make_program(scenario: str, args) -> FieldProgram:
    return FieldProgram(
        scenario, sep=args.sep, elect_grain=args.elect_grain, desc_tau=args.desc_tau,
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
    """Emergent-flocking reward (order + cohesion + goal-tracking - separation).

    VMAS's built-in ``flocking`` reward only shapes pairwise inter-agent distance
    toward a fixed 0.1 target and ignores heading and the scripted target agent
    entirely, so it does not reward the order/target-tracking behaviour this
    experiment reports. Built from field-perception quantities so what SHAC
    optimises matches the metrics/plots. Uses ``true_goal_rel``: the evaluation
    signal may use privileged state even when policies see a masked goal
    (flocking_beacon).
    """
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

    # Soft connectivity cohesion: reward having up to `cohesion_k` neighbours
    # (saturates, so there is no incentive to collapse the flock).
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
    """VMAS's sparse discovery reward, plus a dense pull toward the nearest
    uncovered target.

    Discovery's real reward only fires when >= 2 agents are *simultaneously*
    within ``covering_range`` of a target -- a rare event over a short SHAC
    horizon, so the analytic gradient has essentially nothing to climb before
    agents stumble into range by chance. ``-distance to nearest uncovered
    target`` (0 once none remain) gives a signal at every step.
    """
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
    """Dense arrival reward that pays for *holding* the goal, not just reaching it.

    VMAS navigation's built-in reward is shared *net progress* toward the goal
    (a telescoping distance-reduction term) plus a tiny all-on-goal bonus. Its
    episode integral depends only on how much distance was closed, so it pays
    the same whether an agent parks on its goal or sails through the 0.1 on-goal
    disc at cruising speed and orbits it. Worse, the program's ``brake`` (D-gain)
    term lowers approach speed near the goal, so SHAC *raises* that reward by
    deleting the brake — which is exactly what tanks the reported
    ``on_goal_frac`` (measured full run: trained w_brake 1.5 -> 0.07,
    on_goal_frac 0.50 -> 0.26; reward went *up* 0.044 -> 0.053 as accuracy fell).

    This reward instead peaks when an agent *sits on* its goal. Two bounded
    higher-is-better bumps of the goal distance ``gd``: a wide ``approach`` bump
    gives a usable analytic gradient across the whole arena, and a sharp
    ``hold`` bump (scale ~ the 0.1 on-goal radius) makes the last centimetres —
    and therefore braking to a stop — the highest-paying region, so it moves
    *with* ``on_goal_frac`` instead of against it. Collisions stay penalised via
    the differentiable ``smooth_collision_penalty`` already added to the SHAC
    actor loss for this scenario.
    """
    if p_next.true_goal_rel is None:
        raise ValueError("navigation reward needs true_goal_rel in the perception")
    gd = p_next.true_goal_rel.norm(dim=-1)
    approach = torch.exp(-gd / approach_scale)
    hold = torch.exp(-gd / hold_scale)
    return w_approach * approach + w_hold * hold


def shac_reward(scenario: str, args, env, p_next: Perception, rews: list[Tensor]) -> Tensor:
    """Per-node reward used for both SHAC training and evaluation.

    ``flocking``/``flocking_beacon``, ``discovery`` and ``navigation`` use
    shaped/dense rewards (see above); the remaining scenarios' built-in VMAS
    reward already matches their reported primary metric.
    """
    if scenario == "navigation":
        return navigation_shaped_reward(
            p_next,
            w_approach=args.nav_w_approach,
            approach_scale=args.nav_approach_scale,
            w_hold=args.nav_w_hold,
            hold_scale=args.nav_hold_scale,
        )
    if scenario in ("flocking", "flocking_beacon"):
        # flocking_beacon's task is "track the target most agents cannot see":
        # goal tracking dominates its reward, otherwise an idle clump scores
        # high on alignment+cohesion+spawn proximity (measured: an untrained
        # GNN standing still reached 0.84/episode) and the dissemination
        # question the scenario exists for never gets asked.
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
        # Bounded higher-is-better tracking score (flocking_beacon's primary
        # metric): 1 at the target, 0 beyond unit distance.
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


# ── SHAC ──────────────────────────────────────────────────────────────────────


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
    # Per-update training diagnostics (empty for the untrained expert):
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
    """Self-healing probe: per-step metric series under mid-episode perturbation.

    From step ``steps//2`` onward, communication edges are dropped i.i.d. at
    ``drop_rate`` (sustained outage), and/or — for partial-obs scenarios — the
    goal-knower role moves to a different agent (``swap_knower``), forcing the
    program to re-elect and re-converge its fields. Policies and program see
    the same perturbed graph. Returns per-step series, not averages.
    """
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
        env, policy, program, scenario=scenario, args=args, u_range=u_range,
        steps=args.episode_len, record_trace=True,
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
    """Split a policy's parameters into (interpretable field weights, neural).

    Single source of truth for both the two-LR actor optimizer and the
    per-update grad-norm diagnostics.
    """
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
    """Two LR groups: interpretable field gains tolerate the aggressive field
    LR; the neural part (gates / residual / black-box net) trains at the safe
    neural LR (using the field LR for both destabilised the neural part before
    the program learned anything)."""
    field_params, neural_params = _param_groups(policy)
    groups = []
    if field_params:
        groups.append({"params": field_params, "lr": args.actor_lr_field})
    if neural_params:
        groups.append({"params": neural_params, "lr": args.actor_lr_neural})
    return torch.optim.Adam(groups)


def train_shac(scenario: str, kind: str, *, seed: int, args, env) -> RunResult:  # noqa: PLR0912, PLR0915
    torch.manual_seed(seed)
    device = torch.device(args.device)
    spec = SCENARIO_SPEC[scenario]
    u_range = env_action_range(env)
    program = make_program(scenario, args)

    obs = env.reset()
    p0 = obs_to_perception(obs, scenario=scenario, radius=args.radius)
    policy = build_policy(kind, p0, spec, scenario, hidden=args.hidden).to(device)
    if kind == "expert":
        return _expert_run(
            env, policy, program, scenario=scenario, args=args, u_range=u_range, seed=seed
        )
    n_params = sum(t.numel() for t in policy.parameters())
    feat_dim = _local_features(p0).shape[-1]
    critic = Critic(feat_dim, hidden=args.hidden).to(device)
    target_critic = Critic(feat_dim, hidden=args.hidden).to(device)
    target_critic.load_state_dict(critic.state_dict())

    opt_a = _make_actor_optimizer(policy, args)
    field_params, neural_params = _param_groups(policy)
    # NB (2026-07-15): a cosine LR anneal on the actor was tried to shrink the
    # neural policy's late-training reward oscillation (which widens its
    # across-seed CI). It reliably tightened flocking's neural CI (0.24->0.15)
    # but was REMOVED because on scenarios where the neural policy is still
    # improving at update 150 (navigation: on_goal_frac was still climbing) the
    # decayed LR CAPS it — measured navigation neural 0.28->0.14, i.e. the anneal
    # made the black-box baseline look worse than it truly is. Nerfing the
    # baseline to tidy a variance band is not a fair comparison, and no single
    # decay rate helps the oscillating-and-converged scenarios without hurting
    # the still-climbing ones. The neural SHAC across-seed variance is left as
    # the honest property of black-box analytic-gradient RL it is; the ONE
    # clearly-artifactual high-variance result (deep imitation students
    # diverging) is fixed in train_imitation, not here.
    opt_c = torch.optim.Adam(critic.parameters(), lr=args.critic_lr)
    modulated = policy if isinstance(policy, ModulatedFieldPolicy) else None
    residual_hybrid = policy if isinstance(policy, HybridFieldPolicy) else None

    eval_steps, eval_reward, eval_metric, weight_traj = [], [], [], []
    actor_losses, critic_losses, gnorm_field, gnorm_neural = [], [], [], []
    has_weights = hasattr(policy, "weights")

    t0 = time.perf_counter()
    for update in range(args.updates):
        frac = update / max(1, args.updates - 1)
        explore_std = args.act_scale * (
            args.explore_frac_start + frac * (args.explore_frac_end - args.explore_frac_start)
        )
        try:
            obs = _reset_rollout(env, program, scenario=scenario, args=args)
            obs = [o.detach() for o in obs]

            # Phase-uniform window start (see module docstring): advance the
            # rollout no-grad to a random episode phase so late-phase payoffs
            # (arrival, hold) reach the analytic gradient too. The phase cap
            # ramps in over the first phase_ramp_frac of updates so early
            # training keeps the spawn-anchored bootstrap distribution.
            ramp = 1.0
            if args.phase_ramp_frac > 0:
                ramp = min(1.0, update / max(1.0, args.phase_ramp_frac * (args.updates - 1)))
            t_max = round(ramp * (args.episode_len - args.horizon))
            prefix = int(torch.randint(0, t_max + 1, (1,)).item())
            if prefix > 0:
                with torch.no_grad():
                    for _ in range(prefix):
                        p = obs_to_perception(obs, scenario=scenario, radius=args.radius)
                        ft = program.step(p)
                        force = policy(p, ft)
                        if explore_std > 0:
                            force = force + explore_std * torch.randn_like(force)
                        acts = pack_actions(
                            force, batch_size=p.batch_size, n_agents=p.n_agents, u_range=u_range
                        )
                        obs, _, _, _ = env.step(acts)
                obs = [o.detach() for o in obs]

            obs_seq, rew_seq = [], []
            actor_loss = torch.zeros((), device=device)
            discount = 1.0
            for _ in range(args.horizon):
                p = obs_to_perception(obs, scenario=scenario, radius=args.radius)
                ft = program.step(p)
                force = policy(p, ft)
                if explore_std > 0:
                    force = force + explore_std * torch.randn_like(force)
                acts = pack_actions(
                    force, batch_size=p.batch_size, n_agents=p.n_agents, u_range=u_range
                )
                obs_seq.append([o.detach() for o in obs])
                obs, rews, _, _ = env.step(acts)
                p_next = obs_to_perception(obs, scenario=scenario, radius=args.radius)
                r = shac_reward(scenario, args, env, p_next, rews)
                actor_loss = actor_loss - discount * r.mean()
                if modulated is not None and args.gate_l2 > 0:
                    # Keep the controller a *modulation*: gates are pulled
                    # toward 1 (the plain program), so they only deviate where
                    # deviation earns reward — the policy stays readable.
                    pen_gate = (modulated.gates(p) - 1.0).pow(2).mean()
                    actor_loss = actor_loss + discount * args.gate_l2 * pen_gate
                if residual_hybrid is not None and args.residual_l2 > 0:
                    pen_res = residual_hybrid.residual(p).pow(2).mean()
                    actor_loss = actor_loss + discount * args.residual_l2 * pen_res
                if spec.smooth_collision and args.smooth_collision_weight > 0:
                    pen = smooth_collision_penalty(
                        p_next.pos,
                        batch_size=p.batch_size,
                        n_agents=p.n_agents,
                        min_dist=args.min_dist,
                    )
                    actor_loss = actor_loss + discount * args.smooth_collision_weight * pen
                rew_seq.append(r.detach())
                discount *= args.gamma

            p_term = obs_to_perception(obs, scenario=scenario, radius=args.radius)
            actor_loss = actor_loss - discount * target_critic(p_term).mean()
            opt_a.zero_grad()
            actor_loss.backward()
            actor_losses.append(float(actor_loss.item()))
            if field_params:
                gnorm_field.append(_grad_norm(field_params))
            if neural_params:
                gnorm_neural.append(_grad_norm(neural_params))
            nn.utils.clip_grad_norm_(policy.parameters(), args.grad_clip)
            opt_a.step()

            with torch.no_grad():
                feats = [
                    _local_features(obs_to_perception(o, scenario=scenario, radius=args.radius))
                    for o in obs_seq
                ]
                last_v = target_critic(p_term).detach()
                vals = [target_critic.net(f).squeeze(-1) for f in feats]
                targets = _td_lambda(rew_seq, vals, last_v, args.gamma, args.lam)
            critic_loss_last = 0.0
            for _ in range(args.critic_epochs):
                loss_c = torch.zeros((), device=device)
                for f, tgt in zip(feats, targets, strict=True):
                    loss_c = loss_c + F.mse_loss(critic.net(f).squeeze(-1), tgt)
                loss_c = loss_c / len(feats)
                opt_c.zero_grad()
                loss_c.backward()
                opt_c.step()
                critic_loss_last = float(loss_c.item())
            critic_losses.append(critic_loss_last)
            with torch.no_grad():
                for tp, sp in zip(target_critic.parameters(), critic.parameters(), strict=True):
                    tp.mul_(1 - args.tau).add_(args.tau * sp)
        except RuntimeError as exc:  # skip rare flaky autograd/CUDA faults, keep training
            opt_a.zero_grad(set_to_none=True)
            opt_c.zero_grad(set_to_none=True)
            print(f"  [warn] {scenario}/{kind} seed={seed} update {update} skipped: {exc}")
            continue

        if update % args.eval_every == 0 or update == args.updates - 1:
            ev, _, _ = evaluate(
                env, policy, program, scenario=scenario, args=args, u_range=u_range,
                steps=args.episode_len,
            )
            eval_steps.append(update)
            eval_reward.append(ev["reward"])
            eval_metric.append(ev["primary"])
            if has_weights:
                weight_traj.append(policy_weights(policy))
            print(
                f"  [{scenario}/{kind}] seed={seed} upd={update:>3}/{args.updates}  "
                f"reward={ev['reward']:.3f}  {spec.primary_metric}={ev['primary']:.3f}"
            )

    train_time = time.perf_counter() - t0
    k = max(1, min(args.final_window, len(eval_reward)))
    final_ev, trace, gate_trace = evaluate(
        env, policy, program, scenario=scenario, args=args, u_range=u_range,
        steps=args.episode_len, record_trace=True, record_gates=True,
    )
    policy_state = {k2: v.detach().cpu().clone() for k2, v in policy.state_dict().items()}
    result = RunResult(
        seed=seed,
        eval_steps=eval_steps,
        eval_reward=eval_reward,
        eval_metric=eval_metric,
        weight_traj=weight_traj,
        final={
            "reward": sum(eval_reward[-k:]) / k,
            "primary": sum(eval_metric[-k:]) / k,
            **{m: v for m, v in final_ev.items() if m not in ("reward", "primary")},
        },
        weights=policy_weights(policy) if has_weights else None,
        pos_trace=trace,
        train_time_s=train_time,
        policy_state=policy_state,
        n_params=n_params,
        gate_trace=gate_trace,
        actor_loss_curve=actor_losses,
        critic_loss_curve=critic_losses,
        grad_norm_field=gnorm_field,
        grad_norm_neural=gnorm_neural,
    )
    del policy, critic, target_critic, opt_a, opt_c
    return result


# ── Imitation (behaviour cloning + parameter recovery) ────────────────────────


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
    torch.manual_seed(seed)
    device = torch.device(args.device)
    spec = SCENARIO_SPEC[scenario]
    u_range = env_action_range(env)
    program = make_program(scenario, args)
    expert = make_expert_policy(scenario).to(device)

    # Roll the expert out once; store obs + the (stateful) program's field
    # terms + the expert force, so every student trains against exactly the
    # fields the expert acted on.
    dataset: list[tuple[list[Tensor], FieldTerms, Tensor]] = []
    obs = _reset_rollout(env, program, scenario=scenario, args=args)
    with torch.no_grad():
        for _ in range(args.bc_rollout):
            p = obs_to_perception(obs, scenario=scenario, radius=args.radius)
            ft = program.step(p)
            force = expert(p, ft)
            dataset.append(([o.detach() for o in obs], detach_terms(ft), force.detach()))
            acts = pack_actions(
                force, batch_size=p.batch_size, n_agents=p.n_agents, u_range=u_range
            )
            obs, _, _, _ = env.step(acts)

    sample_p = obs_to_perception(dataset[0][0], scenario=scenario, radius=args.radius)
    student = build_policy(kind, sample_p, spec, scenario, hidden=args.hidden).to(device)
    # The aggressive recovery LR gets every student to a good minimum fast, but
    # the deep message-passing students then blow out of it late in training
    # (measured: discovery neural_d3 reached loss 0.29 mid-run then ended ~0.84
    # on 4/5 seeds, one seed spiking 1e4; beacon neural_d3 similar) — a gradient
    # explosion that turns a settled ~0.03 result into a giant across-seed CI
    # misreading as "deep GNNs can't imitate". Grad clipping caps the explosion
    # and a cosine LR decay lets the student settle into the minimum instead of
    # bouncing around it, without slowing the initial descent (the program
    # students, which converge in a few epochs, are unaffected).
    #
    # Hybrid gets a SLOWER LR on its neural gate than on the field weights (not
    # the SHAC-style faster field / same-as-baseline neural split): at a flat
    # bc_lr for both, the gate net races ahead of the still-settling field
    # weights and locks the pair into a bad joint local minimum — measured
    # (probe, 2026-07-16): with matched LRs the loss drops fast then plateaus
    # hard at ~0.025 for the full 120-epoch budget (vs. parametric's ~0 by
    # epoch ~20, despite the hybrid being strictly more expressive); slowing
    # only the gate to bc_lr*0.25 lets the field weights settle first and the
    # combination reaches the exact 0 floor by epoch ~30.
    field_p, neural_p = _param_groups(student)
    groups = []
    if field_p:
        groups.append({"params": field_p, "lr": args.bc_lr})
    if neural_p:
        lr_neural = args.bc_lr * args.bc_lr_neural_frac if field_p else args.bc_lr
        groups.append({"params": neural_p, "lr": lr_neural})
    opt = torch.optim.Adam(groups)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=args.bc_epochs, eta_min=args.bc_lr * args.bc_lr_min_frac
    )

    loss_curve = []
    for epoch in range(args.bc_epochs):
        ep_loss = 0.0
        for obs_t, ft, target_force in dataset:
            p = obs_to_perception(obs_t, scenario=scenario, radius=args.radius)
            loss = F.mse_loss(student(p, ft), target_force)
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(student.parameters(), args.grad_clip)
            opt.step()
            ep_loss += float(loss.item())
        sched.step()
        ep_loss /= len(dataset)
        loss_curve.append(ep_loss)
        if epoch % max(1, args.bc_epochs // 8) == 0 or epoch == args.bc_epochs - 1:
            print(f"  [imit {scenario}/{kind}] seed={seed} epoch={epoch:>3} loss={ep_loss:.5f}")

    # Student rollout for the demo GIF: the student drives, the program runs live.
    obs = _reset_rollout(env, program, scenario=scenario, args=args)
    trace = []
    with torch.no_grad():
        for _ in range(args.bc_rollout):
            p = obs_to_perception(obs, scenario=scenario, radius=args.radius)
            ft = program.step(p)
            trace.append(env0_positions(obs))
            acts = pack_actions(
                student(p, ft), batch_size=p.batch_size, n_agents=p.n_agents, u_range=u_range
            )
            obs, _, _, _ = env.step(acts)

    result = ImitationResult(
        kind=kind,
        seed=seed,
        loss_curve=loss_curve,
        recovered=policy_weights(student),
        expert=EXPERT_WEIGHTS[scenario],
        final_loss=loss_curve[-1],
        pos_trace=trace,
    )
    del student, expert, dataset
    return result
