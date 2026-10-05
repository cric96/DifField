"""SHAC training loop for VMAS policies."""

from __future__ import annotations

import time

import torch
import torch.nn.functional as F  # noqa: N812
from torch import nn
from vmas_diffield.policies import (
    Critic,
    HybridFieldPolicy,
    ModulatedFieldPolicy,
    _local_features,
    build_policy,
    policy_weights,
)
from vmas_diffield.scenarios import SCENARIO_SPEC
from vmas_diffield.train import (
    RunResult,
    _expert_run,
    _grad_norm,
    _make_actor_optimizer,
    _param_groups,
    _reset_rollout,
    _td_lambda,
    evaluate,
    make_program,
    shac_reward,
    smooth_collision_penalty,
)
from vmas_diffield.vmas_env import env_action_range, obs_to_perception, pack_actions


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
                reward = shac_reward(scenario, args, env, p_next, rews)
                actor_loss = actor_loss - discount * reward.mean()
                if modulated is not None and args.gate_l2 > 0:
                    actor_loss = (
                        actor_loss
                        + discount * args.gate_l2 * (modulated.gates(p) - 1.0).pow(2).mean()
                    )
                if residual_hybrid is not None and args.residual_l2 > 0:
                    penalty = residual_hybrid.residual(p).pow(2).mean()
                    actor_loss = actor_loss + discount * args.residual_l2 * penalty
                if spec.smooth_collision and args.smooth_collision_weight > 0:
                    penalty = smooth_collision_penalty(
                        p_next.pos,
                        batch_size=p.batch_size,
                        n_agents=p.n_agents,
                        min_dist=args.min_dist,
                    )
                    actor_loss = actor_loss + discount * args.smooth_collision_weight * penalty
                rew_seq.append(reward.detach())
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
                for features, target in zip(feats, targets, strict=True):
                    loss_c = loss_c + F.mse_loss(critic.net(features).squeeze(-1), target)
                loss_c = loss_c / len(feats)
                opt_c.zero_grad()
                loss_c.backward()
                opt_c.step()
                critic_loss_last = float(loss_c.item())
            critic_losses.append(critic_loss_last)
            with torch.no_grad():
                for target_param, source_param in zip(
                    target_critic.parameters(), critic.parameters(), strict=True
                ):
                    target_param.mul_(1 - args.tau).add_(args.tau * source_param)
        except RuntimeError as exc:
            opt_a.zero_grad(set_to_none=True)
            opt_c.zero_grad(set_to_none=True)
            print(f"  [warn] {scenario}/{kind} seed={seed} update {update} skipped: {exc}")
            continue

        if update % args.eval_every == 0 or update == args.updates - 1:
            ev, _, _ = evaluate(
                env,
                policy,
                program,
                scenario=scenario,
                args=args,
                u_range=u_range,
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
        env,
        policy,
        program,
        scenario=scenario,
        args=args,
        u_range=u_range,
        steps=args.episode_len,
        record_trace=True,
        record_gates=True,
    )
    policy_state = {key: value.detach().cpu().clone() for key, value in policy.state_dict().items()}
    result = RunResult(
        seed=seed,
        eval_steps=eval_steps,
        eval_reward=eval_reward,
        eval_metric=eval_metric,
        weight_traj=weight_traj,
        final={
            "reward": sum(eval_reward[-k:]) / k,
            "primary": sum(eval_metric[-k:]) / k,
            **{
                metric: value
                for metric, value in final_ev.items()
                if metric not in ("reward", "primary")
            },
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
