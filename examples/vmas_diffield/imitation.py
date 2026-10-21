"""Behaviour-cloning trainer for VMAS policies."""

from __future__ import annotations

import torch
import torch.nn.functional as F  # noqa: N812
from torch import nn
from vmas_diffield.policies import (
    EXPERT_WEIGHTS,
    build_policy,
    make_expert_policy,
    policy_weights,
)
from vmas_diffield.scenarios import SCENARIO_SPEC
from vmas_diffield.train import (
    ImitationResult,
    _param_groups,
    _reset_rollout,
    detach_terms,
    make_program,
)
from vmas_diffield.vmas_env import (
    env0_positions,
    env_action_range,
    obs_to_perception,
    pack_actions,
)


def train_imitation(scenario: str, kind: str, *, seed: int, args, env) -> ImitationResult:
    torch.manual_seed(seed)
    device = torch.device(args.device)
    spec = SCENARIO_SPEC[scenario]
    u_range = env_action_range(env)
    program = make_program(scenario, args)
    expert = make_expert_policy(scenario).to(device)

    dataset = []
    obs = _reset_rollout(env, program, scenario=scenario, args=args)
    with torch.no_grad():
        for _ in range(args.bc_rollout):
            p = obs_to_perception(obs, scenario=scenario, radius=args.radius)
            terms = program.step(p)
            force = expert(p, terms)
            dataset.append(([item.detach() for item in obs], detach_terms(terms), force.detach()))
            acts = pack_actions(
                force, batch_size=p.batch_size, n_agents=p.n_agents, u_range=u_range
            )
            obs, _, _, _ = env.step(acts)

    sample_p = obs_to_perception(dataset[0][0], scenario=scenario, radius=args.radius)
    student = build_policy(kind, sample_p, spec, scenario, hidden=args.hidden).to(device)
    field_params, neural_params = _param_groups(student)
    groups = []
    if field_params:
        groups.append({"params": field_params, "lr": args.bc_lr})
    if neural_params:
        lr_neural = args.bc_lr * args.bc_lr_neural_frac if field_params else args.bc_lr
        groups.append({"params": neural_params, "lr": lr_neural})
    optimizer = torch.optim.Adam(groups)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.bc_epochs, eta_min=args.bc_lr * args.bc_lr_min_frac
    )

    loss_curve = []
    for epoch in range(args.bc_epochs):
        epoch_loss = 0.0
        for obs_t, terms, target_force in dataset:
            p = obs_to_perception(obs_t, scenario=scenario, radius=args.radius)
            loss = F.mse_loss(student(p, terms), target_force)
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(student.parameters(), args.grad_clip)
            optimizer.step()
            epoch_loss += float(loss.item())
        scheduler.step()
        epoch_loss /= len(dataset)
        loss_curve.append(epoch_loss)
        if epoch % max(1, args.bc_epochs // 8) == 0 or epoch == args.bc_epochs - 1:
            print(f"  [imit {scenario}/{kind}] seed={seed} epoch={epoch:>3} loss={epoch_loss:.5f}")

    obs = _reset_rollout(env, program, scenario=scenario, args=args)
    trace = []
    with torch.no_grad():
        for _ in range(args.bc_rollout):
            p = obs_to_perception(obs, scenario=scenario, radius=args.radius)
            terms = program.step(p)
            trace.append(env0_positions(obs))
            acts = pack_actions(
                student(p, terms), batch_size=p.batch_size, n_agents=p.n_agents, u_range=u_range
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
