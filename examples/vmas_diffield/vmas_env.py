"""VMAS environment construction, observation adaptation, and action packing."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import vmas
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
    ScenarioSpec,
)

__all__ = [
    "DISCOVERY_LIDAR_RANGE",
    "SAMPLING_AGENT_LIDAR_RAYS",
    "SCENARIO_SPEC",
    "FieldTerms",
    "Perception",
    "ScenarioSpec",
    "brake_term",
    "detach_env",
    "env0_positions",
    "env_action_range",
    "explore_term",
    "grid_sense_term",
    "lidar_sense_term",
    "make_diff_env",
    "obs_to_perception",
    "pack_actions",
    "soft_normalize",
]

_POS = slice(0, 2)
_VEL = slice(2, 4)
_GOAL_REL = slice(4, 6)


def make_diff_env(scenario, *, num_envs, n_agents, device, max_steps):
    """Create a differentiable VMAS environment for a supported scenario."""
    if scenario not in SCENARIO_SPEC:
        raise ValueError(f"scenario must be one of {tuple(SCENARIO_SPEC)}, got {scenario}")
    vmas_name = SCENARIO_SPEC[scenario].vmas_name or scenario
    return vmas.make_env(
        scenario=vmas_name,
        num_envs=num_envs,
        device=str(device),
        continuous_actions=True,
        grad_enabled=True,
        n_agents=n_agents,
        max_steps=max_steps,
    )


def detach_env(env) -> None:
    """Detach persistent environment tensors between optimization updates."""
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
    """Flattened node observations, with masked policy inputs and eval-only goals."""

    pos: Tensor
    vel: Tensor
    extra: Tensor  # obs tail (masked goal_rel + lidar + ... [+ knows bit])
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
    """Flatten observations and build a per-environment radius graph."""
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
        # Mask the goal in policy features and append the knower indicator.
        extra = torch.cat([extra[:, :2] * mask, extra[:, 2:], mask], dim=-1)
    return Perception(
        pos=pos,
        vel=flat[:, _VEL],
        extra=extra,
        goal_rel=goal_rel,
        true_goal_rel=true_goal_rel,
        edge_index=_batched_radius_graph(pos.reshape(b, n, 2), radius),
        num_nodes=b * n,
        batch_size=b,
        n_agents=n,
        knows=knows,
    )


def _batched_radius_graph(pos: Tensor, radius: float) -> Tensor:
    """Build directed neighbor-to-self edges without connecting environments."""
    _, n, _ = pos.shape
    dist = torch.cdist(pos, pos)
    adj = (dist <= radius) & (dist > 0.0)
    env_idx, i_idx, j_idx = adj.nonzero(as_tuple=True)
    return torch.stack([env_idx * n + j_idx, env_idx * n + i_idx], dim=0).to(pos.device)


def pack_actions(force_flat, *, batch_size, n_agents, u_range):
    """Map flattened forces to VMAS actions using a smooth bounded squash."""
    action = (u_range * torch.tanh(force_flat / u_range)).reshape(batch_size, n_agents, 2)
    return [action[:, i, :] for i in range(n_agents)]


def env_action_range(env) -> float:
    return float(env.agents[0].u_range)


def env0_positions(obs_list: list[Tensor]) -> Tensor:
    """Agent positions of parallel env 0 as ``[N, 2]`` (for trajectory/GIF)."""
    return torch.stack(obs_list, dim=1)[0, :, _POS].detach().cpu()
