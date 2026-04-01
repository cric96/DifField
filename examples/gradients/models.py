"""Reusable models for gradient examples."""

from __future__ import annotations

import torch
import torch.nn as nn

from aggregate_gnn import SpatialScenario, mux, nbr, rep
from aggregate_gnn.dsl import AggregateContext, field

try:
    from .common import run_gradient_program
except ImportError:
    from common import run_gradient_program

MAX_DIST = 100.0
LEAKY_RELU_SLOPE = 0.2
MASKED_LOGIT = -1e9
ATTENTION_DENOM_EPS = 1e-8


class GradientModel(nn.Module):
    def __init__(self, scenario, rounds: int, init_w: float = 3.0):
        super().__init__()
        self.scenario = scenario
        self.rounds = rounds
        self.w = nn.Parameter(torch.tensor(init_w))

    def forward(self, source: torch.Tensor) -> torch.Tensor:
        output, _ = run_gradient_program(self.scenario, source, rounds=self.rounds, weight=self.w)
        return output


class AttentionMinAggr(nn.Module):
    def __init__(self, tau_init: float = 1.0):
        super().__init__()
        self.a = nn.Parameter(torch.tensor(1.0))
        self.b = nn.Parameter(torch.tensor(0.0))
        self._log_tau = nn.Parameter(torch.tensor(float(tau_init)).log())

    @property
    def tau(self):
        return self._log_tau.exp()

    def forward(self, msg: torch.Tensor, index: torch.Tensor, num_nodes: int) -> torch.Tensor:
        tau = self.tau
        finite = msg.isfinite()
        safe_msg = torch.where(finite, msg, torch.zeros_like(msg))

        score = torch.nn.functional.leaky_relu(self.a * safe_msg + self.b, LEAKY_RELU_SLOPE)
        neg_score = -score / tau
        neg_score = torch.where(finite, neg_score, torch.full_like(neg_score, MASKED_LOGIT))

        max_s = neg_score.new_full((num_nodes,), MASKED_LOGIT)
        max_s.scatter_reduce_(0, index, neg_score, reduce="amax", include_self=True)
        exp_s = (neg_score - max_s[index]).exp()
        sum_exp = neg_score.new_zeros(num_nodes)
        sum_exp.scatter_add_(0, index, exp_s)
        alpha = exp_s / sum_exp[index].clamp(min=ATTENTION_DENOM_EPS)

        out = safe_msg.new_zeros(num_nodes)
        out.scatter_add_(0, index, alpha * safe_msg)

        has_finite = safe_msg.new_zeros(num_nodes)
        has_finite.scatter_add_(0, index, finite.float())
        return torch.where(has_finite > 0, out, torch.tensor(float("inf")))


class AttentionGradientModel(nn.Module):
    def __init__(self, scenario, rounds: int):
        super().__init__()
        self.scenario = scenario
        self.rounds = rounds
        self.w = nn.Parameter(torch.tensor(1.5))
        self.attn_aggr = AttentionMinAggr(tau_init=1.0)

    def forward(self, source: torch.Tensor) -> torch.Tensor:
        output, _ = run_gradient_program(
            self.scenario,
            source,
            rounds=self.rounds,
            weight=self.w,
            aggr=self.attn_aggr,
        )
        return output


def pairwise_shortest_hop(positions: torch.Tensor, source: int, radius: float) -> torch.Tensor:
    num_nodes = positions.shape[0]
    dist = torch.full((num_nodes,), float("inf"), dtype=torch.float32, device=positions.device)
    dist[source] = 0.0

    pairwise_dist = torch.cdist(positions, positions)
    adj = (pairwise_dist <= radius) & (pairwise_dist > 0)

    frontier = torch.zeros(num_nodes, dtype=torch.bool, device=positions.device)
    frontier[source] = True
    for step in range(1, num_nodes + 1):
        neigh = (adj[frontier]).any(dim=0)
        newly = neigh & torch.isinf(dist)
        dist[newly] = float(step)
        if not newly.any():
            break
        frontier = newly

    return torch.nan_to_num(dist, nan=MAX_DIST, posinf=MAX_DIST, neginf=0.0)


class MotionPolicy(nn.Module):
    def __init__(self, hidden: int = 32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(4, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
            nn.Linear(hidden, 2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class LearnableMovingGradient(nn.Module):
    def __init__(self, positions: torch.Tensor, radius: float, source_idx: int, learn_mode: str):
        super().__init__()
        self.positions0 = positions
        self.radius = radius
        self.source_idx = source_idx
        self.learn_mode = learn_mode
        self.motion_policy = MotionPolicy()
        self.w_raw = nn.Parameter(torch.tensor(0.54))
        self.dt = 0.1
        self.max_speed = 0.04

    @property
    def w(self) -> torch.Tensor:
        return torch.nn.functional.softplus(self.w_raw) + 1e-3

    def _clip_box(self, positions: torch.Tensor, velocities: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        for dim in (0, 1):
            low = positions[:, dim] < 0.0
            high = positions[:, dim] > 1.0
            velocities[low | high, dim] *= -1.0
        return positions.clamp(0.0, 1.0), velocities

    def forward(self, rounds: int) -> tuple[torch.Tensor, torch.Tensor]:
        positions = self.positions0.clone()
        velocities = torch.zeros_like(positions)
        scenario = SpatialScenario(positions=positions, edge_radius=self.radius)
        ctx = AggregateContext(scenario.edge_index, scenario.num_nodes, edge_weight=scenario.edge_weight)

        source = torch.zeros(positions.shape[0], dtype=torch.float32, device=positions.device)
        source[self.source_idx] = 1.0

        pred_seq = []
        for _ in range(rounds):
            scenario.sync_context(ctx._ctx)
            with ctx.round():
                dist = rep(
                    "dist",
                    MAX_DIST,
                    lambda dist_old: mux(source, field.of(0.0), nbr(dist_old + self.w, aggr="min")),
                )

            dist_feat = torch.nan_to_num(dist, nan=0.0, posinf=10.0, neginf=0.0)
            features = torch.cat([positions, dist_feat.unsqueeze(-1), velocities.norm(dim=1, keepdim=True)], dim=1)
            dv = self.motion_policy(features)
            new_vel = velocities + self.dt * dv
            speed = new_vel.norm(dim=1, keepdim=True).clamp_min(1e-8)
            new_vel = new_vel * torch.clamp(self.max_speed / speed, max=1.0)

            if self.learn_mode == "ac":
                new_vel = new_vel.detach()
            elif self.learn_mode == "motion":
                dist = dist.detach()

            positions = positions + self.dt * new_vel
            positions, velocities = self._clip_box(positions, new_vel)
            scenario.update_positions(positions, refresh_topology=True)
            pred_seq.append(dist)

        return torch.stack(pred_seq, dim=0), positions


def step_teacher_positions(pos: torch.Tensor, vel: torch.Tensor, dt: float = 0.1) -> tuple[torch.Tensor, torch.Tensor]:
    new_pos = pos + dt * vel
    new_vel = vel.clone()
    for dim in (0, 1):
        low = new_pos[:, dim] < 0.0
        high = new_pos[:, dim] > 1.0
        new_vel[low | high, dim] *= -1.0
    return new_pos.clamp(0.0, 1.0), new_vel
