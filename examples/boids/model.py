"""Model and simulation primitives for learnable boids."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
import torch.nn as nn
from torch_geometric.utils import degree as pyg_degree
from torch_geometric.utils import scatter as pyg_scatter
from torch_geometric.utils import softmax as pyg_softmax

from aggregate_gnn import (
    SpatialScenario,
    boids_acceleration_dense,
    bounce_in_box,
    limit_speed,
    nbr,
    normalize_vectors,
    rep,
)
from aggregate_gnn.dsl import AggregateContext

if TYPE_CHECKING:
    from .config import ModelSpec, SimulationSpec, TeacherDynamics


def _softplus_param(raw: torch.Tensor, minimum: float = 1e-4) -> torch.Tensor:
    return torch.nn.functional.softplus(raw) + minimum


def _bounded_sigmoid(raw: torch.Tensor, low: float, high: float) -> torch.Tensor:
    return low + (high - low) * torch.sigmoid(raw)


def _inverse_sigmoid_target(value: float) -> float:
    clipped = min(max(float(value), 1e-4), 1.0 - 1e-4)
    return float(torch.logit(torch.tensor(clipped, dtype=torch.float32)).item())


def _inverse_bounded_sigmoid_target(value: float, low: float, high: float) -> float:
    if high <= low:
        raise ValueError("max speed bounds must satisfy high > low")
    clipped = min(max(float(value), low + 1e-6), high - 1e-6)
    scaled = (clipped - low) / (high - low)
    return _inverse_sigmoid_target(scaled)


def _edge_connectivity_stats(edge_index: torch.Tensor, num_nodes: int) -> tuple[float, float, float]:
    if num_nodes <= 0:
        return 0.0, 0.0, 0.0
    if edge_index.numel() == 0:
        return 0.0, 0.0, float(num_nodes)

    src = edge_index[0]
    tgt = edge_index[1]
    min_degree = float(pyg_degree(src, num_nodes=num_nodes).min().item())

    src_list = src.tolist()
    tgt_list = tgt.tolist()
    neighbors = [set() for _ in range(num_nodes)]
    for src_node, tgt_node in zip(src_list, tgt_list):
        neighbors[src_node].add(tgt_node)
        neighbors[tgt_node].add(src_node)
    visited = [False] * num_nodes
    components = 0
    for node in range(num_nodes):
        if visited[node]:
            continue
        components += 1
        stack = [node]
        visited[node] = True
        while stack:
            cur = stack.pop()
            for nxt in neighbors[cur]:
                if not visited[nxt]:
                    visited[nxt] = True
                    stack.append(nxt)
    return float(edge_index.shape[1]), min_degree, float(components)


class EdgeAttentionAggr(nn.Module):
    def __init__(self, feature_dim: int):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(3, 16), nn.ELU(), nn.Linear(16, 1))
        self.log_tau = nn.Parameter(torch.tensor(0.0))
        self.feature_dim = feature_dim

    @property
    def tau(self) -> torch.Tensor:
        return self.log_tau.exp()

    def forward(self, msg: torch.Tensor, index: torch.Tensor, num_nodes: int) -> torch.Tensor:
        if msg.dim() == 1:
            msg = msg.unsqueeze(-1)
        mag = msg.norm(dim=1, keepdim=True)
        edge_feat = torch.cat([msg, mag], dim=1)
        logits = self.net(edge_feat).squeeze(-1) / self.tau.clamp(min=1e-3)
        alpha = pyg_softmax(logits, index=index, num_nodes=num_nodes).unsqueeze(-1)
        out = pyg_scatter(alpha * msg, index=index, dim=0, dim_size=num_nodes, reduce="sum")
        if self.feature_dim == 1:
            return out.squeeze(-1)
        return out


class LearnableAggregateBoids(nn.Module):
    def __init__(
        self,
        positions0: torch.Tensor,
        radius: float,
        sep: float,
        dt: float,
        mode: str,
        init_connectivity: str = "hybrid",
        init_k_neighbors: int = 8,
        init_min_degree: int = 2,
        init_damping_target: float = 0.94,
        init_max_speed_target: float = 0.03,
        max_speed_min: float = 0.002,
        max_speed_max: float = 0.06,
        train_max_speed: bool = True,
    ):
        super().__init__()
        self.positions0 = positions0
        self.radius = radius
        self.sep = sep
        self.dt = dt
        self.mode = mode
        self.init_connectivity = init_connectivity
        self.init_k_neighbors = init_k_neighbors
        self.init_min_degree = init_min_degree
        self.last_init_graph_stats = {"num_edges": float("nan"), "min_degree": float("nan"), "num_components": float("nan")}
        self.last_rollout_graph_health = {"mean_num_edges": float("nan"), "mean_min_degree": float("nan"), "max_num_components": float("nan")}
        self.last_rollout_speed_health = {"mean_pre_clip_speed": float("nan"), "mean_cap_fraction": float("nan")}
        self.max_speed_min = float(max_speed_min)
        self.max_speed_max = float(max_speed_max)
        self.train_max_speed = bool(train_max_speed)

        self.w_sep_raw = nn.Parameter(torch.tensor(1.2))
        self.w_align_raw = nn.Parameter(torch.tensor(0.7))
        self.w_cohesion_raw = nn.Parameter(torch.tensor(0.6))
        self.damping_raw = nn.Parameter(torch.tensor(_inverse_sigmoid_target(init_damping_target)))
        self.max_speed_raw = nn.Parameter(torch.tensor(_inverse_bounded_sigmoid_target(init_max_speed_target, self.max_speed_min, self.max_speed_max)))
        self.align_aggr = EdgeAttentionAggr(feature_dim=2)
        self.cohesion_aggr = EdgeAttentionAggr(feature_dim=2)

    @classmethod
    def from_specs(
        cls,
        *,
        positions0: torch.Tensor,
        simulation: "SimulationSpec",
        model: "ModelSpec",
    ) -> "LearnableAggregateBoids":
        return cls(
            positions0=positions0,
            radius=simulation.radius,
            sep=simulation.sep,
            dt=simulation.dt,
            mode=model.mode,
            init_connectivity=model.init_connectivity,
            init_k_neighbors=model.init_k_neighbors,
            init_min_degree=model.init_min_degree,
            init_damping_target=model.init_damping_target,
            init_max_speed_target=model.init_max_speed_target,
            max_speed_min=model.max_speed_min,
            max_speed_max=model.max_speed_max,
            train_max_speed=model.train_max_speed,
        )

    @property
    def w_sep(self) -> torch.Tensor:
        return _softplus_param(self.w_sep_raw)

    @property
    def w_align(self) -> torch.Tensor:
        return _softplus_param(self.w_align_raw)

    @property
    def w_cohesion(self) -> torch.Tensor:
        return _softplus_param(self.w_cohesion_raw)

    @property
    def damping(self) -> torch.Tensor:
        return torch.sigmoid(self.damping_raw)

    @property
    def max_speed(self) -> torch.Tensor:
        return _bounded_sigmoid(self.max_speed_raw, self.max_speed_min, self.max_speed_max)

    def _velocity_params(self) -> list[nn.Parameter]:
        params = [self.w_sep_raw, self.w_align_raw, self.w_cohesion_raw, self.damping_raw]
        if self.train_max_speed:
            params.append(self.max_speed_raw)
        return params

    def _attention_params(self) -> list[nn.Parameter]:
        return list(self.align_aggr.parameters()) + list(self.cohesion_aggr.parameters())

    def trainable_parameters(self) -> list[nn.Parameter]:
        if self.mode == "weights":
            return self._velocity_params()
        if self.mode == "attention":
            return self._attention_params()
        return self._velocity_params() + self._attention_params()

    def _alignment_aggr(self) -> str | nn.Module:
        return "mean" if self.mode == "weights" else self.align_aggr

    def _cohesion_aggr(self) -> str | nn.Module:
        return "mean" if self.mode == "weights" else self.cohesion_aggr

    def rollout(self, rounds: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        positions = self.positions0.clone()
        scenario = SpatialScenario(
            positions=positions,
            edge_radius=self.radius if self.init_connectivity in {"radius", "hybrid"} else None,
            k_neighbors=self.init_k_neighbors if self.init_connectivity == "knn" else None,
            ensure_init_connected=self.init_connectivity == "hybrid",
            init_min_degree=self.init_min_degree,
            init_k_neighbors=self.init_k_neighbors,
        )
        self.last_init_graph_stats = dict(scenario.init_graph_stats)
        ctx = AggregateContext(scenario.edge_index, scenario.num_nodes, edge_weight=scenario.edge_weight)
        init_vel = torch.zeros_like(positions)
        positions_seq = []
        velocities_seq = []
        edge_counts = []
        min_degrees = []
        components = []
        self._rollout_pre_clip_speeds = []
        self._rollout_cap_fractions = []

        for _ in range(rounds):
            scenario.sync_context(ctx._ctx)
            pos_t = scenario.positions
            with ctx.round():
                vel = rep("vel", init_vel, lambda prev: self._velocity_update(prev, pos_t))

            new_pos = pos_t + self.dt * vel
            new_pos, vel_bounced = bounce_in_box(new_pos, vel)
            scenario.update_positions(new_pos, refresh_topology=True)
            ctx._ctx.state.update("vel", vel_bounced)

            num_edges, min_degree, num_components = _edge_connectivity_stats(scenario.edge_index, scenario.num_nodes)
            edge_counts.append(num_edges)
            min_degrees.append(min_degree)
            components.append(num_components)
            positions_seq.append(scenario.positions)
            velocities_seq.append(vel_bounced)

        self.last_rollout_graph_health = {
            "mean_num_edges": float(sum(edge_counts) / max(1, len(edge_counts))),
            "mean_min_degree": float(sum(min_degrees) / max(1, len(min_degrees))),
            "max_num_components": float(max(components) if components else float("nan")),
        }
        self.last_rollout_speed_health = {
            "mean_pre_clip_speed": float(sum(self._rollout_pre_clip_speeds) / max(1, len(self._rollout_pre_clip_speeds))),
            "mean_cap_fraction": float(sum(self._rollout_cap_fractions) / max(1, len(self._rollout_cap_fractions))),
        }
        del self._rollout_pre_clip_speeds
        del self._rollout_cap_fractions
        return torch.stack(positions_seq, dim=0), torch.stack(velocities_seq, dim=0), scenario.positions

    def _velocity_update(self, vel: torch.Tensor, pos: torch.Tensor) -> torch.Tensor:
        neigh_vel = nbr(vel, aggr=self._alignment_aggr())
        align_force = neigh_vel - vel
        cohesion_force = nbr(pos, aggr=self._cohesion_aggr()) - pos
        dx = pos[:, 0].unsqueeze(1) - pos[:, 0].unsqueeze(0)
        dy = pos[:, 1].unsqueeze(1) - pos[:, 1].unsqueeze(0)
        dist = torch.sqrt(dx.pow(2) + dy.pow(2) + 1e-9)
        close_mask = (dist <= self.sep) & (dist > 0)
        sep_force = torch.stack([(dx * close_mask).sum(dim=1), (dy * close_mask).sum(dim=1)], dim=1)

        acc = self.w_sep * normalize_vectors(sep_force) + self.w_align * align_force + self.w_cohesion * cohesion_force
        pre_clip_vel = self.damping * vel + self.dt * acc
        pre_clip_speed = pre_clip_vel.norm(dim=-1)
        if hasattr(self, "_rollout_pre_clip_speeds"):
            self._rollout_pre_clip_speeds.append(float(pre_clip_speed.mean().detach().item()))
            self._rollout_cap_fractions.append(float((pre_clip_speed > self.max_speed).float().mean().detach().item()))
        return torch.nan_to_num(limit_speed(pre_clip_vel, self.max_speed), nan=0.0, posinf=0.0, neginf=0.0)


@torch.no_grad()
def teacher_rollout(
    positions0: torch.Tensor,
    rounds: int,
    radius: float,
    sep: float,
    dt: float,
    w_sep: float = 1.4,
    w_align: float = 0.8,
    w_cohesion: float = 0.6,
    damping: float = 0.96,
    max_speed: float = 0.014,
) -> tuple[torch.Tensor, torch.Tensor]:
    positions = positions0.clone()
    velocities = torch.zeros_like(positions)
    pos_seq = []
    vel_seq = []
    for _ in range(rounds):
        acc = boids_acceleration_dense(positions, velocities, radius=radius, sep=sep, w_sep=w_sep, w_align=w_align, w_cohesion=w_cohesion)
        velocities = limit_speed(damping * velocities + dt * acc, max_speed)
        positions = positions + dt * velocities
        positions, velocities = bounce_in_box(positions, velocities)
        pos_seq.append(positions.clone())
        vel_seq.append(velocities.clone())
    return torch.stack(pos_seq, dim=0), torch.stack(vel_seq, dim=0)


@torch.no_grad()
def teacher_rollout_from_specs(
    *,
    positions0: torch.Tensor,
    simulation: "SimulationSpec",
    teacher: "TeacherDynamics",
) -> tuple[torch.Tensor, torch.Tensor]:
    return teacher_rollout(
        positions0=positions0,
        rounds=simulation.rounds,
        radius=simulation.radius,
        sep=simulation.sep,
        dt=simulation.dt,
        w_sep=teacher.w_sep,
        w_align=teacher.w_align,
        w_cohesion=teacher.w_cohesion,
        damping=teacher.damping,
        max_speed=teacher.max_speed,
    )


def emergent_regularizers(
    pos_seq: torch.Tensor,
    vel_seq: torch.Tensor,
    cohesion_weight: float,
    alignment_weight: float,
    speed_weight: float,
    accel_weight: float,
) -> torch.Tensor:
    center = pos_seq.mean(dim=1, keepdim=True)
    cohesion_spread = (pos_seq - center).norm(dim=-1).mean()
    mean_vel = vel_seq.mean(dim=1, keepdim=True)
    alignment_dispersion = (vel_seq - mean_vel).norm(dim=-1).mean()
    speed_penalty = vel_seq.norm(dim=-1).mean()
    accel_penalty = (vel_seq[1:] - vel_seq[:-1]).norm(dim=-1).mean() if vel_seq.shape[0] > 1 else vel_seq.new_tensor(0.0)
    return cohesion_weight * cohesion_spread + alignment_weight * alignment_dispersion + speed_weight * speed_penalty + accel_weight * accel_penalty
