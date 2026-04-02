"""Model and simulation primitives for learnable boids."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.utils import degree as pyg_degree
from torch_geometric.utils import scatter as pyg_scatter
from torch_geometric.utils import softmax as pyg_softmax

from aggregate_gnn import (
    SpatialScenario,
    bounce_in_box,
    limit_speed,
    nbr,
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


def _scatter_mean(src: torch.Tensor, index: torch.Tensor, num_nodes: int) -> torch.Tensor:
    if src.numel() == 0:
        shape = (num_nodes,) if src.dim() == 1 else (num_nodes, *src.shape[1:])
        return torch.zeros(shape, dtype=src.dtype, device=src.device)
    return pyg_scatter(src, index, dim=0, dim_size=num_nodes, reduce="mean")


def _soft_separation_force(positions: torch.Tensor, sep: float, sharpness: float = 10.0) -> torch.Tensor:
    num_nodes = positions.shape[0]
    dx = positions[:, 0].unsqueeze(1) - positions[:, 0].unsqueeze(0)
    dy = positions[:, 1].unsqueeze(1) - positions[:, 1].unsqueeze(0)
    dist = torch.sqrt(dx.pow(2) + dy.pow(2) + 1e-9)
    soft_mask = torch.sigmoid(sharpness * (sep - dist))
    soft_mask = soft_mask * (1.0 - torch.eye(num_nodes, device=positions.device, dtype=positions.dtype))
    return torch.stack([(dx * soft_mask).sum(dim=1), (dy * soft_mask).sum(dim=1)], dim=1)


def _build_boids_scenario(
    positions: torch.Tensor,
    *,
    radius: float,
    init_connectivity: str,
    init_k_neighbors: int,
    init_min_degree: int,
) -> SpatialScenario:
    return SpatialScenario(
        positions=positions,
        edge_radius=radius if init_connectivity in {"radius", "hybrid"} else None,
        k_neighbors=init_k_neighbors if init_connectivity == "knn" else None,
        ensure_init_connected=init_connectivity == "hybrid",
        init_min_degree=init_min_degree,
        init_k_neighbors=init_k_neighbors,
        device=positions.device,
    )


def sample_initial_boids_state(
    num_nodes: int,
    *,
    seed: int,
    velocity_scale: float,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    generator = torch.Generator()
    generator.manual_seed(seed)
    positions0 = torch.rand(num_nodes, 2, generator=generator, dtype=torch.float32).to(device)
    if velocity_scale <= 0.0:
        velocities0 = torch.zeros_like(positions0)
    else:
        velocities0 = ((torch.rand(num_nodes, 2, generator=generator, dtype=torch.float32) - 0.5) * velocity_scale).to(device)
    return positions0, velocities0


def trajectory_loss_components(
    pred_pos_seq: torch.Tensor,
    pred_vel_seq: torch.Tensor,
    teacher_pos_seq: torch.Tensor,
    teacher_vel_seq: torch.Tensor,
    *,
    velocity_weight: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    pos_loss = torch.stack([
        F.mse_loss(pred_pos, teacher_pos)
        for pred_pos, teacher_pos in zip(pred_pos_seq, teacher_pos_seq)
    ]).mean()
    vel_loss = torch.stack([
        F.mse_loss(pred_vel, teacher_vel)
        for pred_vel, teacher_vel in zip(pred_vel_seq, teacher_vel_seq)
    ]).mean()
    total_loss = pos_loss + velocity_weight * vel_loss
    return total_loss, pos_loss, vel_loss


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

    def _build_rollout_scenario(self, positions: torch.Tensor) -> SpatialScenario:
        return _build_boids_scenario(
            positions,
            radius=self.radius,
            init_connectivity=self.init_connectivity,
            init_k_neighbors=self.init_k_neighbors,
            init_min_degree=self.init_min_degree,
        )

    def rollout(
        self,
        rounds: int,
        *,
        positions0: torch.Tensor | None = None,
        velocities0: torch.Tensor | None = None,
        trunc_window: int | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        positions = (self.positions0 if positions0 is None else positions0).clone()
        init_vel = torch.zeros_like(positions) if velocities0 is None else velocities0.clone()
        scenario = self._build_rollout_scenario(positions)
        num_edges, min_degree, num_components = _edge_connectivity_stats(scenario.edge_index, scenario.num_nodes)
        self.last_init_graph_stats = {
            "num_edges": num_edges,
            "min_degree": min_degree,
            "num_components": num_components,
        }
        ctx = AggregateContext(scenario.edge_index, scenario.num_nodes, edge_weight=scenario.edge_weight)
        scenario.sync_context(ctx._ctx)
        positions_seq = []
        velocities_seq = []
        self._rollout_pre_clip_speeds = []
        self._rollout_cap_fractions = []

        for round_idx in range(rounds):
            if trunc_window is not None and trunc_window > 0 and round_idx > 0 and round_idx % trunc_window == 0:
                scenario.update_positions(scenario.positions.detach(), refresh_topology=False)
                detached_vel = ctx._ctx.state.get_or_init("vel", init_vel).detach()
                ctx._ctx.state.update("vel", detached_vel)

            scenario.sync_context(ctx._ctx)
            pos_t = scenario.positions
            with ctx.round():
                vel = rep("vel", init_vel, lambda prev: self._velocity_update(prev, pos_t))

            new_pos = pos_t + self.dt * vel
            new_pos, vel_bounced = bounce_in_box(new_pos, vel)
            scenario.update_positions(new_pos, refresh_topology=False)
            ctx._ctx.state.update("vel", vel_bounced)

            positions_seq.append(scenario.positions)
            velocities_seq.append(vel_bounced)

        self.last_rollout_graph_health = {
            "mean_num_edges": num_edges,
            "mean_min_degree": min_degree,
            "max_num_components": num_components,
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
        sep_force = _soft_separation_force(pos, self.sep)

        acc = self.w_sep * sep_force + self.w_align * align_force + self.w_cohesion * cohesion_force
        pre_clip_vel = self.damping * vel + self.dt * acc
        pre_clip_speed = pre_clip_vel.norm(dim=-1)
        if hasattr(self, "_rollout_pre_clip_speeds"):
            self._rollout_pre_clip_speeds.append(float(pre_clip_speed.mean().detach().item()))
            self._rollout_cap_fractions.append(float((pre_clip_speed > self.max_speed).float().mean().detach().item()))
        return torch.nan_to_num(limit_speed(pre_clip_vel, self.max_speed), nan=0.0, posinf=0.0, neginf=0.0)


@torch.no_grad()
def teacher_rollout(
    positions0: torch.Tensor,
    velocities0: torch.Tensor,
    edge_index: torch.Tensor,
    rounds: int,
    sep: float,
    dt: float,
    w_sep: float = 1.4,
    w_align: float = 0.8,
    w_cohesion: float = 0.6,
    damping: float = 0.96,
    max_speed: float = 0.014,
) -> tuple[torch.Tensor, torch.Tensor]:
    positions = positions0.clone()
    velocities = velocities0.clone()
    pos_seq = []
    vel_seq = []
    num_nodes = positions.shape[0]
    src_idx = edge_index[0] if edge_index.numel() > 0 else None
    dst_idx = edge_index[1] if edge_index.numel() > 0 else None
    for _ in range(rounds):
        if src_idx is not None and dst_idx is not None:
            align_force = _scatter_mean(velocities[src_idx], dst_idx, num_nodes) - velocities
            cohesion_force = _scatter_mean(positions[src_idx], dst_idx, num_nodes) - positions
        else:
            align_force = torch.zeros_like(velocities)
            cohesion_force = torch.zeros_like(positions)
        sep_force = _soft_separation_force(positions, sep)
        acc = w_sep * sep_force + w_align * align_force + w_cohesion * cohesion_force
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
    velocities0: torch.Tensor | None = None,
    rounds: int | None = None,
    simulation: "SimulationSpec",
    teacher: "TeacherDynamics",
    model: "ModelSpec",
) -> tuple[torch.Tensor, torch.Tensor]:
    scenario = _build_boids_scenario(
        positions0,
        radius=simulation.radius,
        init_connectivity=model.init_connectivity,
        init_k_neighbors=model.init_k_neighbors,
        init_min_degree=model.init_min_degree,
    )
    return teacher_rollout(
        positions0=positions0,
        velocities0=(torch.zeros_like(positions0) if velocities0 is None else velocities0),
        edge_index=scenario.edge_index,
        rounds=simulation.rounds if rounds is None else rounds,
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
