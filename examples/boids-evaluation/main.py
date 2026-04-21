#!/usr/bin/env python3
"""Simplified boids evaluation example.

Single-file orchestrator with clearly separated sections:
  1. Aggregate programming (gather/scatter/iterate)
  2. Model (learnable parameters)
  3. Losses
  4. Training (fixed horizon, replay traces)
  5. Evaluation
  6. Visualization
  7. Main orchestrator
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "examples"))

from autofield import (
    SpatialScenario,
    bounce_in_box,
    gather_avg,
    gather_sum,
    iterate,
    limit_speed,
    normalize_vectors,
    scatter,
)
from autofield.dsl import AggregateContext

from shared.plotting import export_moving_gif, plot_moving_snapshots, plot_node_trajectories
from shared.training import MetricHistory, grad_norm, parse_int_csv
from shared.diagnostics.csv import save_history_csv
from shared.metrics import mean, std
from generate_plots import generate_boids_plots, plot_train_val_metrics, plot_parameter_recovery_bands

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
except ImportError:
    plt = None


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 1: AGGREGATE PROGRAMMING (gather / scatter / iterate)
# ═════════════════════════════════════════════════════════════════════════════


def boids_step(
    vel: torch.Tensor,
    pos: torch.Tensor,
    *,
    dt: float,
    sep: float,
    w_sep: float | torch.Tensor,
    w_align: float | torch.Tensor,
    w_cohesion: float | torch.Tensor,
    damping: float | torch.Tensor,
    max_speed: float | torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    # 1. Alignment & Cohesion (Neighbor averages)
    neigh_vel = gather_avg(scatter(vel))
    neigh_pos = gather_avg(scatter(pos))
    align_force = neigh_vel - vel
    cohesion_force = neigh_pos - pos

    # 2. Separation (Distance-based repulsion)
    delta = pos - scatter(pos)
    dist = delta.norm(dim=-1)
    sep_mask = ((dist <= sep) & (dist > 0)).pointwise()
    sep_force = normalize_vectors(gather_sum(delta * sep_mask))

    # 3. Acceleration & Integration
    acc = w_sep * sep_force + w_align * align_force + w_cohesion * cohesion_force
    pre_clip_vel = damping * vel + dt * acc
    return pre_clip_vel, limit_speed(pre_clip_vel, max_speed)

# ═════════════════════════════════════════════════════════════════════════════
# SECTION 2: Utils for teacher rollout and trace generation
# ═════════════════════════════════════════════════════════════════════════════

def sample_initial_state(
    num_nodes: int,
    *,  
    seed: int,
    velocity_scale: float,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    gen = torch.Generator()
    gen.manual_seed(seed)
    positions = torch.rand(num_nodes, 2, generator=gen, dtype=torch.float32).to(device)
    if velocity_scale <= 0.0:
        velocities = torch.zeros_like(positions)
    else:
        directions = (
            torch.rand(num_nodes, 2, generator=gen, dtype=torch.float32) * 2.0 - 1.0
        ).to(device)
        velocities = normalize_vectors(directions) * velocity_scale
    return positions, velocities


@torch.no_grad()
def teacher_rollout(
    positions0: torch.Tensor,
    velocities0: torch.Tensor,
    *,
    rounds: int,
    radius: float,
    sep: float,
    dt: float,
    damping: float,
    max_speed: float,
    w_sep: float,
    w_align: float,
    w_cohesion: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    scenario = SpatialScenario(
        positions=positions0,
        edge_radius=radius,
        device=positions0.device,
    )
    ctx = AggregateContext(
        scenario.edge_index, scenario.num_nodes, edge_weight=scenario.edge_weight
    )
    scenario.sync_context(ctx._ctx)
    pos_t = scenario.positions
    pre_clip_seq: list[torch.Tensor] = []

    def velocity_update(prev: torch.Tensor) -> torch.Tensor:
        pre_clip, clipped = boids_step(
            prev, pos_t,
            dt=dt, sep=sep,
            w_sep=w_sep, w_align=w_align, w_cohesion=w_cohesion,
            damping=damping, max_speed=max_speed,
        )
        pre_clip_seq.append(
            torch.nan_to_num(pre_clip, nan=0.0, posinf=0.0, neginf=0.0)
        )
        return torch.nan_to_num(clipped, nan=0.0, posinf=0.0, neginf=0.0)

    positions_seq: list[torch.Tensor] = []
    velocities_seq: list[torch.Tensor] = []
    edge_index_seq: list[torch.Tensor] = []

    for _ in range(rounds):
        scenario.sync_context(ctx._ctx)
        pos_t = scenario.positions
        edge_index_seq.append(scenario.edge_index.clone())
        with ctx.round():
            vel = iterate(velocities0, velocity_update, name="vel")
        new_pos = pos_t + dt * vel
        new_pos, vel_bounced = bounce_in_box(new_pos, vel)
        scenario.update_positions(new_pos, refresh_topology=True)
        ctx._ctx.state.update(vel_bounced, name="vel")
        positions_seq.append(scenario.positions)
        velocities_seq.append(vel_bounced)

    return (
        torch.stack(positions_seq, dim=0),
        torch.stack(velocities_seq, dim=0),
        torch.stack(pre_clip_seq, dim=0),
        edge_index_seq,
    )


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 3: MODEL (learnable parameters)
# ═════════════════════════════════════════════════════════════════════════════


def _inverse_softplus(x: float) -> float:
    import math
    return x + math.log(1.0 - math.exp(-x)) if x < 20.0 else x


def _softplus(x: torch.Tensor) -> torch.Tensor:
    return F.softplus(x)


class LearnableBoids(nn.Module):
    def __init__(
        self,
        *,
        radius: float,
        sep: float,
        dt: float,
        damping: float,
        max_speed: float,
        init_w_sep_target: float = 0.02,
        init_w_align_target: float = 0.08,
        init_w_cohesion_target: float = 1.10,
    ):
        super().__init__()
        self.radius = radius
        self.sep = sep
        self.dt = dt
        self.damping = damping
        self.max_speed = max_speed

        self.w_sep_raw = nn.Parameter(torch.tensor(_inverse_softplus(init_w_sep_target)))
        self.w_align_raw = nn.Parameter(torch.tensor(_inverse_softplus(init_w_align_target)))
        self.w_cohesion_raw = nn.Parameter(torch.tensor(_inverse_softplus(init_w_cohesion_target)))

    @property
    def w_sep(self) -> torch.Tensor:
        return _softplus(self.w_sep_raw)

    @property
    def w_align(self) -> torch.Tensor:
        return _softplus(self.w_align_raw)

    @property
    def w_cohesion(self) -> torch.Tensor:
        return _softplus(self.w_cohesion_raw)

    def trainable_parameters(self) -> list[nn.Parameter]:
        return [self.w_sep_raw, self.w_align_raw, self.w_cohesion_raw]

    def step(
        self,
        positions: torch.Tensor,
        velocities: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        scenario = SpatialScenario(
            positions=positions,
            edge_radius=self.radius,
            device=positions.device,
        )
        ctx = AggregateContext(
            scenario.edge_index, scenario.num_nodes, edge_weight=scenario.edge_weight
        )
        scenario.sync_context(ctx._ctx)
        pos_t = scenario.positions
        pre_clip_holder: list[torch.Tensor] = []

        def velocity_update(prev: torch.Tensor) -> torch.Tensor:
            pre_clip, clipped = boids_step(
                prev, pos_t,
                dt=self.dt, sep=self.sep,
                w_sep=self.w_sep, w_align=self.w_align, w_cohesion=self.w_cohesion,
                damping=self.damping, max_speed=self.max_speed,
            )
            pre_clip_holder.append(
                torch.nan_to_num(pre_clip, nan=0.0, posinf=0.0, neginf=0.0)
            )
            return torch.nan_to_num(clipped, nan=0.0, posinf=0.0, neginf=0.0)

        with ctx.round():
            vel = iterate(velocities, velocity_update, name="vel")

        new_pos = pos_t + self.dt * vel
        new_pos, vel_bounced = bounce_in_box(new_pos, vel)
        return new_pos, vel_bounced, pre_clip_holder[0]

    @torch.no_grad()
    def rollout(
        self,
        rounds: int,
        *,
        positions0: torch.Tensor,
        velocities0: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, list[torch.Tensor]]:
        positions = positions0.clone()
        velocities = velocities0.clone()
        positions_seq: list[torch.Tensor] = []
        velocities_seq: list[torch.Tensor] = []
        edge_index_seq: list[torch.Tensor] = []

        for step_idx in range(rounds):
            # We need to compute edge_index here for consistency
            scenario = SpatialScenario(
                positions=positions,
                edge_radius=self.radius,
                device=positions.device,
            )
            edge_index_seq.append(scenario.edge_index.clone())

            pred_pos, pred_vel, _ = self.step(
                positions, velocities,
            )
            positions_seq.append(pred_pos)
            velocities_seq.append(pred_vel)
            positions = pred_pos
            velocities = pred_vel

        return (
            torch.stack(positions_seq, dim=0),
            torch.stack(velocities_seq, dim=0),
            edge_index_seq,
        )


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 4: LOSSES
# ═════════════════════════════════════════════════════════════════════════════


def trajectory_mse_loss(
    pred_pos_seq: torch.Tensor,
    pred_vel_seq: torch.Tensor,
    teacher_pos_seq: torch.Tensor,
    teacher_vel_seq: torch.Tensor,
    *,
    velocity_weight: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    pos_loss = F.mse_loss(pred_pos_seq, teacher_pos_seq)
    vel_loss = F.mse_loss(pred_vel_seq, teacher_vel_seq)
    total = pos_loss + velocity_weight * vel_loss
    return total, pos_loss, vel_loss


def close_pair_distance_loss(
    pred_pos: torch.Tensor,
    teacher_pos: torch.Tensor,
    *,
    sep: float,
    focus_scale: float = 1.5,
    sharpness: float = 24.0,
) -> torch.Tensor:
    focus_radius = focus_scale * sep
    pred_dist = torch.cdist(pred_pos, pred_pos)
    teacher_dist = torch.cdist(teacher_pos, teacher_pos)
    near_weight = torch.maximum(
        torch.sigmoid(sharpness * (focus_radius - pred_dist)),
        torch.sigmoid(sharpness * (focus_radius - teacher_dist)),
    )
    mask = 1.0 - torch.eye(pred_pos.shape[0], device=pred_pos.device, dtype=pred_pos.dtype)
    weight = near_weight * mask
    return (
        (((pred_dist - teacher_dist).pow(2)) * weight).sum()
        / weight.sum().clamp_min(1e-6)
    )


def teacher_forced_step_losses(
    model: LearnableBoids,
    *,
    positions0: torch.Tensor,
    velocities0: torch.Tensor,
    teacher_pos_seq: torch.Tensor,
    teacher_vel_seq: torch.Tensor,
    teacher_preclip_seq: torch.Tensor,
    velocity_loss_weight: float,
    sep: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    total_loss = torch.zeros((), device=positions0.device)
    total_pos_loss = torch.zeros((), device=positions0.device)
    total_vel_loss = torch.zeros((), device=positions0.device)
    total_sep_focus = torch.zeros((), device=positions0.device)

    state_pos = positions0
    state_vel = velocities0
    num_steps = int(teacher_pos_seq.shape[0])

    for step_idx in range(num_steps):
        target_pos = teacher_pos_seq[step_idx]
        target_vel = teacher_vel_seq[step_idx]
        target_preclip = teacher_preclip_seq[step_idx]

        pred_pos, _, pred_preclip = model.step(
            state_pos, state_vel,
        )
        pos_loss = F.mse_loss(pred_pos, target_pos)
        vel_loss = F.mse_loss(pred_preclip, target_preclip)
        loss = pos_loss + velocity_loss_weight * vel_loss

        sep_focus_loss = close_pair_distance_loss(
            pred_pos, target_pos, sep=sep,
        )

        total_loss = total_loss + loss
        total_pos_loss = total_pos_loss + pos_loss
        total_vel_loss = total_vel_loss + vel_loss
        total_sep_focus = total_sep_focus # + sep_focus_loss

        state_pos = target_pos
        state_vel = target_vel

    scale = 1.0 / max(1, num_steps)
    return (
        total_loss * scale,
        total_pos_loss * scale,
        total_vel_loss * scale,
        total_sep_focus * scale,
    )


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 5: TRAINING (fixed horizon, replay traces)
# ═════════════════════════════════════════════════════════════════════════════


@dataclass(frozen=True)
class ReplayTrace:
    positions0: torch.Tensor
    velocities0: torch.Tensor
    pos_seq: torch.Tensor
    vel_seq: torch.Tensor
    preclip_seq: torch.Tensor
    edge_index_seq: list[torch.Tensor]
    seed: int


def generate_or_load_trace(
    *,
    seed: int,
    num_nodes: int,
    rounds: int,
    radius: float,
    sep: float,
    dt: float,
    damping: float,
    max_speed: float,
    w_sep: float,
    w_align: float,
    w_cohesion: float,
    velocity_scale: float,
    trace_dir: Path,
    device: torch.device,
) -> ReplayTrace:
    trace_path = trace_dir / f"seed{seed}_r{rounds}.pt"
    if trace_path.exists():
        payload = torch.load(trace_path, map_location=device, weights_only=False)
        return ReplayTrace(
            positions0=payload["positions0"],
            velocities0=payload["velocities0"],
            pos_seq=payload["pos_seq"],
            vel_seq=payload["vel_seq"],
            preclip_seq=payload["preclip_seq"],
            edge_index_seq=payload.get("edge_index_seq", []),
            seed=int(payload["metadata"]["seed"]),
        )

    positions0, velocities0 = sample_initial_state(
        num_nodes, seed=seed, velocity_scale=velocity_scale, device=device,
    )
    pos_seq, vel_seq, preclip_seq, edge_index_seq = teacher_rollout(
        positions0, velocities0,
        rounds=rounds, radius=radius, sep=sep, dt=dt,
        damping=damping, max_speed=max_speed,
        w_sep=w_sep, w_align=w_align, w_cohesion=w_cohesion,
    )
    trace_dir.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "positions0": positions0.detach().cpu(),
            "velocities0": velocities0.detach().cpu(),
            "pos_seq": pos_seq.detach().cpu(),
            "vel_seq": vel_seq.detach().cpu(),
            "preclip_seq": preclip_seq.detach().cpu(),
            "edge_index_seq": [ei.detach().cpu() for ei in edge_index_seq],
            "metadata": {"seed": seed, "rounds": rounds, "num_nodes": num_nodes},
        },
        trace_path,
    )
    return ReplayTrace(
        positions0=positions0,
        velocities0=velocities0,
        pos_seq=pos_seq,
        vel_seq=vel_seq,
        preclip_seq=preclip_seq,
        edge_index_seq=edge_index_seq,
        seed=seed,
    )


def train_one_seed(
    *,
    seed: int,
    trace: ReplayTrace,
    eval_seed: int,
    eval_positions0: torch.Tensor,
    eval_velocities0: torch.Tensor,
    eval_teacher_pos_seq: torch.Tensor,
    eval_teacher_vel_seq: torch.Tensor,
    eval_teacher_preclip_seq: torch.Tensor,
    epochs: int,
    lr: float,
    velocity_loss_weight: float,
    separation_loss_weight: float,
    eval_every: int,
    validation_dir: Path,
    radius: float,
    sep: float,
    dt: float,
    damping: float,
    max_speed: float,
    teacher_w_sep: float,
    teacher_w_align: float,
    teacher_w_cohesion: float,
    velocity_scale: float,
    num_nodes: int,
    rounds: int,
    device: torch.device,
    highlight_node: int,
    show_links: bool,
    links_alpha: float,
    links_width: float,
    gif_fps: int,
) -> tuple[LearnableBoids, dict[str, list[float]], dict[int, torch.Tensor]]:
    torch.manual_seed(seed)

    model = LearnableBoids(
        radius=radius, sep=sep, dt=dt, damping=damping, max_speed=max_speed,
    ).to(device)

    optimizer = torch.optim.Adam(model.trainable_parameters(), lr=lr)

    history_keys = [
        "epoch", "total_loss", "pos_loss", "vel_loss",
        "pos_error", "w_sep", "w_align", "w_cohesion",
        "w_sep_abs_error", "w_sep_rel_error",
        "w_align_abs_error", "w_align_rel_error",
        "w_cohesion_abs_error", "w_cohesion_rel_error",
        "grad_norm", "lr",
        "val_total_loss", "val_pos_loss", "val_vel_loss", "val_pos_error",
    ]
    history = MetricHistory.from_keys(history_keys)

    eval_trajectories = {}

    teacher_params = {
        "w_sep": teacher_w_sep,
        "w_align": teacher_w_align,
        "w_cohesion": teacher_w_cohesion,
    }

    for epoch in range(epochs):
        optimizer.zero_grad()
        total, pos, vel, sep_focus = teacher_forced_step_losses(
            model,
            positions0=trace.positions0,
            velocities0=trace.velocities0,
            teacher_pos_seq=trace.pos_seq,
            teacher_vel_seq=trace.vel_seq,
            teacher_preclip_seq=trace.preclip_seq,
            velocity_loss_weight=velocity_loss_weight,
            sep=sep,
        )
        objective = total + separation_loss_weight * sep_focus
        objective.backward()
        g_norm = grad_norm(model.trainable_parameters())
        torch.nn.utils.clip_grad_norm_(model.trainable_parameters(), 5.0)
        optimizer.step()

        with torch.no_grad():
            pred_pos_seq, _, _ = model.rollout(
                rounds,
                positions0=trace.positions0,
                velocities0=trace.velocities0,
            )
        pos_error = (trace.pos_seq - pred_pos_seq).norm(dim=-1).mean().item()

        learned_params = {
            "w_sep": float(model.w_sep.item()),
            "w_align": float(model.w_align.item()),
            "w_cohesion": float(model.w_cohesion.item()),
        }
        recovery = compute_recovery(teacher_params, learned_params)

        do_eval = (epoch + 1) % max(1, eval_every) == 0 or epoch == 0 or epoch == epochs - 1 or epoch == 3 or epoch == 4

        val_total, val_pos, val_vel, val_pos_error = float("nan"), float("nan"), float("nan"), float("nan")
        if do_eval:
            with torch.no_grad():
                val_pred_pos, val_pred_vel, val_edge_seq = model.rollout(
                    rounds,
                    positions0=eval_positions0,
                    velocities0=eval_velocities0,
                )
            val_total, val_pos, val_vel = trajectory_mse_loss(
                val_pred_pos, val_pred_vel,
                eval_teacher_pos_seq, eval_teacher_vel_seq,
                velocity_weight=velocity_loss_weight,
            )
            val_pos_error = (eval_teacher_pos_seq - val_pred_pos).norm(dim=-1).mean().item()
            val_total, val_pos, val_vel = (
                float(val_total.item()), float(val_pos.item()), float(val_vel.item()),
            )

            _save_validation_gif(
                val_pred_pos, val_pred_vel, val_edge_seq,
                epoch=epoch,
                validation_dir=validation_dir,
                eval_seed=eval_seed,
                highlight_node=highlight_node,
                show_links=show_links,
                links_alpha=links_alpha,
                links_width=links_width,
                gif_fps=gif_fps,
                learned_params=learned_params,
                teacher_params=teacher_params,
                recovery=recovery,
                rounds=rounds,
                show_source=False,
                phantom_pos_seq=eval_teacher_pos_seq,
            )
            eval_trajectories[epoch] = val_pred_pos.detach().cpu()

        history.append(
            epoch=float(epoch + 1),
            total_loss=float(total.item()),
            pos_loss=float(pos.item()),
            vel_loss=float(vel.item()),
            pos_error=pos_error,
            w_sep=learned_params["w_sep"],
            w_align=learned_params["w_align"],
            w_cohesion=learned_params["w_cohesion"],
            w_sep_abs_error=recovery["w_sep"]["abs_error"],
            w_sep_rel_error=recovery["w_sep"]["rel_error"],
            w_align_abs_error=recovery["w_align"]["abs_error"],
            w_align_rel_error=recovery["w_align"]["rel_error"],
            w_cohesion_abs_error=recovery["w_cohesion"]["abs_error"],
            w_cohesion_rel_error=recovery["w_cohesion"]["rel_error"],
            grad_norm=g_norm,
            lr=lr,
            val_total_loss=val_total,
            val_pos_loss=val_pos,
            val_vel_loss=val_vel,
            val_pos_error=val_pos_error,
        )

        if (epoch + 1) % 5 == 0 or epoch == 0:
            msg = (
                f"epoch={epoch + 1:3d} loss={total.item():.6f} pos_err={pos_error:.6f} "
                f"err_sep={recovery['w_sep']['rel_error']:.1%} "
                f"err_align={recovery['w_align']['rel_error']:.1%} "
                f"err_coh={recovery['w_cohesion']['rel_error']:.1%}"
            )
            if not torch.isnan(torch.tensor(val_total)):
                msg += f" val={val_total:.6f}"
            print(msg)

    return model, history.to_dict(), eval_trajectories


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 6: EVALUATION
# ═════════════════════════════════════════════════════════════════════════════


def compute_recovery(
    teacher_params: dict[str, float],
    learned_params: dict[str, float],
) -> dict[str, dict[str, float]]:
    recovery: dict[str, dict[str, float]] = {}
    for name, tv in teacher_params.items():
        lv = learned_params[name]
        abs_err = abs(lv - tv)
        recovery[name] = {
            "teacher": tv,
            "learned": lv,
            "abs_error": abs_err,
            "rel_error": abs_err / max(abs(tv), 1e-9),
        }
    return recovery


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 7: VISUALIZATION
# ═════════════════════════════════════════════════════════════════════════════


def _save_validation_gif(
    pred_pos_seq: torch.Tensor,
    pred_vel_seq: torch.Tensor,
    edge_index_seq: list[torch.Tensor],
    *,
    epoch: int,
    validation_dir: Path,
    eval_seed: int,
    highlight_node: int,
    show_links: bool,
    links_alpha: float,
    links_width: float,
    gif_fps: int,
    learned_params: dict[str, float],
    teacher_params: dict[str, float],
    recovery: dict[str, dict[str, float]],
    rounds: int,
    show_source: bool = True,
    phantom_pos_seq: torch.Tensor | None = None,
) -> None:
    epoch_dir = validation_dir / f"epoch_{epoch + 1:04d}"
    epoch_dir.mkdir(parents=True, exist_ok=True)

    positions_by_round = {r: pred_pos_seq[r] for r in range(rounds)}
    values_by_round = {r: pred_vel_seq[r].norm(dim=1) for r in range(rounds)}
    edge_index_by_round = {r: edge_index_seq[r] for r in range(rounds)}

    gif_path = epoch_dir / f"validation_seed{eval_seed}_pred.gif"
    export_moving_gif(
        positions_by_round=positions_by_round,
        values_by_round=values_by_round,
        edge_index_by_round=edge_index_by_round,
        source_idx=highlight_node,
        output_path=str(gif_path),
        title=f"Validation Seed {eval_seed} epoch {epoch + 1}",
        fps=gif_fps,
        show_links=show_links,
        links_alpha=links_alpha,
        links_width=links_width,
        show_source=show_source,
        phantom_pos_seq=phantom_pos_seq,
    )

    traj_path = epoch_dir / f"validation_seed{eval_seed}_trajectories.png"
    plot_node_trajectories(
        positions_over_time=[pred_pos_seq[r] for r in range(rounds)],
        source_idx=highlight_node,
        output_path=str(traj_path),
        title=f"Validation Seed {eval_seed} epoch {epoch + 1} Trajectories",
        show_source=show_source,
        phantom_pos_seq=phantom_pos_seq,
    )

    with (epoch_dir / "learned_parameters.json").open("w", encoding="utf-8") as f:
        json.dump(
            {"target": teacher_params, "learned": learned_params, "recovery": recovery},
            f, indent=2,
        )


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 7: MAIN ORCHESTRATOR
# ═════════════════════════════════════════════════════════════════════════════


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Simplified boids evaluation")
    parser.add_argument("--seeds", type=str, default="5,7,11")
    parser.add_argument("--eval-seeds", type=str, default="101")
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--rounds", type=int, default=24)
    parser.add_argument("--num-nodes", type=int, default=50)
    parser.add_argument("--lr", type=float, default=0.05)
    parser.add_argument("--radius", type=float, default=0.23)
    parser.add_argument("--sep", type=float, default=0.06)
    parser.add_argument("--dt", type=float, default=1.0)
    parser.add_argument("--damping", type=float, default=0.94)
    parser.add_argument("--max-speed", type=float, default=0.014)
    parser.add_argument("--velocity-scale", type=float, default=0.014)
    parser.add_argument("--teacher-w-sep", type=float, default=0.05)
    parser.add_argument("--teacher-w-align", type=float, default=0.90)
    parser.add_argument("--teacher-w-cohesion", type=float, default=0.35)
    parser.add_argument("--velocity-loss-weight", type=float, default=1.0)
    parser.add_argument("--separation-loss-weight", type=float, default=0.0)
    parser.add_argument("--eval-every", type=int, default=5)
    parser.add_argument("--out-dir", type=str, default="generated/boids-evaluation")
    parser.add_argument("--highlight-node", type=int, default=0)
    parser.add_argument("--gif-fps", type=int, default=8)
    parser.add_argument("--hide-links", action="store_true")
    parser.add_argument("--links-alpha", type=float, default=0.15)
    parser.add_argument("--links-width", type=float, default=0.6)
    parser.add_argument("--device", type=str, default="")
    return parser.parse_args()


def _get_device(device_str: str) -> torch.device:
    if device_str:
        return torch.device(device_str)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main() -> None:
    args = parse_args()
    device = _get_device(args.device)
    seeds = parse_int_csv(args.seeds)
    eval_seeds = parse_int_csv(args.eval_seeds)
    eval_seed = eval_seeds[0] if eval_seeds else seeds[-1]

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    trace_dir = out_dir / "replay-traces"
    validation_dir = out_dir / "validation"

    print(f"=== Boids Evaluation: {len(seeds)} training seeds, eval seed {eval_seed} ===")
    print(f"  device={device}  epochs={args.epochs}  rounds={args.rounds}  nodes={args.num_nodes}")
    print(f"  teacher: w_sep={args.teacher_w_sep}  w_align={args.teacher_w_align}  w_cohesion={args.teacher_w_cohesion}")

    # Pre-generate evaluation trace
    eval_trace = generate_or_load_trace(
        seed=eval_seed, num_nodes=args.num_nodes, rounds=args.rounds,
        radius=args.radius, sep=args.sep, dt=args.dt,
        damping=args.damping, max_speed=args.max_speed,
        w_sep=args.teacher_w_sep, w_align=args.teacher_w_align,
        w_cohesion=args.teacher_w_cohesion, velocity_scale=args.velocity_scale,
        trace_dir=trace_dir, device=device,
    )

    # Teacher Reference Generation
    teacher_dir = out_dir / "teacher"
    teacher_dir.mkdir(parents=True, exist_ok=True)
    print(f"--- Generating Teacher Reference at {teacher_dir} ---")
    plot_node_trajectories(
        positions_over_time=[eval_trace.pos_seq[r] for r in range(args.rounds)],
        source_idx=args.highlight_node,
        output_path=str(teacher_dir / "teacher_trajectories.png"),
        title="Teacher Ground Truth Trajectories",
        show_source=False,
    )
    export_moving_gif(
        positions_by_round={r: eval_trace.pos_seq[r] for r in range(args.rounds)},
        values_by_round={r: eval_trace.vel_seq[r].norm(dim=1) for r in range(args.rounds)},
        edge_index_by_round={r: eval_trace.edge_index_seq[r] for r in range(args.rounds)},
        source_idx=args.highlight_node,
        output_path=str(teacher_dir / "teacher_reference.gif"),
        title="Teacher Ground Truth Reference",
        fps=args.gif_fps,
        show_links=not args.hide_links,
        links_alpha=args.links_alpha,
        links_width=args.links_width,
        show_source=False,
    )

    all_histories: list[dict[str, list[float]]] = []
    final_models: dict[int, LearnableBoids] = {}
    all_eval_trajectories: list[dict[int, torch.Tensor]] = []

    for seed in seeds:
        print(f"\n--- Training seed {seed} ---")
        trace = generate_or_load_trace(
            seed=seed, num_nodes=args.num_nodes, rounds=args.rounds,
            radius=args.radius, sep=args.sep, dt=args.dt,
            damping=args.damping, max_speed=args.max_speed,
            w_sep=args.teacher_w_sep, w_align=args.teacher_w_align,
            w_cohesion=args.teacher_w_cohesion, velocity_scale=args.velocity_scale,
            trace_dir=trace_dir, device=device,
        )

        model, history, eval_trajectories = train_one_seed(
            seed=seed,
            trace=trace,
            eval_seed=eval_seed,
            eval_positions0=eval_trace.positions0,
            eval_velocities0=eval_trace.velocities0,
            eval_teacher_pos_seq=eval_trace.pos_seq,
            eval_teacher_vel_seq=eval_trace.vel_seq,
            eval_teacher_preclip_seq=eval_trace.preclip_seq,
            epochs=args.epochs,
            lr=args.lr,
            velocity_loss_weight=args.velocity_loss_weight,
            separation_loss_weight=args.separation_loss_weight,
            eval_every=args.eval_every,
            validation_dir=validation_dir,
            radius=args.radius,
            sep=args.sep,
            dt=args.dt,
            damping=args.damping,
            max_speed=args.max_speed,
            teacher_w_sep=args.teacher_w_sep,
            teacher_w_align=args.teacher_w_align,
            teacher_w_cohesion=args.teacher_w_cohesion,
            velocity_scale=args.velocity_scale,
            num_nodes=args.num_nodes,
            rounds=args.rounds,
            device=device,
            highlight_node=args.highlight_node,
            show_links=not args.hide_links,
            links_alpha=args.links_alpha,
            links_width=args.links_width,
            gif_fps=args.gif_fps,
        )
        all_histories.append(history)
        final_models[seed] = model
        all_eval_trajectories.append(eval_trajectories)

    # Final visualization suite
    teacher_params = {
        "w_sep": args.teacher_w_sep,
        "w_align": args.teacher_w_align,
        "w_cohesion": args.teacher_w_cohesion,
    }

    plot_train_val_metrics(all_histories, out_dir)
    plot_parameter_recovery_bands(all_histories, teacher_params, out_dir)

    for i, (seed, model) in enumerate(final_models.items()):
        seed_dir = out_dir / f"seed_{seed}"
        seed_dir.mkdir(parents=True, exist_ok=True)
        
        # Save individual seed history
        history = all_histories[i]
        save_history_csv(history, seed_dir / "history.csv")

        # Evaluate final model on evaluation seed for consistent phantom overlay
        with torch.no_grad():
            pred_pos, pred_vel, pred_edge_seq = model.rollout(
                args.rounds,
                positions0=eval_trace.positions0,
                velocities0=eval_trace.velocities0,
            )

        positions_over_time = [pred_pos[r] for r in range(args.rounds)]
        highlight_idx = max(0, min(args.highlight_node, args.num_nodes - 1))

        # Extract data for reproducing the figures
        extracted_data = {
            "pred_pos": pred_pos.detach().cpu(),
            "pred_vel": pred_vel.detach().cpu(),
            "eval_trace_pos": eval_trace.pos_seq.detach().cpu(),
            "eval_trace_vel": eval_trace.vel_seq.detach().cpu(),
            "pred_edge_seq": [ei.detach().cpu() for ei in pred_edge_seq],
            "highlight_node": highlight_idx,
            "rounds": args.rounds,
            "num_nodes": args.num_nodes,
            "history": history,
            "teacher_params": teacher_params,
            "eval_trajectories": all_eval_trajectories[i]
        }
        torch.save(extracted_data, seed_dir / "eval_data.pt")

        generate_boids_plots(
            data=extracted_data,
            seed=seed,
            seed_dir=seed_dir,
            hide_links=args.hide_links,
            links_alpha=args.links_alpha,
            links_width=args.links_width,
            gif_fps=args.gif_fps,
        )

    # Save aggregated history and summary
    avg_history = {}
    for key in all_histories[0]:
        vals = [h[key] for h in all_histories if key in h]
        if vals and all(len(v) == len(vals[0]) for v in vals):
            avg_history[key] = [mean([v[i] for v in vals]) for i in range(len(vals[0]))]

    save_history_csv(avg_history, out_dir / "history.csv")

    final_learned = {
        seed: {
            "w_sep": float(m.w_sep.item()),
            "w_align": float(m.w_align.item()),
            "w_cohesion": float(m.w_cohesion.item()),
        }
        for seed, m in final_models.items()
    }
    recovery_summary = {
        str(seed): compute_recovery(teacher_params, params)
        for seed, params in final_learned.items()
    }

    summary = {
        "seeds": seeds,
        "eval_seed": eval_seed,
        "epochs": args.epochs,
        "rounds": args.rounds,
        "num_nodes": args.num_nodes,
        "teacher_params": teacher_params,
        "learned_params": final_learned,
        "recovery": recovery_summary,
    }
    with (out_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"\n=== Summary ===")
    print(f"{'seed':>6} | {'w_sep':>10} | {'w_align':>10} | {'w_cohesion':>10} | {'err_sep':>8} | {'err_align':>8} | {'err_coh':>8}")
    print("-" * 80)
    for seed in seeds:
        lp = final_learned[seed]
        rec = recovery_summary[str(seed)]
        print(
            f"{seed:>6} | {lp['w_sep']:>10.4f} | {lp['w_align']:>10.4f} | {lp['w_cohesion']:>10.4f} | "
            f"{rec['w_sep']['rel_error']:>8.1%} | {rec['w_align']['rel_error']:>8.1%} | {rec['w_cohesion']['rel_error']:>8.1%}"
        )

    print(f"\nArtifacts saved to {out_dir}")


if __name__ == "__main__":
    main()
