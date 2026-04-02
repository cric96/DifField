"""Training workflows for gradient examples."""

from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT / "examples") not in sys.path:
    sys.path.insert(0, str(ROOT / "examples"))

from aggregate_gnn import SnapshotRecorder
from shared.plotting import save_grid_simulation_gif, export_moving_gif

try:
    from .common import auto_rounds, build_corner_source_grid, run_gradient_program
    from .models import (
        MAX_DIST,
        AttentionGradientModel,
        GradientModel,
        LearnableMovingGradient,
        pairwise_shortest_hop,
        step_teacher_positions,
    )
    from .specs import AttentionGradientSpec, LearnableGradientSpec, MovingGradientSpec
except ImportError:
    from common import auto_rounds, build_corner_source_grid, run_gradient_program
    from models import (
        MAX_DIST,
        AttentionGradientModel,
        GradientModel,
        LearnableMovingGradient,
        pairwise_shortest_hop,
        step_teacher_positions,
    )
    from specs import AttentionGradientSpec, LearnableGradientSpec, MovingGradientSpec


class LearnableGradientWorkflow:
    def __init__(self, spec: LearnableGradientSpec):
        self.spec = spec

    def run(
        self,
        device: torch.device | None = None,
        gif: bool = False,
        viz: bool = True,
        viz_prefix: str = "examples/gradient_learnable",
        gif_fps: int = 10,
    ) -> tuple[GradientModel, torch.Tensor, torch.Tensor, torch.Tensor]:
        scenario, source, target = build_corner_source_grid(
            self.spec.grid.rows,
            self.spec.grid.cols,
            connectivity=self.spec.grid.connectivity,
            device=device,
        )
        rounds = auto_rounds(self.spec.grid.rows, self.spec.grid.cols, 0)
        model = GradientModel(scenario, rounds, init_w=self.spec.initial_weight).to(scenario.device)
        optimizer = torch.optim.Adam(model.parameters(), lr=self.spec.training.lr)

        print("=== Gradient (learnable w) training ===")
        print(f"Initial w = {model.w.item():.4f}")
        print()

        for epoch in range(self.spec.training.epochs):
            optimizer.zero_grad()
            pred = model(source)
            mask = pred.isfinite()
            loss = nn.functional.mse_loss(pred[mask], target[mask])
            loss.backward()
            optimizer.step()

            if (epoch + 1) % 50 == 0 or epoch == 0:
                print(f"Epoch {epoch + 1:3d}  loss={loss.item():.6f}  w={model.w.item():.4f}")

        print()
        print(f"Final w = {model.w.item():.4f}  (expected: 1.0)")

        recorder = None
        if gif and viz:
            recorder = SnapshotRecorder(state_fields=["dist"], capture_output=True)

        with torch.no_grad():
            if gif and viz:
                pred, _ = run_gradient_program(
                    scenario,
                    source,
                    rounds=rounds,
                    weight=model.w,
                    recorder=recorder,
                )
                save_grid_simulation_gif(
                    recorder.records,
                    "dist",
                    self.spec.grid.rows,
                    self.spec.grid.cols,
                    f"{viz_prefix}_evolution.gif",
                    title="Learnable Gradient Evolution (post-training)",
                    fps=gif_fps,
                )
            else:
                pred = model(source)
                
        return model, source, target, pred


class AttentionGradientWorkflow:
    def __init__(self, spec: AttentionGradientSpec):
        self.spec = spec

    def run(
        self,
        device: torch.device | None = None,
        gif: bool = False,
        viz: bool = True,
        viz_prefix: str = "examples/gradient_attention",
        gif_fps: int = 10,
    ) -> tuple[AttentionGradientModel, torch.Tensor]:
        scenario, source, target = build_corner_source_grid(
            self.spec.grid.rows,
            self.spec.grid.cols,
            connectivity=self.spec.grid.connectivity,
            device=device,
        )
        rounds = auto_rounds(self.spec.grid.rows, self.spec.grid.cols, 0)
        model = AttentionGradientModel(scenario, rounds).to(scenario.device)
        optimizer = torch.optim.Adam(model.parameters(), lr=self.spec.training.lr)

        print("=== Gradient with Learnable Attention Aggregator ===")
        print(f"Grid: {self.spec.grid.rows}x{self.spec.grid.cols}  Rounds: {model.rounds}")
        print("AC:   rep(inf)(d => mux(src, 0, nbr(d + w, aggr=attn_min)))")
        print("GNN:  GAT-style Bellman-Ford MPNN")
        print("Params: w (hop cost), a/b (attention), tau (temperature)")
        print(
            f"  Initial w={model.w.item():.2f}, "
            f"tau={model.attn_aggr.tau.item():.2f}, "
            f"a={model.attn_aggr.a.item():.2f}"
        )
        print()

        for epoch in range(self.spec.training.epochs):
            optimizer.zero_grad()
            pred = model(source)
            mask = pred.isfinite()
            loss = nn.functional.mse_loss(pred[mask], target[mask])
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
            optimizer.step()

            if (epoch + 1) % 5 == 0:
                print(pred.view(self.spec.grid.rows, self.spec.grid.cols).detach().cpu().numpy().round(2))
                print(
                    f"Epoch {epoch + 1:3d}  loss={loss.item():8.4f}  "
                    f"w={model.w.item():.4f}  tau={model.attn_aggr.tau.item():.4f}  "
                    f"a={model.attn_aggr.a.item():.4f}"
                )

        print()
        print(f"Final w = {model.w.item():.4f}")

        recorder = None
        if gif and viz:
            recorder = SnapshotRecorder(state_fields=["dist"], capture_output=True)

        with torch.no_grad():
            if gif and viz:
                pred, _ = run_gradient_program(
                    scenario,
                    source,
                    rounds=rounds,
                    weight=model.w,
                    aggr=model.attn_aggr,
                    recorder=recorder,
                )
                save_grid_simulation_gif(
                    recorder.records,
                    "dist",
                    self.spec.grid.rows,
                    self.spec.grid.cols,
                    f"{viz_prefix}_evolution.gif",
                    title="Attention Gradient Evolution (post-training)",
                    fps=gif_fps,
                )
            else:
                pred = model(source)

        return model, pred


class MovingGradientWorkflow:
    def __init__(self, spec: MovingGradientSpec):
        self.spec = spec

    def run(
        self,
        device: torch.device | None = None,
        gif: bool = False,
        viz: bool = True,
        viz_prefix: str = "examples/gradient_moving_nodes",
        gif_fps: int = 10,
    ) -> tuple[LearnableMovingGradient, torch.Tensor, torch.Tensor]:
        torch.manual_seed(self.spec.seed)
        positions0 = torch.rand(self.spec.num_nodes, 2, device=device)
        model = LearnableMovingGradient(
            positions=positions0,
            radius=self.spec.radius,
            source_idx=self.spec.source,
            learn_mode=self.spec.learn,
        ).to(device)

        teacher_vel = torch.zeros_like(positions0)
        teacher_vel[:, 0] = 0.01 * torch.sin(2.0 * torch.pi * positions0[:, 1])
        teacher_vel[:, 1] = 0.01 * torch.cos(2.0 * torch.pi * positions0[:, 0])

        if self.spec.learn == "motion":
            params = list(model.motion_policy.parameters())
        elif self.spec.learn == "ac":
            params = [model.w_raw]
        else:
            params = list(model.motion_policy.parameters()) + [model.w_raw]

        opt = torch.optim.Adam(params, lr=self.spec.lr)

        print("=== Learnable Moving Nodes Gradient ===")
        print(f"mode={self.spec.learn} init_w={model.w.item():.4f}")

        for epoch in range(self.spec.epochs):
            opt.zero_grad()
            pred_seq, final_pos = model(self.spec.rounds)

            targets = []
            pos = positions0.clone()
            vel = teacher_vel.clone()
            for _ in range(self.spec.rounds):
                targets.append(pairwise_shortest_hop(pos, self.spec.source, self.spec.radius))
                pos, vel = step_teacher_positions(pos, vel, dt=0.1)
            tgt_seq = torch.stack(targets, dim=0)

            pred_norm = pred_seq / MAX_DIST
            tgt_norm = tgt_seq / MAX_DIST
            loss = nn.functional.mse_loss(pred_norm, tgt_norm)

            source_pos = final_pos[self.spec.source]
            goal_loss = (final_pos[self.spec.target] - source_pos).pow(2).sum()
            w_reg = (model.w - 1.0).pow(2)
            total = loss + 0.1 * goal_loss + 0.01 * w_reg

            total.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()

            if (epoch + 1) % 20 == 0 or epoch == 0:
                print(
                    f"epoch={epoch + 1:3d} loss={total.item():.5f} "
                    f"mse={loss.item():.5f} w={model.w.item():.4f}"
                )

        print("Training complete.")
        
        final_pred_seq, final_pos = model(self.spec.rounds)

        if gif and viz:
            print("Generating evolution GIF...")
            pos_by_round = {}
            val_by_round = {}
            edge_index_by_round = {}
            
            positions = positions0.clone()
            velocities = torch.zeros_like(positions)
            scenario = SpatialScenario(positions=positions, edge_radius=model.radius, device=positions.device)
            ctx = AggregateContext(scenario.edge_index, scenario.num_nodes, edge_weight=scenario.edge_weight)
            source = torch.zeros(positions.shape[0], dtype=torch.float32, device=positions.device)
            source[model.source_idx] = 1.0

            with torch.no_grad():
                for r in range(self.spec.rounds):
                    scenario.sync_context(ctx._ctx)
                    with ctx.round():
                        dist = rep(
                            "dist",
                            MAX_DIST,
                            lambda dist_old: mux(source, field.of(0.0), nbr(dist_old + model.w, aggr="min")),
                        )
                    
                    pos_by_round[r] = positions.clone()
                    val_by_round[r] = dist.clone()
                    edge_index_by_round[r] = scenario.edge_index.clone()

                    dist_feat = torch.nan_to_num(dist, nan=0.0, posinf=10.0, neginf=0.0)
                    features = torch.cat([positions, dist_feat.unsqueeze(-1), velocities.norm(dim=1, keepdim=True)], dim=1)
                    dv = model.motion_policy(features)
                    new_vel = velocities + model.dt * dv
                    speed = new_vel.norm(dim=1, keepdim=True).clamp_min(1e-8)
                    new_vel = new_vel * torch.clamp(model.max_speed / speed, max=1.0)
                    
                    positions = positions + model.dt * new_vel
                    positions, velocities = model._clip_box(positions, new_vel)
                    scenario.update_positions(positions, refresh_topology=True)

            export_moving_gif(
                positions_by_round=pos_by_round,
                values_by_round=val_by_round,
                source_idx=model.source_idx,
                output_path=f"{viz_prefix}_evolution.gif",
                title="Moving Nodes Gradient Evolution (post-training)",
                edge_index_by_round=edge_index_by_round,
                show_links=True,
                fps=gif_fps,
            )

        return model, final_pred_seq[-1], final_pos
