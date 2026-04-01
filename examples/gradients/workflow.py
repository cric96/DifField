"""Training workflows for gradient examples."""

from __future__ import annotations

import torch
import torch.nn as nn

try:
    from .common import auto_rounds, build_corner_source_grid
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
    from common import auto_rounds, build_corner_source_grid
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

    def run(self) -> tuple[GradientModel, torch.Tensor, torch.Tensor, torch.Tensor]:
        scenario, source, target = build_corner_source_grid(
            self.spec.grid.rows,
            self.spec.grid.cols,
            connectivity=self.spec.grid.connectivity,
        )
        rounds = auto_rounds(self.spec.grid.rows, self.spec.grid.cols, 0)
        model = GradientModel(scenario, rounds, init_w=self.spec.initial_weight)
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

        with torch.no_grad():
            pred = model(source)
        return model, source, target, pred


class AttentionGradientWorkflow:
    def __init__(self, spec: AttentionGradientSpec):
        self.spec = spec

    def run(self) -> tuple[AttentionGradientModel, torch.Tensor]:
        scenario, source, target = build_corner_source_grid(
            self.spec.grid.rows,
            self.spec.grid.cols,
            connectivity=self.spec.grid.connectivity,
        )
        rounds = auto_rounds(self.spec.grid.rows, self.spec.grid.cols, 0)
        model = AttentionGradientModel(scenario, rounds)
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
                print(pred.view(self.spec.grid.rows, self.spec.grid.cols).detach().numpy().round(2))
                print(
                    f"Epoch {epoch + 1:3d}  loss={loss.item():8.4f}  "
                    f"w={model.w.item():.4f}  tau={model.attn_aggr.tau.item():.4f}  "
                    f"a={model.attn_aggr.a.item():.4f}"
                )

        print()
        print(f"Final:  w={model.w.item():.4f}  (target: 1.0)")
        print(f"        tau={model.attn_aggr.tau.item():.4f}  (lower = sharper)")
        print(f"        a={model.attn_aggr.a.item():.4f}  (positive = attend to low values)")

        with torch.no_grad():
            pred = model(source)
        return model, pred


class MovingGradientWorkflow:
    def __init__(self, spec: MovingGradientSpec):
        self.spec = spec

    def run(self) -> LearnableMovingGradient:
        torch.manual_seed(self.spec.seed)
        positions0 = torch.rand(self.spec.num_nodes, 2)
        model = LearnableMovingGradient(
            positions=positions0,
            radius=self.spec.radius,
            source_idx=self.spec.source,
            learn_mode=self.spec.learn,
        )

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
        return model
