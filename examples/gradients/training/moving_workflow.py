"""Training workflow for gradients in moving node networks."""

from __future__ import annotations

import torch
import torch.nn as nn
from ..domain.moving_logic import (
    pairwise_shortest_hop,
    step_teacher_positions,
    MAX_DIST,
)
from ..model.moving_model import LearnableMovingGradient
from ..visualization.moving import render_moving_gradient_evolution


class MovingGradientWorkflow:
    """Manages integrated learning of communication and motion policies."""

    def __init__(self, spec):
        self.spec = spec

    def run(
        self,
        device: torch.device | None = None,
        gif: bool = False,
        viz: bool = True,
        viz_prefix: str = "generated/gradient_moving_nodes",
        gif_fps: int = 10,
    ) -> tuple[LearnableMovingGradient, torch.Tensor, torch.Tensor]:
        """Execute the training process."""
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
                targets.append(
                    pairwise_shortest_hop(pos, self.spec.source, self.spec.radius)
                )
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
                    f"epoch={epoch + 1:3d} loss={total.item():.5f} mse={loss.item():.5f} w={model.w.item():.4f}"
                )

        print("Training complete.")

        if gif and viz:
            print("Generating evolution GIF...")
            render_moving_gradient_evolution(
                model, self.spec.rounds, f"{viz_prefix}_evolution.gif", fps=gif_fps
            )

        with torch.no_grad():
            final_pred_seq, final_pos = model(self.spec.rounds)

        return model, final_pred_seq[-1], final_pos
