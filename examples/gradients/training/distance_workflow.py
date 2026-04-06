"""Training workflow for simple learnable gradients."""

from __future__ import annotations

import torch
import torch.nn as nn
from ..domain.grid import build_corner_source_grid
from ..domain.program import auto_rounds
from ..model.distance_model import GradientModel
from ..visualization.grid import render_gradient_evolution


class LearnableGradientWorkflow:
    """Manages the training of a simple gradient model with learnable weights."""

    def __init__(self, spec):
        self.spec = spec

    def run(
        self,
        device: torch.device | None = None,
        gif: bool = False,
        viz: bool = True,
        viz_prefix: str = "generated/gradient_learnable",
        gif_fps: int = 10,
    ) -> tuple[GradientModel, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Execute the training process."""
        scenario, source, target = build_corner_source_grid(
            self.spec.grid.rows,
            self.spec.grid.cols,
            connectivity=self.spec.grid.connectivity,
            device=device,
        )
        rounds = auto_rounds(self.spec.grid.rows, self.spec.grid.cols, 0)
        model = GradientModel(scenario, rounds, init_w=self.spec.initial_weight).to(
            scenario.device
        )
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
                print(
                    f"Epoch {epoch + 1:3d}  loss={loss.item():.6f}  w={model.w.item():.4f}"
                )

        print()
        print(f"Final w = {model.w.item():.4f}  (expected: 1.0)")

        if gif and viz:
            render_gradient_evolution(
                scenario,
                source,
                rounds,
                model.w,
                self.spec.grid.rows,
                self.spec.grid.cols,
                f"{viz_prefix}_evolution.gif",
                title="Learnable Gradient Evolution (post-training)",
                fps=gif_fps,
            )

        with torch.no_grad():
            final_pred = model(source)

        return model, source, target, final_pred
