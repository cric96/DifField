"""Training workflow for gradients with attention aggregation."""

from __future__ import annotations

import torch
import torch.nn as nn
from ..domain.grid import build_corner_source_grid
from ..domain.program import auto_rounds
from ..model.distance_model import AttentionGradientModel
from ..visualization.grid import render_gradient_evolution


class AttentionGradientWorkflow:
    """Manages the training of a gradient model with learnable attention aggregation."""

    def __init__(self, spec):
        self.spec = spec

    def run(
        self,
        device: torch.device | None = None,
        gif: bool = False,
        viz: bool = True,
        viz_prefix: str = "generated/gradient_attention",
        gif_fps: int = 10,
    ) -> tuple[AttentionGradientModel, torch.Tensor]:
        """Execute the training process."""
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
        print(
            f"Grid: {self.spec.grid.rows}x{self.spec.grid.cols}  Rounds: {model.rounds}"
        )

        for epoch in range(self.spec.training.epochs):
            optimizer.zero_grad()
            pred = model(source)
            mask = pred.isfinite()
            loss = nn.functional.mse_loss(pred[mask], target[mask])
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
            optimizer.step()

            if (epoch + 1) % 5 == 0:
                print(
                    f"Epoch {epoch + 1:3d}  loss={loss.item():8.4f}  w={model.w.item():.4f}  tau={model.attn_aggr.tau.item():.4f}"
                )

        print()
        print(f"Final w = {model.w.item():.4f}")

        if gif and viz:
            render_gradient_evolution(
                scenario,
                source,
                rounds,
                model.w,
                self.spec.grid.rows,
                self.spec.grid.cols,
                f"{viz_prefix}_evolution.gif",
                aggr=model.attn_aggr,
                title="Attention Gradient Evolution (post-training)",
                fps=gif_fps,
            )

        with torch.no_grad():
            final_pred = model(source)

        return model, final_pred
