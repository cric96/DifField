"""Tests specific to the learnable gradient example family."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from examples.gradients.domain.program import auto_rounds
from examples.gradients.domain.grid import build_corner_source_grid
from examples.gradients.model.distance_model import GradientModel


def test_gradient_model_training_decreases_loss_and_recovers_unit_weight():
    torch.manual_seed(0)
    scenario, source, target = build_corner_source_grid(
        4, 4, connectivity=4, device=torch.device("cpu")
    )
    rounds = auto_rounds(4, 4, 0)
    model = GradientModel(scenario, rounds, init_w=3.0)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.05)

    losses = []
    for _ in range(80):
        optimizer.zero_grad()
        pred = model(source)
        mask = pred.isfinite()
        loss = F.mse_loss(pred[mask], target[mask])
        loss.backward()
        assert model.w.grad is not None
        assert torch.isfinite(model.w.grad)
        optimizer.step()
        losses.append(float(loss.item()))

    with torch.no_grad():
        pred = model(source)

    assert losses[-1] < losses[0] * 0.05
    assert abs(float(model.w.item()) - 1.0) < 0.1
    assert torch.max((pred - target).abs()).item() < 0.3
