"""Metric functions for territory imitation learning."""

from __future__ import annotations

from typing import TYPE_CHECKING
import torch
import torch.nn.functional as F

if TYPE_CHECKING:
    from ..domain.layout import TerritoryOutputs


def summary_metrics(
    pred: "TerritoryOutputs",
    target: "TerritoryOutputs",
    *,
    risk_loss_weight: float,
) -> dict[str, torch.Tensor]:
    """Compute training metrics comparing predicted to target outputs."""
    load_loss = F.mse_loss(pred.sink_loads, target.sink_loads)
    risk_loss = F.mse_loss(pred.sink_risks, target.sink_risks)
    total = load_loss + risk_loss_weight * risk_loss
    owner_agreement = (pred.hard_owner == target.hard_owner).float().mean()
    conservation_error = (pred.sink_loads.sum() - target.sink_loads.sum()).abs()

    return {
        "total": total,
        "load_loss": load_loss,
        "risk_loss": risk_loss,
        "owner_agreement": owner_agreement,
        "conservation_error": conservation_error,
    }
