"""Evaluation engine for learnable territory models."""

from __future__ import annotations

from typing import TYPE_CHECKING
import torch

from ..domain.factory import (
    build_scenario_from_spec,
    build_teacher_model_from_spec,
    build_layout_from_spec,
)
from ..domain.program import rollout_summary

try:
    from examples.shared.metrics import mean, std
except ImportError:
    from shared.metrics import mean, std

if TYPE_CHECKING:
    from ..model.territory_model import LearnableTerritoryModel
    from ..domain.specs import LearnableTerritoriesSpec


METRIC_KEYS = (
    "total",
    "load_loss",
    "risk_loss",
    "owner_agreement",
    "conservation_error",
)


def _metric_bundle(records: list[dict[str, float]], reducer) -> dict[str, float]:
    """Aggregate a list of metric dictionaries into a single one."""
    if not records:
        return {key: float("nan") for key in METRIC_KEYS}
    return {
        key: float(reducer([float(record[key]) for record in records]))
        for key in METRIC_KEYS
    }


@torch.no_grad()
def evaluate_seed(
    model: "LearnableTerritoryModel",
    *,
    seed: int,
    spec: "LearnableTerritoriesSpec",
    device: torch.device | None = None,
) -> dict[str, float | int | list[float]]:
    """Evaluate the model on a specific random seed."""
    eval_device = device
    if eval_device is None:
        try:
            eval_device = next(model.parameters()).device
        except StopIteration:
            eval_device = torch.device("cpu")

    scenario = build_scenario_from_spec(spec, device=eval_device)
    teacher = build_teacher_model_from_spec(spec, device=scenario.device)
    layout = build_layout_from_spec(spec, scenario, seed=seed)

    target = rollout_summary(
        spec, scenario, layout, teacher, name_prefix="territories_teacher_eval"
    )
    pred = rollout_summary(
        spec, scenario, layout, model, name_prefix=f"territories_{spec.model.mode}_eval"
    )

    from ..training.metrics import summary_metrics

    metrics = summary_metrics(
        pred, target, risk_loss_weight=spec.training.risk_loss_weight
    )

    return {
        "seed": int(seed),
        "total": float(metrics["total"].item()),
        "load_loss": float(metrics["load_loss"].item()),
        "risk_loss": float(metrics["risk_loss"].item()),
        "owner_agreement": float(metrics["owner_agreement"].item()),
        "conservation_error": float(metrics["conservation_error"].item()),
        "pred_sink_loads": [
            float(value) for value in pred.sink_loads.detach().cpu().tolist()
        ],
        "teacher_sink_loads": [
            float(value) for value in target.sink_loads.detach().cpu().tolist()
        ],
        "pred_sink_risks": [
            float(value) for value in pred.sink_risks.detach().cpu().tolist()
        ],
        "teacher_sink_risks": [
            float(value) for value in target.sink_risks.detach().cpu().tolist()
        ],
    }


@torch.no_grad()
def evaluate_seeds(
    model: "LearnableTerritoryModel",
    *,
    seeds: tuple[int, ...],
    spec: "LearnableTerritoriesSpec",
    device: torch.device | None = None,
) -> dict[str, object]:
    """Evaluate the model across multiple random seeds and aggregate results."""
    per_seed = [
        evaluate_seed(model, seed=seed, spec=spec, device=device) for seed in seeds
    ]
    return {
        "seeds": [int(seed) for seed in seeds],
        "mean": _metric_bundle(per_seed, mean),
        "std": _metric_bundle(per_seed, std),
        "per_seed": per_seed,
    }
