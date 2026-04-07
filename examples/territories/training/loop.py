"""Orchestration of the territories training loop."""

from __future__ import annotations

from typing import TYPE_CHECKING
import torch

from ..domain.factory import (
    build_layout,
    build_scenario_from_spec,
    build_teacher_model_from_spec,
)
from ..domain.program import rollout_summary
from ..model.territory_model import LearnableTerritoryModel

try:
    from ..shared import MetricHistory, grad_norm
except ImportError:
    from shared import MetricHistory, grad_norm
from .metrics import summary_metrics
from ..evaluation.evaluator import evaluate_seeds

if TYPE_CHECKING:
    from ..domain.specs import LearnableTerritoriesSpec


class TerritoriesTrainingLoop:
    """Manages the optimization process for learnable territory parameters."""

    def __init__(self, spec: "LearnableTerritoriesSpec"):
        self.spec = spec

    def run(
        self, on_epoch_end=None, *, save_eval_checkpoints: bool = False
    ) -> tuple[LearnableTerritoryModel, dict[str, list[float]], list[dict]]:
        """Execute the full training process."""
        spec = self.spec
        scenario = build_scenario_from_spec(spec)
        train_layout = build_layout(
            scenario,
            num_sinks=spec.program.num_sinks,
            seed=spec.training.train_seed,
            scenario_preset=spec.program.scenario_preset,
            sink_positions=spec.program.sink_positions,
        )
        teacher = build_teacher_model_from_spec(spec).to(scenario.device)
        model = LearnableTerritoryModel.from_spec(spec.model).to(scenario.device)
        optimizer = torch.optim.Adam(model.trainable_parameters(), lr=spec.training.lr)

        history = MetricHistory.from_keys(
            [
                "epoch",
                "total",
                "load_loss",
                "risk_loss",
                "owner_agreement",
                "conservation_error",
                "grad_norm",
                "range_weight",
                "risk_weight",
                "assignment_tau",
                "surcharge_weight",
                "val_total",
                "val_load_loss",
                "val_risk_loss",
                "val_owner_agreement",
                "val_conservation_error",
            ]
        )

        # Pre-compute teacher target
        with torch.no_grad():
            target_train = rollout_summary(
                spec,
                scenario,
                train_layout,
                teacher,
                name_prefix="territories_teacher_train",
            )

        best_score = float("inf")
        best_state = {
            key: value.detach().clone() for key, value in model.state_dict().items()
        }
        eval_checkpoints: list[dict] = []

        for epoch in range(spec.training.epochs):
            optimizer.zero_grad()
            pred_train = rollout_summary(
                spec,
                scenario,
                train_layout,
                model,
                name_prefix=f"territories_{spec.model.mode}_train",
            )
            train_metrics = summary_metrics(
                pred_train,
                target_train,
                risk_loss_weight=spec.training.risk_loss_weight,
            )

            train_metrics["total"].backward()
            trainable_params = model.trainable_parameters()
            epoch_grad_norm = grad_norm(trainable_params)
            torch.nn.utils.clip_grad_norm_(
                trainable_params, max_norm=spec.training.clip_grad_norm
            )
            optimizer.step()

            # Evaluation
            if (
                epoch + 1
            ) % spec.training.eval_every == 0 or epoch + 1 == spec.training.epochs:
                eval_report = evaluate_seeds(
                    model,
                    seeds=spec.evaluation.eval_seeds,
                    spec=spec,
                    device=scenario.device,
                )
                eval_metrics = eval_report["mean"]
            else:
                eval_metrics = {
                    k: float("nan")
                    for k in [
                        "total",
                        "load_loss",
                        "risk_loss",
                        "owner_agreement",
                        "conservation_error",
                    ]
                }

            score = float(eval_metrics["total"])
            if not torch.isfinite(torch.tensor(score)):
                score = float(train_metrics["total"].detach().item())

            if score < best_score:
                best_score = score
                best_state = {
                    key: value.detach().clone()
                    for key, value in model.state_dict().items()
                }

            # Record
            params = model.current_parameters()
            history.append(
                epoch=float(epoch + 1),
                total=float(train_metrics["total"].item()),
                load_loss=float(train_metrics["load_loss"].item()),
                risk_loss=float(train_metrics["risk_loss"].item()),
                owner_agreement=float(train_metrics["owner_agreement"].item()),
                conservation_error=float(train_metrics["conservation_error"].item()),
                grad_norm=epoch_grad_norm,
                range_weight=params["range_weight"],
                risk_weight=params["risk_weight"],
                assignment_tau=params["assignment_tau"],
                surcharge_weight=params["surcharge_weight"],
                val_total=float(eval_metrics["total"]),
                val_load_loss=float(eval_metrics["load_loss"]),
                val_risk_loss=float(eval_metrics["risk_loss"]),
                val_owner_agreement=float(eval_metrics["owner_agreement"]),
                val_conservation_error=float(eval_metrics["conservation_error"]),
            )

            if on_epoch_end:
                on_epoch_end(epoch, model, pred_train, eval_metrics)

            if save_eval_checkpoints and (
                (epoch + 1) % spec.training.eval_every == 0 or epoch + 1 == spec.training.epochs
            ):
                eval_checkpoints.append({
                    "epoch": epoch + 1,
                    "state_dict": {k: v.detach().clone() for k, v in model.state_dict().items()},
                })

        model.load_state_dict(best_state)
        return model, history.to_dict(), eval_checkpoints
