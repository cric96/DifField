"""Hold-out evaluation helpers for learnable territories experiments."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from autofield import GridScenario, SimulationEngine, SnapshotRecorder

try:
    from examples.shared.metrics import mean, std
except ImportError:
    from shared.metrics import mean, std

try:
    from .core import TerritoryLayout, TerritoryOutputs, auto_rounds, build_layout, decode_territory_output, make_territory_program
    from .models import LearnableTerritoryModel
    from .specs import LearnableTerritoriesSpec
except ImportError:
    from core import TerritoryLayout, TerritoryOutputs, auto_rounds, build_layout, decode_territory_output, make_territory_program
    from models import LearnableTerritoryModel
    from specs import LearnableTerritoriesSpec


METRIC_KEYS = (
    "total",
    "load_loss",
    "risk_loss",
    "owner_agreement",
    "conservation_error",
)


def build_layout_from_spec(
    spec: LearnableTerritoriesSpec,
    scenario: GridScenario,
    *,
    seed: int,
) -> TerritoryLayout:
    return build_layout(
        scenario,
        num_sinks=spec.program.num_sinks,
        seed=seed,
        scenario_preset=spec.program.scenario_preset,
        sink_positions=spec.program.sink_positions,
    )


def build_scenario_from_spec(
    spec: LearnableTerritoriesSpec,
    *,
    device: torch.device | None = None,
) -> GridScenario:
    return GridScenario(
        spec.grid.rows,
        spec.grid.cols,
        connectivity=spec.grid.connectivity,
        device=torch.device("cpu") if device is None else device,
        edge_weight=1.0,
    )


def build_teacher_model_from_spec(
    spec: LearnableTerritoriesSpec,
    *,
    device: torch.device | None = None,
) -> LearnableTerritoryModel:
    model = LearnableTerritoryModel(
        mode="scalars",
        init_range_weight_target=spec.teacher.range_weight,
        init_risk_weight_target=spec.teacher.risk_weight,
        init_assignment_tau_target=spec.teacher.assignment_tau,
    )
    if device is not None:
        model = model.to(device)
    return model


def rollout_summary(
    spec: LearnableTerritoriesSpec,
    scenario: GridScenario,
    layout: TerritoryLayout,
    model: LearnableTerritoryModel,
    *,
    name_prefix: str,
    recorder: SnapshotRecorder | None = None,
) -> TerritoryOutputs:
    if layout.num_sinks != spec.program.num_sinks:
        raise ValueError(
            f"Layout provides {layout.num_sinks} sinks, but spec expects {spec.program.num_sinks}",
        )
    engine = SimulationEngine.from_scenario(scenario)
    local_correction = model.local_correction(layout.node_features)
    output, _ = engine.run(
        rounds=auto_rounds(spec.grid.rows, spec.grid.cols, spec.program.rounds),
        program=make_territory_program(
            layout,
            range_weight=model.range_weight,
            risk_weight=model.risk_weight,
            assignment_tau=model.assignment_tau,
            local_correction=local_correction,
            name_prefix=name_prefix,
        ),
        signals={
            "demand": layout.demand,
            "risk": layout.risk,
        },
        recorder=recorder,
    )
    return decode_territory_output(
        output,
        layout=layout,
    )


def summary_metrics(
    pred: TerritoryOutputs,
    target: TerritoryOutputs,
    *,
    risk_loss_weight: float,
) -> dict[str, torch.Tensor]:
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


def _metric_bundle(records: list[dict[str, float]], reducer) -> dict[str, float]:
    if not records:
        return {key: float("nan") for key in METRIC_KEYS}
    return {
        key: float(reducer([float(record[key]) for record in records]))
        for key in METRIC_KEYS
    }


@torch.no_grad()
def evaluate_seed(
    model: LearnableTerritoryModel,
    *,
    seed: int,
    spec: LearnableTerritoriesSpec,
    device: torch.device | None = None,
) -> dict[str, float | int | list[float]]:
    eval_device = device
    if eval_device is None:
        try:
            eval_device = next(model.parameters()).device
        except StopIteration:
            eval_device = torch.device("cpu")

    scenario = build_scenario_from_spec(spec, device=eval_device)
    teacher = build_teacher_model_from_spec(spec, device=scenario.device)
    layout = build_layout_from_spec(spec, scenario, seed=seed)
    target = rollout_summary(spec, scenario, layout, teacher, name_prefix="territories_teacher_eval")
    pred = rollout_summary(spec, scenario, layout, model, name_prefix=f"territories_{spec.model.mode}_eval")
    metrics = summary_metrics(pred, target, risk_loss_weight=spec.training.risk_loss_weight)

    return {
        "seed": int(seed),
        "total": float(metrics["total"].item()),
        "load_loss": float(metrics["load_loss"].item()),
        "risk_loss": float(metrics["risk_loss"].item()),
        "owner_agreement": float(metrics["owner_agreement"].item()),
        "conservation_error": float(metrics["conservation_error"].item()),
        "pred_sink_loads": [float(value) for value in pred.sink_loads.detach().cpu().tolist()],
        "teacher_sink_loads": [float(value) for value in target.sink_loads.detach().cpu().tolist()],
        "pred_sink_risks": [float(value) for value in pred.sink_risks.detach().cpu().tolist()],
        "teacher_sink_risks": [float(value) for value in target.sink_risks.detach().cpu().tolist()],
    }


@torch.no_grad()
def evaluate_seeds(
    model: LearnableTerritoryModel,
    *,
    seeds: tuple[int, ...],
    spec: LearnableTerritoriesSpec,
    device: torch.device | None = None,
) -> dict[str, object]:
    per_seed = [
        evaluate_seed(model, seed=seed, spec=spec, device=device)
        for seed in seeds
    ]
    return {
        "seeds": [int(seed) for seed in seeds],
        "mean": _metric_bundle(per_seed, mean),
        "std": _metric_bundle(per_seed, std),
        "per_seed": per_seed,
    }