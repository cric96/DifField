"""Tests for the learnable territories example family."""

from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

from autofield import GridScenario, SimulationEngine
from examples.territories.domain import (
    auto_rounds,
    build_layout,
    decode_territory_output,
    make_territory_program,
)
from examples.territories.evaluation import evaluate_seeds
from examples.territories.model import LearnableTerritoryModel
from examples.territories.reporting import compute_parameter_recovery_metrics
from examples.territories.domain.specs import (
    GridSpec,
    LearnableTerritoriesSpec,
    TerritoryEvaluationSpec,
    TerritoryModelSpec,
    TerritoryProgramSpec,
    TerritoryTeacherSpec,
    TerritoryTrainingSpec,
)


def _rollout_summary(
    scenario: GridScenario,
    *,
    seed: int,
    num_sinks: int,
    model: LearnableTerritoryModel,
    scenario_preset: str = "balanced",
    sink_positions: tuple[tuple[int, int], ...] | None = None,
) -> tuple[torch.Tensor, object]:
    layout = build_layout(
        scenario,
        num_sinks=num_sinks,
        seed=seed,
        scenario_preset=scenario_preset,
        sink_positions=sink_positions,
    )
    engine = SimulationEngine.from_scenario(scenario)
    output, _ = engine.run(
        rounds=auto_rounds(scenario.rows, scenario.cols, 0),
        program=make_territory_program(
            layout,
            range_weight=model.range_weight,
            risk_weight=model.risk_weight,
            assignment_tau=model.assignment_tau,
            local_correction=model.local_correction(layout.node_features),
            name_prefix="territory_test",
        ),
        signals={"demand": layout.demand, "risk": layout.risk},
    )
    decoded = decode_territory_output(output, layout=layout)
    return layout, decoded


@pytest.mark.parametrize(
    ("num_sinks", "scenario_preset"),
    [
        (2, "balanced"),
        (3, "balanced"),
        (4, "asymmetric_canyon"),
    ],
)
def test_territory_program_conserves_total_demand_at_sinks(
    num_sinks: int, scenario_preset: str
):
    scenario = GridScenario(
        7, 7, connectivity=4, device=torch.device("cpu"), edge_weight=1.0
    )
    model = LearnableTerritoryModel(
        mode="scalars",
        init_range_weight_target=1.0,
        init_risk_weight_target=1.8,
        init_assignment_tau_target=0.25,
    )

    layout, decoded = _rollout_summary(
        scenario,
        seed=3,
        num_sinks=num_sinks,
        model=model,
        scenario_preset=scenario_preset,
    )

    assert ((decoded.hard_owner >= 0) & (decoded.hard_owner < layout.num_sinks)).all()
    assert torch.equal(
        decoded.hard_owner[torch.tensor(layout.sink_indices, dtype=torch.long)],
        torch.arange(layout.num_sinks, dtype=torch.long),
    )
    assert (
        abs(float(decoded.sink_loads.sum().item()) - float(layout.demand.sum().item()))
        < 5e-3
    )
    assert decoded.sink_loads.shape[0] == layout.num_sinks
    assert torch.isfinite(decoded.sink_risks).all()


def test_final_only_summary_training_decreases_loss():
    scenario = GridScenario(
        8, 8, connectivity=4, device=torch.device("cpu"), edge_weight=1.0
    )
    teacher = LearnableTerritoryModel(
        mode="scalars",
        init_range_weight_target=1.0,
        init_risk_weight_target=1.8,
        init_assignment_tau_target=0.25,
    )
    learner = LearnableTerritoryModel(
        mode="scalars",
        init_range_weight_target=0.40,
        init_risk_weight_target=0.10,
        init_assignment_tau_target=0.95,
    )
    optimizer = torch.optim.Adam(learner.trainable_parameters(), lr=0.20)

    with torch.no_grad():
        _, teacher_summary = _rollout_summary(
            scenario,
            seed=5,
            num_sinks=4,
            model=teacher,
            scenario_preset="asymmetric_canyon",
        )

    losses = []
    for _ in range(80):
        optimizer.zero_grad()
        _, pred_summary = _rollout_summary(
            scenario,
            seed=5,
            num_sinks=4,
            model=learner,
            scenario_preset="asymmetric_canyon",
        )
        load_loss = F.mse_loss(pred_summary.sink_loads, teacher_summary.sink_loads)
        risk_loss = F.mse_loss(pred_summary.sink_risks, teacher_summary.sink_risks)
        total = load_loss + 0.5 * risk_loss
        total.backward()
        for parameter in learner.trainable_parameters():
            assert parameter.grad is not None
            assert torch.isfinite(parameter.grad).all()
        optimizer.step()
        losses.append(float(total.item()))

    assert losses[-1] < losses[0] * 0.85
    assert min(losses) < losses[0] * 0.80


def test_evaluate_seeds_returns_seed_breakdown_and_dispersion():
    spec = LearnableTerritoriesSpec(
        grid=GridSpec(rows=6, cols=6, connectivity=4),
        program=TerritoryProgramSpec(
            rounds=0, num_sinks=4, scenario_preset="asymmetric_canyon"
        ),
        teacher=TerritoryTeacherSpec(
            range_weight=1.0, risk_weight=1.8, assignment_tau=0.25
        ),
        model=TerritoryModelSpec(
            mode="scalars",
            hidden_dim=16,
            init_range_weight_target=1.0,
            init_risk_weight_target=1.8,
            init_assignment_tau_target=0.25,
        ),
        training=TerritoryTrainingSpec(
            epochs=4,
            lr=0.05,
            risk_loss_weight=0.5,
            train_seed=3,
            eval_every=1,
            print_every=1,
            clip_grad_norm=5.0,
        ),
        evaluation=TerritoryEvaluationSpec(eval_seeds=(3,)),
    )
    model = LearnableTerritoryModel(
        mode="scalars",
        init_range_weight_target=1.0,
        init_risk_weight_target=1.8,
        init_assignment_tau_target=0.25,
    )

    report = evaluate_seeds(model, seeds=(3,), spec=spec, device=torch.device("cpu"))

    assert report["per_seed"][0]["seed"] == 3
    assert len(report["per_seed"][0]["pred_sink_loads"]) == 4
    assert report["std"]["total"] == 0.0
    assert report["mean"]["conservation_error"] < 1e-4


def test_build_layout_accepts_explicit_sink_positions():
    scenario = GridScenario(
        9, 9, connectivity=4, device=torch.device("cpu"), edge_weight=1.0
    )
    explicit_positions = ((1, 1), (2, 6), (6, 2), (7, 7))

    layout = build_layout(
        scenario,
        num_sinks=4,
        seed=4,
        scenario_preset="balanced",
        sink_positions=explicit_positions,
    )

    assert layout.sink_positions == explicit_positions
    assert layout.num_sinks == 4
    assert torch.isfinite(layout.demand).all()
    assert torch.isfinite(layout.risk).all()


def test_decode_territory_output_validates_width():
    output = torch.zeros(10, 5)
    scenario = GridScenario(
        9, 9, connectivity=4, device=torch.device("cpu"), edge_weight=1.0
    )
    layout = build_layout(scenario, num_sinks=4, seed=4, scenario_preset="balanced")

    with pytest.raises(ValueError, match="width"):
        decode_territory_output(output[:, :4], layout=layout)


def test_parameter_recovery_metrics_report_relative_error():
    recovery = compute_parameter_recovery_metrics(
        {"range_weight": 1.0, "risk_weight": 2.0, "assignment_tau": 0.5},
        {"range_weight": 0.75, "risk_weight": 2.5, "assignment_tau": 0.25},
    )

    assert recovery["range_weight"]["abs_error"] == 0.25
    assert abs(recovery["risk_weight"]["rel_error"] - 0.25) < 1e-9
    assert abs(recovery["assignment_tau"]["rel_error"] - 0.5) < 1e-9
