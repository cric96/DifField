"""Factory functions for creating territory layouts and scenarios from specs."""

from __future__ import annotations

from typing import TYPE_CHECKING
import torch
from autofield import GridScenario

from .layout import TerritoryLayout
from .geometry import meshgrid
from .fields import build_demand_field, build_risk_field, normalize_scenario_preset
from .sinks import resolve_sink_positions

if TYPE_CHECKING:
    from ..domain.specs import LearnableTerritoriesSpec


def build_layout(
    scenario: GridScenario,
    *,
    num_sinks: int,
    seed: int,
    scenario_preset: str = "asymmetric_canyon",
    sink_positions: tuple[tuple[int, int], ...] | None = None,
) -> TerritoryLayout:
    """Build a complete TerritoryLayout from scenario parameters."""
    resolved_preset = normalize_scenario_preset(scenario_preset)
    sink_positions = resolve_sink_positions(
        scenario.rows,
        scenario.cols,
        num_sinks=num_sinks,
        scenario_preset=resolved_preset,
        sink_positions=sink_positions,
    )
    sink_indices = tuple(scenario.pos_to_idx(row, col) for row, col in sink_positions)
    sink_fields = tuple(scenario.marker(row, col) for row, col in sink_positions)
    sink_mask = torch.zeros(
        scenario.num_nodes, dtype=torch.float32, device=scenario.device
    )
    for sink_index in sink_indices:
        sink_mask[sink_index] = 1.0

    demand = build_demand_field(
        scenario.rows,
        scenario.cols,
        seed=seed,
        device=scenario.device,
        scenario_preset=resolved_preset,
    )
    risk = build_risk_field(
        scenario.rows,
        scenario.cols,
        seed=seed,
        device=scenario.device,
        scenario_preset=resolved_preset,
    )

    grid_row, grid_col = meshgrid(scenario.rows, scenario.cols, scenario.device)
    node_features = torch.stack(
        (
            grid_row.reshape(-1),
            grid_col.reshape(-1),
            demand,
            risk,
        ),
        dim=-1,
    )
    sink_coordinate_field = torch.zeros(
        (scenario.num_nodes, 2), dtype=torch.float32, device=scenario.device
    )
    sink_coordinate_field[list(sink_indices)] = node_features[list(sink_indices), :2]

    return TerritoryLayout(
        scenario_preset=resolved_preset,
        sink_positions=sink_positions,
        sink_indices=sink_indices,
        sink_fields=sink_fields,
        sink_mask=sink_mask,
        sink_coordinate_field=sink_coordinate_field,
        demand=demand,
        risk=risk,
        node_features=node_features,
    )


def build_layout_from_spec(
    spec: "LearnableTerritoriesSpec",
    scenario: GridScenario,
    *,
    seed: int,
) -> TerritoryLayout:
    """Build layout using parameters from a LearnableTerritoriesSpec."""
    return build_layout(
        scenario,
        num_sinks=spec.program.num_sinks,
        seed=seed,
        scenario_preset=spec.program.scenario_preset,
        sink_positions=spec.program.sink_positions,
    )


def build_scenario_from_spec(
    spec: "LearnableTerritoriesSpec",
    *,
    device: torch.device | None = None,
) -> GridScenario:
    """Build a GridScenario from a LearnableTerritoriesSpec."""
    return GridScenario(
        spec.grid.rows,
        spec.grid.cols,
        connectivity=spec.grid.connectivity,
        device=torch.device("cpu") if device is None else device,
        edge_weight=1.0,
    )


def build_teacher_model_from_spec(
    spec: "LearnableTerritoriesSpec",
    *,
    device: torch.device | None = None,
) -> "LearnableTerritoryModel":
    """Build a teacher LearnableTerritoryModel from specs."""
    from ..model.territory_model import LearnableTerritoryModel

    model = LearnableTerritoryModel(
        mode="scalars",
        init_range_weight_target=spec.teacher.range_weight,
        init_risk_weight_target=spec.teacher.risk_weight,
        init_assignment_tau_target=spec.teacher.assignment_tau,
    )
    if device is not None:
        model = model.to(device)
    return model
