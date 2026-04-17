"""Aggregate programs for territory formation and load collection."""

from __future__ import annotations

from typing import TYPE_CHECKING
import torch
from autofield import collect_cast, gradient, gradient_cast, scatter_range, SimulationEngine
from autofield.dsl import field
from .layout import TerritoryLayout, TerritoryOutputs

if TYPE_CHECKING:
    from autofield import GridScenario, SnapshotRecorder
    from ..domain.specs import LearnableTerritoriesSpec
    from ..model.territory_model import LearnableTerritoryModel


def auto_rounds(rows: int, cols: int, rounds: int) -> int:
    """Determine the number of computation rounds automatically if set to 0."""
    return rounds if rounds > 0 else 2 * (rows + cols)


def territory_program(
    layout: TerritoryLayout,
    *,
    range_weight: torch.Tensor,
    risk_weight: torch.Tensor,
    assignment_tau: torch.Tensor,
    local_correction: torch.Tensor | None = None,
    name_prefix: str = "territories",
) -> torch.Tensor:
    """Core aggregate program for territory potential and load collection."""
    correction = field.zeros() if local_correction is None else local_correction
    step_cost = range_weight * scatter_range() + risk_weight * layout.risk + correction
    tau = assignment_tau.clamp_min(1e-3)

    # 1. Distance potential to sinks
    potential = gradient(
        layout.sink_mask,
        weight=step_cost,
        name=f"{name_prefix}_potential",
        mode="soft",
        tau=tau,
    )

    # 2. Sink coordinate propagation (ownership)
    owner_coords = gradient_cast(
        layout.sink_mask,
        layout.sink_coordinate_field,
        accumulation=lambda value: value,
        weight=step_cost,
        name=f"{name_prefix}_owner_coords",
        mode="soft",
        tau=tau,
    )

    # 3. Load collection
    collected_load = collect_cast(
        potential,
        layout.demand,
        field.zeros(),
        torch.add,
        weight=step_cost,
        name=f"{name_prefix}_load",
        mode="hard",
    )

    # 4. Risk collection
    collected_risk = collect_cast(
        potential,
        layout.demand * layout.risk,
        field.zeros(),
        torch.add,
        weight=step_cost,
        name=f"{name_prefix}_risk",
        mode="hard",
    )

    return torch.cat(
        (
            potential.unsqueeze(-1),
            owner_coords,
            collected_load.unsqueeze(-1),
            collected_risk.unsqueeze(-1),
        ),
        dim=-1,
    )


def decode_territory_output(
    output: torch.Tensor,
    *,
    layout: TerritoryLayout,
) -> TerritoryOutputs:
    """Decode the raw tensor output of the territory program into typed outputs."""
    expected_width = 5
    if output.dim() != 2:
        raise ValueError(
            f"Expected territory output shaped [num_nodes, {expected_width}], got {tuple(output.shape)}"
        )
    if output.shape[1] != expected_width:
        raise ValueError(
            f"Expected territory output width {expected_width}, got {output.shape[1]}",
        )

    potential = output[:, 0]
    owner_coords = output[:, 1:3]
    collected_load = output[:, 3]
    collected_risk = output[:, 4]

    sink_index_tensor = torch.tensor(
        layout.sink_indices, device=output.device, dtype=torch.long
    )
    sink_coords = layout.node_features[sink_index_tensor, :2]

    assignment_temperature = output.new_tensor(0.01)
    squared_distance = (
        (owner_coords.unsqueeze(1) - sink_coords.unsqueeze(0)).square().sum(dim=-1)
    )
    assignment = torch.softmax(-squared_distance / assignment_temperature, dim=-1)

    hard_owner = assignment.argmax(dim=-1).to(dtype=torch.long)
    sink_loads = (layout.demand.unsqueeze(-1) * assignment).sum(dim=0)
    sink_risks = ((layout.demand * layout.risk).unsqueeze(-1) * assignment).sum(dim=0)

    return TerritoryOutputs(
        potential=potential,
        owner_coords=owner_coords,
        hard_owner=hard_owner,
        collected_load=collected_load,
        collected_risk=collected_risk,
        sink_loads=sink_loads,
        sink_risks=sink_risks,
    )


def make_territory_program(
    layout: TerritoryLayout,
    *,
    range_weight: torch.Tensor,
    risk_weight: torch.Tensor,
    assignment_tau: torch.Tensor,
    local_correction: torch.Tensor | None = None,
    name_prefix: str = "territories",
) -> Callable:
    """Factory creating a territory program compatible with SimulationEngine."""

    def program(_runtime):
        return territory_program(
            layout,
            range_weight=range_weight,
            risk_weight=risk_weight,
            assignment_tau=assignment_tau,
            local_correction=local_correction,
            name_prefix=name_prefix,
        )

    return program


def rollout_summary(
    spec: "LearnableTerritoriesSpec",
    scenario: "GridScenario",
    layout: TerritoryLayout,
    model: "LearnableTerritoryModel",
    *,
    name_prefix: str,
    recorder: "SnapshotRecorder" | None = None,
) -> TerritoryOutputs:
    """Run a full territory program rollout and return summary results."""
    if layout.num_sinks != spec.program.num_sinks:
        raise ValueError(
            f"Layout provides {layout.num_sinks} sinks, but spec expects {spec.program.num_sinks}",
        )
    engine = SimulationEngine.from_scenario(scenario)
    local_correction = model.local_correction(layout.node_features)

    def program(_runtime):
        return territory_program(
            layout,
            range_weight=model.range_weight,
            risk_weight=model.risk_weight,
            assignment_tau=model.assignment_tau,
            local_correction=local_correction,
            name_prefix=name_prefix,
        )

    output, _ = engine.run(
        rounds=auto_rounds(spec.grid.rows, spec.grid.cols, spec.program.rounds),
        program=program,
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
