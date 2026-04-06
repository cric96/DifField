"""Aggregate core for multi-sink territory formation and load collection."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import torch

from autofield import collect_cast, gradient, gradient_cast, nbr_range


@dataclass(frozen=True)
class TerritoryLayout:
    scenario_preset: str
    sink_positions: tuple[tuple[int, int], ...]
    sink_indices: tuple[int, ...]
    sink_fields: tuple[torch.Tensor, ...]
    sink_mask: torch.Tensor
    sink_coordinate_field: torch.Tensor
    demand: torch.Tensor
    risk: torch.Tensor
    node_features: torch.Tensor

    @property
    def num_sinks(self) -> int:
        return len(self.sink_indices)


@dataclass(frozen=True)
class TerritoryOutputs:
    potential: torch.Tensor
    owner_coords: torch.Tensor
    hard_owner: torch.Tensor
    collected_load: torch.Tensor
    collected_risk: torch.Tensor
    sink_loads: torch.Tensor
    sink_risks: torch.Tensor


def auto_rounds(rows: int, cols: int, rounds: int) -> int:
    return rounds if rounds > 0 else 2 * (rows + cols)


def _seeded_value(seed: int, index: int, scale: float) -> float:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed) * 997 + index * 131)
    return float((torch.rand((), generator=generator).item() * 2.0 - 1.0) * scale)


def _meshgrid(rows: int, cols: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    row_coords = torch.linspace(0.0, 1.0, rows, device=device)
    col_coords = torch.linspace(0.0, 1.0, cols, device=device)
    grid_row, grid_col = torch.meshgrid(row_coords, col_coords, indexing="ij")
    return grid_row, grid_col


def _normalize_scenario_preset(scenario_preset: str) -> str:
    preset = scenario_preset.strip().lower()
    aliases = {
        "default": "balanced",
        "paper": "asymmetric_canyon",
    }
    normalized = aliases.get(preset, preset)
    if normalized not in {"balanced", "asymmetric_canyon"}:
        raise ValueError(f"Unsupported territories scenario preset: {scenario_preset}")
    return normalized


def _interior_index(size: int, ratio: float) -> int:
    if size <= 2:
        return max(0, min(size - 1, int(round(ratio * max(size - 1, 0)))))
    lower = 1
    upper = size - 2
    raw = int(round(ratio * upper))
    return max(lower, min(upper, raw))


def _grid_position(rows: int, cols: int, row_ratio: float, col_ratio: float) -> tuple[int, int]:
    return (
        _interior_index(rows, row_ratio),
        _interior_index(cols, col_ratio),
    )


def _candidate_positions(rows: int, cols: int) -> list[tuple[int, int]]:
    row_range = range(rows) if rows <= 2 else range(1, rows - 1)
    col_range = range(cols) if cols <= 2 else range(1, cols - 1)
    return [(row, col) for row in row_range for col in col_range]


def _squared_distance(a: tuple[int, int], b: tuple[int, int]) -> int:
    return (a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2


def _fill_with_farthest_positions(
    rows: int,
    cols: int,
    positions: list[tuple[int, int]],
    num_sinks: int,
) -> list[tuple[int, int]]:
    candidates = [candidate for candidate in _candidate_positions(rows, cols) if candidate not in positions]
    if num_sinks > len(positions) + len(candidates):
        raise ValueError(f"Cannot place {num_sinks} unique sinks on a {rows}x{cols} grid")
    while len(positions) < num_sinks:
        if positions:
            next_position = max(
                candidates,
                key=lambda candidate: min(_squared_distance(candidate, existing) for existing in positions),
            )
        else:
            center = ((rows - 1) / 2.0, (cols - 1) / 2.0)
            next_position = max(
                candidates,
                key=lambda candidate: (candidate[0] - center[0]) ** 2 + (candidate[1] - center[1]) ** 2,
            )
        positions.append(next_position)
        candidates.remove(next_position)
    return positions


def _positions_from_anchors(
    rows: int,
    cols: int,
    anchors: tuple[tuple[float, float], ...],
    num_sinks: int,
) -> tuple[tuple[int, int], ...]:
    ordered: list[tuple[int, int]] = []
    seen: set[tuple[int, int]] = set()
    for row_ratio, col_ratio in anchors:
        position = _grid_position(rows, cols, row_ratio, col_ratio)
        if position in seen:
            continue
        seen.add(position)
        ordered.append(position)
        if len(ordered) == num_sinks:
            return tuple(ordered)
    return tuple(_fill_with_farthest_positions(rows, cols, ordered, num_sinks))


def balanced_sink_positions(rows: int, cols: int, num_sinks: int) -> tuple[tuple[int, int], ...]:
    anchor_count = max(num_sinks, 8)
    anchors = tuple(
        (
            0.50 + 0.32 * torch.sin(torch.tensor(2.0 * torch.pi * idx / anchor_count + 0.35)).item(),
            0.50 + 0.38 * torch.cos(torch.tensor(2.0 * torch.pi * idx / anchor_count + 0.35)).item(),
        )
        for idx in range(anchor_count)
    )
    return _positions_from_anchors(rows, cols, anchors, num_sinks)


def asymmetric_canyon_sink_positions(rows: int, cols: int, num_sinks: int) -> tuple[tuple[int, int], ...]:
    anchors = (
        (0.16, 0.17),
        (0.74, 0.22),
        (0.28, 0.83),
        (0.86, 0.68),
        (0.55, 0.56),
        (0.11, 0.64),
        (0.62, 0.12),
        (0.40, 0.91),
        (0.82, 0.42),
        (0.48, 0.28),
    )
    return _positions_from_anchors(rows, cols, anchors, num_sinks)


def resolve_sink_positions(
    rows: int,
    cols: int,
    *,
    num_sinks: int,
    scenario_preset: str = "asymmetric_canyon",
    sink_positions: tuple[tuple[int, int], ...] | None = None,
) -> tuple[tuple[int, int], ...]:
    if sink_positions is not None:
        positions = tuple((int(row), int(col)) for row, col in sink_positions)
    else:
        preset = _normalize_scenario_preset(scenario_preset)
        if preset == "balanced":
            positions = balanced_sink_positions(rows, cols, num_sinks)
        else:
            positions = asymmetric_canyon_sink_positions(rows, cols, num_sinks)

    if len(positions) != num_sinks:
        raise ValueError(f"Resolved {len(positions)} sink positions but expected {num_sinks}")
    if len(set(positions)) != len(positions):
        raise ValueError("Sink positions must be unique")
    for row, col in positions:
        if row < 0 or row >= rows or col < 0 or col >= cols:
            raise ValueError(f"Sink position {(row, col)} is outside the {rows}x{cols} grid")
    return positions


def build_demand_field(
    rows: int,
    cols: int,
    seed: int,
    device: torch.device,
    *,
    scenario_preset: str = "asymmetric_canyon",
) -> torch.Tensor:
    grid_row, grid_col = _meshgrid(rows, cols, device)
    preset = _normalize_scenario_preset(scenario_preset)
    if preset == "asymmetric_canyon":
        centers = (
            (0.17 + _seeded_value(seed, 1, 0.05), 0.24 + _seeded_value(seed, 2, 0.05), 1.05),
            (0.79 + _seeded_value(seed, 3, 0.04), 0.27 + _seeded_value(seed, 4, 0.05), 0.72),
            (0.33 + _seeded_value(seed, 5, 0.05), 0.83 + _seeded_value(seed, 6, 0.05), 1.18),
            (0.82 + _seeded_value(seed, 7, 0.04), 0.72 + _seeded_value(seed, 8, 0.06), 0.52),
        )
        demand = torch.full_like(grid_row, 0.02)
        for row_center, col_center, weight in centers:
            demand = demand + weight * torch.exp(
                -((grid_row - row_center) / 0.15).square() - ((grid_col - col_center) / 0.17).square()
            )
        canyon_flow = 0.24 * torch.exp(
            -((grid_col - (0.47 + 0.12 * torch.sin(2.0 * torch.pi * (grid_row - 0.08)))) / 0.16).square()
            - ((grid_row - 0.58) / 0.30).square()
        )
        lower_ridge = 0.18 * torch.exp(-((grid_row - 0.78) / 0.10).square() - ((grid_col - 0.34) / 0.20).square())
        demand = demand + canyon_flow + lower_ridge
        demand = demand.reshape(-1)
        return demand / demand.sum().clamp_min(1e-6)

    centers = (
        (0.20 + _seeded_value(seed, 1, 0.06), 0.70 + _seeded_value(seed, 2, 0.05), 1.00),
        (0.68 + _seeded_value(seed, 3, 0.06), 0.61 + _seeded_value(seed, 4, 0.05), 0.85),
        (0.48 + _seeded_value(seed, 5, 0.05), 0.33 + _seeded_value(seed, 6, 0.06), 0.65),
    )
    demand = torch.full_like(grid_row, 0.025)
    for row_center, col_center, weight in centers:
        demand = demand + weight * torch.exp(
            -((grid_row - row_center) / 0.16).square() - ((grid_col - col_center) / 0.18).square()
        )
    corridor = 0.20 * torch.exp(-((grid_col - 0.52) / 0.28).square() - ((grid_row - 0.52) / 0.24).square())
    demand = demand + corridor
    demand = demand.reshape(-1)
    return demand / demand.sum().clamp_min(1e-6)


def build_risk_field(
    rows: int,
    cols: int,
    seed: int,
    device: torch.device,
    *,
    scenario_preset: str = "asymmetric_canyon",
) -> torch.Tensor:
    grid_row, grid_col = _meshgrid(rows, cols, device)
    preset = _normalize_scenario_preset(scenario_preset)
    if preset == "asymmetric_canyon":
        gate_upper = 0.21 + _seeded_value(seed, 11, 0.04)
        gate_middle = 0.57 + _seeded_value(seed, 12, 0.05)
        curve = 0.50 + 0.15 * torch.sin(2.0 * torch.pi * (grid_row - 0.08)) + 0.04 * torch.cos(5.0 * torch.pi * grid_row)
        canyon = torch.exp(-((grid_col - curve) / 0.055).square())
        gate_mask = torch.exp(-((grid_row - gate_upper) / 0.08).square()) + torch.exp(-((grid_row - gate_middle) / 0.10).square())
        spur = 0.35 * torch.exp(-((grid_row - 0.74) / 0.09).square() - ((grid_col - 0.34) / 0.14).square())
        basin = 0.22 * torch.exp(-((grid_row - 0.33) / 0.16).square() - ((grid_col - 0.77) / 0.18).square())
        diagonal = 0.16 * torch.exp(-(((grid_row + 0.18) - grid_col) / 0.14).square())
        risk = canyon * (1.0 - 0.88 * gate_mask.clamp(max=1.0)) + spur + basin + diagonal
        risk = risk.clamp_min(0.0)
        risk = risk / risk.max().clamp_min(1e-6)
        return risk.reshape(-1)

    gap_upper = 0.26 + _seeded_value(seed, 7, 0.06)
    gap_lower = 0.75 + _seeded_value(seed, 8, 0.06)
    diagonal = 0.08 + _seeded_value(seed, 9, 0.05)
    barrier = torch.exp(-((grid_col - 0.53) / 0.08).square())
    gap_mask = torch.exp(-((grid_row - gap_upper) / 0.10).square()) + torch.exp(-((grid_row - gap_lower) / 0.12).square())
    diagonal_risk = torch.exp(-(((grid_row - grid_col) - diagonal) / 0.16).square())
    risk = barrier * (1.0 - 0.82 * gap_mask.clamp(max=1.0)) + 0.22 * diagonal_risk
    risk = risk.clamp_min(0.0)
    risk = risk / risk.max().clamp_min(1e-6)
    return risk.reshape(-1)


def build_layout(
    scenario,
    *,
    num_sinks: int,
    seed: int,
    scenario_preset: str = "asymmetric_canyon",
    sink_positions: tuple[tuple[int, int], ...] | None = None,
) -> TerritoryLayout:
    resolved_preset = _normalize_scenario_preset(scenario_preset)
    sink_positions = resolve_sink_positions(
        scenario.rows,
        scenario.cols,
        num_sinks=num_sinks,
        scenario_preset=resolved_preset,
        sink_positions=sink_positions,
    )
    sink_indices = tuple(scenario.pos_to_idx(row, col) for row, col in sink_positions)
    sink_fields = tuple(scenario.marker(row, col) for row, col in sink_positions)
    sink_mask = torch.zeros(scenario.num_nodes, dtype=torch.float32, device=scenario.device)
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

    grid_row, grid_col = _meshgrid(scenario.rows, scenario.cols, scenario.device)
    node_features = torch.stack(
        (
            grid_row.reshape(-1),
            grid_col.reshape(-1),
            demand,
            risk,
        ),
        dim=-1,
    )
    sink_coordinate_field = torch.zeros((scenario.num_nodes, 2), dtype=torch.float32, device=scenario.device)
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


def territory_program(
    layout: TerritoryLayout,
    *,
    range_weight: torch.Tensor,
    risk_weight: torch.Tensor,
    assignment_tau: torch.Tensor,
    local_correction: torch.Tensor | None = None,
    name_prefix: str = "territories",
) -> torch.Tensor:
    correction = 0.0 if local_correction is None else local_correction
    step_cost = range_weight * nbr_range() + risk_weight * layout.risk + correction
    tau = assignment_tau.clamp_min(1e-3)
    potential = gradient(
        layout.sink_mask,
        weight=step_cost,
        name=f"{name_prefix}_potential",
        mode="soft",
        tau=tau,
    )
    owner_coords = gradient_cast(
        layout.sink_mask,
        layout.sink_coordinate_field,
        accumulation=lambda value: value,
        weight=step_cost,
        name=f"{name_prefix}_owner_coords",
        mode="soft",
        tau=tau,
    )
    collected_load = collect_cast(
        potential,
        layout.demand,
        0.0,
        torch.add,
        weight=step_cost,
        name=f"{name_prefix}_load",
        mode="hard",
    )
    collected_risk = collect_cast(
        potential,
        layout.demand * layout.risk,
        0.0,
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


def make_territory_program(
    layout: TerritoryLayout,
    *,
    range_weight: torch.Tensor,
    risk_weight: torch.Tensor,
    assignment_tau: torch.Tensor,
    local_correction: torch.Tensor | None = None,
    name_prefix: str = "territories",
) -> Callable:
    def program(_runtime) -> torch.Tensor:
        return territory_program(
            layout,
            range_weight=range_weight,
            risk_weight=risk_weight,
            assignment_tau=assignment_tau,
            local_correction=local_correction,
            name_prefix=name_prefix,
        )

    return program


def decode_territory_output(
    output: torch.Tensor,
    *,
    layout: TerritoryLayout,
) -> TerritoryOutputs:
    num_sinks = layout.num_sinks
    expected_width = 5
    if output.dim() != 2:
        raise ValueError(f"Expected territory output shaped [num_nodes, {expected_width}], got {tuple(output.shape)}")
    if output.shape[1] != expected_width:
        raise ValueError(
            f"Expected territory output width {expected_width} for the constant-width multi-source program, got {output.shape[1]}",
        )
    potential = output[:, 0]
    owner_coords = output[:, 1:3]
    collected_load = output[:, 3]
    collected_risk = output[:, 4]
    sink_index_tensor = torch.tensor(layout.sink_indices, device=output.device, dtype=torch.long)
    sink_coords = layout.node_features[sink_index_tensor, :2]
    assignment_temperature = output.new_tensor(0.01)
    squared_distance = (owner_coords.unsqueeze(1) - sink_coords.unsqueeze(0)).square().sum(dim=-1)
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