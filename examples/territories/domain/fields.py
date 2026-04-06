"""Generation of demand and risk fields for territories."""

from __future__ import annotations

import torch
from .geometry import meshgrid, seeded_value


def normalize_scenario_preset(scenario_preset: str) -> str:
    """Standardize scenario preset names."""
    preset = scenario_preset.strip().lower()
    aliases = {
        "default": "balanced",
        "paper": "asymmetric_canyon",
    }
    normalized = aliases.get(preset, preset)
    if normalized not in {"balanced", "asymmetric_canyon"}:
        raise ValueError(f"Unsupported territories scenario preset: {scenario_preset}")
    return normalized


def build_demand_field(
    rows: int,
    cols: int,
    seed: int,
    device: torch.device,
    *,
    scenario_preset: str = "asymmetric_canyon",
) -> torch.Tensor:
    """Generate a demand field based on the selected scenario."""
    grid_row, grid_col = meshgrid(rows, cols, device)
    preset = normalize_scenario_preset(scenario_preset)
    if preset == "asymmetric_canyon":
        centers = (
            (
                0.17 + seeded_value(seed, 1, 0.05),
                0.24 + seeded_value(seed, 2, 0.05),
                1.05,
            ),
            (
                0.79 + seeded_value(seed, 3, 0.04),
                0.27 + seeded_value(seed, 4, 0.05),
                0.72,
            ),
            (
                0.33 + seeded_value(seed, 5, 0.05),
                0.83 + seeded_value(seed, 6, 0.05),
                1.18,
            ),
            (
                0.82 + seeded_value(seed, 7, 0.04),
                0.72 + seeded_value(seed, 8, 0.06),
                0.52,
            ),
        )
        demand = torch.full_like(grid_row, 0.02)
        for row_center, col_center, weight in centers:
            demand = demand + weight * torch.exp(
                -((grid_row - row_center) / 0.15).square()
                - ((grid_col - col_center) / 0.17).square()
            )
        canyon_flow = 0.24 * torch.exp(
            -(
                (
                    grid_col
                    - (0.47 + 0.12 * torch.sin(2.0 * torch.pi * (grid_row - 0.08)))
                )
                / 0.16
            ).square()
            - ((grid_row - 0.58) / 0.30).square()
        )
        lower_ridge = 0.18 * torch.exp(
            -((grid_row - 0.78) / 0.10).square() - ((grid_col - 0.34) / 0.20).square()
        )
        demand = demand + canyon_flow + lower_ridge
        demand = demand.reshape(-1)
        return demand / demand.sum().clamp_min(1e-6)

    centers = (
        (0.20 + seeded_value(seed, 1, 0.06), 0.70 + seeded_value(seed, 2, 0.05), 1.00),
        (0.68 + seeded_value(seed, 3, 0.06), 0.61 + seeded_value(seed, 4, 0.05), 0.85),
        (0.48 + seeded_value(seed, 5, 0.05), 0.33 + seeded_value(seed, 6, 0.06), 0.65),
    )
    demand = torch.full_like(grid_row, 0.025)
    for row_center, col_center, weight in centers:
        demand = demand + weight * torch.exp(
            -((grid_row - row_center) / 0.16).square()
            - ((grid_col - col_center) / 0.18).square()
        )
    corridor = 0.20 * torch.exp(
        -((grid_col - 0.52) / 0.28).square() - ((grid_row - 0.52) / 0.24).square()
    )
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
    """Generate a risk field based on the selected scenario."""
    grid_row, grid_col = meshgrid(rows, cols, device)
    preset = normalize_scenario_preset(scenario_preset)
    if preset == "asymmetric_canyon":
        gate_upper = 0.21 + seeded_value(seed, 11, 0.04)
        gate_middle = 0.57 + seeded_value(seed, 12, 0.05)
        curve = (
            0.50
            + 0.15 * torch.sin(2.0 * torch.pi * (grid_row - 0.08))
            + 0.04 * torch.cos(5.0 * torch.pi * grid_row)
        )
        canyon = torch.exp(-((grid_col - curve) / 0.055).square())
        gate_mask = torch.exp(-((grid_row - gate_upper) / 0.08).square()) + torch.exp(
            -((grid_row - gate_middle) / 0.10).square()
        )
        spur = 0.35 * torch.exp(
            -((grid_row - 0.74) / 0.09).square() - ((grid_col - 0.34) / 0.14).square()
        )
        basin = 0.22 * torch.exp(
            -((grid_row - 0.33) / 0.16).square() - ((grid_col - 0.77) / 0.18).square()
        )
        diagonal = 0.16 * torch.exp(-(((grid_row + 0.18) - grid_col) / 0.14).square())
        risk = (
            canyon * (1.0 - 0.88 * gate_mask.clamp(max=1.0)) + spur + basin + diagonal
        )
        risk = risk.clamp_min(0.0)
        risk = risk / risk.max().clamp_min(1e-6)
        return risk.reshape(-1)

    gap_upper = 0.26 + seeded_value(seed, 7, 0.06)
    gap_lower = 0.75 + seeded_value(seed, 8, 0.06)
    diagonal = 0.08 + seeded_value(seed, 9, 0.05)
    barrier = torch.exp(-((grid_col - 0.53) / 0.08).square())
    gap_mask = torch.exp(-((grid_row - gap_upper) / 0.10).square()) + torch.exp(
        -((grid_row - gap_lower) / 0.12).square()
    )
    diagonal_risk = torch.exp(-(((grid_row - grid_col) - diagonal) / 0.16).square())
    risk = barrier * (1.0 - 0.82 * gap_mask.clamp(max=1.0)) + 0.22 * diagonal_risk
    risk = risk.clamp_min(0.0)
    risk = risk / risk.max().clamp_min(1e-6)
    return risk.reshape(-1)
