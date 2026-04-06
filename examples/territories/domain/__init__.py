"""Territories domain layer: layout, fields, and aggregate programs."""

from .layout import TerritoryLayout, TerritoryOutputs
from .geometry import meshgrid, squared_distance, grid_position, seeded_value
from .fields import build_demand_field, build_risk_field, normalize_scenario_preset
from .sinks import resolve_sink_positions
from .factory import build_layout
from .program import (
    auto_rounds,
    territory_program,
    decode_territory_output,
    make_territory_program,
)

__all__ = [
    "TerritoryLayout",
    "TerritoryOutputs",
    "auto_rounds",
    "build_demand_field",
    "build_layout",
    "build_risk_field",
    "decode_territory_output",
    "grid_position",
    "make_territory_program",
    "meshgrid",
    "normalize_scenario_preset",
    "resolve_sink_positions",
    "seeded_value",
    "squared_distance",
    "territory_program",
]
