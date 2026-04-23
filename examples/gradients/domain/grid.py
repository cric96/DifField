"""Grid construction and expected distance computation for gradients."""

from __future__ import annotations

import torch

from diffield.sim import GridScenario
from diffield.utils import get_grid_distances


def build_corner_source_grid(
    rows: int,
    cols: int,
    *,
    connectivity: int = 4,
    device: torch.device | None = None,
) -> tuple[GridScenario, torch.Tensor, torch.Tensor]:
    """Create a grid scenario with a single source at the top-left corner."""
    scenario_kwargs = {"connectivity": connectivity}
    if device is not None:
        scenario_kwargs["device"] = device

    scenario = GridScenario(rows, cols, **scenario_kwargs)
    source = scenario.marker(0, 0)
    expected = get_grid_distances(
        rows, cols, src_r=0, src_c=0, connectivity=connectivity
    )
    if device is not None:
        expected = expected.to(device)
    return scenario, source, expected
