"""Grid construction and expected distance computation for gradients."""

from __future__ import annotations

from typing import TYPE_CHECKING

from diffield.sim import GridScenario
from diffield.utils import get_grid_distances

if TYPE_CHECKING:
    import torch


def build_corner_source_grid(
    rows: int,
    cols: int,
    *,
    connectivity: int = 4,
    device: torch.device | None = None,
) -> tuple[GridScenario, torch.Tensor, torch.Tensor]:
    """Create a grid scenario with a single source at the top-left corner."""
    scenario = GridScenario(rows, cols, connectivity=connectivity, device=device or "cpu")
    source = scenario.marker(0, 0)
    expected = get_grid_distances(
        rows, cols, src_r=0, src_c=0, connectivity=connectivity
    )
    if device is not None:
        expected = expected.to(device)
    return scenario, source, expected
