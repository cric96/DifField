"""Sink placement and resolution for territories."""

from __future__ import annotations

import torch
from .geometry import grid_position, squared_distance
from .fields import normalize_scenario_preset


def _candidate_positions(rows: int, cols: int) -> list[tuple[int, int]]:
    """Get all valid interior grid positions for sink placement."""
    row_range = range(rows) if rows <= 2 else range(1, rows - 1)
    col_range = range(cols) if cols <= 2 else range(1, cols - 1)
    return [(row, col) for row in row_range for col in col_range]


def _fill_with_farthest_positions(
    rows: int,
    cols: int,
    positions: list[tuple[int, int]],
    num_sinks: int,
) -> list[tuple[int, int]]:
    """Fill the remaining sink slots using a farthest-point heuristic."""
    candidates = [
        candidate
        for candidate in _candidate_positions(rows, cols)
        if candidate not in positions
    ]
    if num_sinks > len(positions) + len(candidates):
        raise ValueError(
            f"Cannot place {num_sinks} unique sinks on a {rows}x{cols} grid"
        )
    while len(positions) < num_sinks:
        if positions:
            next_position = max(
                candidates,
                key=lambda candidate: min(
                    squared_distance(candidate, existing) for existing in positions
                ),
            )
        else:
            center = ((rows - 1) / 2.0, (cols - 1) / 2.0)
            next_position = max(
                candidates,
                key=lambda candidate: (
                    (candidate[0] - center[0]) ** 2 + (candidate[1] - center[1]) ** 2
                ),
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
    """Map normalized anchors to grid positions and fill remaining slots."""
    ordered: list[tuple[int, int]] = []
    seen: set[tuple[int, int]] = set()
    for row_ratio, col_ratio in anchors:
        position = grid_position(rows, cols, row_ratio, col_ratio)
        if position in seen:
            continue
        seen.add(position)
        ordered.append(position)
        if len(ordered) == num_sinks:
            return tuple(ordered)
    return tuple(_fill_with_farthest_positions(rows, cols, ordered, num_sinks))


def balanced_sink_positions(
    rows: int, cols: int, num_sinks: int
) -> tuple[tuple[int, int], ...]:
    """Sinks placed in a circular pattern around the center."""
    anchor_count = max(num_sinks, 8)
    anchors = tuple(
        (
            0.50
            + 0.32
            * torch.sin(
                torch.tensor(2.0 * torch.pi * idx / anchor_count + 0.35)
            ).item(),
            0.50
            + 0.38
            * torch.cos(
                torch.tensor(2.0 * torch.pi * idx / anchor_count + 0.35)
            ).item(),
        )
        for idx in range(anchor_count)
    )
    return _positions_from_anchors(rows, cols, anchors, num_sinks)


def asymmetric_canyon_sink_positions(
    rows: int, cols: int, num_sinks: int
) -> tuple[tuple[int, int], ...]:
    """Sinks placed according to a pre-defined asymmetric pattern."""
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
    """Resolve sink positions from either explicit input or scenario presets."""
    if sink_positions is not None:
        positions = tuple((int(row), int(col)) for row, col in sink_positions)
    else:
        preset = normalize_scenario_preset(scenario_preset)
        if preset == "balanced":
            positions = balanced_sink_positions(rows, cols, num_sinks)
        else:
            positions = asymmetric_canyon_sink_positions(rows, cols, num_sinks)

    if len(positions) != num_sinks:
        raise ValueError(
            f"Resolved {len(positions)} sink positions but expected {num_sinks}"
        )
    if len(set(positions)) != len(positions):
        raise ValueError("Sink positions must be unique")
    for row, col in positions:
        if row < 0 or row >= rows or col < 0 or col >= cols:
            raise ValueError(
                f"Sink position {(row, col)} is outside the {rows}x{cols} grid"
            )
    return positions
