"""Utilities for accessing deeply nested dictionary values."""

from __future__ import annotations

from typing import Any


def nested_get(payload: dict[str, Any], path: str, default: Any = None) -> Any:
    """Read nested values using dot notation, e.g. 'training.final_traj_loss'."""
    current: Any = payload
    for part in path.split("."):
        if not isinstance(current, dict) or part not in current:
            return default
        current = current[part]
    return current
