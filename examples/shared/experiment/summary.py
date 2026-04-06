"""Logic for flattening nested experiment summaries for CSV export."""

from __future__ import annotations

import json


def flatten_summary_for_csv(
    payload: dict[str, object], prefix: str = ""
) -> dict[str, float | int | str | bool]:
    """Flatten nested summary payloads into scalar CSV columns."""
    flat: dict[str, float | int | str | bool] = {}
    for key, value in payload.items():
        full_key = f"{prefix}.{key}" if prefix else key
        if isinstance(value, dict):
            flat.update(flatten_summary_for_csv(value, full_key))
        elif isinstance(value, (str, bool, int, float)):
            flat[full_key] = value
        else:
            flat[full_key] = json.dumps(value, sort_keys=True)
    return flat
