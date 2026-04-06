"""Utilities for saving experiment results to CSV files."""

from __future__ import annotations

import csv
from pathlib import Path


def save_history_csv(history: dict[str, list[float]], output_path: str | Path) -> None:
    """Save training history (epoch-wise metrics) to a CSV file."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    keys = list(history.keys())
    length = len(history[keys[0]]) if keys else 0

    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        for index in range(length):
            row = {key: history[key][index] for key in keys}
            writer.writerow(row)


def save_summary_csv(
    summary: dict[str, float | int | str | bool], output_path: str | Path
) -> None:
    """Save a single summary record to a CSV file."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary.keys()))
        writer.writeheader()
        writer.writerow(summary)
