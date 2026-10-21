"""Generic training utilities shared across examples."""

from __future__ import annotations


def parse_int_csv(seed_csv: str) -> list[int]:
    """Parse a comma-separated string of integers into a list."""
    text = seed_csv.strip()
    return (
        []
        if not text
        else [int(item.strip()) for item in text.split(",") if item.strip()]
    )

