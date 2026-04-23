"""History collection primitives for example training loops."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class MetricHistory:
    """Typed wrapper over dict-of-lists metric histories.

    The class centralizes key validation and append semantics so callers do not
    spread low-level dictionary manipulation across training code.
    """

    _data: dict[str, list[float]]

    @classmethod
    def from_keys(cls, keys: list[str]) -> MetricHistory:
        """Initialize an empty history with a set of known keys."""
        unique_keys = list(dict.fromkeys(keys))
        return cls({key: [] for key in unique_keys})

    def append(self, **metrics: float) -> None:
        """Add a new round of metrics to the history, validating keys."""
        unknown_keys = [key for key in metrics if key not in self._data]
        if unknown_keys:
            raise KeyError(f"Unknown history keys: {unknown_keys}")

        missing_keys = [key for key in self._data if key not in metrics]
        if missing_keys:
            raise KeyError(f"Missing history keys: {missing_keys}")

        for key, value in metrics.items():
            self._data[key].append(float(value))

    def to_dict(self) -> dict[str, list[float]]:
        """Return a copy of the history data as a dictionary."""
        return {key: values[:] for key, values in self._data.items()}
