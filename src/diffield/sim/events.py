"""Event scheduling utilities for time-varying simulations."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass


@dataclass
class SimulationRuntime:
    """Mutable runtime object shared across events and program steps."""

    scenario: object
    signals: dict[str, object]
    metadata: dict[str, object]
    round_idx: int = 0


@dataclass(frozen=True)
class ScheduledEvent:
    """An event callback triggered at a specific round."""

    round_idx: int
    callback: Callable[[SimulationRuntime], None]
    name: str = ""


class EventSchedule:
    """Round-indexed event schedule."""

    def __init__(self, events: list[ScheduledEvent] | None = None):
        self._events_by_round: dict[int, list[ScheduledEvent]] = {}
        for event in events or []:
            self.add(event)

    def add(self, event: ScheduledEvent) -> None:
        self._events_by_round.setdefault(event.round_idx, []).append(event)

    def at(self, round_idx: int) -> list[ScheduledEvent]:
        return self._events_by_round.get(round_idx, [])

    def apply(self, round_idx: int, runtime: SimulationRuntime) -> None:
        for event in self.at(round_idx):
            event.callback(runtime)
