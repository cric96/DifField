"""Recording helpers for simulation rounds and internal fields."""

from __future__ import annotations

from dataclasses import dataclass, field
import time

import torch


@dataclass
class SnapshotRecorder:
    """Capture selected states/exports at configured rounds."""

    state_fields: list[str] | None = None
    export_fields: list[str] | None = None
    capture_output: bool = True
    record_rounds: set[int] | None = None
    records: dict[int, dict[str, torch.Tensor]] = field(default_factory=dict)
    timings: dict[int, float] = field(default_factory=dict)

    def should_record(self, round_idx: int) -> bool:
        if self.record_rounds is None:
            return True
        return round_idx in self.record_rounds

    def record(
        self,
        *,
        round_idx: int,
        round_ctx,
        output: torch.Tensor,
        started_at: float,
    ) -> None:
        if not self.should_record(round_idx):
            return

        payload: dict[str, torch.Tensor] = {}
        states = round_ctx.state._states
        exports = round_ctx.exports

        if self.state_fields is None:
            for key, value in states.items():
                payload[key] = value.detach().clone()
        else:
            for key in self.state_fields:
                if key in states:
                    payload[key] = states[key].detach().clone()

        if self.export_fields:
            for key in self.export_fields:
                if key in exports:
                    payload[key] = exports[key].detach().clone()

        if self.capture_output:
            payload["output"] = output.detach().clone()

        self.records[round_idx] = payload
        self.timings[round_idx] = time.time() - started_at

    def get(self, round_idx: int, name: str) -> torch.Tensor:
        return self.records[round_idx][name]
