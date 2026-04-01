"""Configuration objects for channel examples."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class GridSpec:
    rows: int
    cols: int

    @property
    def num_nodes(self) -> int:
        return self.rows * self.cols


@dataclass(frozen=True)
class ChannelProgramSpec:
    rounds: int
    tolerance: float


@dataclass(frozen=True)
class SmallChannelSpec:
    grid: GridSpec
    program: ChannelProgramSpec
    noise_scale: float
    seed: int


@dataclass(frozen=True)
class LargeChannelSpec:
    grid: GridSpec
    program: ChannelProgramSpec
