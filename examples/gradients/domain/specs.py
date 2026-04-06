"""Configuration objects for gradient example workflows."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class GridSpec:
    rows: int
    cols: int
    connectivity: int = 4


@dataclass(frozen=True)
class TrainingSpec:
    epochs: int
    lr: float


@dataclass(frozen=True)
class LearnableGradientSpec:
    grid: GridSpec
    rounds: int
    training: TrainingSpec
    initial_weight: float


@dataclass(frozen=True)
class AttentionGradientSpec:
    grid: GridSpec
    rounds: int
    training: TrainingSpec


@dataclass(frozen=True)
class MovingGradientSpec:
    num_nodes: int
    rounds: int
    epochs: int
    radius: float
    seed: int
    lr: float
    source: int
    target: int
    learn: str
