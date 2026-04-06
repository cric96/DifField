"""Configuration objects for learnable territories experiments."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class GridSpec:
    rows: int
    cols: int
    connectivity: int = 4

    @property
    def num_nodes(self) -> int:
        return self.rows * self.cols


@dataclass(frozen=True)
class TerritoryProgramSpec:
    rounds: int
    num_sinks: int
    scenario_preset: str = "asymmetric_canyon"
    sink_positions: tuple[tuple[int, int], ...] | None = None

    def __post_init__(self) -> None:
        if self.rounds < 0:
            raise ValueError("rounds must be non-negative")
        if self.num_sinks < 1:
            raise ValueError("num_sinks must be positive")
        if not self.scenario_preset.strip():
            raise ValueError("scenario_preset must be non-empty")
        if self.sink_positions is not None and len(self.sink_positions) != self.num_sinks:
            raise ValueError("num_sinks must match the number of explicit sink positions")


@dataclass(frozen=True)
class TerritoryTeacherSpec:
    range_weight: float
    risk_weight: float
    assignment_tau: float


@dataclass(frozen=True)
class TerritoryModelSpec:
    mode: str
    hidden_dim: int
    init_range_weight_target: float
    init_risk_weight_target: float
    init_assignment_tau_target: float
    init_surcharge_weight_target: float = 0.10


@dataclass(frozen=True)
class TerritoryTrainingSpec:
    epochs: int
    lr: float
    risk_loss_weight: float
    train_seed: int
    eval_every: int
    print_every: int
    clip_grad_norm: float


@dataclass(frozen=True)
class TerritoryEvaluationSpec:
    eval_seeds: tuple[int, ...]


@dataclass(frozen=True)
class LearnableTerritoriesSpec:
    grid: GridSpec
    program: TerritoryProgramSpec
    teacher: TerritoryTeacherSpec
    model: TerritoryModelSpec
    training: TerritoryTrainingSpec
    evaluation: TerritoryEvaluationSpec