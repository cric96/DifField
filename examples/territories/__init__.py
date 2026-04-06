"""Learnable multi-sink territories example family."""

from .domain import (
    TerritoryLayout,
    TerritoryOutputs,
    auto_rounds,
    build_layout,
    decode_territory_output,
    make_territory_program,
)
from .evaluation import evaluate_seed, evaluate_seeds
from .model import LearnableTerritoryModel
from .reporting import TerritoriesSummaryBuilder, compute_parameter_recovery_metrics
from .domain.specs import (
    GridSpec,
    LearnableTerritoriesSpec,
    TerritoryEvaluationSpec,
    TerritoryModelSpec,
    TerritoryProgramSpec,
    TerritoryTeacherSpec,
    TerritoryTrainingSpec,
)

__all__ = [
    "GridSpec",
    "LearnableTerritoriesSpec",
    "LearnableTerritoryModel",
    "TerritoryEvaluationSpec",
    "TerritoryLayout",
    "TerritoryModelSpec",
    "TerritoryOutputs",
    "TerritoryProgramSpec",
    "TerritoryTeacherSpec",
    "TerritoryTrainingSpec",
    "auto_rounds",
    "build_layout",
    "compute_parameter_recovery_metrics",
    "decode_territory_output",
    "evaluate_seed",
    "evaluate_seeds",
    "make_territory_program",
    "TerritoriesSummaryBuilder",
]
