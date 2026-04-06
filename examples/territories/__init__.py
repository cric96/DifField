"""Learnable multi-sink territories example family."""

from .core import (
    TerritoryLayout,
    TerritoryOutputs,
    auto_rounds,
    build_layout,
    decode_territory_output,
    make_territory_program,
)
from .evaluation_utils import evaluate_seed, evaluate_seeds
from .models import LearnableTerritoryModel
from .reporting import TerritoriesSummaryBuilder, compute_parameter_recovery_metrics
from .specs import (
    GridSpec,
    LearnableTerritoriesSpec,
    TerritoryEvaluationSpec,
    TerritoryModelSpec,
    TerritoryProgramSpec,
    TerritoryTeacherSpec,
    TerritoryTrainingSpec,
)
from .workflow import LearnableTerritoriesWorkflow

__all__ = [
    "GridSpec",
    "LearnableTerritoriesSpec",
    "LearnableTerritoriesWorkflow",
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