"""Boids runner layer: subprocess orchestration for experiments."""

from .subprocess import (
    LearnableRunOptions,
    RunOutcome,
    build_learnable_command,
    run_learnable_subprocess,
)

__all__ = [
    "LearnableRunOptions",
    "RunOutcome",
    "build_learnable_command",
    "run_learnable_subprocess",
]
