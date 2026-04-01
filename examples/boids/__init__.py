"""Boids example family."""

from .learnable import main as learnable_main
from .model import LearnableAggregateBoids, teacher_rollout
from .simple import main as simple_main

__all__ = [
    "LearnableAggregateBoids",
    "learnable_main",
    "simple_main",
    "teacher_rollout",
]