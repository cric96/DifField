"""Boids example family.

Keep package exports lazy so importing submodules like `boids.runner`
does not eagerly import the whole example stack.
"""

from __future__ import annotations

from typing import Any

__all__ = [
    "LearnableAggregateBoids",
    "learnable_main",
    "simple_main",
    "teacher_rollout",
]


def __getattr__(name: str) -> Any:
    if name == "LearnableAggregateBoids":
        from .model.boids_model import LearnableAggregateBoids

        return LearnableAggregateBoids
    if name == "teacher_rollout":
        from .domain.teacher import teacher_rollout

        return teacher_rollout
    if name == "learnable_main":
        from .learnable import main

        return main
    if name == "simple_main":
        from .simple import main

        return main
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(__all__)
