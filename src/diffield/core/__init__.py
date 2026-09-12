"""Core execution and state-management utilities for diffield."""

from .alignment import (
    AlignedDict,
    Aligner,
    AlignmentError,
    aggregate,
    current_aligner,
    get_alignment_check,
    resolve_label,
    set_alignment_check,
)
from .context import RoundContext, sub_context
from .device import DeviceContext
from .execution import AggregateContext
from .mode import get_default_mode, set_default_mode, with_mode
from .stack import (
    context_stack,
    current_context,
    pop_context,
    push_context,
    resolve_context,
    with_context,
)
from .state import StateManager

__all__ = [
    "AggregateContext",
    "AlignedDict",
    "Aligner",
    "AlignmentError",
    "DeviceContext",
    "RoundContext",
    "StateManager",
    "aggregate",
    "context_stack",
    "current_aligner",
    "current_context",
    "get_alignment_check",
    "get_default_mode",
    "pop_context",
    "push_context",
    "resolve_context",
    "resolve_label",
    "set_alignment_check",
    "set_default_mode",
    "sub_context",
    "with_context",
    "with_mode",
]
