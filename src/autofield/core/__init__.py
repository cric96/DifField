"""Core execution and state-management utilities for autofield."""

from .context import RoundContext, sub_context
from .device import DeviceContext
from .execution import AggregateContext
from .stack import context_stack, current_context, pop_context, push_context, resolve_context
from .state import StateManager

__all__ = [
    "AggregateContext",
    "DeviceContext",
    "RoundContext",
    "StateManager",
    "context_stack",
    "current_context",
    "push_context",
    "pop_context",
    "resolve_context",
    "sub_context",
]