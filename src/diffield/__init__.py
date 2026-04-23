"""diffield — aggregate computing and differentiable field calculus."""

from .core import (
    AggregateContext,
    DeviceContext,
    RoundContext,
    StateManager,
    get_default_mode,
    set_default_mode,
    with_mode,
)
from .pyg_backend import HAS_PYG

__all__ = [
    # Core
    "AggregateContext",
    "DeviceContext",
    "RoundContext",
    "StateManager",
    # Mode configuration
    "get_default_mode",
    "set_default_mode",
    "with_mode",
    # Backend
    "HAS_PYG",
]
