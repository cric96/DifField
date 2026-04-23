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
    "HAS_PYG",
    "AggregateContext",
    "DeviceContext",
    "RoundContext",
    "StateManager",
    "get_default_mode",
    "set_default_mode",
    "with_mode",
]
