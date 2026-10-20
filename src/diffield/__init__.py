"""diffield — aggregate computing and differentiable field calculus.

The top level holds the execution context and the mode; the primitives live in
:mod:`diffield.dsl`, the simulator in :mod:`diffield.sim`, the device-by-device
runtime in :mod:`diffield.decentralized`.
"""

from .core import (
    AggregateContext,
    DeviceContext,
    RoundContext,
    StateManager,
    aggregate,
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
    "aggregate",
    "get_default_mode",
    "set_default_mode",
    "with_mode",
]
