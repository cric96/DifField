"""Thread-local default mode configuration."""

from __future__ import annotations

import threading
from contextlib import contextmanager

_thread_local = threading.local()


def get_default_mode() -> str:
    """Return the current default mode (``"hard"`` or ``"soft"``)."""
    return getattr(_thread_local, "default_mode", "hard")


def set_default_mode(mode: str) -> None:
    """Set the default mode for the current thread."""
    if mode not in ("hard", "soft"):
        raise ValueError(f"mode must be 'hard' or 'soft', got {mode!r}")
    _thread_local.default_mode = mode


def get_default_tau(fallback: float) -> float:
    """Return the temperature set by :func:`with_mode`, or *fallback*.

    Used by aggregation-style relaxations (lower is sharper); conditional
    selections (``mux``, ``branch``) keep their own opposite convention.
    """
    tau = getattr(_thread_local, "default_tau", None)
    return fallback if tau is None else tau


@contextmanager
def with_mode(mode: str, tau: float | None = None):
    """Temporarily override the default mode (and relaxation temperature)."""
    prev_mode = get_default_mode()
    prev_tau = getattr(_thread_local, "default_tau", None)
    set_default_mode(mode)
    _thread_local.default_tau = tau
    try:
        yield
    finally:
        _thread_local.default_mode = prev_mode
        _thread_local.default_tau = prev_tau
