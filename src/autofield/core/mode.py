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


@contextmanager
def with_mode(mode: str):
    """Temporarily override the default mode inside a context block."""
    prev = get_default_mode()
    set_default_mode(mode)
    try:
        yield
    finally:
        _thread_local.default_mode = prev
