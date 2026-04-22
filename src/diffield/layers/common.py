"""Shared helpers for nn.Module-based aggregate layers."""

from __future__ import annotations

from typing import Callable

import torch.nn as nn


def register_callable(
    module: nn.Module,
    attr_name: str,
    fn: Callable | nn.Module,
    child_name: str,
) -> None:
    """Assign *fn* as ``module.<attr_name>`` and register it if needed."""
    setattr(module, attr_name, fn)
    if isinstance(fn, nn.Module):
        module.add_module(child_name, fn)