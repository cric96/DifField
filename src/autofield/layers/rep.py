"""Recurrent layer implementing the ``rep`` primitive."""

from __future__ import annotations

import inspect
from typing import Callable

import torch.nn as nn
from torch import Tensor

from ..core import RoundContext, resolve_context
from .common import register_callable


class RepLayer(nn.Module):
    r"""Temporal recurrence: ``s_i^(t) = update_fn(s_i^(t-1), x_i, ctx)``."""

    def __init__(
        self,
        name: str,
        init_value: float | Tensor,
        update_fn: Callable[[Tensor, Tensor, RoundContext], Tensor] | nn.Module,
    ) -> None:
        super().__init__()
        self.name = name
        self.init_value = init_value
        register_callable(self, "update_fn", update_fn, "_update_fn")

    def forward(self, x: Tensor, ctx: RoundContext | None = None) -> Tensor:
        """Run one step of the recurrence."""
        ctx = resolve_context(ctx)
        state = ctx.state.get_or_init(self.name, self.init_value)
        try:
            target = self.update_fn.forward if isinstance(self.update_fn, nn.Module) else self.update_fn
            signature = inspect.signature(target)
            num_positional_params = len(
                [param for param in signature.parameters.values() if param.default is inspect.Parameter.empty]
            )
        except (ValueError, TypeError):
            num_positional_params = 3

        if num_positional_params <= 1:
            new_state = self.update_fn(state)
        elif num_positional_params == 2:
            new_state = self.update_fn(state, x)
        else:
            new_state = self.update_fn(state, x, ctx)

        ctx.state.update(self.name, new_state)
        return new_state