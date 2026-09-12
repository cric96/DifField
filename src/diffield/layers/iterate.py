"""Recurrent layer implementing the ``iterate`` primitive."""

from __future__ import annotations

import inspect
from collections.abc import Callable

from torch import Tensor, nn

from ..core import RoundContext, resolve_context
from .common import register_callable


class IterateLayer(nn.Module):
    r"""Temporal recurrence: ``s_i^(t) = update_fn(s_i^(t-1), x_i, ctx)``.

    ``name`` is an optional alignment label; identity otherwise comes from the
    occurrence's position in the evaluation tree.
    """

    def __init__(
        self,
        init_value: Tensor,
        update_fn: Callable[[Tensor, Tensor, RoundContext], Tensor] | nn.Module,
        *,
        name: str | None = None,
    ) -> None:
        super().__init__()
        self.name = name
        self.init_value = init_value
        register_callable(self, "update_fn", update_fn, "_update_fn")

        try:
            target = (
                self.update_fn.forward
                if isinstance(self.update_fn, nn.Module)
                else self.update_fn
            )
            signature = inspect.signature(target)
            self._num_positional_params = len(
                [
                    param
                    for param in signature.parameters.values()
                    if param.default is inspect.Parameter.empty
                ]
            )
        except (ValueError, TypeError):
            self._num_positional_params = 3

    def forward(self, x: Tensor, ctx: RoundContext | None = None) -> Tensor:
        """Run one step of the recurrence.

        The occurrence opens its own alignment frame, so constructs inside the
        update body nest below it rather than becoming its siblings.
        """
        ctx = resolve_context(ctx)
        with ctx.align.scope("it", self.name) as key:
            state = ctx.state.get_or_init(self.init_value, name=key)

            if self._num_positional_params <= 1:
                new_state = self.update_fn(state)
            elif self._num_positional_params == 2:
                new_state = self.update_fn(state, x)
            else:
                new_state = self.update_fn(state, x, ctx)

        ctx.state.update(new_state, name=key)
        return new_state
