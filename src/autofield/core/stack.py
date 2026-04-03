"""Thread-local execution stack for aggregate contexts."""

from __future__ import annotations

import threading
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .context import RoundContext


_thread_local = threading.local()


def context_stack() -> list[RoundContext]:
    if not hasattr(_thread_local, "ctx_stack"):
        _thread_local.ctx_stack = []
    return _thread_local.ctx_stack


def push_context(ctx: RoundContext) -> None:
    context_stack().append(ctx)


def pop_context() -> RoundContext:
    stack = context_stack()
    if not stack:
        raise RuntimeError("No active AggregateContext. Use `with ctx.round(): ...`")
    return stack.pop()


def current_context() -> RoundContext:
    stack = context_stack()
    if not stack:
        raise RuntimeError("No active AggregateContext. Use `with ctx.round(): ...`")
    return stack[-1]


def resolve_context(ctx: RoundContext | None) -> RoundContext:
    return ctx if ctx is not None else current_context()