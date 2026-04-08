"""Edge-wise neighbor expressions and range primitives."""

from __future__ import annotations

from typing import Callable

import torch
from torch import Tensor

from ..core import RoundContext, current_context
from .helpers import edge_sources_targets, ensure_field


class NeighborExpr:
    """Edge-wise neighborhood expression evaluated before aggregation."""

    __array_priority__ = 1000

    def __init__(
        self,
        evaluator: Callable[[RoundContext, Tensor, Tensor | None], Tensor],
        *,
        tag: str | None = None,
        source_field: Tensor | None = None,
        _repr: str | None = None,
    ) -> None:
        self._evaluator = evaluator
        self.tag = tag
        self.source_field = source_field
        self._repr = _repr

    def __str__(self) -> str:
        return self._repr if self._repr is not None else "NeighborExpr"

    def __repr__(self) -> str:
        base = self.__str__()
        if self.source_field is not None:
            # We display a compressed version of the source field tensor
            sf_str = repr(self.source_field).replace("\n", "").replace("       ", " ")

            shape = list(self.source_field.shape)
            # Explain dimensions: row = node, col = feature
            shape_desc = f"nodes={shape[0]}" + (
                f", features={shape[1]}" if len(shape) > 1 else ""
            )

            return f"<{base} source(shape=[{shape_desc}])={sf_str}>"
        return f"<{base}>"

    def full_repr(self, max_nodes: int = 8) -> str:
        """Evaluate the expression in the current context and return a visual grid representation of the messages."""
        from ..core import context_stack

        if not context_stack():
            return f"<{self.__str__()} (no active context for evaluation)>"

        try:
            ctx = current_context()
            val = self.evaluate(
                ctx=ctx, edge_index=ctx.edge_index, edge_weight=ctx.edge_weight
            )
            edge_index = ctx.edge_index
            num_nodes = ctx.num_nodes

            shape = list(val.shape)
            shape_desc = f"edges={shape[0]}" + (
                f", features={shape[1]}" if len(shape) > 1 else ""
            )

            lines = [
                f"{self.__str__()} evaluated to edge tensor:",
                f"  Shape: [{shape_desc}]",
                f"  Message Grid (Row=Source, Col=Target):",
            ]

            disp_nodes = min(num_nodes, max_nodes)

            # Create a dense grid of strings
            grid = [["-" for _ in range(disp_nodes)] for _ in range(disp_nodes)]

            # Fill the grid
            for i in range(edge_index.shape[1]):
                src = edge_index[0, i].item()
                tgt = edge_index[1, i].item()

                if src < disp_nodes and tgt < disp_nodes:
                    v = val[i]
                    if v.ndim == 0:
                        s = f"{v.item():.4g}"
                    elif v.numel() <= 4:
                        # compact list
                        lst = [f"{x:.2g}" for x in v.tolist()]
                        s = f"[{', '.join(lst)}]"
                    else:
                        s = f"[..{v.shape[-1]}f..]"
                    grid[src][tgt] = s

            # Determine column widths
            col_widths = [len(str(j)) for j in range(disp_nodes)]
            for c in range(disp_nodes):
                for r in range(disp_nodes):
                    col_widths[c] = max(col_widths[c], len(grid[r][c]))

            # Format header
            header_cols = " | ".join(
                f"{str(c):>{col_widths[c]}}" for c in range(disp_nodes)
            )
            lines.append(f"      tgt | {header_cols}")
            lines.append(f"  src     |")

            sep_line = "  --------+" + "-" * (len(header_cols) + 2)
            lines.append(sep_line)

            for r in range(disp_nodes):
                row_cols = " | ".join(
                    f"{grid[r][c]:>{col_widths[c]}}" for c in range(disp_nodes)
                )
                row_str = f"    {r:<5} | {row_cols}"
                lines.append(row_str)

            if num_nodes > max_nodes:
                lines.append(
                    f"  ... and {num_nodes - max_nodes} more nodes not shown. (Showing top-left {max_nodes}x{max_nodes} corner)"
                )

            return "\n".join(lines)
        except Exception as e:
            return f"<{self.__str__()} (evaluation failed: {e})>"

    def evaluate(
        self,
        *,
        ctx: RoundContext,
        edge_index: Tensor,
        edge_weight: Tensor | None,
    ) -> Tensor:
        return self._evaluator(ctx, edge_index, edge_weight)

    def _binary(
        self,
        other: float | Tensor | "NeighborExpr",
        op: Callable[[Tensor, Tensor], Tensor],
        op_str: str,
    ) -> "NeighborExpr":
        other_expr = as_nbr_expr(other)
        return NeighborExpr(
            lambda ctx, edge_index, edge_weight: op(
                self.evaluate(ctx=ctx, edge_index=edge_index, edge_weight=edge_weight),
                other_expr.evaluate(
                    ctx=ctx, edge_index=edge_index, edge_weight=edge_weight
                ),
            ),
            _repr=f"({self} {op_str} {other_expr})",
        )

    def _rbinary(
        self,
        other: float | Tensor | "NeighborExpr",
        op: Callable[[Tensor, Tensor], Tensor],
        op_str: str,
    ) -> "NeighborExpr":
        other_expr = as_nbr_expr(other)
        return NeighborExpr(
            lambda ctx, edge_index, edge_weight: op(
                other_expr.evaluate(
                    ctx=ctx, edge_index=edge_index, edge_weight=edge_weight
                ),
                self.evaluate(ctx=ctx, edge_index=edge_index, edge_weight=edge_weight),
            ),
            _repr=f"({other_expr} {op_str} {self})",
        )

    def __add__(self, other: float | Tensor | "NeighborExpr") -> "NeighborExpr":
        return self._binary(other, torch.add, "+")

    def __radd__(self, other: float | Tensor | "NeighborExpr") -> "NeighborExpr":
        return self._rbinary(other, torch.add, "+")

    def __sub__(self, other: float | Tensor | "NeighborExpr") -> "NeighborExpr":
        return self._binary(other, torch.sub, "-")

    def __rsub__(self, other: float | Tensor | "NeighborExpr") -> "NeighborExpr":
        return self._rbinary(other, torch.sub, "-")

    def __mul__(self, other: float | Tensor | "NeighborExpr") -> "NeighborExpr":
        return self._binary(other, torch.mul, "*")

    def __rmul__(self, other: float | Tensor | "NeighborExpr") -> "NeighborExpr":
        return self._rbinary(other, torch.mul, "*")

    def __truediv__(self, other: float | Tensor | "NeighborExpr") -> "NeighborExpr":
        return self._binary(other, torch.div, "/")

    def __rtruediv__(self, other: float | Tensor | "NeighborExpr") -> "NeighborExpr":
        return self._rbinary(other, torch.div, "/")

    def __neg__(self) -> "NeighborExpr":
        return NeighborExpr(
            lambda ctx, edge_index, edge_weight: (
                -self.evaluate(
                    ctx=ctx,
                    edge_index=edge_index,
                    edge_weight=edge_weight,
                )
            ),
            _repr=f"(-{self})",
        )


def as_nbr_expr(value: float | Tensor | NeighborExpr) -> NeighborExpr:
    """Convert a scalar, tensor, or NeighborExpr into an edge-wise NeighborExpr.

    For scalars and tensors this creates an expression that gathers source
    node values (i.e. ``field_value[source_nodes]``). This is mainly used to
    support arithmetic inside neighbor expressions, such as ``nbr(x) + 1`` or
    ``nbr(x) * weight_field``. If *value* is already
    a :class:`NeighborExpr` it is returned unchanged.

    The optional ``source_field`` metadata is resolved from the active round
    context when available; outside a round context it falls back to the raw
    tensor (if any), otherwise ``None``.
    """
    if isinstance(value, NeighborExpr):
        return value

    def evaluate(
        ctx: RoundContext, edge_index: Tensor, _edge_weight: Tensor | None
    ) -> Tensor:
        source_nodes, _target_nodes = edge_sources_targets(edge_index)
        if isinstance(value, Tensor):
            field_value = ensure_field(value, ctx)
        else:
            field_value = torch.full(
                (ctx.num_nodes,),
                float(value),
                dtype=torch.float32,
                device=ctx.edge_index.device,
            )
        return field_value[source_nodes]

    # Handle source field resolution carefully to avoid crashing when not in round
    try:
        source_field = (
            ensure_field(value, current_context())
            if isinstance(value, Tensor)
            else None
        )
    except (RuntimeError, ValueError):
        source_field = value if isinstance(value, Tensor) else None

    repr_str = "field" if isinstance(value, Tensor) else str(value)
    return NeighborExpr(evaluate, source_field=source_field, _repr=repr_str)


def nbr(value: Tensor, *, tag: str | None = None) -> NeighborExpr:
    """Create an edge-wise expression that gathers neighbour values.

    The returned :class:`NeighborExpr` represents ``x_j`` for every edge
    ``(j -> i)`` and can be combined with other expressions (including
    :func:`nbr_range`) using arithmetic operators.  The actual gathering
    happens lazily when the expression is passed to :func:`hood`.

    Parameters
    ----------
    value:
        Node field tensor. Passing an existing
        :class:`NeighborExpr` raises :exc:`TypeError`.
    tag:
        Optional identifier used by :func:`hood` to export the original
        field and support message overrides.

    Examples
    --------
    >>> hood(nbr(x) + nbr_range(), aggr="min")
    >>> hood(nbr(x) * 2, aggr="sum")
    """
    if isinstance(value, NeighborExpr):
        raise TypeError(
            "nbr() expects a tensor node field, "
            "not a NeighborExpr.  Use the expression directly with hood()."
        )

    ctx = current_context()
    field_value = ensure_field(value, ctx)

    def evaluator(
        _ctx: RoundContext, edge_index: Tensor, _edge_weight: Tensor | None
    ) -> Tensor:
        source_nodes, _ = edge_sources_targets(edge_index)
        return field_value[source_nodes]

    repr_str = f"nbr({tag})" if tag else "nbr(field)"
    return NeighborExpr(evaluator, tag=tag, source_field=field_value, _repr=repr_str)


def nbr_range() -> NeighborExpr:
    """Return the current edge metric / range as an edge-wise expression."""

    def evaluate(
        ctx: RoundContext, edge_index: Tensor, edge_weight: Tensor | None
    ) -> Tensor:
        if edge_weight is not None:
            return edge_weight
        return torch.ones(
            edge_index.shape[1], device=ctx.edge_index.device, dtype=torch.float32
        )

    return NeighborExpr(evaluate, _repr="nbr_range()")
