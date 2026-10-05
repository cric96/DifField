"""Edge-wise neighbor expressions and range primitives."""

from __future__ import annotations

from collections.abc import Callable

import torch
from torch import Tensor

from ..core import RoundContext, current_context
from .helpers import edge_sources_targets, ensure_field


class LinkField:
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
        return self._repr if self._repr is not None else "LinkField"

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
                "  Message Grid (Row=Source, Col=Target):",
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
                f"{c!s:>{col_widths[c]}}" for c in range(disp_nodes)
            )
            lines.append(f"      tgt | {header_cols}")
            lines.append("  src     |")

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
        other: float | Tensor | LinkField,
        op: Callable[[Tensor, Tensor], Tensor],
        op_str: str,
    ) -> LinkField:
        other_expr = as_scatter_expr(other)
        return LinkField(
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
        other: float | Tensor | LinkField,
        op: Callable[[Tensor, Tensor], Tensor],
        op_str: str,
    ) -> LinkField:
        other_expr = as_scatter_expr(other)
        return LinkField(
            lambda ctx, edge_index, edge_weight: op(
                other_expr.evaluate(
                    ctx=ctx, edge_index=edge_index, edge_weight=edge_weight
                ),
                self.evaluate(ctx=ctx, edge_index=edge_index, edge_weight=edge_weight),
            ),
            _repr=f"({other_expr} {op_str} {self})",
        )

    def __add__(self, other: float | Tensor | LinkField) -> LinkField:
        return self._binary(other, torch.add, "+")

    def __radd__(self, other: float | Tensor | LinkField) -> LinkField:
        return self._rbinary(other, torch.add, "+")

    def __sub__(self, other: float | Tensor | LinkField) -> LinkField:
        return self._binary(other, torch.sub, "-")

    def __rsub__(self, other: float | Tensor | LinkField) -> LinkField:
        return self._rbinary(other, torch.sub, "-")

    def __mul__(self, other: float | Tensor | LinkField) -> LinkField:
        return self._binary(other, torch.mul, "*")

    def __rmul__(self, other: float | Tensor | LinkField) -> LinkField:
        return self._rbinary(other, torch.mul, "*")

    def __truediv__(self, other: float | Tensor | LinkField) -> LinkField:
        return self._binary(other, torch.div, "/")

    def __rtruediv__(self, other: float | Tensor | LinkField) -> LinkField:
        return self._rbinary(other, torch.div, "/")

    def __neg__(self) -> LinkField:
        return LinkField(
            lambda ctx, edge_index, edge_weight: (
                -self.evaluate(
                    ctx=ctx,
                    edge_index=edge_index,
                    edge_weight=edge_weight,
                )
            ),
            _repr=f"(-{self})",
        )

    def abs(self) -> LinkField:
        return LinkField(
            lambda ctx, edge_index, edge_weight: (
                torch.abs(self.evaluate(
                    ctx=ctx,
                    edge_index=edge_index,
                    edge_weight=edge_weight,
                ))
            ),
            _repr=f"abs({self})",
        )

    def _comparison(
        self,
        other: float | Tensor | LinkField,
        op: Callable[[Tensor, Tensor], Tensor],
        op_str: str,
    ) -> LinkField:
        other_expr = as_scatter_expr(other)
        return LinkField(
            lambda ctx, edge_index, edge_weight: op(
                self.evaluate(ctx=ctx, edge_index=edge_index, edge_weight=edge_weight),
                other_expr.evaluate(
                    ctx=ctx, edge_index=edge_index, edge_weight=edge_weight
                ),
            ).float(),
            _repr=f"({self} {op_str} {other_expr})",
        )

    def __le__(self, other: float | Tensor | LinkField) -> LinkField:
        return self._comparison(other, torch.le, "<=")

    def __lt__(self, other: float | Tensor | LinkField) -> LinkField:
        return self._comparison(other, torch.lt, "<")

    def __ge__(self, other: float | Tensor | LinkField) -> LinkField:
        return self._comparison(other, torch.ge, ">=")

    def __gt__(self, other: float | Tensor | LinkField) -> LinkField:
        return self._comparison(other, torch.gt, ">")

    def _logical(
        self,
        other: float | Tensor | LinkField,
        op: Callable[[Tensor, Tensor], Tensor],
        op_str: str,
    ) -> LinkField:
        other_expr = as_scatter_expr(other)
        return LinkField(
            lambda ctx, edge_index, edge_weight: op(
                self.evaluate(ctx=ctx, edge_index=edge_index, edge_weight=edge_weight),
                other_expr.evaluate(
                    ctx=ctx, edge_index=edge_index, edge_weight=edge_weight
                ),
            ).float(),
            _repr=f"({self} {op_str} {other_expr})",
        )

    def __and__(self, other: float | Tensor | LinkField) -> LinkField:
        return self._logical(other, torch.logical_and, "&")

    def __or__(self, other: float | Tensor | LinkField) -> LinkField:
        return self._logical(other, torch.logical_or, "|")

    def pointwise(self) -> LinkField:
        """Treat this scalar field as a factor for pointwise multiplication with vectors."""
        return LinkField(
            lambda ctx, edge_index, edge_weight: self.evaluate(
                ctx=ctx, edge_index=edge_index, edge_weight=edge_weight
            ).unsqueeze(-1),
            _repr=f"pointwise({self})",
        )

    def unsqueeze(self, dim: int) -> LinkField:
        return LinkField(
            lambda ctx, edge_index, edge_weight: self.evaluate(
                ctx=ctx, edge_index=edge_index, edge_weight=edge_weight
            ).unsqueeze(dim),
            _repr=f"unsqueeze({self}, {dim})",
        )

    def map(self, fn: Callable[[Tensor], Tensor], label: str = "map") -> LinkField:
        """Apply *fn* pointwise to this expression, staying in the link domain.

        The function sees the edge-wise tensor, so a transform of a neighbour's
        value is applied per message rather than per node.
        """
        return LinkField(
            lambda ctx, edge_index, edge_weight: fn(
                self.evaluate(
                    ctx=ctx, edge_index=edge_index, edge_weight=edge_weight
                )
            ),
            _repr=f"{label}({self})",
        )

    def norm(self, dim: int = -1, keepdim: bool = False) -> LinkField:
        return LinkField(
            lambda ctx, edge_index, edge_weight: self.evaluate(
                ctx=ctx, edge_index=edge_index, edge_weight=edge_weight
            ).norm(dim=dim, keepdim=keepdim),
            _repr=f"norm({self})",
        )

    def sum(self, dim: int = -1, keepdim: bool = False) -> LinkField:
        return self.map(lambda t: t.sum(dim=dim, keepdim=keepdim), "sum")

    def relu(self) -> LinkField:
        return self.map(torch.relu, "relu")

    def exp(self) -> LinkField:
        return self.map(torch.exp, "exp")


def as_scatter_expr(value: float | Tensor | LinkField) -> LinkField:
    """Convert a scalar, tensor, or LinkField into an edge-wise LinkField.

    For scalars and tensors this creates an expression that gathers **target**
    node values (i.e. ``field_value[target_nodes]``). This is mainly used to
    support arithmetic inside neighbor expressions, such as ``scatter(x) + 1`` or
    ``scatter(x) * weight_field``. If *value* is already
    a :class:`LinkField` it is returned unchanged.

    The distinction between source and target matters: ``scatter(x)`` yields
    ``x_j`` (outgoing value) while a bare tensor ``x`` yields ``x_i`` (incoming
    value). Thus ``scatter(x) - x`` computes ``x_j - x_i`` on each edge.

    The optional ``source_field`` metadata is resolved from the active round
    context when available; outside a round context it falls back to the raw
    tensor (if any), otherwise ``None``.
    """
    if isinstance(value, LinkField):
        return value

    def evaluate(
        ctx: RoundContext, edge_index: Tensor, _edge_weight: Tensor | None
    ) -> Tensor:
        _source_nodes, target_nodes = edge_sources_targets(edge_index)
        if isinstance(value, Tensor):
            field_value = ensure_field(value, ctx)
        else:
            field_value = torch.full(
                (ctx.num_nodes,),
                float(value),
                dtype=torch.float32,
                device=ctx.edge_index.device,
            )
        return field_value[target_nodes]

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
    return LinkField(evaluate, source_field=source_field, _repr=repr_str)


def scatter(value: Tensor, *, tag: str | None = None) -> LinkField:
    """Create an edge-wise expression that gathers neighbour values.

    The returned :class:`LinkField` represents ``x_j`` for every edge
    ``(j -> i)`` and can be combined with other expressions (including
    :func:`scatter_range`) using arithmetic operators.  The actual gathering
    happens lazily when the expression is passed to :func:`gather`.

    Parameters
    ----------
    value:
        Node field tensor. Passing an existing
        :class:`LinkField` raises :exc:`TypeError`.
    tag:
        Optional identifier used by :func:`gather` to export the original
        field and support message overrides.

    Examples
    --------
    >>> gather(scatter(x) + scatter_range(), aggr="min")
    >>> gather(scatter(x) * 2, aggr="sum")
    """
    if isinstance(value, LinkField):
        raise TypeError(
            "scatter() expects a tensor node field, "
            "not a LinkField.  Use the expression directly with gather()."
        )

    ctx = current_context()
    field_value = ensure_field(value, ctx)

    def evaluator(
        _ctx: RoundContext, edge_index: Tensor, _edge_weight: Tensor | None
    ) -> Tensor:
        source_nodes, _ = edge_sources_targets(edge_index)
        return field_value[source_nodes]

    repr_str = f"scatter({tag})" if tag else "scatter(field)"
    return LinkField(evaluator, tag=tag, source_field=field_value, _repr=repr_str)


def scatter_range() -> LinkField:
    """Return the current edge metric / range as an edge-wise expression."""

    def evaluate(
        ctx: RoundContext, edge_index: Tensor, edge_weight: Tensor | None
    ) -> Tensor:
        if edge_weight is not None:
            return edge_weight
        return torch.ones(
            edge_index.shape[1], device=ctx.edge_index.device, dtype=torch.float32
        )

    return LinkField(evaluate, _repr="scatter_range()")


def link_cat(exprs: list[LinkField | Tensor | float], dim: int = -1) -> LinkField:
    """Concatenate link expressions along their feature dimension.

    Scalar (per-edge) operands are promoted to a trailing feature axis, so a
    cost and a payload can be packed into one message and reduced by a single
    :func:`~diffield.dsl.primitives.gather`.
    """
    if not exprs:
        raise ValueError("link_cat requires at least one expression")
    parts = [as_scatter_expr(expr) for expr in exprs]

    def evaluate(
        ctx: RoundContext, edge_index: Tensor, edge_weight: Tensor | None
    ) -> Tensor:
        values = [
            part.evaluate(ctx=ctx, edge_index=edge_index, edge_weight=edge_weight)
            for part in parts
        ]
        if dim == -1:
            values = [
                value.unsqueeze(-1) if value.dim() == 1 else value for value in values
            ]
        return torch.cat(values, dim=dim)

    return LinkField(
        evaluate, _repr="link_cat(" + ", ".join(str(part) for part in parts) + ")"
    )


def link_map(fn: Callable[..., Tensor], *operands: LinkField | Tensor | float) -> LinkField:
    """Apply *fn* edge-wise to evaluated operands: ``fn(op_1[e], ..., op_n[e])``.

    Lets an ordinary tensor function (e.g. a learned metric module) define a
    link field from several neighbour expressions.
    """
    parts = [as_scatter_expr(operand) for operand in operands]

    def evaluate(ctx: RoundContext, edge_index: Tensor, edge_weight: Tensor | None) -> Tensor:
        return fn(
            *(
                part.evaluate(ctx=ctx, edge_index=edge_index, edge_weight=edge_weight)
                for part in parts
            )
        )

    return LinkField(evaluate, _repr=f"{getattr(fn, '__name__', type(fn).__name__)}(...)")


def membership() -> LinkField:
    """Soft region membership of each link: 1 outside a soft ``aligned_on``."""
    ctx = current_context()
    return ctx.membership if ctx.membership is not None else as_scatter_expr(1.0)
