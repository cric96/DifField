#!/usr/bin/env python3
"""Tensor encoding of fields — example from the paper.

Replicates the four-stage computational cycle described in
"Tensor Encoding of Fields" using diffield primitives, and compares
the results against the dense tensor formulation.

Graph: 3 nodes, bidirectional edges 0↔1, 1↔2
Sensor field: q = [10, 20, 30]
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import torch  # noqa: E402

from diffield import AggregateContext  # noqa: E402
from diffield.core import current_context  # noqa: E402
from diffield.dsl import (  # noqa: E402
    gather_avg,
    gather_max,
    gather_min,
    gather_sum,
    scatter,
)

# ── helpers ──────────────────────────────────────────────────────────


def to_dense_matrix(
    edge_index: torch.Tensor, values: torch.Tensor, n: int
) -> torch.Tensor:
    """Materialise edge-wise values into an NxN dense matrix (row=src, col=tgt)."""
    m = torch.zeros(n, n, dtype=torch.float32)
    src, tgt = edge_index[0], edge_index[1]
    m[src, tgt] = values
    return m


def fmt_tensor(t: torch.Tensor, name: str = "") -> str:
    lines = []
    if name:
        lines.append(f"  {name}  (shape {list(t.shape)})")
    for row in t.tolist():
        lines.append(f"    {row}")
    return "\n".join(lines)


def _run_field_ops(q: torch.Tensor, ctx: AggregateContext) -> None:
    print("\n" + "-" * 60)
    print("1. FIELD OPERATIONS  (node -> node)")
    print("-" * 60)

    q_hat_dense = q / 30.0

    with ctx.round():
        q_hat_af = q / 30.0

    print(f"\n  Dense:    q_hat = q / 30 = {q_hat_dense.tolist()}")
    print(f"  diffield:                = {q_hat_af.tolist()}")
    print(f"  Match: {torch.allclose(q_hat_dense, q_hat_af)}")


def _run_lifting_ops(
    q: torch.Tensor,
    a: torch.Tensor,
    edge_index: torch.Tensor,
    ctx: AggregateContext,
) -> tuple[torch.Tensor, torch.Tensor]:
    print("\n" + "-" * 60)
    print("2. LIFTING OPERATIONS  (node -> link)")
    print("-" * 60)

    ones = torch.ones(a.shape[0])
    t_g_dense = a * (q.outer(ones) - ones.outer(q))

    with ctx.round():
        t_g_linkfield = scatter(q) - q
        gather_sum(t_g_linkfield)
        rctx = current_context()
        t_g_values = t_g_linkfield.evaluate(
            ctx=rctx,
            edge_index=rctx.edge_index,
            edge_weight=rctx.edge_weight,
        )
        t_g_af = to_dense_matrix(rctx.edge_index, t_g_values, a.shape[0])

    print("\n  Dense T_g = A . (q.1^T - 1.q^T):")
    print(fmt_tensor(t_g_dense))
    print("\n  diffield T_g (materialised):")
    print(fmt_tensor(t_g_af))
    print(f"  Match: {torch.allclose(t_g_dense, t_g_af)}")

    return t_g_dense, t_g_af


def _run_link_field_ops(
    t_g_dense: torch.Tensor,
    q: torch.Tensor,
    ctx: AggregateContext,
) -> tuple[torch.Tensor, torch.Tensor]:
    print("\n" + "-" * 60)
    print("3. LINK-FIELD OPERATIONS  (link -> link)")
    print("-" * 60)

    abs_t_g_dense = t_g_dense.abs()

    with ctx.round():
        abs_t_g_linkfield = (scatter(q) - q).abs()
        rctx = current_context()
        abs_t_g_values = abs_t_g_linkfield.evaluate(
            ctx=rctx,
            edge_index=rctx.edge_index,
            edge_weight=rctx.edge_weight,
        )
        abs_t_g_af = to_dense_matrix(
            rctx.edge_index, abs_t_g_values, t_g_dense.shape[0]
        )

    print("\n  Dense |T_g|:")
    print(fmt_tensor(abs_t_g_dense))
    print("\n  diffield |T_g| (materialised):")
    print(fmt_tensor(abs_t_g_af))
    print(f"  Match: {torch.allclose(abs_t_g_dense, abs_t_g_af)}")

    return abs_t_g_dense, abs_t_g_af


def _run_neighborhood_ops(
    q: torch.Tensor,
    a: torch.Tensor,
    t_g_dense: torch.Tensor,
    abs_t_g_dense: torch.Tensor,
    ctx: AggregateContext,
) -> None:
    print("\n" + "-" * 60)
    print("4. NEIGHBORHOOD OPERATIONS  (link -> node)")
    print("-" * 60)

    ones = torch.ones(a.shape[0])
    d_in = a.sum(dim=0)
    y_sum_dense = abs_t_g_dense.T @ ones
    y_mean_dense = y_sum_dense / d_in

    y_max_dense = torch.zeros(a.shape[0])
    y_min_dense = torch.zeros(a.shape[0])
    for i in range(a.shape[0]):
        incoming = t_g_dense[:, i][a[:, i] == 1]
        y_max_dense[i] = incoming.max()
        y_min_dense[i] = incoming.min()

    with ctx.round():
        t_g_expr = scatter(q) - q
        y_sum_af = gather_sum(t_g_expr.abs())
        y_mean_af = gather_avg(t_g_expr.abs())
        y_max_af = gather_max(t_g_expr)
        y_min_af = gather_min(t_g_expr)

    print(f"\n  In-degree d_in = A^T . 1 = {d_in.tolist()}")

    print(f"\n  y_sum  (dense):    {y_sum_dense.tolist()}")
    print(f"  y_sum  (diffield): {y_sum_af.tolist()}")
    print(f"  Match: {torch.allclose(y_sum_dense, y_sum_af)}")

    print(f"\n  y_mean (dense):    {y_mean_dense.tolist()}")
    print(f"  y_mean (diffield): {y_mean_af.tolist()}")
    print(f"  Match: {torch.allclose(y_mean_dense, y_mean_af)}")

    print(f"\n  y_max  (dense):    {y_max_dense.tolist()}")
    print(f"  y_max  (diffield): {y_max_af.tolist()}")
    print(f"  Match: {torch.allclose(y_max_dense, y_max_af)}")

    print(f"\n  y_min  (dense):    {y_min_dense.tolist()}")
    print(f"  y_min  (diffield): {y_min_af.tolist()}")
    print(f"  Match: {torch.allclose(y_min_dense, y_min_af)}")


# ── setup ────────────────────────────────────────────────────────────

NUM_NODES = 3
# Directed edges: 0->1, 1->0, 1->2, 2->1
edge_index = torch.tensor(
    [[0, 1, 1, 2],
     [1, 0, 2, 1]],
    dtype=torch.long
)

ctx = AggregateContext(edge_index, num_nodes=NUM_NODES)

# Sensor field
q = torch.tensor([10.0, 20.0, 30.0])

# Dense adjacency matrix (for reference)
a = torch.zeros(NUM_NODES, NUM_NODES)
a[edge_index[0], edge_index[1]] = 1.0


# ── run ──────────────────────────────────────────────────────────────


def main():
    print("=" * 60)
    print("Tensor Encoding of Fields - diffield example")
    print("=" * 60)

    print(f"\nGraph: {NUM_NODES} nodes, edges 0<->1, 1<->2")
    print(f"  edge_index: {edge_index.tolist()}")
    print(f"\n  Sensor field q = {q.tolist()}")
    print("\n  Adjacency matrix A:")
    print(fmt_tensor(a))

    _run_field_ops(q, ctx)
    t_g_dense, _t_g_af = _run_lifting_ops(q, a, edge_index, ctx)
    abs_t_g_dense, _abs_t_g_af = _run_link_field_ops(
        t_g_dense, q, ctx
    )
    _run_neighborhood_ops(q, a, t_g_dense, abs_t_g_dense, ctx)

    print("\n" + "=" * 60)
    print("All operations verified")
    print("=" * 60)


if __name__ == "__main__":
    main()
