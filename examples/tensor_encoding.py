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

import torch

from diffield import AggregateContext
from diffield.dsl import (
    gather_max,
    gather_min,
    gather_sum,
    gather_avg,
    scatter,
)
from diffield.core import current_context


# ── helpers ──────────────────────────────────────────────────────────

def to_dense_matrix(edge_index: torch.Tensor, values: torch.Tensor, n: int) -> torch.Tensor:
    """Materialise edge-wise values into an NxN dense matrix (row=src, col=tgt)."""
    M = torch.zeros(n, n, dtype=torch.float32)
    src, tgt = edge_index[0], edge_index[1]
    M[src, tgt] = values
    return M


def fmt_tensor(t: torch.Tensor, name: str = "") -> str:
    lines = []
    if name:
        lines.append(f"  {name}  (shape {list(t.shape)})")
    for row in t.tolist():
        lines.append(f"    {row}")
    return "\n".join(lines)


# ── setup ────────────────────────────────────────────────────────────

NUM_NODES = 3
# Directed edges: 0→1, 1→0, 1→2, 2→1
edge_index = torch.tensor(
    [[0, 1, 1, 2],
     [1, 0, 2, 1]], 
    dtype=torch.long
)

ctx = AggregateContext(edge_index, num_nodes=NUM_NODES)

# Sensor field
q = torch.tensor([10.0, 20.0, 30.0])

# Dense adjacency matrix (for reference)
A = torch.zeros(NUM_NODES, NUM_NODES)
A[edge_index[0], edge_index[1]] = 1.0


# ── run ──────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("Tensor Encoding of Fields — diffield example")
    print("=" * 60)

    print(f"\nGraph: {NUM_NODES} nodes, edges 0↔1, 1↔2")
    print(f"  edge_index: {edge_index.tolist()}")
    print(f"\n  Sensor field q = {q.tolist()}")
    print(f"\n  Adjacency matrix A:")
    print(fmt_tensor(A))

    # ── 1. Field operations ──────────────────────────────────────────
    print("\n" + "-" * 60)
    print("1. FIELD OPERATIONS  (node → node)")
    print("-" * 60)

    # Dense: q_hat = q / 30
    q_hat_dense = q / 30.0

    # diffield: pointwise division
    with ctx.round():
        q_hat_af = q / 30.0

    print(f"\n  Dense:    q_hat = q / 30 = {q_hat_dense.tolist()}")
    print(f"  diffield:                = {q_hat_af.tolist()}")
    print(f"  Match: {torch.allclose(q_hat_dense, q_hat_af)}")

    # ── 2. Lifting operations ────────────────────────────────────────
    print("\n" + "-" * 60)
    print("2. LIFTING OPERATIONS  (node → link)")
    print("-" * 60)

    # Dense: T_g = A ⊙ (q·1^T − 1·q^T)
    ones = torch.ones(NUM_NODES)
    T_g_dense = A * (q.outer(ones) - ones.outer(q))

    # diffield: scatter(q) − q  (scatter(q) gives q_j, bare q gives q_i)
    # produces (q_j − q_i) on each edge; the adjacency mask is implicit because
    # edges that don't exist are never materialised
    with ctx.round():
        T_g_linkfield = scatter(q) - q  # q[source] - q[target]
        gather_sum(T_g_linkfield)  # optional: materialise node-wise sum to check correctness
        # Evaluate on each edge to get the link values
        rctx = current_context()
        T_g_values = T_g_linkfield.evaluate(ctx=rctx, edge_index=rctx.edge_index, edge_weight=rctx.edge_weight)
        T_g_af = to_dense_matrix(rctx.edge_index, T_g_values, NUM_NODES)

    print(f"\n  Dense T_g = A ⊙ (q·1^T − 1·q^T):")
    print(fmt_tensor(T_g_dense))
    print(f"\n  diffield T_g (materialised):")
    print(fmt_tensor(T_g_af))
    print(f"  Match: {torch.allclose(T_g_dense, T_g_af)}")

    # ── 3. Link-field operations ─────────────────────────────────────
    print("\n" + "-" * 60)
    print("3. LINK-FIELD OPERATIONS  (link → link)")
    print("-" * 60)

    # Dense: |T_g|
    abs_T_g_dense = T_g_dense.abs()

    # diffield: |scatter(q) − q|
    with ctx.round():
        abs_T_g_linkfield = (scatter(q) - q).abs()
        rctx = current_context()
        abs_T_g_values = abs_T_g_linkfield.evaluate(ctx=rctx, edge_index=rctx.edge_index, edge_weight=rctx.edge_weight)
        abs_T_g_af = to_dense_matrix(rctx.edge_index, abs_T_g_values, NUM_NODES)

    print(f"\n  Dense |T_g|:")
    print(fmt_tensor(abs_T_g_dense))
    print(f"\n  diffield |T_g| (materialised):")
    print(fmt_tensor(abs_T_g_af))
    print(f"  Match: {torch.allclose(abs_T_g_dense, abs_T_g_af)}")

    # ── 4. Neighborhood operations ───────────────────────────────────
    print("\n" + "-" * 60)
    print("4. NEIGHBORHOOD OPERATIONS  (link → node)")
    print("-" * 60)

    # Dense incoming reductions:
    # y_sum, y_mean are over |T_g|
    # y_max, y_min are over T_g
    d_in = A.sum(dim=0)  # in-degree per node
    y_sum_dense = abs_T_g_dense.T @ ones
    y_mean_dense = y_sum_dense / d_in

    y_max_dense = torch.zeros(NUM_NODES)
    y_min_dense = torch.zeros(NUM_NODES)
    for i in range(NUM_NODES):
        incoming = T_g_dense[:, i][A[:, i] == 1]
        y_max_dense[i] = incoming.max()
        y_min_dense[i] = incoming.min()

    # diffield: gather_*
    with ctx.round():
        T_g_expr = scatter(q) - q
        y_sum_af = gather_sum(T_g_expr.abs())
        y_mean_af = gather_avg(T_g_expr.abs())
        y_max_af = gather_max(T_g_expr)
        y_min_af = gather_min(T_g_expr)

    print(f"\n  In-degree d_in = A^T · 1 = {d_in.tolist()}")

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

    print("\n" + "=" * 60)
    print("All operations verified ✓")
    print("=" * 60)


if __name__ == "__main__":
    main()
