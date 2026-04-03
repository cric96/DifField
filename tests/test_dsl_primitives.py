"""Tests for the public DSL primitives and field helpers."""

from __future__ import annotations

import torch

from aggregate_gnn import (
    AggregateContext,
    branch,
    broadcast,
    const,
    gradient,
    mid,
    mux,
    nbr,
    nbrRange,
    rep,
)
from aggregate_gnn.dsl import field
from aggregate_gnn.utils import make_grid_graph
from tests.support import line_graph, triangle_graph


class TestRep:
    def test_accumulates(self):
        edge_index, n = line_graph()
        ctx = AggregateContext(edge_index, n)
        results = []
        for _ in range(5):
            with ctx.round():
                val = rep("counter", 0.0, lambda s: s + 1)
            results.append(val[0].item())
        assert results == [1.0, 2.0, 3.0, 4.0, 5.0]

    def test_per_node_state(self):
        edge_index, n = line_graph()
        ctx = AggregateContext(edge_index, n)
        init = torch.arange(n, dtype=torch.float32)
        for _ in range(3):
            with ctx.round():
                val = rep("state", init, lambda s: s + 1)
        assert torch.allclose(val, init + 3)

    def test_nested_rep_multiple_nbr(self):
        edge_index, n = line_graph()
        ctx = AggregateContext(edge_index, n)

        results = []
        for _ in range(3):
            with ctx.round():
                val = rep(
                    "outer",
                    torch.zeros(n),
                    lambda outer_s: rep(
                        "inner",
                        torch.zeros(n),
                        lambda inner_s: nbr(outer_s, aggr="sum") + nbr(inner_s, aggr="sum") + 1.0,
                    ),
                )
            results.append(val.clone())

        assert torch.allclose(results[0], torch.tensor([1.0, 1.0, 1.0, 1.0]))
        assert torch.allclose(results[1], torch.tensor([3.0, 5.0, 5.0, 3.0]))
        assert torch.allclose(results[2], torch.tensor([11.0, 17.0, 17.0, 11.0]))


class TestNbr:
    def test_sum_triangle(self):
        edge_index, n = triangle_graph()
        ctx = AggregateContext(edge_index, n)
        x = torch.tensor([1.0, 2.0, 3.0])
        with ctx.round():
            m = nbr(x, aggr="sum")
        assert torch.allclose(m, torch.tensor([5.0, 4.0, 3.0]))

    def test_min_triangle(self):
        edge_index, n = triangle_graph()
        ctx = AggregateContext(edge_index, n)
        x = torch.tensor([10.0, 2.0, 5.0])
        with ctx.round():
            m = nbr(x, aggr="min")
        assert torch.allclose(m, torch.tensor([2.0, 5.0, 2.0]))

    def test_ignores_context_edge_weight_by_default(self):
        edge_index, n = triangle_graph()
        edge_weight = torch.tensor([2.0, 3.0, 4.0, 5.0, 6.0, 7.0])
        ctx = AggregateContext(edge_index, n, edge_weight=edge_weight)
        x = torch.tensor([1.0, 2.0, 3.0])

        with ctx.round():
            m = nbr(x, aggr="sum")

        assert torch.allclose(m, torch.tensor([5.0, 4.0, 3.0]))

    def test_explicit_edge_weight_still_scales_messages(self):
        edge_index, n = triangle_graph()
        ctx = AggregateContext(edge_index, n)
        x = torch.tensor([1.0, 2.0, 3.0])
        message_weight = torch.tensor([2.0, 3.0, 4.0, 5.0, 6.0, 7.0])

        with ctx.round():
            m = nbr(x, aggr="sum", edge_weight=message_weight)

        assert torch.allclose(m, torch.tensor([27.0, 17.0, 14.0]))

    def test_nbr_range_supports_weighted_shortest_paths(self):
        edge_index, _ = triangle_graph()
        edge_weight = torch.tensor([2.0, 2.0, 2.0, 2.0, 10.0, 10.0])
        source = torch.tensor([1.0, 0.0, 0.0])
        ctx = AggregateContext(edge_index, 3, edge_weight=edge_weight)

        for _ in range(4):
            with ctx.round():
                dist = rep(
                    "weighted_dist",
                    float("inf"),
                    lambda dist_old: mux(source, field.of(0.0), nbr(dist_old + nbrRange(), aggr="min")),
                )

        assert torch.allclose(dist, torch.tensor([0.0, 2.0, 4.0]))

    def test_tagged_nbr_exports_original_field(self):
        edge_index, n = triangle_graph()
        ctx = AggregateContext(edge_index, n)
        x = torch.tensor([1.0, 2.0, 3.0])

        with ctx.round() as round_ctx:
            _ = nbr(x, aggr="sum", tag="scores")

        assert torch.allclose(round_ctx.exports["scores"], x)


class TestBranch:
    def test_isolation(self):
        edge_index, n = line_graph()
        ctx = AggregateContext(edge_index, n)
        cond = torch.tensor([True, True, False, False])
        x = torch.tensor([1.0, 2.0, 10.0, 20.0])

        with ctx.round():
            result = branch(
                cond,
                lambda: nbr(x, aggr="sum"),
                lambda: nbr(x, aggr="sum"),
            )
        assert torch.allclose(result, torch.tensor([2.0, 1.0, 20.0, 10.0]))

    def test_different_nbr_in_branches(self):
        edge_index, n = line_graph()
        ctx = AggregateContext(edge_index, n)
        cond = torch.tensor([True, True, False, False])
        x = torch.tensor([1.0, 2.0, 30.0, 10.0])

        with ctx.round():
            result = branch(
                cond,
                lambda: nbr(x, aggr="sum"),
                lambda: nbr(x, aggr="min"),
            )
        assert torch.allclose(result, torch.tensor([2.0, 1.0, 10.0, 30.0]))

    def test_state_reset_on_switch(self):
        edge_index, n = line_graph()
        ctx = AggregateContext(edge_index, n)

        cond1 = torch.tensor([True, True, True, True])
        with ctx.round():
            branch(
                cond1,
                lambda: rep("val", 0.0, lambda s: s + 1),
                lambda: rep("val", 0.0, lambda s: s + 1),
                reset_states={"val": 0.0},
            )
        assert torch.allclose(ctx._ctx.state._states["val"], torch.tensor([1.0, 1.0, 1.0, 1.0]))

        cond2 = torch.tensor([True, True, True, False])
        with ctx.round():
            branch(
                cond2,
                lambda: rep("val", 0.0, lambda s: s + 1),
                lambda: rep("val", 0.0, lambda s: s + 1),
                reset_states={"val": 0.0},
            )
        assert torch.allclose(ctx._ctx.state._states["val"], torch.tensor([2.0, 2.0, 2.0, 1.0]))

    def test_branch_rep_nbr_nested(self):
        edge_index, n = line_graph()
        ctx = AggregateContext(edge_index, n)
        cond = torch.tensor([True, True, False, False])

        results = []
        for _ in range(3):
            with ctx.round():
                val = branch(
                    cond,
                    lambda: rep("true_val", torch.zeros(n), lambda s: nbr(s, aggr="sum") + 1.0),
                    lambda: rep("false_val", torch.zeros(n), lambda s: nbr(s, aggr="max") + 10.0),
                )
            results.append(val.clone())

        assert torch.allclose(results[0], torch.tensor([1.0, 1.0, 10.0, 10.0]))
        assert torch.allclose(results[1], torch.tensor([2.0, 2.0, 20.0, 20.0]))
        assert torch.allclose(results[2], torch.tensor([3.0, 3.0, 30.0, 30.0]))


class TestMux:
    def test_no_isolation(self):
        edge_index, n = line_graph()
        ctx = AggregateContext(edge_index, n)
        cond = torch.tensor([True, True, False, False])
        x = torch.tensor([1.0, 2.0, 10.0, 20.0])

        with ctx.round():
            result = mux(
                cond.float(),
                lambda: nbr(x, aggr="sum"),
                lambda: nbr(x, aggr="sum"),
            )
        assert torch.allclose(result, torch.tensor([2.0, 11.0, 22.0, 10.0]))

    def test_selection(self):
        edge_index, n = line_graph()
        ctx = AggregateContext(edge_index, n)
        cond = torch.tensor([1.0, 1.0, 0.0, 0.0])

        with ctx.round():
            result = mux(
                cond,
                torch.tensor([10.0, 20.0, 30.0, 40.0]),
                torch.tensor([100.0, 200.0, 300.0, 400.0]),
            )
        assert torch.allclose(result, torch.tensor([10.0, 20.0, 300.0, 400.0]))


class TestGradient:
    def test_gradient_fixed(self):
        rows, cols = 5, 5
        edge_index, n = make_grid_graph(rows, cols)

        source = torch.zeros(n)
        source[0] = 1.0
        w = torch.tensor(1.0)

        ctx = AggregateContext(edge_index, n)
        for _ in range(rows + cols):
            with ctx.round():
                d = rep("dist", float("inf"), lambda dist: mux(source, torch.zeros(n), nbr(dist + w, aggr="min")))

        expected = torch.zeros(n)
        for row_idx in range(rows):
            for col_idx in range(cols):
                expected[row_idx * cols + col_idx] = float(row_idx + col_idx)

        assert torch.allclose(d, expected)

    def test_differentiability(self):
        rows, cols = 3, 3
        edge_index, n = make_grid_graph(rows, cols)

        source = torch.zeros(n)
        source[0] = 1.0
        w = torch.tensor(1.0, requires_grad=True)

        ctx = AggregateContext(edge_index, n)
        for _ in range(6):
            with ctx.round():
                d = rep("dist", float("inf"), lambda dist: mux(source, torch.zeros(n), nbr(dist + w, aggr="min")))

        loss = d[d.isfinite()].sum()
        loss.backward()
        assert w.grad is not None
        assert w.grad.item() != 0.0

    def test_convenience_gradient_uses_edge_weight_by_default(self):
        edge_index, _ = triangle_graph()
        edge_weight = torch.tensor([2.0, 2.0, 2.0, 2.0, 10.0, 10.0])
        source = torch.tensor([1.0, 0.0, 0.0])
        ctx = AggregateContext(edge_index, 3, edge_weight=edge_weight)

        for _ in range(4):
            with ctx.round():
                d = gradient(source, name="weighted")

        assert torch.allclose(d, torch.tensor([0.0, 2.0, 4.0]))

    def test_nbr_range_is_differentiable_with_edge_weight_tensor(self):
        edge_index, _ = triangle_graph()
        edge_weight = torch.tensor([2.0, 2.0, 2.0, 2.0, 10.0, 10.0], requires_grad=True)
        source = torch.tensor([1.0, 0.0, 0.0])
        ctx = AggregateContext(edge_index, 3, edge_weight=edge_weight)

        for _ in range(4):
            with ctx.round():
                d = rep(
                    "weighted_dist",
                    float("inf"),
                    lambda dist_old: mux(source, field.of(0.0), nbr(dist_old + nbrRange(), aggr="min")),
                )

        loss = d[d.isfinite()].sum()
        loss.backward()

        assert edge_weight.grad is not None
        assert torch.isfinite(edge_weight.grad).all()
        assert edge_weight.grad.abs().sum().item() > 0.0

    def test_nested_rep_multiple_nbr_differentiability(self):
        rows, cols = 3, 3
        edge_index, n = make_grid_graph(rows, cols)

        w1 = torch.tensor(0.5, requires_grad=True)
        w2 = torch.tensor(0.5, requires_grad=True)

        ctx = AggregateContext(edge_index, n)
        for _ in range(3):
            with ctx.round():
                val = rep(
                    "outer",
                    torch.zeros(n),
                    lambda outer_s: rep(
                        "inner",
                        torch.zeros(n),
                        lambda inner_s: nbr(outer_s * w1, aggr="sum") + nbr(inner_s * w2, aggr="sum") + 1.0,
                    ),
                )

        loss = val.sum()
        loss.backward()

        assert w1.grad is not None
        assert w2.grad is not None
        assert w1.grad.item() != 0.0
        assert w2.grad.item() != 0.0


class TestField:
    def test_const(self):
        edge_index, n = triangle_graph()
        ctx = AggregateContext(edge_index, n)
        with ctx.round():
            value = const(2.5)
        assert torch.allclose(value, torch.full((n,), 2.5))

    def test_of(self):
        edge_index, n = triangle_graph()
        ctx = AggregateContext(edge_index, n)
        with ctx.round():
            f = field.of(3.14)
        assert f.shape == (n,)
        assert torch.allclose(f, torch.tensor([3.14, 3.14, 3.14]))

    def test_zeros_ones_inf(self):
        edge_index, n = triangle_graph()
        ctx = AggregateContext(edge_index, n)
        with ctx.round():
            z = field.zeros()
            o = field.ones()
            i = field.inf()
        assert torch.allclose(z, torch.zeros(n))
        assert torch.allclose(o, torch.ones(n))
        assert (i == float("inf")).all()

    def test_mid_returns_node_ids(self):
        edge_index, _ = line_graph()
        ctx = AggregateContext(edge_index, 4)
        with ctx.round():
            node_ids = mid()
        assert torch.allclose(node_ids, torch.tensor([0.0, 1.0, 2.0, 3.0]))

    def test_in_gradient_program(self):
        edge_index = torch.tensor(
            [
                [0, 1, 1, 2, 0, 1, 2],
                [1, 0, 2, 1, 0, 1, 2],
            ],
            dtype=torch.long,
        )
        n = 3
        source = torch.tensor([1.0, 0.0, 0.0])
        w = torch.tensor(1.0)

        ctx = AggregateContext(edge_index, n)
        for _ in range(4):
            with ctx.round():
                d = rep("dist", float("inf"), lambda dist: mux(source, field.of(0.0), nbr(dist + w, aggr="min")))
        assert torch.allclose(d, torch.tensor([0.0, 1.0, 2.0]))

    def test_broadcast_propagates_root_value(self):
        edge_index, n = line_graph()
        ctx = AggregateContext(edge_index, n)
        source = torch.tensor([True, False, False, False])
        payload = torch.tensor([10.0, 0.0, 0.0, 0.0])

        for _ in range(4):
            with ctx.round():
                out = broadcast(source, payload, name="payload")

        assert torch.allclose(out, torch.tensor([10.0, 10.0, 10.0, 10.0]))