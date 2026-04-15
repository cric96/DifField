"""Tests for the public DSL primitives and field helpers."""

from __future__ import annotations

import torch
import pytest

from autofield import (
    AggregateContext,
    branch,
    broadcast,
    const,
    gradient,
    mid,
    mux,
    nbr,
    nbr_range,
    rep,
    foldhood,
    minhood,
    maxhood,
    sumhood,
)
from autofield.dsl import field
from autofield.utils import make_grid_graph
from conftest import (
    field_from_values,
    field_mid,
    field_of,
    field_ones,
    field_zeros,
    field_with_overrides,
)

ROUNDS = 3
GRADIENT_ROUNDS = 4


def values(*items: float) -> torch.Tensor:
    return torch.tensor(items, dtype=torch.float32)


def flags(*items: bool) -> torch.Tensor:
    return torch.tensor(items, dtype=torch.bool)


class TestRep:
    def test_accumulates(self, line_ctx):
        results = []
        for _ in range(5):
            with line_ctx.round():
                val = rep(field.zeros(), lambda s: s + 1)
            results.append(val[0].item())
        assert results == [1.0, 2.0, 3.0, 4.0, 5.0]

    def test_per_node_state(self, line_ctx):
        n = line_ctx.num_nodes
        init = torch.arange(n, dtype=torch.float32)
        for _ in range(3):
            with line_ctx.round():
                val = rep(init, lambda s: s + 1)
        assert torch.allclose(val, init + 3)

    def test_nested_rep_multiple_nbr(self, line_ctx):
        zero_field = field_zeros(line_ctx)
        results = []
        for _ in range(ROUNDS):
            with line_ctx.round():
                val = rep(
                    zero_field,
                    lambda outer_s: rep(
                        zero_field,
                        lambda inner_s: (
                            sumhood(nbr(outer_s)) + sumhood(nbr(inner_s)) + 1.0
                        ),
                    ),
                )
            results.append(val.clone())

        assert torch.allclose(results[0], values(1.0, 1.0, 1.0, 1.0))
        assert torch.allclose(results[1], values(3.0, 5.0, 5.0, 3.0))
        assert torch.allclose(results[2], values(11.0, 17.0, 17.0, 11.0))

    def test_rep_isolated_nodes(self, isolated_topology):
        edge_index, n = isolated_topology
        ctx = AggregateContext(edge_index, n)
        for _ in range(ROUNDS):
            with ctx.round():
                val = rep(field.zeros(), lambda s: s + 1)
        assert torch.allclose(val, values(3.0, 3.0, 3.0, 3.0))

    def test_rep_empty_topology(self, empty_topology):
        edge_index, n = empty_topology
        ctx = AggregateContext(edge_index, n)
        for _ in range(3):
            with ctx.round():
                val = rep(field.zeros(), lambda s: s + 1)
        assert val.shape == (0,)


class TestNbr:
    def test_sum_triangle(self, triangle_ctx):
        x = field_from_values(triangle_ctx, [1.0, 2.0, 3.0])
        with triangle_ctx.round():
            m = sumhood(nbr(x))
        assert torch.allclose(m, values(5.0, 4.0, 3.0))

    def test_min_triangle(self, triangle_ctx):
        x = field_from_values(triangle_ctx, [10.0, 2.0, 5.0])
        with triangle_ctx.round():
            m = minhood(nbr(x))
        assert torch.allclose(m, values(2.0, 5.0, 2.0))

    def test_ignores_context_edge_weight_by_default(self, triangle_topology):
        edge_index, n = triangle_topology
        edge_weight = torch.tensor([2.0, 3.0, 4.0, 5.0, 6.0, 7.0])
        ctx = AggregateContext(edge_index, n, edge_weight=edge_weight)
        x = field_from_values(ctx, [1.0, 2.0, 3.0])

        with ctx.round():
            m = sumhood(nbr(x))

        assert torch.allclose(m, values(5.0, 4.0, 3.0))

    def test_explicit_edge_weight_still_scales_messages(self, triangle_ctx):
        x = field_from_values(triangle_ctx, [1.0, 2.0, 3.0])
        # message_weight is now an edge-level tensor.
        # We can test multiplication inside hood by evaluating it explicitly
        # or just test that an explicit nbr() multiplication works.
        # Let's test multiplying a node field by another edge-level scalar.
        message_weight = torch.tensor([2.0, 3.0, 4.0, 5.0, 6.0, 7.0])
        from autofield.dsl.neighbor import NeighborExpr

        # Manually create an edge-wise expression for the weight
        weight_expr = NeighborExpr(lambda ctx, ei, ew: message_weight)

        with triangle_ctx.round():
            m = foldhood(nbr(x) * weight_expr, aggr="sum")

        assert torch.allclose(m, values(27.0, 17.0, 14.0))

    def test_nbr_range_supports_weighted_shortest_paths(self, triangle_topology):
        edge_index, _ = triangle_topology
        edge_weight = torch.tensor([2.0, 2.0, 2.0, 2.0, 10.0, 10.0])
        ctx = AggregateContext(edge_index, 3, edge_weight=edge_weight)
        source = field_with_overrides(ctx, ((0, 1.0),))

        for _ in range(GRADIENT_ROUNDS):
            with ctx.round():
                dist = rep(
                    field.inf(),
                    lambda dist_old: mux(
                        source, field.of(0.0), minhood(nbr(dist_old) + nbr_range())
                    ),
                    name="weighted_dist",
                )

        assert torch.allclose(dist, values(0.0, 2.0, 4.0))

    def test_tagged_nbr_exports_original_field(self, triangle_ctx):
        x = field_from_values(triangle_ctx, [1.0, 2.0, 3.0])

        with triangle_ctx.round() as round_ctx:
            _ = sumhood(nbr(x, tag="scores"))

        assert torch.allclose(round_ctx.exports["scores"], x)

    def test_include_self_can_force_self_aggregation(self):
        edge_index = torch.tensor(
            [
                [0, 1, 1, 2, 0, 1, 2],
                [1, 0, 2, 1, 0, 1, 2],
            ],
            dtype=torch.long,
        )
        ctx = AggregateContext(edge_index, 3)
        x = field_from_values(ctx, [1.0, 2.0, 3.0])

        with ctx.round():
            no_self = foldhood(nbr(x), aggr="sum", include_self=False)
        with ctx.round():
            with_self = foldhood(nbr(x), aggr="sum", include_self=True)

        assert torch.allclose(no_self, values(2.0, 4.0, 2.0))
        assert torch.allclose(with_self, values(3.0, 6.0, 5.0))

    def test_include_self_works_for_neighbor_expr(self, line_ctx):
        with line_ctx.round():
            no_self = foldhood(nbr_range() * 0.0 + 1.0, aggr="sum", include_self=False)
        with line_ctx.round():
            with_self = foldhood(nbr_range() * 0.0 + 1.0, aggr="sum", include_self=True)

        assert torch.allclose(no_self, values(1.0, 2.0, 2.0, 1.0))
        assert torch.allclose(with_self, values(2.0, 3.0, 3.0, 2.0))

    def test_nbr_isolated_nodes(self, isolated_topology):
        edge_index, n = isolated_topology
        ctx = AggregateContext(edge_index, n)
        x = field_from_values(ctx, [1.0, 2.0, 3.0, 4.0])
        with ctx.round():
            m = sumhood(nbr(x))
        assert torch.allclose(m, values(0.0, 0.0, 0.0, 0.0))


class TestBranch:
    def test_isolation(self, line_ctx):
        cond = flags(True, True, False, False)
        x = field_from_values(line_ctx, [1.0, 2.0, 10.0, 20.0])

        with line_ctx.round():
            result = branch(
                cond,
                lambda: sumhood(nbr(x)),
                lambda: sumhood(nbr(x)),
            )
        assert torch.allclose(result, values(2.0, 1.0, 20.0, 10.0))

    def test_different_nbr_in_branches(self, line_ctx):
        cond = flags(True, True, False, False)
        x = field_from_values(line_ctx, [1.0, 2.0, 30.0, 10.0])

        with line_ctx.round():
            result = branch(
                cond,
                lambda: sumhood(nbr(x)),
                lambda: minhood(nbr(x)),
            )
        assert torch.allclose(result, values(2.0, 1.0, 10.0, 30.0))

    def test_state_reset_on_switch(self, line_ctx):
        cond1 = flags(True, True, True, True)
        with line_ctx.round():
            branch(
                cond1,
                lambda: rep(field.zeros(), lambda s: s + 1, name="val"),
                lambda: rep(field.zeros(), lambda s: s + 1, name="val"),
                reset_states={"val": field.zeros()},
            )
        assert torch.allclose(
            line_ctx.state.get_state(name="val"), values(1.0, 1.0, 1.0, 1.0)
        )

        cond2 = flags(True, True, True, False)
        with line_ctx.round():
            branch(
                cond2,
                lambda: rep(field.zeros(), lambda s: s + 1, name="val"),
                lambda: rep(field.zeros(), lambda s: s + 1, name="val"),
                reset_states={"val": field.zeros()},
            )
        assert torch.allclose(
            line_ctx.state.get_state(name="val"), values(2.0, 2.0, 2.0, 1.0)
        )

    def test_branch_rep_nbr_nested(self, line_ctx):
        zero_field = field_zeros(line_ctx)
        cond = flags(True, True, False, False)

        results = []
        for _ in range(ROUNDS):
            with line_ctx.round():
                val = branch(
                    cond,
                    lambda: rep(
                        zero_field,
                        lambda s: sumhood(nbr(s)) + 1.0,
                        name="true_val",
                    ),
                    lambda: rep(
                        zero_field,
                        lambda s: maxhood(nbr(s)) + 10.0,
                        name="false_val",
                    ),
                )
            results.append(val.clone())

        assert torch.allclose(results[0], values(1.0, 1.0, 10.0, 10.0))
        assert torch.allclose(results[1], values(2.0, 2.0, 20.0, 20.0))
        assert torch.allclose(results[2], values(3.0, 3.0, 30.0, 30.0))


class TestMux:
    def test_no_isolation(self, line_ctx):
        cond = flags(True, True, False, False)
        x = field_from_values(line_ctx, [1.0, 2.0, 10.0, 20.0])

        with line_ctx.round():
            result = mux(
                cond.float(),
                lambda: sumhood(nbr(x)),
                lambda: sumhood(nbr(x)),
            )
        assert torch.allclose(result, values(2.0, 11.0, 22.0, 10.0))

    def test_selection(self, line_ctx):
        cond = values(1.0, 1.0, 0.0, 0.0)
        if_true = field_from_values(line_ctx, [10.0, 20.0, 30.0, 40.0])
        if_false = field_from_values(line_ctx, [100.0, 200.0, 300.0, 400.0])

        with line_ctx.round():
            result = mux(cond, if_true, if_false)
        assert torch.allclose(result, values(10.0, 20.0, 300.0, 400.0))


class TestGradient:
    def test_gradient_fixed(self):
        rows, cols = 5, 5
        edge_index, n = make_grid_graph(rows, cols)
        w = torch.tensor(1.0)

        ctx = AggregateContext(edge_index, n)
        source = field_with_overrides(ctx, ((0, 1.0),))
        zero_field = field_zeros(ctx)
        for _ in range(rows + cols):
            with ctx.round():
                d = rep(
                    field.inf(),
                    lambda dist: mux(source, zero_field, minhood(nbr(dist) + w)),
                )

        expected = field_zeros(ctx)
        for row_idx in range(rows):
            for col_idx in range(cols):
                expected[row_idx * cols + col_idx] = float(row_idx + col_idx)

        assert torch.allclose(d, expected)

    def test_differentiability(self):
        rows, cols = 3, 3
        edge_index, n = make_grid_graph(rows, cols)
        w = torch.tensor(1.0, requires_grad=True)

        ctx = AggregateContext(edge_index, n)
        source = field_with_overrides(ctx, ((0, 1.0),))
        zero_field = field_zeros(ctx)
        for _ in range(6):
            with ctx.round():
                d = rep(
                    field.inf(),
                    lambda dist: mux(source, zero_field, minhood(nbr(dist) + w)),
                )

        loss = d[d.isfinite()].sum()
        loss.backward()
        assert w.grad is not None
        assert w.grad.item() != 0.0

    def test_convenience_gradient_uses_edge_weight_by_default(self, triangle_topology):
        edge_index, _ = triangle_topology
        edge_weight = torch.tensor([2.0, 2.0, 2.0, 2.0, 10.0, 10.0])
        ctx = AggregateContext(edge_index, 3, edge_weight=edge_weight)
        source = field_with_overrides(ctx, ((0, 1.0),))

        for _ in range(GRADIENT_ROUNDS):
            with ctx.round():
                d = gradient(source, name="weighted")

        assert torch.allclose(d, values(0.0, 2.0, 4.0))

    def test_nbr_range_is_differentiable_with_edge_weight_tensor(
        self, triangle_topology
    ):
        edge_index, _ = triangle_topology
        edge_weight = torch.tensor([2.0, 2.0, 2.0, 2.0, 10.0, 10.0], requires_grad=True)
        ctx = AggregateContext(edge_index, 3, edge_weight=edge_weight)
        source = field_with_overrides(ctx, ((0, 1.0),))

        for _ in range(GRADIENT_ROUNDS):
            with ctx.round():
                d = rep(
                    field.inf(),
                    lambda dist_old: mux(
                        source, field.of(0.0), minhood(nbr(dist_old) + nbr_range())
                    ),
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
        zero_field = field_zeros(ctx)
        for _ in range(ROUNDS):
            with ctx.round():
                val = rep(
                    zero_field,
                    lambda outer_s: rep(
                        zero_field,
                        lambda inner_s: (
                            sumhood(nbr(outer_s) * w1)
                            + sumhood(nbr(inner_s) * w2)
                            + 1.0
                        ),
                    ),
                )

        loss = val.sum()
        loss.backward()

        assert w1.grad is not None
        assert w2.grad is not None
        assert w1.grad.item() != 0.0
        assert w2.grad.item() != 0.0


class TestField:
    def test_const(self, triangle_ctx):
        with triangle_ctx.round():
            value = const(2.5)
        assert torch.allclose(value, field_of(triangle_ctx, 2.5))

    def test_of(self, triangle_ctx):
        expected = field_of(triangle_ctx, 3.14)
        with triangle_ctx.round():
            f = field.of(3.14)
        assert f.shape == expected.shape
        assert torch.allclose(f, expected)

    def test_zeros_ones_inf(self, triangle_ctx):
        with triangle_ctx.round():
            z = field.zeros()
            o = field.ones()
            i = field.inf()
        assert torch.allclose(z, field_zeros(triangle_ctx))
        assert torch.allclose(o, field_ones(triangle_ctx))
        assert (i == float("inf")).all()

    def test_mid_returns_node_ids(self, line_ctx):
        expected = field_mid(line_ctx)
        with line_ctx.round():
            node_ids = mid()
        assert torch.allclose(node_ids, expected)

    def test_in_gradient_program(self):
        edge_index = torch.tensor(
            [
                [0, 1, 1, 2, 0, 1, 2],
                [1, 0, 2, 1, 0, 1, 2],
            ],
            dtype=torch.long,
        )
        n = 3
        w = torch.tensor(1.0)

        ctx = AggregateContext(edge_index, n)
        source = field_with_overrides(ctx, ((0, 1.0),))
        zero_field = field_zeros(ctx)
        for _ in range(GRADIENT_ROUNDS):
            with ctx.round():
                d = rep(
                    field.inf(),
                    lambda dist: mux(source, zero_field, minhood(nbr(dist) + w)),
                )
        assert torch.allclose(d, values(0.0, 1.0, 2.0))

    def test_broadcast_propagates_root_value(self, line_ctx):
        source = flags(True, False, False, False)
        payload = field_from_values(line_ctx, [10.0, 0.0, 0.0, 0.0])

        for _ in range(GRADIENT_ROUNDS):
            with line_ctx.round():
                out = broadcast(source, payload, name="payload")

        assert torch.allclose(out, values(10.0, 10.0, 10.0, 10.0))


class TestAutoNaming:
    def test_rep_without_name_accumulates(self, line_ctx):
        results = []
        for _ in range(3):
            with line_ctx.round():
                val = rep(field.zeros(), lambda s: s + 1)
            results.append(val[0].item())
        assert results == [1.0, 2.0, 3.0]

    def test_rep_same_position_same_name_in_branch(self, line_ctx):
        zero_field = field_zeros(line_ctx)
        cond = flags(True, True, False, False)

        results = []
        for _ in range(ROUNDS):
            with line_ctx.round():
                val = branch(
                    cond,
                    lambda: rep(zero_field, lambda s: sumhood(nbr(s)) + 1.0),
                    lambda: rep(zero_field, lambda s: sumhood(nbr(s)) + 10.0),
                )
            results.append(val.clone())

        assert torch.allclose(results[0], values(1.0, 1.0, 10.0, 10.0))
        assert torch.allclose(results[1], values(2.0, 2.0, 20.0, 20.0))
        assert torch.allclose(results[2], values(3.0, 3.0, 30.0, 30.0))

    def test_rep_different_positions_different_names(self, line_ctx):
        zero_field = field_zeros(line_ctx)
        with line_ctx.round():
            a = rep(zero_field, lambda s: s + 1.0)
            b = rep(zero_field, lambda s: s + 10.0)
        assert torch.allclose(a, field_ones(line_ctx))
        assert torch.allclose(b, field_of(line_ctx, 10.0))

    def test_two_nbr_same_level_different_tags(self, triangle_ctx):
        x = field_from_values(triangle_ctx, [1.0, 2.0, 3.0])
        with triangle_ctx.round() as round_ctx:
            a = sumhood(nbr(x))
            b = maxhood(nbr(x))
        assert torch.allclose(a, values(5.0, 4.0, 3.0))
        assert torch.allclose(b, values(3.0, 3.0, 2.0))
        tags = list(round_ctx.exports.keys())
        assert len(tags) == 2
        assert tags[0] != tags[1]

    def test_explicit_name_overrides_auto(self, line_ctx):
        with line_ctx.round():
            rep(field.zeros(), lambda s: s + 1, name="my_state")
        assert line_ctx.get_state(name="my_state") is not None
        assert torch.allclose(line_ctx.get_state(name="my_state"), field_ones(line_ctx))
