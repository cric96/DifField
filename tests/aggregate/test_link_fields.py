"""Tests for neighbor expressions: scatter, scatter_range, tags, edge weights, include_self."""

from __future__ import annotations

import torch

from diffield import AggregateContext
from diffield.dsl import (
    branch,
    mux,
    scatter,
    scatter_range,
    iterate,
    gather,
    gather_min,
    gather_max,
    gather_sum,
    field,
)
from diffield.dsl.scattering import LinkField
from conftest import field_from_values, field_with_overrides
from tests.aggregate.support import (
    GRADIENT_ROUNDS,
    ROUNDS,
    flags,
    values,
)


class TestScatter:
    def test_sum_triangle(self, triangle_ctx):
        x = field_from_values(triangle_ctx, [1.0, 2.0, 3.0])
        with triangle_ctx.round():
            m = gather_sum(scatter(x))
        assert torch.allclose(m, values(5.0, 4.0, 3.0))

    def test_min_triangle(self, triangle_ctx):
        x = field_from_values(triangle_ctx, [10.0, 2.0, 5.0])
        with triangle_ctx.round():
            m = gather_min(scatter(x))
        assert torch.allclose(m, values(2.0, 5.0, 2.0))

    def test_ignores_context_edge_weight_by_default(self, triangle_topology):
        edge_index, n = triangle_topology
        edge_weight = torch.tensor([2.0, 3.0, 4.0, 5.0, 6.0, 7.0])
        ctx = AggregateContext(edge_index, n, edge_weight=edge_weight)
        x = field_from_values(ctx, [1.0, 2.0, 3.0])

        with ctx.round():
            m = gather_sum(scatter(x))

        assert torch.allclose(m, values(5.0, 4.0, 3.0))

    def test_explicit_edge_weight_still_scales_messages(self, triangle_ctx):
        x = field_from_values(triangle_ctx, [1.0, 2.0, 3.0])
        message_weight = torch.tensor([2.0, 3.0, 4.0, 5.0, 6.0, 7.0])

        weight_expr = LinkField(lambda ctx, ei, ew: message_weight)

        with triangle_ctx.round():
            m = gather(scatter(x) * weight_expr, aggr="sum")

        assert torch.allclose(m, values(27.0, 17.0, 14.0))

    def test_scatter_isolated_nodes(self, isolated_topology):
        edge_index, n = isolated_topology
        ctx = AggregateContext(edge_index, n)
        x = field_from_values(ctx, [1.0, 2.0, 3.0, 4.0])
        with ctx.round():
            m = gather_sum(scatter(x))
        assert torch.allclose(m, values(0.0, 0.0, 0.0, 0.0))


class TestScatterRange:
    def test_supports_weighted_shortest_paths(self, triangle_topology):
        edge_index, _ = triangle_topology
        edge_weight = torch.tensor([2.0, 2.0, 2.0, 2.0, 10.0, 10.0])
        ctx = AggregateContext(edge_index, 3, edge_weight=edge_weight)
        source = field_with_overrides(ctx, ((0, 1.0),))

        for _ in range(GRADIENT_ROUNDS):
            with ctx.round():
                dist = iterate(
                    field.inf(),
                    lambda dist_old: mux(
                        source, field.of(0.0), gather_min(scatter(dist_old) + scatter_range())
                    ),
                    name="weighted_dist",
                )

        assert torch.allclose(dist, values(0.0, 2.0, 4.0))


class TestScatterTags:
    def test_tagged_scatter_exports_original_field(self, triangle_ctx):
        x = field_from_values(triangle_ctx, [1.0, 2.0, 3.0])

        with triangle_ctx.round() as round_ctx:
            _ = gather_sum(scatter(x, tag="scores"))

        assert torch.allclose(round_ctx.exports["scores"], x)


class TestIncludeSelf:
    def test_can_force_self_aggregation(self):
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
            no_self = gather(scatter(x), aggr="sum", include_self=False)
        with ctx.round():
            with_self = gather(scatter(x), aggr="sum", include_self=True)

        assert torch.allclose(no_self, values(2.0, 4.0, 2.0))
        assert torch.allclose(with_self, values(3.0, 6.0, 5.0))

    def test_works_for_neighbor_expr(self, line_ctx):
        with line_ctx.round():
            no_self = gather(scatter_range() * 0.0 + 1.0, aggr="sum", include_self=False)
        with line_ctx.round():
            with_self = gather(scatter_range() * 0.0 + 1.0, aggr="sum", include_self=True)

        assert torch.allclose(no_self, values(1.0, 2.0, 2.0, 1.0))
        assert torch.allclose(with_self, values(2.0, 3.0, 3.0, 2.0))


class TestScatterInsideBranch:
    def test_ignores_cross_partition_neighbors(self, line_ctx):
        cond = flags(True, True, False, False)
        x = field_from_values(line_ctx, [10.0, 20.0, 30.0, 40.0])

        with line_ctx.round():
            result = branch(
                cond,
                lambda: gather_sum(scatter(x)),
                lambda: gather_max(scatter(x)),
                branch_name="isolation_test",
            )

        assert torch.allclose(result, values(20.0, 10.0, 40.0, 30.0))

    def test_iterate_with_scatter_in_branch_isolates_accumulation(self, line_ctx):
        cond = flags(True, True, False, False)

        results = []
        for _ in range(ROUNDS):
            with line_ctx.round():
                val = branch(
                    cond,
                    lambda: iterate(field.zeros(), lambda s: gather_sum(scatter(s)) + 1.0),
                    lambda: iterate(field.zeros(), lambda s: gather_sum(scatter(s)) + 10.0),
                    branch_name="iterate_scatter_iso",
                )
            results.append(val.clone())

        assert torch.allclose(results[0], values(1.0, 1.0, 10.0, 10.0))
        assert torch.allclose(results[1], values(2.0, 2.0, 20.0, 20.0))
        assert torch.allclose(results[2], values(3.0, 3.0, 30.0, 30.0))
