"""Tests for core DSL primitives: rep, foldhood, and hood variants."""

from __future__ import annotations

import torch

from autofield import AggregateContext, nbr, rep, foldhood, minhood, maxhood, sumhood, avghood
from autofield.dsl import field
from conftest import field_from_values
from tests.aggregate.support import ROUNDS, values


class TestRep:
    def test_accumulates(self, line_ctx):
        results = []
        for _ in range(ROUNDS):
            with line_ctx.round():
                val = rep(field.zeros(), lambda s: s + 1)
            results.append(val[0].item())
        assert results == [1.0, 2.0, 3.0]

    def test_per_node_state(self, line_ctx):
        n = line_ctx.num_nodes
        init = torch.arange(n, dtype=torch.float32)
        for _ in range(ROUNDS):
            with line_ctx.round():
                val = rep(init, lambda s: s + 1)
        assert torch.allclose(val, init + 3)

    def test_isolated_nodes(self, isolated_topology):
        edge_index, n = isolated_topology
        ctx = AggregateContext(edge_index, n)
        for _ in range(ROUNDS):
            with ctx.round():
                val = rep(field.zeros(), lambda s: s + 1)
        assert torch.allclose(val, values(3.0, 3.0, 3.0, 3.0))

    def test_empty_topology(self, empty_topology):
        edge_index, n = empty_topology
        ctx = AggregateContext(edge_index, n)
        for _ in range(3):
            with ctx.round():
                val = rep(field.zeros(), lambda s: s + 1)
        assert val.shape == (0,)


class TestFoldhood:
    def test_sum_triangle(self, triangle_ctx):
        x = field_from_values(triangle_ctx, [1.0, 2.0, 3.0])
        with triangle_ctx.round():
            m = foldhood(nbr(x), aggr="sum")
        assert torch.allclose(m, values(5.0, 4.0, 3.0))

    def test_min_triangle(self, triangle_ctx):
        x = field_from_values(triangle_ctx, [10.0, 2.0, 5.0])
        with triangle_ctx.round():
            m = foldhood(nbr(x), aggr="min")
        assert torch.allclose(m, values(2.0, 5.0, 2.0))

    def test_max_triangle(self, triangle_ctx):
        x = field_from_values(triangle_ctx, [1.0, 2.0, 3.0])
        with triangle_ctx.round():
            m = foldhood(nbr(x), aggr="max")
        assert torch.allclose(m, values(3.0, 3.0, 2.0))

    def test_mean_triangle(self, triangle_ctx):
        x = field_from_values(triangle_ctx, [1.0, 2.0, 3.0])
        with triangle_ctx.round():
            m = foldhood(nbr(x), aggr="mean")
        assert torch.allclose(m, values(2.5, 2.0, 1.5))


class TestHoodVariants:
    def test_sumhood(self, triangle_ctx):
        x = field_from_values(triangle_ctx, [1.0, 2.0, 3.0])
        with triangle_ctx.round():
            m = sumhood(nbr(x))
        assert torch.allclose(m, values(5.0, 4.0, 3.0))

    def test_minhood(self, triangle_ctx):
        x = field_from_values(triangle_ctx, [10.0, 2.0, 5.0])
        with triangle_ctx.round():
            m = minhood(nbr(x))
        assert torch.allclose(m, values(2.0, 5.0, 2.0))

    def test_maxhood(self, triangle_ctx):
        x = field_from_values(triangle_ctx, [1.0, 2.0, 3.0])
        with triangle_ctx.round():
            m = maxhood(nbr(x))
        assert torch.allclose(m, values(3.0, 3.0, 2.0))

    def test_avghood(self, triangle_ctx):
        x = field_from_values(triangle_ctx, [1.0, 2.0, 3.0])
        with triangle_ctx.round():
            m = avghood(nbr(x))
        assert torch.allclose(m, values(2.5, 2.0, 1.5))
