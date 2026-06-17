"""Tests for layer implementations: IterateLayer, GatherLayer, BranchLayer, MuxLayer."""

from __future__ import annotations

import torch

from conftest import field_from_values, field_with_overrides, field_zeros
from diffield.dsl import field, iterate, mux
from diffield.layers import BranchLayer, GatherLayer, IterateLayer, MuxLayer
from tests.aggregate.support import AddConstant, flags, values

COMPOSITION_ROUNDS = 5


class TestGatherLayer:
    def test_sum_triangle(self, triangle_ctx):
        layer = GatherLayer(aggr="sum")
        x = field_from_values(triangle_ctx, [1.0, 2.0, 3.0])
        with triangle_ctx.round():
            m = layer(x)
        assert torch.allclose(m, values(5.0, 4.0, 3.0))

    def test_min_triangle(self, triangle_ctx):
        layer = GatherLayer(aggr="min", fill_value=float("inf"))
        x = field_from_values(triangle_ctx, [10.0, 2.0, 5.0])
        with triangle_ctx.round():
            m = layer(x)
        assert torch.allclose(m, values(2.0, 5.0, 2.0))


class TestIterateLayer:
    def test_iterate_layer_in_dsl(self, line_ctx):
        zero_field = field_zeros(line_ctx)
        iterate_layer = IterateLayer(zero_field, lambda s: s + 1, name="counter")
        results = []
        for _ in range(COMPOSITION_ROUNDS):
            with line_ctx.round():
                val = iterate_layer(zero_field)
            results.append(val[0].item())
        assert results == [1.0, 2.0, 3.0, 4.0, 5.0]


class TestBranchLayer:
    def test_explicit_ctx_isolates_partitions(self, line_ctx):
        cond = flags(True, True, False, False)
        x = field_from_values(line_ctx, [1.0, 2.0, 30.0, 10.0])
        layer = BranchLayer(
            GatherLayer(aggr="sum"),
            GatherLayer(aggr="max"),
            branch_name="standalone",
        )

        with line_ctx.round() as round_ctx:
            result = layer(x, cond, ctx=round_ctx)

        assert torch.allclose(result, values(2.0, 1.0, 10.0, 30.0))


class TestMuxLayer:
    def test_explicit_ctx_selects_outputs(self, line_ctx):
        cond = values(1.0, 1.0, 0.0, 0.0)
        x = field_from_values(line_ctx, [1.0, 2.0, 3.0, 4.0])
        layer = MuxLayer(AddConstant(1.0), AddConstant(100.0))

        with line_ctx.round() as round_ctx:
            result = layer(x, cond, ctx=round_ctx)

        assert torch.allclose(result, values(2.0, 3.0, 103.0, 104.0))


class TestLayersInDsl:
    def test_gather_layer_in_dsl(self, triangle_ctx):
        gather_layer = GatherLayer(aggr="sum")
        x = field_from_values(triangle_ctx, [1.0, 2.0, 3.0])
        with triangle_ctx.round():
            m = gather_layer(x)
        assert torch.allclose(m, values(5.0, 4.0, 3.0))

    def test_layers_compose_with_dsl(self, triangle_ctx):
        gather_layer = GatherLayer(aggr="min", fill_value=float("inf"))
        x = field_from_values(triangle_ctx, [10.0, 2.0, 5.0])
        cond = field_from_values(triangle_ctx, [1.0, 0.0, 1.0])
        with triangle_ctx.round():
            messages = gather_layer(x)
            result = mux(cond, messages, field.zeros())
        assert torch.allclose(result, values(2.0, 0.0, 2.0))

    def test_gradient_with_layers(self, line_ctx):
        gather_min_layer = GatherLayer(aggr="min")
        source = field_with_overrides(line_ctx, ((0, 1.0),))

        for _ in range(COMPOSITION_ROUNDS):
            with line_ctx.round():
                d = iterate(
                    field.inf(),
                    lambda s: mux(source, field.zeros(), gather_min_layer(s + 1)),
                )
        assert torch.allclose(d, values(0.0, 1.0, 2.0, 3.0))
