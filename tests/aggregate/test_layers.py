"""Tests for layer implementations: RepLayer, FoldhoodLayer, BranchLayer, MuxLayer."""

from __future__ import annotations

import torch

from autofield import mux, rep
from autofield.layers import BranchLayer, MuxLayer, FoldhoodLayer, RepLayer
from autofield.dsl import field
from conftest import field_from_values, field_with_overrides, field_zeros
from tests.aggregate.support import AddConstant, flags, values

COMPOSITION_ROUNDS = 5


class TestFoldhoodLayer:
    def test_sum_triangle(self, triangle_ctx):
        layer = FoldhoodLayer(aggr="sum")
        x = field_from_values(triangle_ctx, [1.0, 2.0, 3.0])
        with triangle_ctx.round():
            m = layer(x)
        assert torch.allclose(m, values(5.0, 4.0, 3.0))

    def test_min_triangle(self, triangle_ctx):
        layer = FoldhoodLayer(aggr="min", fill_value=float("inf"))
        x = field_from_values(triangle_ctx, [10.0, 2.0, 5.0])
        with triangle_ctx.round():
            m = layer(x)
        assert torch.allclose(m, values(2.0, 5.0, 2.0))


class TestRepLayer:
    def test_rep_layer_in_dsl(self, line_ctx):
        zero_field = field_zeros(line_ctx)
        rep_layer = RepLayer(zero_field, lambda s: s + 1, name="counter")
        results = []
        for _ in range(COMPOSITION_ROUNDS):
            with line_ctx.round():
                val = rep_layer(zero_field)
            results.append(val[0].item())
        assert results == [1.0, 2.0, 3.0, 4.0, 5.0]


class TestBranchLayer:
    def test_explicit_ctx_isolates_partitions(self, line_ctx):
        cond = flags(True, True, False, False)
        x = field_from_values(line_ctx, [1.0, 2.0, 30.0, 10.0])
        layer = BranchLayer(
            FoldhoodLayer(aggr="sum"),
            FoldhoodLayer(aggr="max"),
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
    def test_nbr_layer_in_dsl(self, triangle_ctx):
        nbr_layer = FoldhoodLayer(aggr="sum")
        x = field_from_values(triangle_ctx, [1.0, 2.0, 3.0])
        with triangle_ctx.round():
            m = nbr_layer(x)
        assert torch.allclose(m, values(5.0, 4.0, 3.0))

    def test_layers_compose_with_dsl(self, triangle_ctx):
        nbr_layer = FoldhoodLayer(aggr="min", fill_value=float("inf"))
        x = field_from_values(triangle_ctx, [10.0, 2.0, 5.0])
        cond = field_from_values(triangle_ctx, [1.0, 0.0, 1.0])
        with triangle_ctx.round():
            messages = nbr_layer(x)
            result = mux(cond, messages, field.zeros())
        assert torch.allclose(result, values(2.0, 0.0, 2.0))

    def test_gradient_with_layers(self, line_ctx):
        nbr_min = FoldhoodLayer(aggr="min")
        source = field_with_overrides(line_ctx, ((0, 1.0),))

        for _ in range(COMPOSITION_ROUNDS):
            with line_ctx.round():
                d = rep(
                    field.inf(),
                    lambda s: mux(source, field.zeros(), nbr_min(s + 1)),
                )
        assert torch.allclose(d, values(0.0, 1.0, 2.0, 3.0))
