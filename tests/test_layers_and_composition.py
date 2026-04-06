"""Tests for layer APIs and higher-order composition patterns."""

from __future__ import annotations

import torch
import torch.nn as nn

from autofield import AggregateContext, branch, mux, nbr, rep
from autofield.layers import BranchLayer, MuxLayer, NbrLayer, RepLayer


class AddConstant(nn.Module):
    def __init__(self, value: float):
        super().__init__()
        self.value = value

    def forward(self, x: torch.Tensor, ctx=None) -> torch.Tensor:
        return x + self.value


class TestLayersInDSL:
    def test_nbr_layer_in_dsl(self, triangle_ctx):
        nbr_layer = NbrLayer(aggr="sum")
        x = torch.tensor([1.0, 2.0, 3.0])
        with triangle_ctx.round():
            m = nbr_layer(x)
        assert torch.allclose(m, torch.tensor([5.0, 4.0, 3.0]))

    def test_rep_layer_in_dsl(self, line_ctx):
        n = line_ctx._ctx.num_nodes
        rep_layer = RepLayer("counter", 0.0, lambda s: s + 1)
        results = []
        for _ in range(3):
            with line_ctx.round():
                val = rep_layer(torch.zeros(n))
            results.append(val[0].item())
        assert results == [1.0, 2.0, 3.0]

    def test_layers_compose_with_dsl(self, triangle_ctx):
        n = triangle_ctx._ctx.num_nodes
        nbr_layer = NbrLayer(aggr="min", fill_value=float("inf"))
        x = torch.tensor([10.0, 2.0, 5.0])
        with triangle_ctx.round():
            messages = nbr_layer(x)
            result = mux(
                torch.tensor([1.0, 0.0, 1.0]),
                messages,
                torch.zeros(n),
            )
        assert torch.allclose(result, torch.tensor([2.0, 0.0, 2.0]))

    def test_gradient_with_layers(self, line_ctx):
        n = line_ctx._ctx.num_nodes
        source = torch.tensor([1.0, 0.0, 0.0, 0.0])
        nbr_min = NbrLayer(aggr="min")

        for _ in range(4):
            with line_ctx.round():
                d = rep(
                    "d",
                    float("inf"),
                    lambda s: mux(source, torch.zeros(n), nbr_min(s + 1)),
                )
        assert torch.allclose(d, torch.tensor([0.0, 1.0, 2.0, 3.0]))


class TestStandaloneLayers:
    def test_branch_layer_explicit_ctx_isolates_partitions(self, line_ctx):
        cond = torch.tensor([True, True, False, False])
        x = torch.tensor([1.0, 2.0, 30.0, 10.0])
        layer = BranchLayer(
            NbrLayer(aggr="sum"), NbrLayer(aggr="max"), branch_name="standalone"
        )

        with line_ctx.round() as round_ctx:
            result = layer(x, cond, ctx=round_ctx)

        assert torch.allclose(result, torch.tensor([2.0, 1.0, 10.0, 30.0]))

    def test_mux_layer_explicit_ctx_selects_outputs(self, line_ctx):
        cond = torch.tensor([1.0, 1.0, 0.0, 0.0])
        x = torch.tensor([1.0, 2.0, 3.0, 4.0])
        layer = MuxLayer(AddConstant(1.0), AddConstant(100.0))

        with line_ctx.round() as round_ctx:
            result = layer(x, cond, ctx=round_ctx)

        assert torch.allclose(result, torch.tensor([2.0, 3.0, 103.0, 104.0]))


class TestComposition:
    def test_mux_rep_branch_composition(self, line_ctx):
        n = line_ctx._ctx.num_nodes

        cond_mux = torch.tensor([True, True, False, False])
        cond_branch = torch.tensor([True, True, True, False])

        results = []
        for _ in range(3):
            with line_ctx.round():
                val = mux(
                    cond_mux,
                    rep(
                        "outer_rep",
                        torch.zeros(n),
                        lambda outer_s: branch(
                            cond_branch,
                            lambda: nbr(outer_s, aggr="sum") + 1.0,
                            lambda: nbr(outer_s, aggr="max") + 10.0,
                        ),
                    ),
                    rep(
                        "other_rep",
                        torch.zeros(n),
                        lambda s: nbr(s, aggr="sum") + 100.0,
                    ),
                )
            results.append(val.clone())

        assert torch.allclose(results[0], torch.tensor([1.0, 1.0, 100.0, 100.0]))
        assert torch.allclose(results[1], torch.tensor([2.0, 3.0, 300.0, 200.0]))
        assert torch.allclose(results[2], torch.tensor([4.0, 5.0, 600.0, 400.0]))

    def test_rep_branch_rep_composition(self, line_ctx):
        n = line_ctx._ctx.num_nodes
        cond = torch.tensor([True, True, False, False])

        results = []
        for _ in range(3):
            with line_ctx.round():
                val = rep(
                    "outer",
                    torch.zeros(n),
                    lambda outer_s: branch(
                        cond,
                        lambda: rep(
                            "inner_true",
                            torch.zeros(n),
                            lambda inner_s: (
                                nbr(outer_s, aggr="sum")
                                + nbr(inner_s, aggr="sum")
                                + 1.0
                            ),
                        ),
                        lambda: rep(
                            "inner_false",
                            torch.zeros(n),
                            lambda inner_s: (
                                nbr(outer_s, aggr="max")
                                + nbr(inner_s, aggr="max")
                                + 10.0
                            ),
                        ),
                    ),
                )
            results.append(val.clone())

        assert torch.allclose(results[0], torch.tensor([1.0, 1.0, 10.0, 10.0]))
        assert torch.allclose(results[1], torch.tensor([3.0, 3.0, 30.0, 30.0]))
        assert torch.allclose(results[2], torch.tensor([7.0, 7.0, 70.0, 70.0]))

    def test_multiple_assignments_composition(self, line_ctx):
        n = line_ctx._ctx.num_nodes
        cond = torch.tensor([True, True, False, False])

        results_x = []
        results_y = []
        results_z = []

        for _ in range(3):
            with line_ctx.round():
                x = rep("x_rep", torch.zeros(n), lambda s: s + 1.0)
                y = branch(
                    cond,
                    lambda: rep(
                        "y_true", torch.zeros(n), lambda s: nbr(s, aggr="sum") + x
                    ),
                    lambda: rep(
                        "y_false", torch.zeros(n), lambda s: nbr(s, aggr="max") + x * 2
                    ),
                )
                z = rep("z_rep", torch.zeros(n), lambda s: nbr(s + y, aggr="sum"))

            results_x.append(x.clone())
            results_y.append(y.clone())
            results_z.append(z.clone())

        assert torch.allclose(results_x[0], torch.tensor([1.0, 1.0, 1.0, 1.0]))
        assert torch.allclose(results_y[0], torch.tensor([1.0, 1.0, 2.0, 2.0]))
        assert torch.allclose(results_z[0], torch.tensor([1.0, 3.0, 3.0, 2.0]))
        assert torch.allclose(results_x[1], torch.tensor([2.0, 2.0, 2.0, 2.0]))
        assert torch.allclose(results_y[1], torch.tensor([3.0, 3.0, 6.0, 6.0]))
        assert torch.allclose(results_z[1], torch.tensor([6.0, 13.0, 14.0, 9.0]))
        assert torch.allclose(results_x[2], torch.tensor([3.0, 3.0, 3.0, 3.0]))
        assert torch.allclose(results_y[2], torch.tensor([6.0, 6.0, 12.0, 12.0]))
        assert torch.allclose(results_z[2], torch.tensor([19.0, 38.0, 40.0, 26.0]))
