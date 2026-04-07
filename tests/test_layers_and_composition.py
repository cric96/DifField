"""Tests for layer APIs and higher-order composition patterns."""

from __future__ import annotations

import torch
import torch.nn as nn
import pytest

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
        n = line_ctx.num_nodes
        rep_layer = RepLayer(0.0, lambda s: s + 1, name="counter")
        results = []
        for _ in range(3):
            with line_ctx.round():
                val = rep_layer(torch.zeros(n))
            results.append(val[0].item())
        assert results == [1.0, 2.0, 3.0]

    def test_layers_compose_with_dsl(self, triangle_ctx):
        n = triangle_ctx.num_nodes
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
        n = line_ctx.num_nodes
        source = torch.tensor([1.0, 0.0, 0.0, 0.0])
        nbr_min = NbrLayer(aggr="min")

        for _ in range(4):
            with line_ctx.round():
                d = rep(
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
        n = line_ctx.num_nodes

        cond_mux = torch.tensor([True, True, False, False])
        cond_branch = torch.tensor([True, True, True, False])

        results = []
        for _ in range(3):
            with line_ctx.round():
                val = mux(
                    cond_mux,
                    rep(
                        torch.zeros(n),
                        lambda outer_s: branch(
                            cond_branch,
                            lambda: nbr(outer_s, aggr="sum") + 1.0,
                            lambda: nbr(outer_s, aggr="max") + 10.0,
                        ),
                    ),
                    rep(
                        torch.zeros(n),
                        lambda s: nbr(s, aggr="sum") + 100.0,
                    ),
                )
            results.append(val.clone())

        assert torch.allclose(results[0], torch.tensor([1.0, 1.0, 100.0, 100.0]))
        assert torch.allclose(results[1], torch.tensor([2.0, 3.0, 300.0, 200.0]))
        assert torch.allclose(results[2], torch.tensor([4.0, 5.0, 600.0, 400.0]))

    def test_rep_branch_rep_composition(self, line_ctx):
        n = line_ctx.num_nodes
        cond = torch.tensor([True, True, False, False])

        results = []
        for _ in range(3):
            with line_ctx.round():
                val = rep(
                    torch.zeros(n),
                    lambda outer_s: branch(
                        cond,
                        lambda: rep(
                            torch.zeros(n),
                            lambda inner_s: (
                                nbr(outer_s, aggr="sum")
                                + nbr(inner_s, aggr="sum")
                                + 1.0
                            ),
                        ),
                        lambda: rep(
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
        n = line_ctx.num_nodes
        cond = torch.tensor([True, True, False, False])

        results_x = []
        results_y = []
        results_z = []

        for _ in range(3):
            with line_ctx.round():
                x = rep(torch.zeros(n), lambda s: s + 1.0)
                y = branch(
                    cond,
                    lambda: rep(torch.zeros(n), lambda s: nbr(s, aggr="sum") + x),
                    lambda: rep(torch.zeros(n), lambda s: nbr(s, aggr="max") + x * 2),
                )
                z = rep(torch.zeros(n), lambda s: nbr(s + y, aggr="sum"))

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


class TestLocalAggregateBranchNbr:
    def test_nbr_inside_branch_ignores_cross_partition_neighbors(self, line_ctx):
        n = line_ctx.num_nodes
        x = torch.tensor([10.0, 20.0, 30.0, 40.0])
        cond = torch.tensor([True, True, False, False])

        with line_ctx.round():
            result = branch(
                cond,
                lambda: nbr(x, aggr="sum"),
                lambda: nbr(x, aggr="max"),
                branch_name="isolation_test",
            )

        assert torch.allclose(result, torch.tensor([20.0, 10.0, 40.0, 30.0]))

    def test_rep_with_nbr_in_branch_isolates_accumulation(self, line_ctx):
        n = line_ctx.num_nodes
        cond = torch.tensor([True, True, False, False])

        results = []
        for _ in range(3):
            with line_ctx.round():
                val = branch(
                    cond,
                    lambda: rep(torch.zeros(n), lambda s: nbr(s, aggr="sum") + 1.0),
                    lambda: rep(torch.zeros(n), lambda s: nbr(s, aggr="sum") + 10.0),
                    branch_name="rep_nbr_iso",
                )
            results.append(val.clone())

        assert torch.allclose(results[0], torch.tensor([1.0, 1.0, 10.0, 10.0]))
        assert torch.allclose(results[1], torch.tensor([2.0, 2.0, 20.0, 20.0]))
        assert torch.allclose(results[2], torch.tensor([3.0, 3.0, 30.0, 30.0]))

    def test_full_local_aggregate_round_pattern(self, line_ctx):
        n = line_ctx.num_nodes
        cond = torch.tensor([True, True, False, False])
        local_input = torch.tensor([1.0, 2.0, 3.0, 4.0])

        results = []
        for _ in range(3):
            with line_ctx.round():
                val = rep(
                    torch.zeros(n),
                    lambda s: branch(
                        cond,
                        lambda: nbr(s, aggr="sum") + local_input,
                        lambda: (
                            nbr(s, aggr="max", fill_value=float("-inf"))
                            + local_input * 2
                        ),
                        branch_name="local_round",
                    ),
                )
            results.append(val.clone())

        assert torch.allclose(results[0], torch.tensor([1.0, 2.0, 6.0, 8.0]))
        assert torch.allclose(results[1], torch.tensor([3.0, 3.0, 14.0, 14.0]))
        assert torch.allclose(results[2], torch.tensor([4.0, 5.0, 20.0, 22.0]))

    def test_soft_branch_nbr_approximates_hard_isolation(self, line_topology):
        edge_index, n = line_topology
        x = torch.tensor([10.0, 20.0, 30.0, 40.0])
        cond = torch.tensor([True, True, False, False])

        hard_ctx = AggregateContext(edge_index, n)
        with hard_ctx.round():
            hard = branch(
                cond,
                lambda: nbr(x, aggr="sum"),
                lambda: nbr(x, aggr="max"),
                branch_name="soft_vs_hard",
                mode="hard",
            )

        soft_ctx = AggregateContext(edge_index, n)
        with soft_ctx.round():
            soft = branch(
                cond,
                lambda: nbr(x, aggr="sum"),
                lambda: nbr(x, aggr="max"),
                branch_name="soft_vs_hard",
                mode="soft",
                tau=0.05,
            )

        assert soft.shape == hard.shape
        assert torch.isfinite(soft).all()
        assert (soft - hard).abs().max() < 25.0

    def test_branch_state_reset_on_partition_switch(self, line_ctx):
        n = line_ctx.num_nodes
        cond1 = torch.tensor([True, True, False, False])
        cond2 = torch.tensor([True, False, False, False])

        with line_ctx.round():
            branch(
                cond1,
                lambda: rep(torch.zeros(n), lambda s: s + 5.0, name="reset_rep"),
                lambda: rep(torch.zeros(n), lambda s: s + 1.0, name="reset_rep"),
                branch_name="switch_test",
                reset_states={"reset_rep": 0.0},
            )

        with line_ctx.round():
            result = branch(
                cond2,
                lambda: rep(torch.zeros(n), lambda s: s + 5.0, name="reset_rep"),
                lambda: rep(torch.zeros(n), lambda s: s + 1.0, name="reset_rep"),
                branch_name="switch_test",
                reset_states={"reset_rep": 0.0},
            )

        assert torch.allclose(result, torch.tensor([10.0, 1.0, 2.0, 2.0]))
