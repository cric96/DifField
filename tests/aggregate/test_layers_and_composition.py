"""Tests for layer APIs and higher-order composition patterns."""

from __future__ import annotations

import torch
import torch.nn as nn
import pytest

from autofield import (
    AggregateContext,
    branch,
    mux,
    nbr,
    rep,
    foldhood,
    minhood,
    maxhood,
    sumhood,
)
from autofield.layers import BranchLayer, MuxLayer, FoldhoodLayer, RepLayer
from autofield.dsl import field
from conftest import field_from_values, field_with_overrides, field_zeros

ROUNDS = 3
SOFT_BRANCH_TAU = 0.05
SOFT_BRANCH_MAX_ERROR = 26.0


def values(*items: float) -> torch.Tensor:
    return torch.tensor(items, dtype=torch.float32)


def flags(*items: bool) -> torch.Tensor:
    return torch.tensor(items, dtype=torch.bool)


class AddConstant(nn.Module):
    def __init__(self, value: float):
        super().__init__()
        self.value = value

    def forward(self, x: torch.Tensor, ctx=None) -> torch.Tensor:
        return x + self.value


class TestLayersInDSL:
    def test_nbr_layer_in_dsl(self, triangle_ctx):
        nbr_layer = FoldhoodLayer(aggr="sum")
        x = field_from_values(triangle_ctx, [1.0, 2.0, 3.0])
        with triangle_ctx.round():
            m = nbr_layer(x)
        assert torch.allclose(m, values(5.0, 4.0, 3.0))

    def test_rep_layer_in_dsl(self, line_ctx):
        zero_field = field_zeros(line_ctx)
        rep_layer = RepLayer(zero_field, lambda s: s + 1, name="counter")
        results = []
        for _ in range(ROUNDS):
            with line_ctx.round():
                val = rep_layer(zero_field)
            results.append(val[0].item())
        assert results == [1.0, 2.0, 3.0]

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

        for _ in range(4):
            with line_ctx.round():
                d = rep(
                    field.inf(),
                    lambda s: mux(source, field.zeros(), nbr_min(s + 1)),
                )
        assert torch.allclose(d, values(0.0, 1.0, 2.0, 3.0))


class TestStandaloneLayers:
    def test_branch_layer_explicit_ctx_isolates_partitions(self, line_ctx):
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

    def test_mux_layer_explicit_ctx_selects_outputs(self, line_ctx):
        cond = values(1.0, 1.0, 0.0, 0.0)
        x = field_from_values(line_ctx, [1.0, 2.0, 3.0, 4.0])
        layer = MuxLayer(AddConstant(1.0), AddConstant(100.0))

        with line_ctx.round() as round_ctx:
            result = layer(x, cond, ctx=round_ctx)

        assert torch.allclose(result, values(2.0, 3.0, 103.0, 104.0))


class TestComposition:
    def test_mux_rep_branch_composition(self, line_ctx):
        cond_mux = flags(True, True, False, False)
        cond_branch = flags(True, True, True, False)

        results = []
        for _ in range(ROUNDS):
            with line_ctx.round():
                val = mux(
                    cond_mux,
                    rep(
                        field.zeros(),
                        lambda outer_s: branch(
                            cond_branch,
                            lambda: sumhood(nbr(outer_s)) + 1.0,
                            lambda: maxhood(nbr(outer_s)) + 10.0,
                        ),
                    ),
                    rep(
                        field.zeros(),
                        lambda s: sumhood(nbr(s)) + 100.0,
                    ),
                )
            results.append(val.clone())

        assert torch.allclose(results[0], values(1.0, 1.0, 100.0, 100.0))
        assert torch.allclose(results[1], values(2.0, 3.0, 300.0, 200.0))
        assert torch.allclose(results[2], values(4.0, 5.0, 600.0, 400.0))

    def test_rep_branch_rep_composition(self, line_ctx):
        cond = flags(True, True, False, False)

        results = []
        for _ in range(ROUNDS):
            with line_ctx.round():
                val = rep(
                    field.zeros(),
                    lambda outer_s: branch(
                        cond,
                        lambda: rep(
                            field.zeros(),
                            lambda inner_s: (
                                sumhood(nbr(outer_s)) + sumhood(nbr(inner_s)) + 1.0
                            ),
                        ),
                        lambda: rep(
                            field.zeros(),
                            lambda inner_s: (
                                maxhood(nbr(outer_s)) + maxhood(nbr(inner_s)) + 10.0
                            ),
                        ),
                    ),
                )
            results.append(val.clone())

        assert torch.allclose(results[0], values(1.0, 1.0, 10.0, 10.0))
        assert torch.allclose(results[1], values(3.0, 3.0, 30.0, 30.0))
        assert torch.allclose(results[2], values(7.0, 7.0, 70.0, 70.0))

    def test_multiple_assignments_composition(self, line_ctx):
        cond = flags(True, True, False, False)

        results_x = []
        results_y = []
        results_z = []

        for _ in range(ROUNDS):
            with line_ctx.round():
                x = rep(field.zeros(), lambda s: s + 1.0)
                y = branch(
                    cond,
                    lambda: rep(field.zeros(), lambda s: sumhood(nbr(s)) + x),
                    lambda: rep(field.zeros(), lambda s: maxhood(nbr(s)) + x * 2),
                )
                z = rep(field.zeros(), lambda s: sumhood(nbr(s + y)))

            results_x.append(x.clone())
            results_y.append(y.clone())
            results_z.append(z.clone())

        assert torch.allclose(results_x[0], values(1.0, 1.0, 1.0, 1.0))
        assert torch.allclose(results_y[0], values(1.0, 1.0, 2.0, 2.0))
        assert torch.allclose(results_z[0], values(1.0, 3.0, 3.0, 2.0))
        assert torch.allclose(results_x[1], values(2.0, 2.0, 2.0, 2.0))
        assert torch.allclose(results_y[1], values(3.0, 3.0, 6.0, 6.0))
        assert torch.allclose(results_z[1], values(6.0, 13.0, 14.0, 9.0))
        assert torch.allclose(results_x[2], values(3.0, 3.0, 3.0, 3.0))
        assert torch.allclose(results_y[2], values(6.0, 6.0, 12.0, 12.0))
        assert torch.allclose(results_z[2], values(19.0, 38.0, 40.0, 26.0))


class TestLocalAggregateBranchNbr:
    def test_nbr_inside_branch_ignores_cross_partition_neighbors(self, line_ctx):
        cond = flags(True, True, False, False)
        x = field_from_values(line_ctx, [10.0, 20.0, 30.0, 40.0])

        with line_ctx.round():
            result = branch(
                cond,
                lambda: sumhood(nbr(x)),
                lambda: maxhood(nbr(x)),
                branch_name="isolation_test",
            )

        assert torch.allclose(result, values(20.0, 10.0, 40.0, 30.0))

    def test_rep_with_nbr_in_branch_isolates_accumulation(self, line_ctx):
        cond = flags(True, True, False, False)

        results = []
        for _ in range(ROUNDS):
            with line_ctx.round():
                val = branch(
                    cond,
                    lambda: rep(field.zeros(), lambda s: sumhood(nbr(s)) + 1.0),
                    lambda: rep(field.zeros(), lambda s: sumhood(nbr(s)) + 10.0),
                    branch_name="rep_nbr_iso",
                )
            results.append(val.clone())

        assert torch.allclose(results[0], values(1.0, 1.0, 10.0, 10.0))
        assert torch.allclose(results[1], values(2.0, 2.0, 20.0, 20.0))
        assert torch.allclose(results[2], values(3.0, 3.0, 30.0, 30.0))

    def test_full_local_aggregate_round_pattern(self, line_ctx):
        cond = flags(True, True, False, False)
        local_input = field_from_values(line_ctx, [1.0, 2.0, 3.0, 4.0])

        results = []
        for _ in range(ROUNDS):
            with line_ctx.round():
                val = rep(
                    field.zeros(),
                    lambda s: branch(
                        cond,
                        lambda: sumhood(nbr(s)) + local_input,
                        lambda: (
                            maxhood(nbr(s), fill_value=float("-inf")) + local_input * 2
                        ),
                        branch_name="local_round",
                    ),
                )
            results.append(val.clone())

        assert torch.allclose(results[0], values(1.0, 2.0, 6.0, 8.0))
        assert torch.allclose(results[1], values(3.0, 3.0, 14.0, 14.0))
        assert torch.allclose(results[2], values(4.0, 5.0, 20.0, 22.0))

    def test_soft_branch_nbr_approximates_hard_isolation(self, line_topology):
        edge_index, n = line_topology
        cond = flags(True, True, False, False)

        hard_ctx = AggregateContext(edge_index, n)
        hard_x = field_from_values(hard_ctx, [10.0, 20.0, 30.0, 40.0])
        with hard_ctx.round():
            hard = branch(
                cond,
                lambda: sumhood(nbr(hard_x)),
                lambda: maxhood(nbr(hard_x)),
                branch_name="soft_vs_hard",
                mode="hard",
            )

        soft_ctx = AggregateContext(edge_index, n)
        soft_x = field_from_values(soft_ctx, [10.0, 20.0, 30.0, 40.0])
        with soft_ctx.round():
            soft = branch(
                cond,
                lambda: sumhood(nbr(soft_x)),
                lambda: maxhood(nbr(soft_x)),
                branch_name="soft_vs_hard",
                mode="soft",
                tau=SOFT_BRANCH_TAU,
            )

        assert soft.shape == hard.shape
        assert torch.isfinite(soft).all()
        assert (soft - hard).abs().max() < SOFT_BRANCH_MAX_ERROR

    def test_branch_state_reset_on_partition_switch(self, line_ctx):
        cond1 = flags(True, True, False, False)
        cond2 = flags(True, False, False, False)

        with line_ctx.round():
            branch(
                cond1,
                lambda: rep(field.zeros(), lambda s: s + 5.0, name="reset_rep"),
                lambda: rep(field.zeros(), lambda s: s + 1.0, name="reset_rep"),
                branch_name="switch_test",
                reset_states={"reset_rep": field.zeros()},
            )

        with line_ctx.round():
            result = branch(
                cond2,
                lambda: rep(field.zeros(), lambda s: s + 5.0, name="reset_rep"),
                lambda: rep(field.zeros(), lambda s: s + 1.0, name="reset_rep"),
                branch_name="switch_test",
                reset_states={"reset_rep": field.zeros()},
            )

        assert torch.allclose(result, values(10.0, 1.0, 2.0, 2.0))
