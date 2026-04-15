"""Tests for differentiability of aggregate operations."""

from __future__ import annotations

import torch

from autofield import (
    AggregateContext,
    collect_cast,
    gradient,
    gradient_cast,
    mux,
    nbr,
    nbr_range,
    rep,
    minhood,
    sumhood,
)
from autofield.dsl import field
from autofield.utils import make_grid_graph
from conftest import (
    assert_finite_gradients,
    field_zeros,
    field_with_overrides,
)
from tests.aggregate.support import GRADIENT_ROUNDS, ROUNDS, flags, values


class TestRepDifferentiability:
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


class TestGradientCastDifferentiability:
    def test_backward_through_accumulation_parameter(self, line_ctx):
        source = field_with_overrides(line_ctx, ((0, 1.0),))
        step = torch.tensor(1.0, requires_grad=True)

        for _ in range(ROUNDS):
            with line_ctx.round():
                output = gradient_cast(
                    source,
                    field.zeros(),
                    lambda value: value + step,
                    name="backprop",
                )

        loss = output.sum()
        loss.backward()

        assert_finite_gradients([step])
        assert step.grad.item() != 0.0


class TestCollectCastDifferentiability:
    def test_backward_to_local_payloads(self, line_ctx):
        potential = torch.arange(line_ctx.num_nodes, dtype=torch.float32)
        local = torch.ones(line_ctx.num_nodes, requires_grad=True)

        for _ in range(ROUNDS):
            with line_ctx.round():
                output = collect_cast(
                    potential,
                    local,
                    field.zeros(),
                    lambda acc, value: acc + value,
                    name="local_grad",
                )

        output[0].backward()

        assert_finite_gradients([local])
        assert torch.allclose(local.grad, torch.ones(line_ctx.num_nodes))

    def test_soft_backpropagates_through_potential(self, line_ctx):
        potential = torch.arange(
            line_ctx.num_nodes, dtype=torch.float32, requires_grad=True
        )

        for _ in range(ROUNDS):
            with line_ctx.round():
                output = collect_cast(
                    potential,
                    torch.ones(line_ctx.num_nodes),
                    field.zeros(),
                    lambda acc, value: acc + value,
                    name="soft_potential",
                    mode="soft",
                    tau=1.0,
                )

        output[0].backward()

        assert_finite_gradients([potential])
        assert potential.grad.abs().sum().item() > 0.0


class TestNbrRangeDifferentiability:
    def test_nbr_range_is_differentiable_with_edge_weight_tensor(
        self, triangle_topology
    ):
        edge_index, _ = triangle_topology
        edge_weight = torch.tensor(
            [2.0, 2.0, 2.0, 2.0, 10.0, 10.0], requires_grad=True
        )
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


class TestGradientDifferentiability:
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
