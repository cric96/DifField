"""Tests for differentiability of DSL primitives."""

from __future__ import annotations

import torch

from conftest import assert_finite_gradients, field_with_overrides
from diffield import AggregateContext
from diffield.dsl import (
    broadcast,
    collect_cast,
    field,
    gather_min,
    gradient,
    gradient_cast,
    iterate,
    mux,
    scatter,
    scatter_range,
)
from tests.aggregate.support import (
    GRADIENT_ROUNDS,
    LINE_SOURCE,
    PROPAGATION_ROUNDS,
    values,
)


class TestDifferentiability:
    def test_nested_iterate_multiple_scatter_differentiability(self):
        edge_index = torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]], dtype=torch.long)
        ctx = AggregateContext(edge_index, 3)
        source = values(1.0, 0.0, 0.0)
        w = torch.tensor(1.0, requires_grad=True)

        for _ in range(GRADIENT_ROUNDS):
            with ctx.round():
                d = iterate(
                    field.inf(),
                    lambda dist: mux(source, field.zeros(), gather_min(scatter(dist) + w)),
                )

        loss = d[2]
        loss.backward()

        assert_finite_gradients([w])
        assert w.grad is not None
        assert w.grad.item() > 0

    def test_soft_gradient_differentiability(self, triangle_topology):
        edge_index, n = triangle_topology
        ctx = AggregateContext(edge_index, n)
        source = field_with_overrides(ctx, ((0, 1.0),))
        w = torch.tensor(1.5, requires_grad=True)

        for _ in range(GRADIENT_ROUNDS):
            with ctx.round():
                d = gradient(source, weight=scatter(w), mode="soft", tau=1.0)

        loss = d.sum()
        loss.backward()

        assert_finite_gradients([w])
        assert w.grad is not None
        assert w.grad.item() != 0.0

    def test_gradient_cast_differentiability(self, line_ctx):
        source = field_with_overrides(line_ctx, LINE_SOURCE)
        center = torch.ones(line_ctx.num_nodes, requires_grad=True)
        w = torch.tensor(1.0, requires_grad=True)

        for _ in range(PROPAGATION_ROUNDS):
            with line_ctx.round():
                out = gradient_cast(
                    source,
                    center,
                    lambda x: x,
                    weight=scatter(w),
                    mode="soft",
                    tau=1.0,
                )

        loss = out.sum()
        loss.backward()

        assert_finite_gradients([center, w])
        assert center.grad is not None
        assert w.grad is not None

    def test_collect_cast_differentiability(self, line_ctx):
        potential = torch.arange(line_ctx.num_nodes, dtype=torch.float32, requires_grad=True)
        local = torch.ones(line_ctx.num_nodes, requires_grad=True)

        for _ in range(PROPAGATION_ROUNDS):
            with line_ctx.round():
                out = collect_cast(
                    potential,
                    local,
                    field.zeros(),
                    torch.add,
                    mode="soft",
                    tau=1.0,
                )

        loss = out.sum()
        loss.backward()

        assert_finite_gradients([potential, local])

    def test_broadcast_differentiability(self, line_ctx):
        mask = values(1.0, 0.0, 0.0, 0.0)
        value = torch.tensor(10.0, requires_grad=True)

        for _ in range(PROPAGATION_ROUNDS):
            with line_ctx.round():
                out = broadcast(mask, value, mode="soft", tau=1.0)

        loss = out.sum()
        loss.backward()

        assert_finite_gradients([value])
        assert value.grad is not None
        assert value.grad.item() > 0.0


class TestScatterRangeDifferentiability:
    def test_scatter_range_is_differentiable_with_edge_weight_tensor(
        self, triangle_topology
    ):
        edge_index, n = triangle_topology
        edge_weight = torch.tensor(
            [1.0, 1.0, 1.0, 1.0, 1.0, 1.0], requires_grad=True
        )
        ctx = AggregateContext(edge_index, n, edge_weight=edge_weight)

        with ctx.round():
            ranges = gather_min(scatter_range())

        loss = ranges.sum()
        loss.backward()

        assert_finite_gradients([edge_weight])
        assert edge_weight.grad is not None
        assert not torch.allclose(edge_weight.grad, torch.zeros_like(edge_weight))
