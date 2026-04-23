"""Tests for higher-order aggregate building blocks: gradient, gradient_cast, broadcast, collect_cast."""

import torch

from diffield import AggregateContext
from diffield.dsl import (
    collect_cast,
    gradient,
    gradient_cast,
    broadcast,
    scatter_range,
    iterate,
    mux,
    gather_min,
    scatter,
    field,
)
from conftest import assert_finite_gradients, field_from_values, field_with_overrides
from tests.aggregate.support import (
    GRADIENT_ROUNDS,
    PROPAGATION_ROUNDS,
    LINE_SOURCE,
    WEIGHTED_SOURCES,
    SOFT_MATCH_TAU,
    SOFT_MATCH_ATOL,
    values,
)


class TestGradient:
    def test_fixed_hop_count(self):
        from diffield.utils import make_grid_graph

        rows, cols = 5, 5
        edge_index, n = make_grid_graph(rows, cols)
        w = torch.tensor(1.0)

        ctx = AggregateContext(edge_index, n)
        source = field_with_overrides(ctx, ((0, 1.0),))
        for _ in range(rows + cols):
            with ctx.round():
                d = iterate(
                    field.inf(),
                    lambda dist: mux(source, field.zeros(), gather_min(scatter(dist) + w)),
                )

        expected = torch.zeros(n, dtype=torch.float32)
        for row_idx in range(rows):
            for col_idx in range(cols):
                expected[row_idx * cols + col_idx] = float(row_idx + col_idx)

        assert torch.allclose(d, expected)

    def test_convenience_gradient_uses_edge_weight_by_default(self, triangle_topology):
        edge_index, _ = triangle_topology
        edge_weight = torch.tensor([2.0, 2.0, 2.0, 2.0, 10.0, 10.0])
        ctx = AggregateContext(edge_index, 3, edge_weight=edge_weight)
        source = field_with_overrides(ctx, ((0, 1.0),))

        for _ in range(GRADIENT_ROUNDS):
            with ctx.round():
                d = gradient(source, name="weighted")

        assert torch.allclose(d, values(0.0, 2.0, 4.0))

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
        for _ in range(GRADIENT_ROUNDS):
            with ctx.round():
                d = iterate(
                    field.inf(),
                    lambda dist: mux(source, field.zeros(), gather_min(scatter(dist) + w)),
                )
        assert torch.allclose(d, values(0.0, 1.0, 2.0))


class TestGradientCast:
    def test_hop_count_on_line(self, line_ctx):
        source = field_with_overrides(line_ctx, LINE_SOURCE)

        for _ in range(PROPAGATION_ROUNDS):
            with line_ctx.round():
                output = gradient_cast(
                    source,
                    field.zeros(),
                    lambda value: value + 1.0,
                    name="hop_count",
                )

        assert torch.allclose(output, values(0.0, 1.0, 2.0, 3.0))

    def test_payload_propagates_from_source(self, line_ctx):
        source = field_with_overrides(line_ctx, LINE_SOURCE)
        center = field_from_values(line_ctx, [10.0, 20.0, 30.0, 40.0])

        for _ in range(PROPAGATION_ROUNDS):
            with line_ctx.round():
                output = gradient_cast(
                    source,
                    center,
                    lambda value: value + 1.0,
                    name="payload",
                )

        assert torch.allclose(output, values(10.0, 11.0, 12.0, 13.0))

    def test_backward_through_accumulation_parameter(self, line_ctx):
        source = field_with_overrides(line_ctx, LINE_SOURCE)
        step = torch.tensor(1.0, requires_grad=True)

        for _ in range(PROPAGATION_ROUNDS):
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

    def test_soft_matches_hard_on_line(self, line_topology):
        edge_index, n = line_topology

        hard_ctx = AggregateContext(edge_index, n)
        source = field_with_overrides(hard_ctx, LINE_SOURCE)
        for _ in range(PROPAGATION_ROUNDS):
            with hard_ctx.round():
                hard = gradient_cast(
                    source,
                    field.zeros(),
                    lambda value: value + 1.0,
                    name="hard",
                )

        soft_ctx = AggregateContext(edge_index, n)
        soft_source = field_with_overrides(soft_ctx, LINE_SOURCE)
        for _ in range(PROPAGATION_ROUNDS):
            with soft_ctx.round():
                soft = gradient_cast(
                    soft_source,
                    field.zeros(),
                    lambda value: value + 1.0,
                    name="soft",
                    mode="soft",
                    tau=SOFT_MATCH_TAU,
                )

        assert torch.allclose(soft, hard, atol=SOFT_MATCH_ATOL)

    def test_weighted_multi_source_propagates_nearest_root_payload(
        self, weighted_collect_topology
    ):
        edge_index, edge_weight, n = weighted_collect_topology
        ctx = AggregateContext(edge_index, n, edge_weight=edge_weight)
        source = field_with_overrides(ctx, WEIGHTED_SOURCES)
        center = field_from_values(ctx, [10.0, -1.0, -1.0, 20.0])

        for _ in range(PROPAGATION_ROUNDS):
            with ctx.round():
                output = gradient_cast(
                    source,
                    center,
                    lambda value: value,
                    weight=scatter_range(),
                    name="weighted_multi_source",
                )

        assert torch.allclose(output, values(10.0, 10.0, 20.0, 20.0))


class TestBroadcast:
    def test_propagates_root_value(self, line_ctx):
        source = torch.tensor([True, False, False, False], dtype=torch.bool)
        payload = field_from_values(line_ctx, [10.0, 0.0, 0.0, 0.0])

        for _ in range(GRADIENT_ROUNDS):
            with line_ctx.round():
                out = broadcast(source, payload, name="broadcast")

        assert torch.allclose(out, values(10.0, 10.0, 10.0, 10.0))


class TestCollectCast:
    def test_collects_subtree_sizes_on_line(self, line_ctx):
        potential = field_from_values(line_ctx, [0.0, 1.0, 2.0, 3.0])
        local = torch.ones(line_ctx.num_nodes)

        for _ in range(PROPAGATION_ROUNDS):
            with line_ctx.round():
                output = collect_cast(
                    potential,
                    local,
                    field.zeros(),
                    lambda acc, value: acc + value,
                    name="sizes",
                )

        assert torch.allclose(output, values(4.0, 3.0, 2.0, 1.0))

    def test_equal_potentials_leave_nodes_as_roots(self, line_ctx):
        potential = field_from_values(line_ctx, [0.0] * line_ctx.num_nodes)
        local = torch.ones(line_ctx.num_nodes)

        for _ in range(PROPAGATION_ROUNDS):
            with line_ctx.round():
                output = collect_cast(
                    potential,
                    local,
                    field.zeros(),
                    lambda acc, value: acc + value,
                    name="roots",
                )

        assert torch.allclose(
            output, field_from_values(line_ctx, [1.0] * line_ctx.num_nodes)
        )

    def test_backward_to_local_payloads(self, line_ctx):
        potential = torch.arange(line_ctx.num_nodes, dtype=torch.float32)
        local = torch.ones(line_ctx.num_nodes, requires_grad=True)

        for _ in range(PROPAGATION_ROUNDS):
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

        for _ in range(PROPAGATION_ROUNDS):
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

    def test_weighted_collect_uses_shortest_path_parents(
        self, weighted_collect_topology
    ):
        edge_index, edge_weight, n = weighted_collect_topology
        ctx = AggregateContext(edge_index, n, edge_weight=edge_weight)
        potential = field_from_values(ctx, [0.0, 1.0, 2.0, 3.0])
        local = field_from_values(ctx, [0.0, 10.0, 20.0, 1.0])

        for _ in range(PROPAGATION_ROUNDS):
            with ctx.round():
                output = collect_cast(
                    potential,
                    local,
                    field.zeros(),
                    torch.add,
                    name="weighted_sizes",
                )

        assert torch.allclose(output, values(31.0, 10.0, 21.0, 1.0))
