"""Tests for higher-order aggregate building blocks."""

import torch

from autofield import AggregateContext, collect_cast, gradient_cast, nbr_range
from autofield.dsl import field
from conftest import (
    assert_finite_gradients,
    field_from_values,
    field_mid,
    field_ones,
    field_with_overrides,
)

ROUNDS = 4
SOFT_MATCH_TAU = 0.05
SOFT_MATCH_ATOL = 1e-3
LINE_SOURCE_OVERRIDES = ((0, 1.0),)
WEIGHTED_SOURCE_OVERRIDES = ((0, 1.0), (3, 1.0))


def values(*items: float) -> torch.Tensor:
    return torch.tensor(items, dtype=torch.float32)


class TestGradientCast:
    def test_hop_count_on_line(self, line_ctx):
        source = field_with_overrides(line_ctx, LINE_SOURCE_OVERRIDES)

        for _ in range(ROUNDS):
            with line_ctx.round():
                output = gradient_cast(
                    source,
                    field.zeros(),
                    lambda value: value + 1.0,
                    name="hop_count",
                )

        assert torch.allclose(output, values(0.0, 1.0, 2.0, 3.0))

    def test_payload_propagates_from_source(self, line_ctx):
        source = field_with_overrides(line_ctx, LINE_SOURCE_OVERRIDES)
        center = field_from_values(line_ctx, [10.0, 20.0, 30.0, 40.0])

        for _ in range(ROUNDS):
            with line_ctx.round():
                output = gradient_cast(
                    source,
                    center,
                    lambda value: value + 1.0,
                    name="payload",
                )

        assert torch.allclose(output, values(10.0, 11.0, 12.0, 13.0))

    def test_backward_through_accumulation_parameter(self, line_ctx):
        source = field_with_overrides(line_ctx, LINE_SOURCE_OVERRIDES)
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

    def test_soft_matches_hard_on_line(self, line_topology):
        edge_index, n = line_topology

        hard_ctx = AggregateContext(edge_index, n)
        source = field_with_overrides(hard_ctx, LINE_SOURCE_OVERRIDES)
        for _ in range(ROUNDS):
            with hard_ctx.round():
                hard = gradient_cast(
                    source,
                    field.zeros(),
                    lambda value: value + 1.0,
                    name="hard",
                )

        soft_ctx = AggregateContext(edge_index, n)
        soft_source = field_with_overrides(soft_ctx, LINE_SOURCE_OVERRIDES)
        for _ in range(ROUNDS):
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
        source = field_with_overrides(ctx, WEIGHTED_SOURCE_OVERRIDES)
        center = field_from_values(ctx, [10.0, -1.0, -1.0, 20.0])

        for _ in range(ROUNDS):
            with ctx.round():
                output = gradient_cast(
                    source,
                    center,
                    lambda value: value,
                    weight=nbr_range(),
                    name="weighted_multi_source",
                )

        assert torch.allclose(output, values(10.0, 10.0, 20.0, 20.0))


class TestCollectCast:
    def test_collects_subtree_sizes_on_line(self, line_ctx):
        potential = field_mid(line_ctx)
        local = field_ones(line_ctx)

        for _ in range(ROUNDS):
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
        local = field_ones(line_ctx)

        for _ in range(ROUNDS):
            with line_ctx.round():
                output = collect_cast(
                    potential,
                    local,
                    field.zeros(),
                    lambda acc, value: acc + value,
                    name="roots",
                )

        assert torch.allclose(output, field_ones(line_ctx))

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
        assert torch.allclose(local.grad, field_ones(line_ctx))

    def test_soft_backpropagates_through_potential(self, line_ctx):
        potential = torch.arange(
            line_ctx.num_nodes, dtype=torch.float32, requires_grad=True
        )

        for _ in range(ROUNDS):
            with line_ctx.round():
                output = collect_cast(
                    potential,
                    field.ones(),
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

        for _ in range(ROUNDS):
            with ctx.round():
                output = collect_cast(
                    potential,
                    local,
                    field.zeros(),
                    torch.add,
                    name="weighted_sizes",
                )

        assert torch.allclose(output, values(31.0, 10.0, 21.0, 1.0))
