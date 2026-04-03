"""Tests for higher-order aggregate building blocks."""

import sys

sys.path.insert(0, "src")

import torch

from aggregate_gnn import AggregateContext, collect_cast, gradient_cast


def line_graph():
    edge_index = torch.tensor(
        [
            [0, 1, 1, 2, 2, 3],
            [1, 0, 2, 1, 3, 2],
        ],
        dtype=torch.long,
    )
    return edge_index, 4


class TestGradientCast:
    def test_hop_count_on_line(self):
        edge_index, n = line_graph()
        source = torch.tensor([1.0, 0.0, 0.0, 0.0])
        ctx = AggregateContext(edge_index, n)

        for _ in range(4):
            with ctx.round():
                output = gradient_cast(source, 0.0, lambda value: value + 1.0, name="hop_count")

        assert torch.allclose(output, torch.tensor([0.0, 1.0, 2.0, 3.0]))

    def test_payload_propagates_from_source(self):
        edge_index, n = line_graph()
        source = torch.tensor([1.0, 0.0, 0.0, 0.0])
        center = torch.tensor([10.0, 20.0, 30.0, 40.0])
        ctx = AggregateContext(edge_index, n)

        for _ in range(4):
            with ctx.round():
                output = gradient_cast(source, center, lambda value: value + 1.0, name="payload")

        assert torch.allclose(output, torch.tensor([10.0, 11.0, 12.0, 13.0]))

    def test_backward_through_accumulation_parameter(self):
        edge_index, n = line_graph()
        source = torch.tensor([1.0, 0.0, 0.0, 0.0])
        step = torch.tensor(1.0, requires_grad=True)
        ctx = AggregateContext(edge_index, n)

        for _ in range(4):
            with ctx.round():
                output = gradient_cast(source, 0.0, lambda value: value + step, name="backprop")

        loss = output.sum()
        loss.backward()

        assert step.grad is not None
        assert torch.isfinite(step.grad)
        assert step.grad.item() != 0.0

    def test_soft_matches_hard_on_line(self):
        edge_index, n = line_graph()
        source = torch.tensor([1.0, 0.0, 0.0, 0.0])

        hard_ctx = AggregateContext(edge_index, n)
        for _ in range(4):
            with hard_ctx.round():
                hard = gradient_cast(source, 0.0, lambda value: value + 1.0, name="hard")

        soft_ctx = AggregateContext(edge_index, n)
        for _ in range(4):
            with soft_ctx.round():
                soft = gradient_cast(
                    source,
                    0.0,
                    lambda value: value + 1.0,
                    name="soft",
                    mode="soft",
                    tau=0.05,
                )

        assert torch.allclose(soft, hard, atol=1e-3)


class TestCollectCast:
    def test_collects_subtree_sizes_on_line(self):
        edge_index, n = line_graph()
        potential = torch.arange(n, dtype=torch.float32)
        local = torch.ones(n)
        ctx = AggregateContext(edge_index, n)

        for _ in range(4):
            with ctx.round():
                output = collect_cast(potential, local, 0.0, lambda acc, value: acc + value, name="sizes")

        assert torch.allclose(output, torch.tensor([4.0, 3.0, 2.0, 1.0]))

    def test_equal_potentials_leave_nodes_as_roots(self):
        edge_index, n = line_graph()
        potential = torch.zeros(n)
        local = torch.ones(n)
        ctx = AggregateContext(edge_index, n)

        for _ in range(4):
            with ctx.round():
                output = collect_cast(potential, local, 0.0, lambda acc, value: acc + value, name="roots")

        assert torch.allclose(output, torch.ones(n))

    def test_backward_to_local_payloads(self):
        edge_index, n = line_graph()
        potential = torch.arange(n, dtype=torch.float32)
        local = torch.ones(n, requires_grad=True)
        ctx = AggregateContext(edge_index, n)

        for _ in range(4):
            with ctx.round():
                output = collect_cast(potential, local, 0.0, lambda acc, value: acc + value, name="local_grad")

        output[0].backward()

        assert local.grad is not None
        assert torch.isfinite(local.grad).all()
        assert torch.allclose(local.grad, torch.ones(n))

    def test_soft_backpropagates_through_potential(self):
        edge_index, n = line_graph()
        potential = torch.arange(n, dtype=torch.float32, requires_grad=True)
        local = torch.ones(n)
        ctx = AggregateContext(edge_index, n)

        for _ in range(4):
            with ctx.round():
                output = collect_cast(
                    potential,
                    local,
                    0.0,
                    lambda acc, value: acc + value,
                    name="soft_potential",
                    mode="soft",
                    tau=1.0,
                )

        output[0].backward()

        assert potential.grad is not None
        assert torch.isfinite(potential.grad).all()
        assert potential.grad.abs().sum().item() > 0.0
