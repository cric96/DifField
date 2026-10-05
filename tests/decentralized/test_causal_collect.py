"""Causal collection is a protocol, including its transient and gradients."""

from types import SimpleNamespace

import pytest
import torch

from diffield import AggregateContext, collect_cast, gradient
from diffield.decentralized import run_decentralized


def graph(pairs):
    return torch.tensor(sorted({edge for a, b in pairs for edge in [(a, b), (b, a)]})).T


@pytest.mark.parametrize(
    "pairs,n",
    [
        ([(0, 1), (1, 2), (2, 3)], 4),
        ([(0, 1), (0, 2), (1, 3), (2, 3)], 4),
        (
            [
                (i, j)
                for i in range(9)
                for j in range(i + 1, 9)
                if (j == i + 3 or (j == i + 1 and i // 3 == j // 3))
            ],
            9,
        ),
    ],
)
def test_vector_conservation_and_device_equivalence(pairs, n):
    pytest.importorskip("mesa")
    edges = graph(pairs)
    signals = {
        "source": torch.arange(n) == 0,
        "payload": torch.stack([torch.arange(n).float(), torch.ones(n)], -1),
    }

    def program(runtime):
        return collect_cast(
            gradient(runtime.signals["source"], mode="hard"),
            runtime.signals["payload"],
            torch.tensor(0.0),
            torch.add,
            causal=True,
            mode="hard",
        )

    ctx = AggregateContext(edges, n)
    central = []
    for _ in range(n * 4):
        with ctx.round():
            central.append(program(SimpleNamespace(signals=signals)))
    scenario = SimpleNamespace(edge_index=edges, num_nodes=n, self_loops=False)
    local = run_decentralized(scenario=scenario, program=program, signals=signals, rounds=n * 4)
    torch.testing.assert_close(torch.stack(central), torch.stack(local.fields), rtol=0, atol=0)
    torch.testing.assert_close(central[0], signals["payload"])
    torch.testing.assert_close(central[-1][0], signals["payload"].sum(0))


def test_ties_use_stable_ids_not_edge_order():
    edges = graph([(0, 1), (0, 2), (1, 3), (2, 3)])
    for ordered in [edges, edges.flip(1)]:
        ctx = AggregateContext(ordered, 4)
        for _ in range(4):
            with ctx.round() as runtime:
                runtime.node_ids = torch.tensor([50.0, 30.0, 10.0, 40.0])
                out = collect_cast(
                    torch.tensor([0.0, 1.0, 1.0, 2.0]),
                    torch.ones(4),
                    torch.tensor(0.0),
                    torch.add,
                    causal=True,
                    mode="hard",
                )
        torch.testing.assert_close(out, torch.tensor([4.0, 1.0, 2.0, 1.0]))
        (state,) = [ctx.state._states[k] for k in ctx.state.keys() if "converge_cast" in k]  # noqa: SIM118
        assert state[3, 1] == 10  # published parent: the lowest id among equal paths


def test_parent_change_partition_and_rejoin_preserve_contributions():
    edges = graph([(0, 1), (0, 2), (1, 3), (2, 3)])
    ctx = AggregateContext(edges, 4)
    for links, root in [
        (edges, 0),
        (graph([(0, 2), (2, 3), (3, 1)]), 0),
        (graph([(0, 2), (1, 3)]), 0),
        (edges, 3),
    ]:
        ctx.update_topology(links)
        for _ in range(24):
            with ctx.round():
                out = collect_cast(
                    gradient(torch.arange(4) == root, mode="hard"),
                    torch.ones(4),
                    torch.tensor(0.0),
                    torch.add,
                    causal=True,
                    mode="hard",
                )
        expected = 2 if links.shape[1] == 4 else 4
        assert out[root] == expected
    assert ctx.round_num == 96


def test_remote_loss_gradient_through_multiple_rounds():
    edges = graph([(0, 1), (1, 2), (2, 3)])

    def loss(coefficient):
        ctx = AggregateContext(edges, 4)
        local = torch.tensor([0.0, 0.0, 0.0, 1.0], dtype=torch.double) * coefficient
        for _ in range(7):
            with ctx.round():
                out = collect_cast(
                    torch.arange(4, dtype=torch.double),
                    local,
                    torch.tensor(0.0),
                    torch.add,
                    causal=True,
                    mode="hard",
                )
        return out[0].square()

    theta = torch.tensor(0.7, dtype=torch.double, requires_grad=True)
    assert torch.autograd.gradcheck(loss, (theta,))
    assert torch.autograd.grad(loss(theta), theta)[0].abs() > 0.1


def test_causal_soft_keeps_the_hard_tree_and_differentiates_the_parent_gate():
    potential = torch.tensor([0.0, 1.0, 1.2, 2.5], requires_grad=True)
    edges = graph([(0, 1), (0, 2), (1, 3), (2, 3)])
    runs = {}
    for mode in ("hard", "soft"):
        ctx = AggregateContext(edges, 4)
        for _ in range(5):
            with ctx.round():
                runs[mode] = collect_cast(
                    potential,
                    torch.ones(4),
                    torch.tensor(0.0),
                    torch.add,
                    causal=True,
                    mode=mode,
                    tau=0.5,
                )
    assert runs["hard"][0] == 4  # every device reaches the root once
    assert runs["soft"][0] < 4  # node 3 sends the probability of its chosen parent
    (grad,) = torch.autograd.grad(runs["soft"][0], potential)
    assert torch.isfinite(grad).all() and grad.abs().sum() > 0
