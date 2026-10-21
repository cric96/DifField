"""Tests for the S-block and descent building blocks: elect, descend, nbr_count."""

import torch

from diffield import AggregateContext
from diffield.dsl import (
    ELECTION_NONE,
    as_scatter_expr,
    descend,
    elect,
    gradient,
    has_neighbors,
    nbr_count,
)
from tests.aggregate.support import flags, values


def chain_topology(n: int) -> torch.Tensor:
    """Bidirectional chain 0-1-...-(n-1)."""
    src = torch.cat([torch.arange(n - 1), torch.arange(1, n)])
    tgt = torch.cat([torch.arange(1, n), torch.arange(n - 1)])
    return torch.stack([src, tgt])


HOP = as_scatter_expr(1.0)


class TestElect:
    def test_unique_leader_on_chain(self):
        n = 5
        ctx = AggregateContext(chain_topology(n), n)
        key = values(3.0, 1.0, 4.0, 0.5, 2.0)
        eligible = flags(True, True, False, True, True)

        leader = lead = None
        for _ in range(2 * n):
            with ctx.round():
                # grain > 2 * eccentricity so nobody falls in the buffer zone
                leader, lead = elect(key, eligible, grain=8.0, weight=HOP, name="t")

        assert leader is not None and lead is not None
        assert leader.tolist() == [False, False, False, True, False]
        assert torch.allclose(lead, torch.full((n,), 0.5))

    def test_all_nodes_candidate_when_eligible_none(self):
        n = 4
        ctx = AggregateContext(chain_topology(n), n)
        key = values(2.0, 3.0, 1.0, 4.0)

        leader = lead = None
        for _ in range(2 * n):
            with ctx.round():
                leader, lead = elect(key, grain=8.0, weight=HOP, name="t")

        assert leader is not None and lead is not None
        assert leader.tolist() == [False, False, True, False]
        assert torch.allclose(lead, torch.full((n,), 1.0))

    def test_reelection_after_leader_death(self):
        n = 5
        grain = 8.0
        ctx = AggregateContext(chain_topology(n), n)
        key = values(3.0, 1.0, 4.0, 0.5, 2.0)
        eligible = flags(True, True, False, True, True)

        for _ in range(2 * n):
            with ctx.round():
                elect(key, eligible, grain=grain, weight=HOP, name="t")

        # node 3 abdicates: its gradient field, no longer sourced, rises one
        # hop per round; nodes fall through the buffer zone, re-candidate
        # past `grain`, and the next-best key (node 1) floods back.
        without_leader = flags(True, True, False, False, True)
        leader = lead = None
        for _ in range(2 * int(grain) + 2 * n):
            with ctx.round():
                leader, lead = elect(
                    key, without_leader, grain=grain, weight=HOP, name="t"
                )

        assert leader is not None and lead is not None
        assert leader.tolist() == [False, True, False, False, False]
        assert torch.allclose(lead, torch.full((n,), 1.0))

    def test_grain_partitions_far_regions(self):
        # two candidates at the ends of a long chain with a small grain: each
        # end keeps its own leadership, everything past the half-grain disc
        # (including the middle) follows nobody.
        n = 7
        ctx = AggregateContext(chain_topology(n), n)
        key = values(1.0, 10.0, 10.0, 10.0, 10.0, 10.0, 2.0)
        eligible = flags(True, False, False, False, False, False, True)

        leader = lead = None
        for _ in range(2 * n):
            with ctx.round():
                leader, lead = elect(key, eligible, grain=2.0, weight=HOP, name="t")

        assert leader is not None and lead is not None
        assert leader.tolist() == [True, False, False, False, False, False, True]
        assert lead[3].item() >= ELECTION_NONE


class TestDescend:
    def test_points_downhill_on_chain(self):
        n = 5
        ctx = AggregateContext(chain_topology(n), n)
        potential = values(3.0, 2.0, 1.0, 0.0, 1.0)
        pos = torch.tensor([[float(i), 0.0] for i in range(n)])

        with ctx.round():
            direction = descend(potential, pos, tau=0.1)

        # nodes 0..2 move +x (toward the minimum at node 3), node 4 moves -x
        assert torch.all(direction[:3, 0] > 0.9)
        assert direction[4, 0] < -0.9
        assert abs(direction[3, 0]) < 0.1  # symmetric neighbours at the minimum

    def test_infinite_potential_gets_zero_weight(self):
        n = 3
        ctx = AggregateContext(chain_topology(n), n)
        potential = values(float("inf"), 0.0, float("inf"))
        pos = torch.tensor([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]])

        with ctx.round():
            direction = descend(potential, pos, tau=0.1)

        # node 1's neighbours are both unreachable -> zero vector
        assert torch.allclose(direction[1], torch.zeros(2))
        # ends still move toward the finite minimum at node 1
        assert direction[0, 0] > 0.9
        assert direction[2, 0] < -0.9

    def test_differentiable_through_toward(self):
        n = 4
        ctx = AggregateContext(chain_topology(n), n)
        potential = values(3.0, 2.0, 1.0, 0.0)
        pos = torch.tensor(
            [[float(i), 0.0] for i in range(n)], requires_grad=True
        )

        with ctx.round():
            direction = descend(potential.detach(), pos, tau=0.1)
        direction.sum().backward()

        assert pos.grad is not None
        assert torch.isfinite(pos.grad).all()
        assert float(pos.grad.abs().sum()) > 0.0


class TestNbrCount:
    def test_degrees_on_chain(self):
        n = 4
        ctx = AggregateContext(chain_topology(n), n)
        with ctx.round():
            deg = nbr_count()
        assert deg.tolist() == [1.0, 2.0, 2.0, 1.0]

    def test_isolated_node(self):
        # edge only between 0 and 1; node 2 isolated
        edge_index = torch.tensor([[0, 1], [1, 0]])
        ctx = AggregateContext(edge_index, 3)
        with ctx.round():
            deg = nbr_count()
            connected = has_neighbors()
        assert deg.tolist() == [1.0, 1.0, 0.0]
        assert connected.tolist() == [True, True, False]


class TestUpdateTopology:
    def test_state_persists_across_topology_swap(self):
        n = 5
        ctx = AggregateContext(chain_topology(n), n)
        key = values(3.0, 1.0, 4.0, 0.5, 2.0)
        eligible = flags(True, True, False, True, True)

        leader = lead = None
        for _ in range(2 * n):
            with ctx.round():
                leader, lead = elect(key, eligible, grain=8.0, weight=HOP, name="t")
        assert leader is not None
        assert leader.tolist() == [False, False, False, True, False]

        # drop the 3-4 link: the network partitions and each component elects
        # its own leader — node 4, now isolated, sees an unreachable leader
        # (rising gradient) and re-candidates itself within a couple rounds,
        # while the main component keeps node 3.
        src = torch.tensor([0, 1, 1, 2, 2, 3])
        tgt = torch.tensor([1, 0, 2, 1, 3, 2])
        ctx.update_topology(torch.stack([src, tgt]))
        for _ in range(3):
            with ctx.round():
                leader, lead = elect(key, eligible, grain=8.0, weight=HOP, name="t")
        assert leader is not None and lead is not None
        assert leader.tolist() == [False, False, False, True, True]
        assert lead[0].item() == 0.5
        assert lead[4].item() == 2.0

    def test_metric_weights_from_positions(self):
        n = 3
        ctx = AggregateContext(chain_topology(n), n)
        pos = torch.tensor([[0.0, 0.0], [2.0, 0.0], [5.0, 0.0]])
        ctx.update_topology(chain_topology(n), positions=pos)

        source = torch.zeros(n)
        source[0] = 1.0
        dist = None
        for _ in range(n + 1):
            with ctx.round():
                dist = gradient(source, name="g")
        assert dist is not None
        assert torch.allclose(dist, values(0.0, 2.0, 5.0))
