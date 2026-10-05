"""Semantics of the SCR building blocks: nbr, aligned_on, bounded_election, converge_cast."""

from operator import add

import torch

from diffield import (
    AggregateContext,
    aligned_on,
    bounded_election,
    broadcast,
    converge_cast,
    follow,
    gather_sum,
    membership,
    mid,
    nbr,
    scatter_range,
    with_mode,
)
from diffield.dsl.helpers import surrogate_probabilities


def line(n=4):
    pairs = [(i, i + 1) for i in range(n - 1)]
    return torch.tensor(sorted({e for a, b in pairs for e in [(a, b), (b, a)]})).T


def run(edges, n, program, rounds):
    ctx = AggregateContext(edges, n, edge_weight=torch.ones(edges.shape[1]))
    outputs = []
    for _ in range(rounds):
        with ctx.round():
            outputs.append(program())
    return outputs


def test_nbr_reads_the_previous_round_and_a_default_before():
    clock = iter(range(10))

    def program():
        now = float(next(clock))
        return gather_sum(nbr(torch.full((4,), now), default=-1.0), fill_value=0.0)

    first, second, third = run(line(), 4, program, 3)
    assert first.tolist() == [-1, -2, -2, -1]  # nobody has published yet
    assert second.tolist() == [0, 0, 0, 0]  # neighbours' round-0 value
    assert third.tolist() == [1, 2, 2, 1]


def test_aligned_on_excludes_other_keys_and_restarts_a_moved_device():
    keys = iter([[0, 0, 1, 1]] * 3 + [[0, 0, 0, 1]] * 3)

    def program():
        key = torch.tensor(next(keys), dtype=torch.float)
        with aligned_on(key):
            neighbours = gather_sum(nbr(torch.ones(4)), fill_value=0.0)
            age = gather_sum(nbr(mid() * 0 + 1), fill_value=0.0)  # second exchanged field
        return neighbours + 0 * age

    outputs = run(line(), 4, program, 6)
    assert outputs[2].tolist() == [1, 1, 1, 1]  # links 1-2 are cut by the key
    # Node 2 moved to region 0: its region-0 state starts fresh, then links reconnect.
    assert outputs[4].tolist() == [1, 1, 1, 0]
    assert outputs[5].tolist() == [1, 2, 1, 0]


def test_soft_membership_weights_links_and_carries_gradient():
    weight = torch.tensor([1.0, 0.5, 0.25, 1.0], requires_grad=True)

    def program():
        with with_mode("soft"), aligned_on(torch.zeros(4), weight=weight):
            return gather_sum(membership(), fill_value=0.0)

    hard = run(line(), 4, _hard_membership, 2)[-1]
    soft = run(line(), 4, program, 2)[-1]
    assert hard.tolist() == [1, 2, 2, 1]
    torch.testing.assert_close(soft, torch.tensor([0.5, 0.625, 0.375, 0.25]))  # sum of w_j * w_i
    soft.sum().backward()
    assert weight.grad.abs().sum() > 0


def _hard_membership():
    with aligned_on(torch.zeros(4), weight=torch.full((4,), 0.3)):
        return gather_sum(membership(), fill_value=0.0)


def test_bounded_election_radius_self_candidacy_and_priority():
    strength = torch.tensor([0.9, 0.1, 0.2, 0.3, 0.4])

    def program():
        return bounded_election(strength, radius=2.5, metric=scatter_range())

    election = run(line(5), 5, program, 8)[-1]
    # Node 0 wins within distance < 2.5 (nodes 0-2); node 4 is the strongest beyond it.
    assert election.leader.tolist() == [0, 0, 0, 4, 4]
    assert election.distance.tolist() == [0, 1, 2, 1, 0]
    assert election.elected.tolist() == [1, 0, 0, 0, 1]
    assert election.confidence.tolist() == [1, 1, 1, 1, 1]


def test_cold_soft_election_matches_hard():
    strength = torch.tensor([0.9, 0.1, 0.2, 0.3, 0.4])
    hard = run(line(5), 5, lambda: bounded_election(strength, radius=2.5), 8)[-1]
    with with_mode("soft", tau=1e-4):
        soft = run(line(5), 5, lambda: bounded_election(strength, radius=2.5), 8)[-1]
    assert torch.equal(soft.leader, hard.leader)
    torch.testing.assert_close(soft.distance, hard.distance)


def test_two_converge_casts_do_not_share_state_and_tuples_round_trip():
    potential = torch.tensor([0.0, 1.0, 2.0, 3.0])

    def program():
        count = converge_cast(potential, torch.ones(4))
        total, again = converge_cast(potential, (torch.arange(4.0), 1.0), add)
        return count, total, again

    count, total, again = run(line(), 4, program, 8)[-1]
    assert count[0] == 4 and again[0] == 4 and total[0] == 6


def test_broadcast_ties_do_not_depend_on_edge_order():
    root = torch.tensor([True, False, False, True])
    edges = torch.tensor([[0, 1, 1, 2, 2, 3, 0, 3], [1, 0, 2, 1, 3, 2, 3, 0]])  # a ring

    def program():
        return broadcast(root, mid() * 10)

    forward = run(edges, 4, program, 6)[-1]
    backward = run(edges.flip(1), 4, program, 6)[-1]
    assert torch.equal(forward, backward)


def test_relaxed_first_admissible_choice_is_differentiable():
    strength = torch.tensor([-0.2, -0.9, -0.1, -0.3], dtype=torch.double)
    distances = torch.tensor([0.9, 1.1, 0.0, 0.0], dtype=torch.double, requires_grad=True)
    index = torch.tensor([0, 1, 0, 1])
    local = torch.tensor([False, False, True, True])
    valid = torch.ones(4, dtype=torch.bool)
    assert torch.autograd.gradcheck(
        lambda d: surrogate_probabilities(strength, d, valid, local, index, 2, 0.1), (distances,)
    )


def test_follow_is_identity_when_hard_and_mixes_the_next_region_when_soft():
    strength = torch.tensor([0.9, 0.1, 0.2, 0.3, 0.4], requires_grad=True)
    value = torch.tensor([0.0, 0.0, 0.0, 10.0, 10.0])

    def program():
        election = bounded_election(strength, radius=2.5)
        return follow(election, value), election

    hard, election = run(line(5), 5, program, 8)[-1]
    assert election.support is None and torch.equal(hard, value)
    with with_mode("soft", tau=0.3):
        soft, election = run(line(5), 5, program, 8)[-1]
    # Node 2 sits at the boundary: part of its estimate comes from node 3's region.
    assert 0 < soft[2] < 10 and soft[0] < soft[2]
    (grad,) = torch.autograd.grad(soft[2], strength)
    assert grad.abs().sum() > 0
