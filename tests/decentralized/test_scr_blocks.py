"""SCR blocks run identically on a central graph and on independent devices."""

from operator import add
from types import SimpleNamespace

import pytest
import torch

from diffield import (
    AggregateContext,
    aligned_on,
    bounded_election,
    broadcast,
    converge_cast,
    distance_to,
    elect,
    gather_sum,
    gradient,
    mid,
    nbr,
    scatter,
)
from diffield.decentralized import run_decentralized

pytest.importorskip("mesa")


def grid(width=4):
    pairs = [
        (i, j)
        for i in range(width * width)
        for j in (i + 1, i + width)
        if j < width * width and (j == i + width or j // width == i // width)
    ]
    return torch.tensor(sorted({e for a, b in pairs for e in [(a, b), (b, a)]})).T, width * width


def both(program, signals, rounds=24):
    edges, n = grid()
    ctx = AggregateContext(edges, n)
    central = []
    for _ in range(rounds):
        with ctx.round():
            central.append(program(SimpleNamespace(signals=signals)))
    scenario = SimpleNamespace(edge_index=edges, num_nodes=n, self_loops=False)
    local = run_decentralized(scenario=scenario, program=program, signals=signals, rounds=rounds)
    return torch.stack(central), torch.stack(local.fields)


SIGNALS = {
    "source": torch.arange(16) == 0,
    "strength": torch.rand(16, generator=torch.Generator().manual_seed(3)),
    "value": torch.arange(16).float() % 5,
}


def test_nbr_exchanges_this_round_fields_where_scatter_cannot():
    def exchanged(runtime):
        return gather_sum(nbr(gradient(runtime.signals["source"])), fill_value=0.0)

    def local_only(runtime):
        return gather_sum(scatter(gradient(runtime.signals["source"])), fill_value=0.0)

    central, devices = both(exchanged, SIGNALS)
    torch.testing.assert_close(central, devices, rtol=0, atol=0)
    central, devices = both(local_only, SIGNALS)
    assert not torch.equal(central, devices)  # neighbours' gradient is not on the wire


def test_aligned_on_a_computed_key_partitions_identically():
    def program(runtime):
        region = (gradient(runtime.signals["source"]) > 2).float()
        with aligned_on(region):
            return distance_to(mid() % 3 == 0)

    central, devices = both(program, SIGNALS)
    torch.testing.assert_close(central, devices, rtol=0, atol=0)


def test_scr_composition_runs_identically_on_devices():
    def program(runtime):
        election = bounded_election(runtime.signals["strength"], radius=2.5)
        with aligned_on(election.leader):
            source = election.leader == mid()
            potential = distance_to(source)
            total, count = converge_cast(potential, (runtime.signals["value"], 1.0), add)
            mean = broadcast(source, total / count)
        return torch.stack((election.leader, potential, total, count, mean), -1)

    central, devices = both(program, SIGNALS, rounds=40)
    torch.testing.assert_close(central, devices, rtol=0, atol=0)
    leader, _, _, count, mean = central[-1].unbind(-1)
    for region in leader.unique():  # converged: one mean per region, everyone counted
        members = leader == region
        assert count[int(region)] == members.sum()
        expected = SIGNALS["value"][members].mean().expand(int(members.sum()))
        torch.testing.assert_close(mean[members], expected)


def test_elect_reads_neighbour_distances_through_nbr():
    def program(runtime):
        leader, lead = elect(runtime.signals["strength"] * 16 + mid(), grain=3.0)
        return torch.stack((leader.float(), lead), -1)

    central, devices = both(program, SIGNALS)
    torch.testing.assert_close(central, devices, rtol=0, atol=0)
