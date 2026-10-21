"""Programs run device by device on a network whose links change."""

import torch

from diffield.dsl import gather_sum, mid, nbr
from examples.shared.devices import DeviceNetwork

PAIR = torch.tensor([[0, 1], [1, 0]])


def heard(runtime):
    """The sum of the neighbours' IDs plus one, as last heard from them; 0 if not heard yet."""
    return gather_sum(nbr(mid().float() + 1))


def test_a_barrier_delivers_bundles_in_the_next_round_and_mailboxes_at_once():
    synchronous = DeviceNetwork(heard, PAIR, 2, torch.zeros(2))
    asynchronous = DeviceNetwork(heard, PAIR, 2, torch.zeros(2))

    synchronous.run([0, 1])
    first = synchronous.outputs.clone()
    synchronous.run([0, 1])
    asynchronous.run([0, 1], barrier=False)

    assert first.tolist() == [0, 0]
    assert synchronous.outputs.tolist() == [2, 1]
    assert asynchronous.outputs.tolist() == [0, 1]


def test_an_idle_device_keeps_its_output_and_removed_links_stop_delivering():
    network = DeviceNetwork(heard, PAIR, 2, torch.zeros(2))
    network.run([0, 1])
    network.run([0, 1])

    network.run([1])
    kept = network.outputs[0].item()
    network.move(torch.empty(2, 0, dtype=torch.long))
    network.run([0, 1], barrier=False)

    assert kept == 2
    assert network.outputs.tolist() == [0, 0]
