"""Silent neighbours must not acquire state by being simulated locally."""

import torch

from diffield import gather_sum, iterate, scatter
from diffield.decentralized.runtime import DeviceRuntime


def program(runtime):
    value = runtime.signals["value"]
    return iterate(
        torch.zeros_like(value), lambda old: value + gather_sum(scatter(old)), name="sum"
    )


def test_partial_inbox_is_used_on_first_round_and_silent_neighbour_stays_unknown():
    device = DeviceRuntime(
        0, [1, 2], [1.0, 1.0], {"value": torch.tensor([1.0, 10.0, 100.0])}, program
    )
    # Device 1 has spoken; device 2 has never run. Its globally known sensor
    # reading must not masquerade as a message in later local rounds.
    output, _ = device.step({"/it:sum": [torch.tensor(5.0), None]})
    assert output == 6
    for _ in range(3):
        output, _ = device.step()
        assert output == 6
    output, _ = device.step({"/it:sum": [None, torch.tensor(7.0)]})
    assert output == 13


def test_topology_change_preserves_own_state_and_forgets_removed_neighbour():
    def counter(runtime):
        initial = torch.zeros_like(runtime.signals["value"])
        return iterate(initial, lambda old: old + 1, name="clock")

    device = DeviceRuntime(0, [1], [1.0], {"value": torch.ones(3)}, counter)
    device.step()
    device.step()
    device.sync_topology([2], [1.0])
    output, _ = device.step()
    assert output == 3
