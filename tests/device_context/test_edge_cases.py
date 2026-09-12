"""Edge-case tests for DeviceContext."""

from __future__ import annotations

from diffield import gather_sum, scatter
from diffield.dsl import DeviceContext

from .support import (
    ONE_NEIGHBOR,
    THREE_NEIGHBORS,
    TWO_NEIGHBORS,
    VERY_LARGE_INIT,
    iterated,
    run_branched_iterate,
    run_round,
)


class TestDeviceContextEdgeCases:
    def test_tagged_neighbor_messages_override_local_expression(self):
        device = DeviceContext(num_neighbors=TWO_NEIGHBORS, self_loop=True)
        expression = device.local_field(own=10.0, scatter=[10.0, 10.0])

        result = run_round(
            device,
            lambda: gather_sum(scatter(expression), tag="chan"),
            neighbor_messages={"chan": [2.0, 3.0]},
        )

        assert result == 5.0

    def test_include_self_overrides_device_self_loop_topology(self):
        device = DeviceContext(num_neighbors=ONE_NEIGHBOR, self_loop=True)
        expression = device.local_field(own=4.0, scatter=[2.0])

        without_self = run_round(
            device,
            lambda: gather_sum(scatter(expression), include_self=False),
        )
        with_self = run_round(
            device,
            lambda: gather_sum(scatter(expression), include_self=True),
        )

        assert without_self == 2.0
        assert with_self == 6.0

    def test_decentralized_iterate_with_incompatible_init_values(self):
        device = DeviceContext(num_neighbors=THREE_NEIGHBORS, self_loop=True)
        cond_true = device.local_field(own=1.0, scatter=1.0)
        cond_false = device.local_field(own=0.0, scatter=1.0)

        def stable_small(s):
            return gather_sum(scatter(s)) * 0.0 + 1.0

        def stable_large(s):
            return gather_sum(scatter(s)) * 0.0 + VERY_LARGE_INIT

        round_1 = run_branched_iterate(
            device,
            condition=cond_true,
            state_name="scaled",
            true_update=stable_small,
            false_update=stable_large,
            true_init=1.0,
            false_init=VERY_LARGE_INIT,
        )
        assert round_1 == 1.0

        round_2 = run_branched_iterate(
            device,
            condition=cond_true,
            state_name="scaled",
            true_update=stable_small,
            false_update=stable_large,
            true_init=1.0,
            false_init=VERY_LARGE_INIT,
            neighbor_values=iterated(round_1, THREE_NEIGHBORS),
        )
        assert round_2 == 1.0

        round_3 = run_branched_iterate(
            device,
            condition=cond_false,
            state_name="scaled",
            true_update=stable_small,
            false_update=stable_large,
            true_init=1.0,
            false_init=VERY_LARGE_INIT,
            neighbor_values=iterated(round_2, THREE_NEIGHBORS),
        )
        assert round_3 == VERY_LARGE_INIT

    def test_decentralized_scatter_aggregates_incompatible_values_across_partitions(self):
        dev_a = DeviceContext(num_neighbors=THREE_NEIGHBORS, self_loop=True)
        dev_b = DeviceContext(num_neighbors=THREE_NEIGHBORS, self_loop=True)

        cond_a = dev_a.local_field(own=1.0, scatter=1.0)
        cond_b = dev_b.local_field(own=0.0, scatter=1.0)

        def run_cross_pollution(device: DeviceContext, condition, neighbor_values):
            return run_branched_iterate(
                device,
                condition=condition,
                state_name="val",
                true_update=lambda s: gather_sum(scatter(s)) + 0.0,
                false_update=lambda s: gather_sum(scatter(s)) + 0.0,
                true_init=1.0,
                false_init=100.0,
                neighbor_values=neighbor_values,
            )

        round_1 = (
            run_cross_pollution(dev_a, cond_a, None),
            run_cross_pollution(dev_b, cond_b, None),
        )
        assert round_1 == (1.0, 100.0)

        round_2 = (
            run_cross_pollution(dev_a, cond_a, iterated(round_1[1], THREE_NEIGHBORS)),
            run_cross_pollution(dev_b, cond_b, iterated(round_1[0], THREE_NEIGHBORS)),
        )
        assert round_2[0] > 100.0
        assert round_2[1] >= 100.0
