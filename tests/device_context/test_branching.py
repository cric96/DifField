"""Branching tests for DeviceContext."""

from __future__ import annotations

import torch

from autofield import branch, gather_max, scatter, iterate, gather_sum
from autofield.dsl import DeviceContext

from .support import (
    FALSE_BRANCH_VALUE,
    ONE_NEIGHBOR,
    OUTER_STATE,
    POLLUTED_INIT,
    STATE_NAME,
    THREE_NEIGHBORS,
    TWO_NEIGHBORS,
    UNIT_VALUE,
    DeviceNetwork,
    iterated,
    run_branched_iterate,
)


class TestDeviceContextBranching:
    def test_decentralized_branching(self):
        dev_a = DeviceContext(num_neighbors=ONE_NEIGHBOR)
        dev_b = DeviceContext(num_neighbors=TWO_NEIGHBORS)
        dev_c = DeviceContext(num_neighbors=ONE_NEIGHBOR)

        cond_a = dev_a.local_field(own=1.0, scatter=1.0)
        cond_b = dev_b.local_field(own=1.0, scatter=[1.0, 0.0])
        cond_c = dev_c.local_field(own=0.0, scatter=1.0)

        network = DeviceNetwork(dev_a, dev_b, dev_c)

        def run_branching(
            device: DeviceContext, condition: torch.Tensor, neighbor_values
        ):
            return run_branched_iterate(
                device,
                condition=condition,
                state_name=STATE_NAME,
                true_update=lambda s: gather_sum(scatter(s)) + UNIT_VALUE,
                false_update=lambda s: gather_max(scatter(s)) + FALSE_BRANCH_VALUE,
                neighbor_values=neighbor_values,
            )

        round_1 = (
            run_branching(dev_a, cond_a, None),
            run_branching(dev_b, cond_b, None),
            run_branching(dev_c, cond_c, None),
        )
        assert round_1 == (1.0, 1.0, 10.0)

        snapshots = network.snapshot(STATE_NAME)
        round_2 = (
            run_branching(dev_a, cond_a, [snapshots[1][STATE_NAME]]),
            run_branching(
                dev_b,
                cond_b,
                [snapshots[0][STATE_NAME], snapshots[2][STATE_NAME]],
            ),
            run_branching(dev_c, cond_c, [snapshots[1][STATE_NAME]]),
        )
        assert round_2 == (3.0, 3.0, 20.0)

        snapshots = network.snapshot(STATE_NAME)
        round_3 = (
            run_branching(dev_a, cond_a, [snapshots[1][STATE_NAME]]),
            run_branching(
                dev_b,
                cond_b,
                [snapshots[0][STATE_NAME], snapshots[2][STATE_NAME]],
            ),
            run_branching(dev_c, cond_c, [snapshots[1][STATE_NAME]]),
        )
        assert round_3 == (7.0, 7.0, 30.0)

    def test_decentralized_branch_with_auto_named_inner_iterate(self):
        device = DeviceContext(num_neighbors=THREE_NEIGHBORS)
        condition = device.local_field(own=1.0, scatter=1.0)

        def run_with_inner_branch(neighbor_values):
            with device.round(
                neighbor_exports=None
                if neighbor_values is None
                else {OUTER_STATE: neighbor_values}
            ):
                outer = iterate(
                    device.local_field(0.0),
                    lambda s: s + UNIT_VALUE,
                    name=OUTER_STATE,
                )
                inner = branch(
                    condition,
                    lambda: iterate(
                        device.local_field(0.0), lambda s: gather_sum(scatter(s)) + outer
                    ),
                    lambda: iterate(
                        device.local_field(0.0),
                        lambda s: gather_max(scatter(s)) + outer * FALSE_BRANCH_VALUE,
                    ),
                )
            return device.result(outer).item(), device.result(inner).item()

        outer_1, inner_1 = run_with_inner_branch(None)
        assert (outer_1, inner_1) == (1.0, 1.0)

        outer_2, inner_2 = run_with_inner_branch(iterated(outer_1, THREE_NEIGHBORS))
        assert outer_2 == 2.0
        assert inner_2 > inner_1

        outer_3, inner_3 = run_with_inner_branch(iterated(outer_2, THREE_NEIGHBORS))
        assert outer_3 == 3.0
        assert inner_3 > inner_2

    def test_decentralized_branch_incompatible_fields_exposes_errors(self):
        dev_a = DeviceContext(num_neighbors=THREE_NEIGHBORS)
        dev_b = DeviceContext(num_neighbors=THREE_NEIGHBORS)

        cond_a = dev_a.local_field(own=1.0, scatter=1.0)
        cond_b = dev_b.local_field(own=0.0, scatter=1.0)

        def run_without_isolation(
            device: DeviceContext, condition: torch.Tensor, neighbor_values
        ):
            return run_branched_iterate(
                device,
                condition=condition,
                state_name=STATE_NAME,
                true_update=lambda s: gather_sum(scatter(s)) + UNIT_VALUE,
                false_update=lambda s: gather_sum(scatter(s)) + FALSE_BRANCH_VALUE,
                false_init=POLLUTED_INIT,
                neighbor_values=neighbor_values,
            )

        round_1 = (
            run_without_isolation(dev_a, cond_a, None),
            run_without_isolation(dev_b, cond_b, None),
        )
        assert round_1 == (1.0, 1010.0)

        round_2 = (
            run_without_isolation(dev_a, cond_a, iterated(round_1[1], THREE_NEIGHBORS)),
            run_without_isolation(dev_b, cond_b, iterated(round_1[0], THREE_NEIGHBORS)),
        )
        assert round_2[0] > POLLUTED_INIT
        assert round_2[1] > POLLUTED_INIT

    def test_decentralized_branch_with_different_scatter_aggregations(self):
        dev_a = DeviceContext(num_neighbors=THREE_NEIGHBORS)
        dev_b = DeviceContext(num_neighbors=THREE_NEIGHBORS)

        cond_a = dev_a.local_field(own=1.0, scatter=1.0)
        cond_b = dev_b.local_field(own=0.0, scatter=1.0)

        def run_different_aggregation(
            device: DeviceContext, condition: torch.Tensor, neighbor_values
        ):
            return run_branched_iterate(
                device,
                condition=condition,
                state_name=STATE_NAME,
                true_update=lambda s: gather_sum(scatter(s)) + UNIT_VALUE,
                false_update=lambda s: gather_max(scatter(s)) + 100.0,
                neighbor_values=neighbor_values,
            )

        round_1 = (
            run_different_aggregation(dev_a, cond_a, None),
            run_different_aggregation(dev_b, cond_b, None),
        )
        assert round_1 == (1.0, 100.0)

        round_2 = (
            run_different_aggregation(
                dev_a, cond_a, iterated(round_1[1], THREE_NEIGHBORS)
            ),
            run_different_aggregation(
                dev_b, cond_b, iterated(round_1[0], THREE_NEIGHBORS)
            ),
        )
        assert round_2[0] > 100.0
        assert round_2[1] > 100.0

        round_3 = (
            run_different_aggregation(
                dev_a, cond_a, iterated(round_2[1], THREE_NEIGHBORS)
            ),
            run_different_aggregation(
                dev_b, cond_b, iterated(round_2[0], THREE_NEIGHBORS)
            ),
        )
        assert round_3[0] > round_2[0]
        assert round_3[1] > round_2[1]
