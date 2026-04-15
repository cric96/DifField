"""Auto-naming focused DeviceContext tests."""

from __future__ import annotations

import torch

from autofield import minhood, nbr, rep, sumhood
from autofield.dsl import DeviceContext

from .support import (
    BRANCH_STATE,
    COUNTER_STATE,
    FALSE_BRANCH_VALUE,
    THREE_NEIGHBORS,
    TWO_NEIGHBORS,
    UNIT_VALUE,
    repeated,
    run_branched_rep,
    run_named_rep,
)


class TestDeviceContextAutoNamed:
    def test_decentralized_auto_named_rep_accumulates(self):
        dev_a = DeviceContext(num_neighbors=THREE_NEIGHBORS)
        dev_b = DeviceContext(num_neighbors=THREE_NEIGHBORS)

        round_1 = (
            run_named_rep(
                dev_a,
                state_name=COUNTER_STATE,
                update=lambda s: sumhood(nbr(s)) + UNIT_VALUE,
            ),
            run_named_rep(
                dev_b,
                state_name=COUNTER_STATE,
                update=lambda s: sumhood(nbr(s)) + UNIT_VALUE,
            ),
        )
        assert round_1 == (1.0, 1.0)

        round_2 = (
            run_named_rep(
                dev_a,
                state_name=COUNTER_STATE,
                update=lambda s: sumhood(nbr(s)) + UNIT_VALUE,
                neighbor_values=repeated(round_1[1], THREE_NEIGHBORS),
            ),
            run_named_rep(
                dev_b,
                state_name=COUNTER_STATE,
                update=lambda s: sumhood(nbr(s)) + UNIT_VALUE,
                neighbor_values=repeated(round_1[0], THREE_NEIGHBORS),
            ),
        )
        assert round_2 == (5.0, 5.0)

        round_3 = (
            run_named_rep(
                dev_a,
                state_name=COUNTER_STATE,
                update=lambda s: sumhood(nbr(s)) + UNIT_VALUE,
                neighbor_values=repeated(round_2[1], THREE_NEIGHBORS),
            ),
            run_named_rep(
                dev_b,
                state_name=COUNTER_STATE,
                update=lambda s: sumhood(nbr(s)) + UNIT_VALUE,
                neighbor_values=repeated(round_2[0], THREE_NEIGHBORS),
            ),
        )
        assert round_3 == (21.0, 21.0)

    def test_decentralized_nbr_without_tag_uses_local_field(self):
        dev_a = DeviceContext(num_neighbors=THREE_NEIGHBORS)
        dev_b = DeviceContext(num_neighbors=THREE_NEIGHBORS)

        def run_sum(device: DeviceContext, neighbor_values):
            with device.round(
                neighbor_exports=None
                if neighbor_values is None
                else {"x": neighbor_values}
            ):
                x = rep(device.local_field(0.0), lambda s: s + UNIT_VALUE, name="x")
                aggregated = sumhood(nbr(x))
            return device.result(x).item(), device.result(aggregated).item()

        xa1, agg_a1 = run_sum(dev_a, None)
        xb1, agg_b1 = run_sum(dev_b, None)
        assert (xa1, xb1) == (1.0, 1.0)

        xa2, agg_a2 = run_sum(dev_a, repeated(xb1, THREE_NEIGHBORS))
        xb2, agg_b2 = run_sum(dev_b, repeated(xa1, THREE_NEIGHBORS))
        assert (xa2, xb2) == (2.0, 2.0)
        assert agg_a2 > agg_a1
        assert agg_b2 > agg_b1

    def test_decentralized_auto_rep_with_branch(self):
        dev_a = DeviceContext(num_neighbors=THREE_NEIGHBORS)
        dev_b = DeviceContext(num_neighbors=THREE_NEIGHBORS)

        cond_a = dev_a.local_field(own=1.0, nbr=1.0)
        cond_b = dev_b.local_field(own=0.0, nbr=1.0)

        def run_branch_state(
            device: DeviceContext, condition: torch.Tensor, neighbor_values
        ):
            return run_branched_rep(
                device,
                condition=condition,
                state_name=BRANCH_STATE,
                true_update=lambda s: sumhood(nbr(s)) + UNIT_VALUE,
                false_update=lambda s: sumhood(nbr(s)) + FALSE_BRANCH_VALUE,
                neighbor_values=neighbor_values,
            )

        round_1 = (
            run_branch_state(dev_a, cond_a, None),
            run_branch_state(dev_b, cond_b, None),
        )
        assert round_1 == (1.0, 10.0)

        round_2 = (
            run_branch_state(dev_a, cond_a, repeated(round_1[1], THREE_NEIGHBORS)),
            run_branch_state(dev_b, cond_b, repeated(round_1[0], THREE_NEIGHBORS)),
        )
        assert round_2[0] > round_1[0]
        assert round_2[1] > round_1[1]

        round_3 = (
            run_branch_state(dev_a, cond_a, repeated(round_2[1], THREE_NEIGHBORS)),
            run_branch_state(dev_b, cond_b, repeated(round_2[0], THREE_NEIGHBORS)),
        )
        assert round_3[0] > round_2[0]
        assert round_3[1] > round_2[1]

    def test_decentralized_nbr_min_aggregation_without_tag(self):
        dev_a = DeviceContext(num_neighbors=THREE_NEIGHBORS)
        dev_b = DeviceContext(num_neighbors=TWO_NEIGHBORS)

        def run_min(device: DeviceContext, neighbor_values):
            with device.round(
                neighbor_exports=None
                if neighbor_values is None
                else {"val": neighbor_values}
            ):
                x = rep(device.local_field(0.0), lambda s: s + UNIT_VALUE, name="val")
                min_neighbor = minhood(nbr(x))
            return device.result(x).item(), device.result(min_neighbor).item()

        xa1, min_a1 = run_min(dev_a, None)
        xb1, min_b1 = run_min(dev_b, None)
        assert (xa1, xb1) == (1.0, 1.0)

        xa2, min_a2 = run_min(dev_a, repeated(xb1, THREE_NEIGHBORS))
        xb2, min_b2 = run_min(dev_b, repeated(xa1, TWO_NEIGHBORS))
        assert (xa2, xb2) == (2.0, 2.0)
        assert torch.isfinite(torch.tensor(min_a2))
        assert torch.isfinite(torch.tensor(min_b2))
