"""Multiple-assignment DeviceContext tests."""

from __future__ import annotations

from diffield.dsl import DeviceContext, branch, gather_sum, iterate, mux, scatter

from .support import (
    BRANCH_THRESHOLD,
    DENSE_PAIR_NEIGHBOR_INDICES,
    LINE_NEIGHBOR_INDICES,
    MUX_FALLBACK_VALUE,
    ONE_NEIGHBOR,
    THREE_NEIGHBORS,
    TWO_NEIGHBORS,
    UNIT_VALUE,
    X_STATE,
    Y_STATE,
    DeviceNetwork,
    assert_tuple_close,
    run_round,
)

FALSE_BRANCH_INCREMENT = 10.0
NEGATIVE_SENTINEL = -1.0


class TestDeviceContextMultipleAssignments:
    def test_decentralized_multiple_assignments_nested(self):
        dev_0 = DeviceContext(num_neighbors=ONE_NEIGHBOR)
        dev_1 = DeviceContext(num_neighbors=TWO_NEIGHBORS)
        dev_2 = DeviceContext(num_neighbors=ONE_NEIGHBOR)

        src_0 = dev_0.local_field(own=1.0, scatter=0.0)
        src_1 = dev_1.local_field(own=0.0, scatter=[1.0, 0.0])
        src_2 = dev_2.local_field(own=0.0, scatter=0.0)

        network = DeviceNetwork(dev_0, dev_1, dev_2)

        def run_device(device: DeviceContext, source, neighbor_exports):
            return run_round(
                device,
                lambda: _nested_assignment_program(device, source),
                neighbor_exports=neighbor_exports,
            )

        round_1 = (
            run_device(dev_0, src_0, None),
            run_device(dev_1, src_1, None),
            run_device(dev_2, src_2, None),
        )
        assert_tuple_close(round_1[0], (1.0, 2.0, -1.0))
        assert_tuple_close(round_1[1], (1.0, 100.0, -1.0))
        assert_tuple_close(round_1[2], (1.0, 100.0, -1.0))

        snapshots = network.snapshot(X_STATE, Y_STATE)
        exports_2 = network.exports(snapshots, *LINE_NEIGHBOR_INDICES)
        round_2 = (
            run_device(dev_0, src_0, exports_2[0]),
            run_device(dev_1, src_1, exports_2[1]),
            run_device(dev_2, src_2, exports_2[2]),
        )
        assert_tuple_close(round_2[0], (2.0, 9.0, 31.0))
        assert_tuple_close(round_2[1], (2.0, 100.0, 220.0))
        assert_tuple_close(round_2[2], (2.0, 100.0, 220.0))

        snapshots = network.snapshot(X_STATE, Y_STATE)
        exports_3 = network.exports(snapshots, *LINE_NEIGHBOR_INDICES)
        round_3 = (
            run_device(dev_0, src_0, exports_3[0]),
            run_device(dev_1, src_1, exports_3[1]),
            run_device(dev_2, src_2, exports_3[2]),
        )
        assert_tuple_close(round_3[0], (3.0, 28.0, 69.0))
        assert_tuple_close(round_3[1], (3.0, 100.0, 330.0))
        assert_tuple_close(round_3[2], (3.0, 100.0, 330.0))

    def test_decentralized_multiple_auto_iterate_no_tags(self):
        dev_0 = DeviceContext(num_neighbors=THREE_NEIGHBORS)
        dev_1 = DeviceContext(num_neighbors=THREE_NEIGHBORS)

        src_0 = dev_0.local_field(own=1.0, scatter=0.0)
        src_1 = dev_1.local_field(own=0.0, scatter=1.0)

        network = DeviceNetwork(dev_0, dev_1)

        def run_device(device: DeviceContext, source, neighbor_exports):
            return run_round(
                device,
                lambda: _two_assignment_program(device, source),
                neighbor_exports=neighbor_exports,
            )

        round_1 = (
            run_device(dev_0, src_0, None),
            run_device(dev_1, src_1, None),
        )
        assert round_1[0][0] == 1.0
        assert_tuple_close(round_1[1], (1.0, 100.0))

        snapshots = network.snapshot(X_STATE, Y_STATE)
        exports_2 = network.exports(snapshots, *DENSE_PAIR_NEIGHBOR_INDICES)
        round_2 = (
            run_device(dev_0, src_0, exports_2[0]),
            run_device(dev_1, src_1, exports_2[1]),
        )
        assert round_2[0][0] == 2.0
        assert_tuple_close(round_2[1], (2.0, 100.0))

        snapshots = network.snapshot(X_STATE, Y_STATE)
        exports_3 = network.exports(snapshots, *DENSE_PAIR_NEIGHBOR_INDICES)
        round_3 = (
            run_device(dev_0, src_0, exports_3[0]),
            run_device(dev_1, src_1, exports_3[1]),
        )
        assert round_3[0][0] == 3.0
        assert_tuple_close(round_3[1], (3.0, 100.0))


def _nested_assignment_program(device: DeviceContext, source):
    x = iterate(device.local_field(0.0), lambda s: s + UNIT_VALUE, name=X_STATE)
    y = mux(
        source,
        iterate(device.local_field(0.0), lambda s: gather_sum(scatter(s + x)), name=Y_STATE),
        device.local_field(MUX_FALLBACK_VALUE),
    )
    z = branch(
        x > BRANCH_THRESHOLD,
        lambda: iterate(
            device.local_field(0.0),
            lambda s: s + y + FALSE_BRANCH_INCREMENT,
            name="z",
        ),
        lambda: device.local_field(NEGATIVE_SENTINEL),
    )
    return x, y, z


def _two_assignment_program(device: DeviceContext, source):
    x = iterate(device.local_field(0.0), lambda s: s + UNIT_VALUE, name=X_STATE)
    y = mux(
        source,
        iterate(device.local_field(0.0), lambda s: gather_sum(scatter(s)) + x, name=Y_STATE),
        device.local_field(MUX_FALLBACK_VALUE),
    )
    return x, y
