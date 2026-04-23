"""Basic tests for local execution via DeviceContext."""

from __future__ import annotations

import pytest
import torch

from diffield import AggregateContext
from diffield.dsl import (
    gradient,
    mux,
    scatter,
    scatter_range,
    iterate,
    gather_min,
    gather_sum,
    DeviceContext,
    field,
)

from .support import (
    ABS_TOL,
    GLOBAL_SYNC_ROUNDS,
    GRADIENT_EXPORT,
    GRADIENT_STATE,
    ONE_NEIGHBOR,
    TWO_NEIGHBORS,
    UNIT_VALUE,
    assert_float_close,
    collect_round_outputs,
    values,
)


@pytest.mark.parametrize(
    ("own", "neighbors"),
    [
        (1.0, 0.0),
        (5.0, [1.0, 2.0, 3.0]),
        (0.0, [1.0]),
    ],
)
def test_local_field(own, neighbors):
    num_neighbors = 1 if isinstance(neighbors, float) else len(neighbors)
    device = DeviceContext(num_neighbors=num_neighbors)

    local = device.local_field(own=own, scatter=neighbors)

    assert local.shape == (num_neighbors + 1,)
    assert local[0] == own
    if isinstance(neighbors, float):
        assert local[1] == neighbors
    else:
        assert torch.allclose(local[1:], values(*neighbors))


def test_result_returns_own_device_value():
    field_value = values(42.0, 1.0, 2.0)
    assert DeviceContext.result(field_value) == 42.0


class TestDeviceContextBasic:
    def test_local_matches_global(self, simple_triangle_topology):
        edge_index, num_nodes = simple_triangle_topology
        source = values(1.0, 0.0, 0.0)
        weight = torch.tensor(UNIT_VALUE)

        ctx = AggregateContext(edge_index, num_nodes)
        global_states = collect_round_outputs(
            ctx,
            GLOBAL_SYNC_ROUNDS,
            lambda: iterate(
                field.inf(),
                lambda dist: mux(source, field.of(0.0), gather_min(scatter(dist + weight))),
                name=GRADIENT_STATE,
            ),
        )

        observed_device_id = 2
        upstream_neighbor_id = 1
        device = DeviceContext(num_neighbors=ONE_NEIGHBOR)
        source_local = device.local_field(
            own=source[observed_device_id].item(),
            scatter=0.0,
        )

        for round_index in range(GLOBAL_SYNC_ROUNDS):
            neighbor_exports = None
            if round_index > 0:
                neighbor_exports = {
                    GRADIENT_STATE: [
                        global_states[round_index - 1][upstream_neighbor_id].item()
                    ]
                }
            with device.round(neighbor_exports=neighbor_exports):
                local_distance = iterate(
                    field.inf(),
                    lambda dist: mux(
                        source_local,
                        field.of(0.0),
                        gather_min(scatter(dist + weight)),
                    ),
                    name=GRADIENT_STATE,
                )

        assert_float_close(
            device.result(local_distance).item(),
            global_states[-1][observed_device_id].item(),
        )

    def test_isolated_device_stays_at_source_distance(self):
        device = DeviceContext(num_neighbors=0)
        source_local = device.local_field(own=1.0)
        weight = torch.tensor(UNIT_VALUE)

        for _ in range(3):
            with device.round():
                local_distance = iterate(
                    field.inf(),
                    lambda dist: mux(
                        source_local,
                        field.of(0.0),
                        gather_min(scatter(dist + weight)),
                    ),
                    name=GRADIENT_STATE,
                )

        assert device.result(local_distance).item() == 0.0

    def test_neighbor_ranges_available_locally(self):
        device = DeviceContext(num_neighbors=TWO_NEIGHBORS)

        with device.round(neighbor_ranges=[1.5, 2.5]):
            total_range = gather_sum(scatter_range())

        assert_float_close(device.result(total_range).item(), 4.0)

    def test_local_weighted_gradient_matches_global(self, weighted_triangle_topology):
        edge_index, edge_weight, num_nodes = weighted_triangle_topology
        source = values(1.0, 0.0, 0.0)

        ctx = AggregateContext(edge_index, num_nodes, edge_weight=edge_weight)
        global_states = collect_round_outputs(
            ctx,
            GLOBAL_SYNC_ROUNDS,
            lambda: gradient(source, name=GRADIENT_STATE),
        )

        device = DeviceContext(num_neighbors=ONE_NEIGHBOR, self_loop=False)
        source_local = device.local_field(own=0.0, scatter=0.0)
        neighbor_ranges = [3.0]
        observed_device_id = 2
        upstream_neighbor_id = 1

        for round_index in range(GLOBAL_SYNC_ROUNDS):
            neighbor_exports = None
            if round_index > 0:
                neighbor_exports = {
                    GRADIENT_EXPORT: [
                        global_states[round_index - 1][upstream_neighbor_id].item()
                    ]
                }
            with device.round(
                neighbor_exports=neighbor_exports,
                neighbor_ranges=neighbor_ranges,
            ):
                local_distance = gradient(source_local, name=GRADIENT_STATE)

        assert_float_close(
            device.result(local_distance).item(),
            global_states[-1][observed_device_id].item(),
            atol=ABS_TOL,
        )
