"""Tests for single-device local execution via DeviceContext."""

from __future__ import annotations

from typing import Any

import pytest
import torch

from autofield import AggregateContext, branch, gradient, mux, nbr, nbr_range, rep
from autofield.dsl import DeviceContext, field


@pytest.mark.parametrize(
    "own, nbrs",
    [
        (1.0, 0.0),
        (5.0, [1.0, 2.0, 3.0]),
        (0.0, [1.0]),
    ],
)
def test_local_field(own, nbrs):
    num_nbrs = 1 if isinstance(nbrs, float) else len(nbrs)
    device = DeviceContext(num_neighbors=num_nbrs)
    f = device.local_field(own=own, nbr=nbrs)
    assert f.shape == (num_nbrs + 1,)
    assert f[0] == own
    if isinstance(nbrs, float):
        assert f[1] == nbrs
    else:
        assert torch.allclose(f[1:], torch.tensor(nbrs, dtype=torch.float32))


def test_result():
    t = torch.tensor([42.0, 1.0, 2.0])
    assert DeviceContext.result(t) == 42.0


class _DeviceSimulator:
    """Helper for running decentralized device simulations."""

    def __init__(self, *devices: DeviceContext):
        self.devices = devices

    def snapshot(self, *state_names: str) -> list[dict[str, Any]]:
        return [
            {name: device.get_state(name) for name in state_names}
            for device in self.devices
        ]

    @staticmethod
    def make_exports(
        snapshots: list[dict], neighbor_indices: list[int]
    ) -> dict[str, list]:
        first_snapshot = snapshots[0]
        return {
            name: [snap[name] for snap in [snapshots[idx] for idx in neighbor_indices]]
            for name in first_snapshot
        }


class TestDeviceContextBasic:
    def test_local_matches_global(self, simple_triangle_topology):
        edge_index, n = simple_triangle_topology
        source = torch.tensor([1.0, 0.0, 0.0])
        w = torch.tensor(1.0)

        ctx = AggregateContext(edge_index, n)
        global_states = []
        for _ in range(4):
            with ctx.round():
                d = rep(
                    "dist",
                    float("inf"),
                    lambda dist: mux(source, field.of(0.0), nbr(dist + w, aggr="min")),
                )
            global_states.append(d.detach().clone())

        device_id = 2
        device = DeviceContext(num_neighbors=1)
        source_local = device.local_field(own=source[device_id].item(), nbr=0.0)
        for round_idx in range(4):
            neighbor_exports = (
                None
                if round_idx == 0
                else {"dist": [global_states[round_idx - 1][1].item()]}
            )
            with device.round(neighbor_exports=neighbor_exports):
                d_local = rep(
                    "dist",
                    float("inf"),
                    lambda dist: mux(
                        source_local, field.of(0.0), nbr(dist + w, aggr="min")
                    ),
                )

        assert (
            abs(device.result(d_local).item() - global_states[-1][device_id].item())
            < 1e-6
        )

    def test_isolated_device(self):
        device = DeviceContext(num_neighbors=0)
        source_local = device.local_field(own=1.0)
        w = torch.tensor(1.0)
        for _ in range(3):
            with device.round():
                d = rep(
                    "dist",
                    float("inf"),
                    lambda dist: mux(
                        source_local, field.of(0.0), nbr(dist + w, aggr="min")
                    ),
                )
        assert device.result(d).item() == 0.0

    def test_neighbor_ranges_available_locally(self):
        device = DeviceContext(num_neighbors=2)

        with device.round(neighbor_ranges=[1.5, 2.5]):
            ranges = nbr(nbr_range(), aggr="sum")

        assert abs(device.result(ranges).item() - 4.0) < 1e-6

    def test_local_weighted_gradient_matches_global(self, weighted_triangle_topology):
        edge_index, edge_weight, n = weighted_triangle_topology
        source = torch.tensor([1.0, 0.0, 0.0])

        ctx = AggregateContext(edge_index, n, edge_weight=edge_weight)
        global_states = []
        for _ in range(4):
            with ctx.round():
                d = gradient(source, name="dist")
            global_states.append(d.detach().clone())

        device = DeviceContext(num_neighbors=1, self_loop=False)
        source_local = device.local_field(own=0.0, nbr=0.0)
        for round_idx in range(4):
            neighbor_exports = (
                None
                if round_idx == 0
                else {"_grad_dist": [global_states[round_idx - 1][1].item()]}
            )
            with device.round(neighbor_exports=neighbor_exports, neighbor_ranges=[3.0]):
                d_local = gradient(source_local, name="dist")

        assert abs(device.result(d_local).item() - global_states[-1][2].item()) < 1e-6


class TestDeviceContextBranching:
    def test_decentralized_branching(self):
        dev_a = DeviceContext(num_neighbors=1)
        dev_b = DeviceContext(num_neighbors=2)
        dev_c = DeviceContext(num_neighbors=1)

        cond_a = dev_a.local_field(own=1.0, nbr=1.0)
        cond_b = dev_b.local_field(own=1.0, nbr=[1.0, 0.0])
        cond_c = dev_c.local_field(own=0.0, nbr=1.0)

        sim = _DeviceSimulator(dev_a, dev_b, dev_c)

        def run_branching(dev, cond, nbr_states):
            exports = None if nbr_states is None else {"state": nbr_states}
            with dev.round(neighbor_exports=exports):
                val = branch(
                    cond,
                    lambda: rep(
                        "state",
                        dev.local_field(0.0),
                        lambda s: nbr(s, aggr="sum") + 1.0,
                    ),
                    lambda: rep(
                        "state",
                        dev.local_field(0.0),
                        lambda s: nbr(s, aggr="max") + 10.0,
                    ),
                )
            return dev.result(val).item()

        r_a1 = run_branching(dev_a, cond_a, None)
        r_b1 = run_branching(dev_b, cond_b, None)
        r_c1 = run_branching(dev_c, cond_c, None)
        assert r_a1 == 1.0
        assert r_b1 == 1.0
        assert r_c1 == 10.0

        snap = sim.snapshot("state")
        r_a2 = run_branching(dev_a, cond_a, [snap[1]["state"]])
        r_b2 = run_branching(dev_b, cond_b, [snap[0]["state"], snap[2]["state"]])
        r_c2 = run_branching(dev_c, cond_c, [snap[1]["state"]])
        assert r_a2 == 3.0
        assert r_b2 == 3.0
        assert r_c2 == 20.0

        snap = sim.snapshot("state")
        r_a3 = run_branching(dev_a, cond_a, [snap[1]["state"]])
        r_b3 = run_branching(dev_b, cond_b, [snap[0]["state"], snap[2]["state"]])
        r_c3 = run_branching(dev_c, cond_c, [snap[1]["state"]])
        assert r_a3 == 7.0
        assert r_b3 == 7.0
        assert r_c3 == 30.0

    def test_decentralized_branch_with_auto_named_inner_rep(self):
        dev = DeviceContext(num_neighbors=3)
        cond = dev.local_field(own=1.0, nbr=1.0)

        def run_with_inner_branch(nbr_vals):
            exports = None if nbr_vals is None else {"outer": nbr_vals}
            with dev.round(neighbor_exports=exports):
                outer = rep(
                    dev.local_field(0.0),
                    lambda s: s + 1.0,
                    name="outer",
                )
                inner_val = branch(
                    cond,
                    lambda: rep(
                        dev.local_field(0.0),
                        lambda s: nbr(s, aggr="sum") + outer,
                    ),
                    lambda: rep(
                        dev.local_field(0.0),
                        lambda s: nbr(s, aggr="max") + outer * 10,
                    ),
                )
            return dev.result(outer).item(), dev.result(inner_val).item()

        o1, i1 = run_with_inner_branch(None)
        assert o1 == 1.0
        assert i1 == 1.0

        o2, i2 = run_with_inner_branch([o1] * 3)
        assert o2 == 2.0
        assert i2 > i1

        o3, i3 = run_with_inner_branch([o2] * 3)
        assert o3 == 3.0
        assert i3 > i2

    def test_decentralized_branch_incompatible_fields_exposes_errors(self):
        dev_a = DeviceContext(num_neighbors=3)
        dev_b = DeviceContext(num_neighbors=3)

        cond_a = dev_a.local_field(own=1.0, nbr=1.0)
        cond_b = dev_b.local_field(own=0.0, nbr=1.0)

        def run_without_isolation(dev, cond, nbr_vals):
            exports = None if nbr_vals is None else {"state": nbr_vals}
            with dev.round(neighbor_exports=exports):
                val = branch(
                    cond,
                    lambda: rep(
                        dev.local_field(0.0),
                        lambda s: nbr(s, aggr="sum") + 1.0,
                        name="state",
                    ),
                    lambda: rep(
                        dev.local_field(1000.0),
                        lambda s: nbr(s, aggr="sum") + 10.0,
                        name="state",
                    ),
                )
            return dev.result(val).item()

        ra1 = run_without_isolation(dev_a, cond_a, None)
        rb1 = run_without_isolation(dev_b, cond_b, None)
        assert ra1 == 1.0
        assert rb1 == 1010.0

        ra2 = run_without_isolation(dev_a, cond_a, [rb1] * 3)
        rb2 = run_without_isolation(dev_b, cond_b, [ra1] * 3)
        assert ra2 > 1000.0
        assert rb2 > 1000.0

    def test_decentralized_branch_with_different_nbr_aggregations(self):
        dev_a = DeviceContext(num_neighbors=3)
        dev_b = DeviceContext(num_neighbors=3)

        cond_a = dev_a.local_field(own=1.0, nbr=1.0)
        cond_b = dev_b.local_field(own=0.0, nbr=1.0)

        def run_different_aggr(dev, cond, nbr_vals):
            exports = None if nbr_vals is None else {"state": nbr_vals}
            with dev.round(neighbor_exports=exports):
                val = branch(
                    cond,
                    lambda: rep(
                        dev.local_field(0.0),
                        lambda s: nbr(s, aggr="sum") + 1.0,
                        name="state",
                    ),
                    lambda: rep(
                        dev.local_field(0.0),
                        lambda s: nbr(s, aggr="max") + 100.0,
                        name="state",
                    ),
                )
            return dev.result(val).item()

        ra1 = run_different_aggr(dev_a, cond_a, None)
        rb1 = run_different_aggr(dev_b, cond_b, None)
        assert ra1 == 1.0
        assert rb1 == 100.0

        ra2 = run_different_aggr(dev_a, cond_a, [rb1] * 3)
        rb2 = run_different_aggr(dev_b, cond_b, [ra1] * 3)
        assert ra2 > 100.0
        assert rb2 > 100.0

        ra3 = run_different_aggr(dev_a, cond_a, [rb2] * 3)
        rb3 = run_different_aggr(dev_b, cond_b, [ra2] * 3)
        assert ra3 > ra2
        assert rb3 > rb2


class TestDeviceContextMultipleAssignments:
    def test_decentralized_multiple_assignments_nested(self):
        dev0 = DeviceContext(num_neighbors=1)
        dev1 = DeviceContext(num_neighbors=2)
        dev2 = DeviceContext(num_neighbors=1)

        src0 = dev0.local_field(own=1.0, nbr=0.0)
        src1 = dev1.local_field(own=0.0, nbr=[1.0, 0.0])
        src2 = dev2.local_field(own=0.0, nbr=0.0)

        sim = _DeviceSimulator(dev0, dev1, dev2)

        def run_device(dev, src, exports):
            with dev.round(neighbor_exports=exports):
                x = rep("x", dev.local_field(0.0), lambda s: s + 1.0)
                y = mux(
                    src,
                    rep("y", dev.local_field(0.0), lambda s: nbr(s + x, aggr="sum")),
                    dev.local_field(100.0),
                )
                cond = x > 1.5
                z = branch(
                    cond,
                    lambda: rep("z", dev.local_field(0.0), lambda s: s + y + 10.0),
                    lambda: dev.local_field(-1.0),
                )
            return dev.result(x).item(), dev.result(y).item(), dev.result(z).item()

        r0 = run_device(dev0, src0, None)
        r1 = run_device(dev1, src1, None)
        r2 = run_device(dev2, src2, None)
        assert r0 == (1.0, 2.0, -1.0)
        assert r1 == (1.0, 100.0, -1.0)
        assert r2 == (1.0, 100.0, -1.0)

        snap = sim.snapshot("x", "y")
        r0 = run_device(dev0, src0, _DeviceSimulator.make_exports(snap, [1]))
        r1 = run_device(dev1, src1, _DeviceSimulator.make_exports(snap, [0, 2]))
        r2 = run_device(dev2, src2, _DeviceSimulator.make_exports(snap, [1]))
        assert r0 == (2.0, 9.0, 31.0)
        assert r1 == (2.0, 100.0, 220.0)
        assert r2 == (2.0, 100.0, 220.0)

        snap = sim.snapshot("x", "y")
        r0 = run_device(dev0, src0, _DeviceSimulator.make_exports(snap, [1]))
        r1 = run_device(dev1, src1, _DeviceSimulator.make_exports(snap, [0, 2]))
        r2 = run_device(dev2, src2, _DeviceSimulator.make_exports(snap, [1]))
        assert r0 == (3.0, 28.0, 69.0)
        assert r1 == (3.0, 100.0, 330.0)
        assert r2 == (3.0, 100.0, 330.0)

    def test_decentralized_multiple_auto_rep_no_tags(self):
        dev0 = DeviceContext(num_neighbors=3)
        dev1 = DeviceContext(num_neighbors=3)

        src0 = dev0.local_field(own=1.0, nbr=0.0)
        src1 = dev1.local_field(own=0.0, nbr=1.0)

        sim = _DeviceSimulator(dev0, dev1)

        def run_device(dev, src, exports):
            with dev.round(neighbor_exports=exports):
                x = rep(
                    dev.local_field(0.0),
                    lambda s: s + 1.0,
                    name="x",
                )
                y = mux(
                    src,
                    rep(
                        dev.local_field(0.0),
                        lambda s: nbr(s, aggr="sum") + x,
                        name="y",
                    ),
                    dev.local_field(100.0),
                )
            return dev.result(x).item(), dev.result(y).item()

        r0_1 = run_device(dev0, src0, None)
        r1_1 = run_device(dev1, src1, None)
        assert r0_1[0] == 1.0
        assert r1_1[0] == 1.0
        assert r1_1[1] == 100.0

        snap = sim.snapshot("x", "y")
        r0_2 = run_device(dev0, src0, _DeviceSimulator.make_exports(snap, [1, 1, 1]))
        r1_2 = run_device(dev1, src1, _DeviceSimulator.make_exports(snap, [0, 0, 0]))
        assert r0_2[0] == 2.0
        assert r1_2[0] == 2.0
        assert r1_2[1] == 100.0

        snap = sim.snapshot("x", "y")
        r0_3 = run_device(dev0, src0, _DeviceSimulator.make_exports(snap, [1, 1, 1]))
        r1_3 = run_device(dev1, src1, _DeviceSimulator.make_exports(snap, [0, 0, 0]))
        assert r0_3[0] == 3.0
        assert r1_3[0] == 3.0
        assert r1_3[1] == 100.0


class TestDeviceContextAutoNamed:
    def test_decentralized_auto_named_rep_accumulates(self):
        dev_a = DeviceContext(num_neighbors=3)
        dev_b = DeviceContext(num_neighbors=3)

        def run_counter(dev, nbr_vals):
            exports = None if nbr_vals is None else {"counter": nbr_vals}
            with dev.round(neighbor_exports=exports):
                val = rep(
                    dev.local_field(0.0),
                    lambda s: nbr(s, aggr="sum") + 1.0,
                    name="counter",
                )
            return dev.result(val).item()

        r_a1 = run_counter(dev_a, None)
        r_b1 = run_counter(dev_b, None)
        assert r_a1 == 1.0
        assert r_b1 == 1.0

        r_a2 = run_counter(dev_a, [r_b1] * 3)
        r_b2 = run_counter(dev_b, [r_a1] * 3)
        assert r_a2 == 5.0
        assert r_b2 == 5.0

        r_a3 = run_counter(dev_a, [r_b2] * 3)
        r_b3 = run_counter(dev_b, [r_a2] * 3)
        assert r_a3 == 21.0
        assert r_b3 == 21.0

    def test_decentralized_nbr_without_tag_uses_local_field(self):
        dev_a = DeviceContext(num_neighbors=3)
        dev_b = DeviceContext(num_neighbors=3)

        def run_sum(dev, nbr_vals):
            exports = None if nbr_vals is None else {"x": nbr_vals}
            with dev.round(neighbor_exports=exports):
                x = rep(
                    dev.local_field(0.0),
                    lambda s: s + 1.0,
                    name="x",
                )
                aggregated = nbr(x, aggr="sum")
            return dev.result(x).item(), dev.result(aggregated).item()

        xa1, agg_a1 = run_sum(dev_a, None)
        xb1, agg_b1 = run_sum(dev_b, None)
        assert xa1 == 1.0
        assert xb1 == 1.0

        xa2, agg_a2 = run_sum(dev_a, [xb1] * 3)
        xb2, agg_b2 = run_sum(dev_b, [xa1] * 3)
        assert xa2 == 2.0
        assert xb2 == 2.0

        assert agg_a2 > agg_a1
        assert agg_b2 > agg_b1

    def test_decentralized_auto_rep_with_branch(self):
        dev_a = DeviceContext(num_neighbors=3)
        dev_b = DeviceContext(num_neighbors=3)

        cond_a = dev_a.local_field(own=1.0, nbr=1.0)
        cond_b = dev_b.local_field(own=0.0, nbr=1.0)

        def run_branched(dev, cond, nbr_vals):
            exports = None if nbr_vals is None else {"branch_state": nbr_vals}
            with dev.round(neighbor_exports=exports):
                val = branch(
                    cond,
                    lambda: rep(
                        dev.local_field(0.0),
                        lambda s: nbr(s, aggr="sum") + 1.0,
                        name="branch_state",
                    ),
                    lambda: rep(
                        dev.local_field(0.0),
                        lambda s: nbr(s, aggr="sum") + 10.0,
                        name="branch_state",
                    ),
                )
            return dev.result(val).item()

        ra1 = run_branched(dev_a, cond_a, None)
        rb1 = run_branched(dev_b, cond_b, None)
        assert ra1 == 1.0
        assert rb1 == 10.0

        ra2 = run_branched(dev_a, cond_a, [rb1] * 3)
        rb2 = run_branched(dev_b, cond_b, [ra1] * 3)
        assert ra2 > ra1
        assert rb2 > rb1

        ra3 = run_branched(dev_a, cond_a, [rb2] * 3)
        rb3 = run_branched(dev_b, cond_b, [ra2] * 3)
        assert ra3 > ra2
        assert rb3 > rb2

    def test_decentralized_nbr_min_aggregation_without_tag(self):
        dev_a = DeviceContext(num_neighbors=3)
        dev_b = DeviceContext(num_neighbors=2)

        def run_min(dev, nbr_vals):
            exports = None if nbr_vals is None else {"val": nbr_vals}
            with dev.round(neighbor_exports=exports):
                x = rep(
                    dev.local_field(0.0),
                    lambda s: s + 1.0,
                    name="val",
                )
                min_nbr = nbr(x, aggr="min")
            return dev.result(x).item(), dev.result(min_nbr).item()

        xa1, min_a1 = run_min(dev_a, None)
        xb1, min_b1 = run_min(dev_b, None)
        assert xa1 == 1.0
        assert xb1 == 1.0

        xa2, min_a2 = run_min(dev_a, [xb1] * 3)
        xb2, min_b2 = run_min(dev_b, [xa1] * 2)
        assert xa2 == 2.0
        assert xb2 == 2.0

        assert torch.isfinite(torch.tensor(min_a2))
        assert torch.isfinite(torch.tensor(min_b2))


class TestDeviceContextEdgeCases:
    def test_tagged_neighbor_messages_override_local_expression(self):
        device = DeviceContext(num_neighbors=2)
        expr = device.local_field(own=10.0, nbr=[10.0, 10.0])

        with device.round(neighbor_messages={"chan": [2.0, 3.0]}):
            result = nbr(expr, aggr="sum", tag="chan")

        assert device.result(result).item() == 5.0

    def test_include_self_overrides_device_self_loop_topology(self):
        device = DeviceContext(num_neighbors=1, self_loop=True)
        expr = device.local_field(own=4.0, nbr=[2.0])

        with device.round():
            without_self = nbr(expr, aggr="sum", include_self=False)
        with device.round():
            with_self = nbr(expr, aggr="sum", include_self=True)

        assert device.result(without_self).item() == 2.0
        assert device.result(with_self).item() == 6.0

    def test_decentralized_rep_with_incompatible_init_values(self):
        dev = DeviceContext(num_neighbors=3)
        cond = dev.local_field(own=1.0, nbr=1.0)

        def run_scaled_init(nbr_vals):
            exports = None if nbr_vals is None else {"scaled": nbr_vals}
            with dev.round(neighbor_exports=exports):
                val = branch(
                    cond,
                    lambda: rep(
                        dev.local_field(1.0),
                        lambda s: nbr(s, aggr="sum") * 0.0 + 1.0,
                        name="scaled",
                    ),
                    lambda: rep(
                        dev.local_field(10000.0),
                        lambda s: nbr(s, aggr="sum") * 0.0 + 10000.0,
                        name="scaled",
                    ),
                )
            return dev.result(val).item()

        r1 = run_scaled_init(None)
        assert r1 == 1.0

        r2 = run_scaled_init([r1] * 3)
        assert r2 == 1.0

        cond_false = dev.local_field(own=0.0, nbr=1.0)

        def run_scaled_init_false(nbr_vals):
            exports = None if nbr_vals is None else {"scaled": nbr_vals}
            with dev.round(neighbor_exports=exports):
                val = branch(
                    cond_false,
                    lambda: rep(
                        dev.local_field(1.0),
                        lambda s: nbr(s, aggr="sum") * 0.0 + 1.0,
                        name="scaled",
                    ),
                    lambda: rep(
                        dev.local_field(10000.0),
                        lambda s: nbr(s, aggr="sum") * 0.0 + 10000.0,
                        name="scaled",
                    ),
                )
            return dev.result(val).item()

        r3 = run_scaled_init_false([r2] * 3)
        assert r3 == 10000.0

    def test_decentralized_nbr_aggregates_incompatible_values_across_partitions(self):
        dev_a = DeviceContext(num_neighbors=3)
        dev_b = DeviceContext(num_neighbors=3)

        cond_a = dev_a.local_field(own=1.0, nbr=1.0)
        cond_b = dev_b.local_field(own=0.0, nbr=1.0)

        def run_cross_pollution(dev, cond, nbr_vals):
            exports = None if nbr_vals is None else {"val": nbr_vals}
            with dev.round(neighbor_exports=exports):
                val = branch(
                    cond,
                    lambda: rep(
                        dev.local_field(1.0),
                        lambda s: nbr(s, aggr="sum") + 0.0,
                        name="val",
                    ),
                    lambda: rep(
                        dev.local_field(100.0),
                        lambda s: nbr(s, aggr="sum") + 0.0,
                        name="val",
                    ),
                )
            return dev.result(val).item()

        ra1 = run_cross_pollution(dev_a, cond_a, None)
        rb1 = run_cross_pollution(dev_b, cond_b, None)
        assert ra1 == 1.0
        assert rb1 == 100.0

        ra2 = run_cross_pollution(dev_a, cond_a, [rb1] * 3)
        rb2 = run_cross_pollution(dev_b, cond_b, [ra1] * 3)
        assert ra2 > 100.0
        assert rb2 >= 100.0
