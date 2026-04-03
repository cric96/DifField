"""Tests for single-device local execution via DeviceContext."""

from __future__ import annotations

import torch

from autofield import AggregateContext, branch, gradient, mux, nbr, nbr_range, rep
from autofield.dsl import DeviceContext, field


class TestDeviceContext:
    def test_local_field(self):
        device = DeviceContext(num_neighbors=3)
        f = device.local_field(own=1.0, nbr=0.0)
        assert f.shape == (4,)
        assert f[0] == 1.0
        assert (f[1:] == 0.0).all()

    def test_result(self):
        t = torch.tensor([42.0, 1.0, 2.0])
        assert DeviceContext.result(t) == 42.0

    def test_local_matches_global(self):
        edge_index = torch.tensor(
            [
                [0, 1, 1, 2, 0, 1, 2],
                [1, 0, 2, 1, 0, 1, 2],
            ],
            dtype=torch.long,
        )
        source = torch.tensor([1.0, 0.0, 0.0])
        w = torch.tensor(1.0)

        ctx = AggregateContext(edge_index, 3)
        global_states = []
        for _ in range(4):
            with ctx.round():
                d = rep("dist", float("inf"), lambda dist: mux(source, field.of(0.0), nbr(dist + w, aggr="min")))
            global_states.append(d.detach().clone())

        device_id = 2
        device = DeviceContext(num_neighbors=1)
        source_local = device.local_field(own=source[device_id].item(), nbr=0.0)
        for round_idx in range(4):
            neighbor_exports = None if round_idx == 0 else {"dist": [global_states[round_idx - 1][1].item()]}
            with device.round(neighbor_exports=neighbor_exports):
                d_local = rep("dist", float("inf"), lambda dist: mux(source_local, field.of(0.0), nbr(dist + w, aggr="min")))

        assert abs(device.result(d_local).item() - global_states[-1][device_id].item()) < 1e-6

    def test_isolated_device(self):
        device = DeviceContext(num_neighbors=0)
        source_local = device.local_field(own=1.0)
        w = torch.tensor(1.0)
        for _ in range(3):
            with device.round():
                d = rep("dist", float("inf"), lambda dist: mux(source_local, field.of(0.0), nbr(dist + w, aggr="min")))
        assert device.result(d).item() == 0.0

    def test_neighbor_ranges_available_locally(self):
        device = DeviceContext(num_neighbors=2)

        with device.round(neighbor_ranges=[1.5, 2.5]):
            ranges = nbr(nbr_range(), aggr="sum")

        assert abs(device.result(ranges).item() - 4.0) < 1e-6

    def test_local_weighted_gradient_matches_global(self):
        edge_index = torch.tensor(
            [
                [0, 1, 1, 2],
                [1, 0, 2, 1],
            ],
            dtype=torch.long,
        )
        edge_weight = torch.tensor([2.0, 2.0, 3.0, 3.0])
        source = torch.tensor([1.0, 0.0, 0.0])

        ctx = AggregateContext(edge_index, 3, edge_weight=edge_weight)
        global_states = []
        for _ in range(4):
            with ctx.round():
                d = gradient(source, name="dist")
            global_states.append(d.detach().clone())

        device = DeviceContext(num_neighbors=1, self_loop=False)
        source_local = device.local_field(own=0.0, nbr=0.0)
        for round_idx in range(4):
            neighbor_exports = None if round_idx == 0 else {"_grad_dist": [global_states[round_idx - 1][1].item()]}
            with device.round(neighbor_exports=neighbor_exports, neighbor_ranges=[3.0]):
                d_local = gradient(source_local, name="dist")

        assert abs(device.result(d_local).item() - global_states[-1][2].item()) < 1e-6

    def test_decentralized_branching(self):
        dev_a = DeviceContext(num_neighbors=1)
        dev_b = DeviceContext(num_neighbors=2)
        dev_c = DeviceContext(num_neighbors=1)

        cond_a = dev_a.local_field(own=1.0, nbr=1.0)
        cond_b = dev_b.local_field(own=1.0, nbr=[1.0, 0.0])
        cond_c = dev_c.local_field(own=0.0, nbr=1.0)

        def run_branching(dev, cond, nbr_states):
            exports = None if nbr_states is None else {"state": nbr_states}
            with dev.round(neighbor_exports=exports):
                val = branch(
                    cond,
                    lambda: rep("state", dev.local_field(0.0), lambda s: nbr(s, aggr="sum") + 1.0),
                    lambda: rep("state", dev.local_field(0.0), lambda s: nbr(s, aggr="max") + 10.0),
                )
            return dev.result(val).item()

        r_a1 = run_branching(dev_a, cond_a, None)
        r_b1 = run_branching(dev_b, cond_b, None)
        r_c1 = run_branching(dev_c, cond_c, None)
        assert r_a1 == 1.0
        assert r_b1 == 1.0
        assert r_c1 == 10.0

        r_a2 = run_branching(dev_a, cond_a, [r_b1])
        r_b2 = run_branching(dev_b, cond_b, [r_a1, r_c1])
        r_c2 = run_branching(dev_c, cond_c, [r_b1])
        assert r_a2 == 3.0
        assert r_b2 == 3.0
        assert r_c2 == 20.0

        r_a3 = run_branching(dev_a, cond_a, [r_b2])
        r_b3 = run_branching(dev_b, cond_b, [r_a2, r_c2])
        r_c3 = run_branching(dev_c, cond_c, [r_b2])
        assert r_a3 == 7.0
        assert r_b3 == 7.0
        assert r_c3 == 30.0

    def test_decentralized_multiple_assignments_nested(self):
        dev0 = DeviceContext(num_neighbors=1)
        dev1 = DeviceContext(num_neighbors=2)
        dev2 = DeviceContext(num_neighbors=1)

        src0 = dev0.local_field(own=1.0, nbr=0.0)
        src1 = dev1.local_field(own=0.0, nbr=[1.0, 0.0])
        src2 = dev2.local_field(own=0.0, nbr=0.0)

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

        def snapshot_states(*devices):
            return [{"x": device.get_state("x"), "y": device.get_state("y")} for device in devices]

        def make_exports(snap, nbr_indices):
            return {
                "x": [snap[idx]["x"] for idx in nbr_indices],
                "y": [snap[idx]["y"] for idx in nbr_indices],
            }

        r0 = run_device(dev0, src0, None)
        r1 = run_device(dev1, src1, None)
        r2 = run_device(dev2, src2, None)
        assert r0 == (1.0, 2.0, -1.0)
        assert r1 == (1.0, 100.0, -1.0)
        assert r2 == (1.0, 100.0, -1.0)

        snap = snapshot_states(dev0, dev1, dev2)
        r0 = run_device(dev0, src0, make_exports(snap, [1]))
        r1 = run_device(dev1, src1, make_exports(snap, [0, 2]))
        r2 = run_device(dev2, src2, make_exports(snap, [1]))
        assert r0 == (2.0, 9.0, 31.0)
        assert r1 == (2.0, 100.0, 220.0)
        assert r2 == (2.0, 100.0, 220.0)

        snap = snapshot_states(dev0, dev1, dev2)
        r0 = run_device(dev0, src0, make_exports(snap, [1]))
        r1 = run_device(dev1, src1, make_exports(snap, [0, 2]))
        r2 = run_device(dev2, src2, make_exports(snap, [1]))
        assert r0 == (3.0, 28.0, 69.0)
        assert r1 == (3.0, 100.0, 330.0)
        assert r2 == (3.0, 100.0, 330.0)

    def test_tagged_neighbor_messages_override_local_expression(self):
        device = DeviceContext(num_neighbors=2)
        expr = device.local_field(own=10.0, nbr=[10.0, 10.0])

        with device.round(neighbor_messages={"chan": [2.0, 3.0]}):
            result = nbr(expr, aggr="sum", tag="chan")

        assert device.result(result).item() == 5.0