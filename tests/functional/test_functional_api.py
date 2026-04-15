"""Tests for exported functional primitives and edge masking helpers."""

from __future__ import annotations

import pytest
import torch
from hypothesis import given, settings
from hypothesis import strategies as st

from autofield import field_where, mask_edges, mask_edges_for_partition, scatter_aggr
from autofield.functional import scatter_min_by_first


class TestScatterAggr:
    def test_sum(self):
        src = torch.tensor([1.0, 2.0, 3.0])
        index = torch.tensor([0, 0, 1])
        out = scatter_aggr(src, index, 2, aggr="sum")
        assert torch.allclose(out, torch.tensor([3.0, 3.0]))

    def test_mean(self):
        src = torch.tensor([1.0, 3.0, 5.0])
        index = torch.tensor([0, 0, 1])
        out = scatter_aggr(src, index, 2, aggr="mean")
        assert torch.allclose(out, torch.tensor([2.0, 5.0]))

    def test_min_hard(self):
        src = torch.tensor([3.0, 1.0, 5.0])
        index = torch.tensor([0, 0, 1])
        out = scatter_aggr(
            src, index, 2, aggr="min", mode="hard", fill_value=float("inf")
        )
        assert torch.allclose(out, torch.tensor([1.0, 5.0]))

    def test_min_soft(self):
        src = torch.tensor([3.0, 1.0, 5.0])
        index = torch.tensor([0, 0, 1])
        out = scatter_aggr(
            src, index, 2, aggr="min", mode="soft", tau=0.01, fill_value=float("inf")
        )
        assert torch.allclose(out, torch.tensor([1.0, 5.0]), atol=0.05)

    def test_max_hard(self):
        src = torch.tensor([3.0, 1.0, 5.0])
        index = torch.tensor([0, 0, 1])
        out = scatter_aggr(
            src, index, 2, aggr="max", mode="hard", fill_value=float("-inf")
        )
        assert torch.allclose(out, torch.tensor([3.0, 5.0]))

    def test_min_by_first_hard_keeps_last_equal_minimum(self):
        src = torch.tensor([[2.0, 20.0], [1.0, 10.0], [1.0, 30.0], [5.0, 50.0]])
        index = torch.tensor([0, 0, 0, 1])
        fill_row = torch.tensor([[float("inf"), -1.0], [float("inf"), -1.0]])

        out = scatter_min_by_first(src, index, 2, mode="hard", fill_row=fill_row)

        assert torch.allclose(out, torch.tensor([[1.0, 30.0], [5.0, 50.0]]))

    def test_custom_callable(self):
        src = torch.tensor([1.0, 2.0, 3.0])
        index = torch.tensor([0, 0, 1])

        def shifted_sum(msg, bucket_index, num_nodes):
            return scatter_aggr(msg, bucket_index, num_nodes, aggr="sum") + 1.0

        out = scatter_aggr(
            src, index, 2, aggr=shifted_sum, mode="soft", tau=0.1, fill_value=-123.0
        )
        assert torch.allclose(out, torch.tensor([4.0, 4.0]))

    def test_min_fill_value_for_isolated_nodes(self):
        src = torch.tensor([3.0, 1.0])
        index = torch.tensor([0, 0])
        out = scatter_aggr(src, index, 3, aggr="min", mode="hard", fill_value=99.0)
        assert torch.allclose(out, torch.tensor([1.0, 99.0, 99.0]))

    def test_soft_min_small_tau_has_finite_gradients(self):
        src = torch.tensor([3.0, 1.0, 5.0], requires_grad=True)
        index = torch.tensor([0, 0, 1])

        out = scatter_aggr(
            src, index, 2, aggr="min", mode="soft", tau=1e-2, fill_value=float("inf")
        )
        out[0].backward()

        assert src.grad is not None
        assert torch.isfinite(src.grad).all()
        assert src.grad.abs().sum().item() > 0.0

    def test_invalid_aggregation_raises(self):
        with pytest.raises(ValueError, match="Unknown aggregation"):
            scatter_aggr(torch.tensor([1.0]), torch.tensor([0]), 1, aggr="median")

    @settings(deadline=None)
    @given(
        values=st.lists(
            st.floats(min_value=-100, max_value=100), min_size=1, max_size=20
        ),
        indices=st.lists(
            st.integers(min_value=0, max_value=5), min_size=1, max_size=20
        ),
    )
    def test_scatter_sum_property(self, values, indices):
        n = min(len(values), len(indices))
        src = torch.tensor(values[:n], dtype=torch.float32)
        idx = torch.tensor(indices[:n], dtype=torch.long)
        num_nodes = 6

        out = scatter_aggr(src, idx, num_nodes, aggr="sum")

        expected = torch.zeros(num_nodes)
        for i, val in zip(idx.tolist(), src.tolist()):
            expected[i] += val

        assert torch.allclose(out, expected, atol=1e-4)


class TestFieldWhere:
    def test_basic_hard(self):
        cond = torch.tensor([1.0, 0.0, 1.0])
        x = torch.tensor([10.0, 20.0, 30.0])
        y = torch.tensor([100.0, 200.0, 300.0])
        result = field_where(cond, x, y, mode="hard")
        assert torch.allclose(result, torch.tensor([10.0, 200.0, 30.0]))

    def test_boolean_cond_hard(self):
        cond = torch.tensor([True, False])
        x = torch.tensor([1.0, 2.0])
        y = torch.tensor([3.0, 4.0])
        result = field_where(cond, x, y, mode="hard")
        assert torch.allclose(result, torch.tensor([1.0, 4.0]))

    def test_avoids_nan_from_inactive_infinite_branch(self):
        cond = torch.tensor([1.0, 0.0])
        x = torch.tensor([5.0, float("inf")])
        y = torch.tensor([float("inf"), 7.0])
        result = field_where(cond, x, y, mode="hard")
        assert torch.allclose(result, torch.tensor([5.0, 7.0]))

    def test_soft_mode_interpolates(self):
        cond = torch.tensor([1.0, 0.0, 0.5])
        x = torch.tensor([10.0, 20.0, 30.0])
        y = torch.tensor([100.0, 200.0, 300.0])
        result = field_where(cond, x, y, mode="soft", tau=100.0)
        assert torch.allclose(result[0], torch.tensor(10.0), atol=0.01)
        assert torch.allclose(result[1], torch.tensor(200.0), atol=0.01)
        assert torch.allclose(result[2], torch.tensor(165.0), atol=1.0)

    def test_soft_mode_gradient_wrt_cond(self):
        cond = torch.tensor([0.6], requires_grad=True)
        x = torch.tensor([10.0])
        y = torch.tensor([0.0])
        result = field_where(cond, x, y, mode="soft", tau=10.0)
        result.backward()
        assert cond.grad is not None
        assert cond.grad.abs().item() > 0.0

    def test_default_mode_uses_global(self):
        from autofield import get_default_mode, set_default_mode, with_mode

        assert get_default_mode() == "hard"
        cond = torch.tensor([0.5])
        x = torch.tensor([1.0])
        y = torch.tensor([2.0])
        result = field_where(cond, x, y)
        assert torch.allclose(result, torch.tensor([1.0]))

        with with_mode("soft"):
            assert get_default_mode() == "soft"
        assert get_default_mode() == "hard"


class TestMaskEdges:
    def test_hard_mask(self, line_topology):
        edge_index, _ = line_topology
        cond = torch.tensor([True, True, False, False])
        ei_out, edge_weight = mask_edges(edge_index, cond, mode="hard")
        assert ei_out.shape[1] == 4
        assert torch.allclose(edge_weight, torch.ones(4))

    def test_soft_mask_returns_continuous_weights(self, line_topology):
        edge_index, _ = line_topology
        cond = torch.tensor([1.0, 0.8, 0.2, 0.0])
        _, edge_weight = mask_edges(edge_index, cond, mode="soft", tau=8.0)
        assert edge_weight.shape[0] == edge_index.shape[1]
        assert ((edge_weight >= 0.0) & (edge_weight <= 1.0)).all()
        assert edge_weight[0] > edge_weight[2]

    def test_mask_edges_for_partition_hard_filters_auxiliary_weights(
        self, line_topology
    ):
        edge_index, _ = line_topology
        cond = torch.tensor([True, True, False, False])
        edge_weight = torch.tensor([1.0, 1.0, 2.0, 2.0, 3.0, 3.0])
        message_weight = torch.tensor([10.0, 11.0, 20.0, 21.0, 30.0, 31.0])

        ei_out, ew_out, mw_out = mask_edges_for_partition(
            edge_index,
            cond,
            partition=True,
            edge_weight=edge_weight,
            message_weight=message_weight,
            mode="hard",
        )

        assert torch.equal(ei_out, torch.tensor([[0, 1], [1, 0]], dtype=torch.long))
        assert torch.allclose(ew_out, torch.tensor([1.0, 1.0]))
        assert torch.allclose(mw_out, torch.tensor([10.0, 11.0]))

    def test_mask_edges_for_partition_soft_composes_message_weight(self, line_topology):
        edge_index, _ = line_topology
        cond = torch.tensor([1.0, 0.8, 0.2, 0.0])
        message_weight = torch.full((edge_index.shape[1],), 2.0)

        _, _, mw_out = mask_edges_for_partition(
            edge_index,
            cond,
            partition=True,
            message_weight=message_weight,
            mode="soft",
            tau=6.0,
        )

        assert mw_out is not None
        assert ((mw_out >= 0.0) & (mw_out <= 2.0)).all()
        assert mw_out[0] > mw_out[2]
