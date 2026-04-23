"""Tests for field helpers: const, field.of/zeros/ones/inf, mid."""

from __future__ import annotations

import torch

from diffield.dsl import const, mid, field
from conftest import field_zeros, field_ones, field_of, field_mid


class TestConst:
    def test_const(self, triangle_ctx):
        with triangle_ctx.round():
            value = const(2.5)
        assert torch.allclose(value, field_of(triangle_ctx, 2.5))


class TestFieldFactories:
    def test_of(self, triangle_ctx):
        expected = field_of(triangle_ctx, 3.14)
        with triangle_ctx.round():
            f = field.of(3.14)
        assert f.shape == expected.shape
        assert torch.allclose(f, expected)

    def test_zeros_ones_inf(self, triangle_ctx):
        with triangle_ctx.round():
            z = field.zeros()
            o = field.ones()
            i = field.inf()
        assert torch.allclose(z, field_zeros(triangle_ctx))
        assert torch.allclose(o, field_ones(triangle_ctx))
        assert (i == float("inf")).all()


class TestMid:
    def test_returns_node_ids(self, line_ctx):
        expected = field_mid(line_ctx)
        with line_ctx.round():
            node_ids = mid()
        assert torch.allclose(node_ids, expected)
