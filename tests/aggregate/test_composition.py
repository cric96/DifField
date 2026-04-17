"""Tests for multi-primitive composition patterns and integration scenarios."""

from __future__ import annotations

import torch

from autofield import branch, mux, scatter, iterate, gather_max, gather_sum
from autofield.dsl import field
from conftest import field_zeros
from tests.aggregate.support import ROUNDS, flags, values


class TestIterateBranchMuxComposition:
    def test_mux_iterate_branch_composition(self, line_ctx):
        cond_mux = flags(True, True, False, False)
        cond_branch = flags(True, True, True, False)

        results = []
        for _ in range(ROUNDS):
            with line_ctx.round():
                val = mux(
                    cond_mux,
                    iterate(
                        field.zeros(),
                        lambda outer_s: branch(
                            cond_branch,
                            lambda: gather_sum(scatter(outer_s)) + 1.0,
                            lambda: gather_max(scatter(outer_s)) + 10.0,
                        ),
                    ),
                    iterate(
                        field.zeros(),
                        lambda s: gather_sum(scatter(s)) + 100.0,
                    ),
                )
            results.append(val.clone())

        assert torch.allclose(results[0], values(1.0, 1.0, 100.0, 100.0))
        assert torch.allclose(results[1], values(2.0, 3.0, 300.0, 200.0))
        assert torch.allclose(results[2], values(4.0, 5.0, 600.0, 400.0))

    def test_iterate_branch_iterate_composition(self, line_ctx):
        cond = flags(True, True, False, False)

        results = []
        for _ in range(ROUNDS):
            with line_ctx.round():
                val = iterate(
                    field.zeros(),
                    lambda outer_s: branch(
                        cond,
                        lambda: iterate(
                            field.zeros(),
                            lambda inner_s: (
                                gather_sum(scatter(outer_s)) + gather_sum(scatter(inner_s)) + 1.0
                            ),
                        ),
                        lambda: iterate(
                            field.zeros(),
                            lambda inner_s: (
                                gather_max(scatter(outer_s)) + gather_max(scatter(inner_s)) + 10.0
                            ),
                        ),
                    ),
                )
            results.append(val.clone())

        assert torch.allclose(results[0], values(1.0, 1.0, 10.0, 10.0))
        assert torch.allclose(results[1], values(3.0, 3.0, 30.0, 30.0))
        assert torch.allclose(results[2], values(7.0, 7.0, 70.0, 70.0))


class TestMultipleAssignmentsComposition:
    def test_multiple_assignments_composition(self, line_ctx):
        cond = flags(True, True, False, False)

        results_x = []
        results_y = []
        results_z = []

        for _ in range(ROUNDS):
            with line_ctx.round():
                x = iterate(field.zeros(), lambda s: s + 1.0)
                y = branch(
                    cond,
                    lambda: iterate(field.zeros(), lambda s: gather_sum(scatter(s)) + x),
                    lambda: iterate(field.zeros(), lambda s: gather_max(scatter(s)) + x * 2),
                )
                z = iterate(field.zeros(), lambda s: gather_sum(scatter(s + y)))

            results_x.append(x.clone())
            results_y.append(y.clone())
            results_z.append(z.clone())

        assert torch.allclose(results_x[0], values(1.0, 1.0, 1.0, 1.0))
        assert torch.allclose(results_y[0], values(1.0, 1.0, 2.0, 2.0))
        assert torch.allclose(results_z[0], values(1.0, 3.0, 3.0, 2.0))
        assert torch.allclose(results_x[1], values(2.0, 2.0, 2.0, 2.0))
        assert torch.allclose(results_y[1], values(3.0, 3.0, 6.0, 6.0))
        assert torch.allclose(results_z[1], values(6.0, 13.0, 14.0, 9.0))
        assert torch.allclose(results_x[2], values(3.0, 3.0, 3.0, 3.0))
        assert torch.allclose(results_y[2], values(6.0, 6.0, 12.0, 12.0))
        assert torch.allclose(results_z[2], values(19.0, 38.0, 40.0, 26.0))


class TestNestedIterateComposition:
    def test_nested_iterate_multiple_scatter(self, line_ctx):
        zero_field = field_zeros(line_ctx)
        results = []
        for _ in range(ROUNDS):
            with line_ctx.round():
                val = iterate(
                    zero_field,
                    lambda outer_s: iterate(
                        zero_field,
                        lambda inner_s: (
                            gather_sum(scatter(outer_s)) + gather_sum(scatter(inner_s)) + 1.0
                        ),
                    ),
                )
            results.append(val.clone())

        assert torch.allclose(results[0], values(1.0, 1.0, 1.0, 1.0))
        assert torch.allclose(results[1], values(3.0, 5.0, 5.0, 3.0))
        assert torch.allclose(results[2], values(11.0, 17.0, 17.0, 11.0))
