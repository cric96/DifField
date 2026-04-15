"""Tests for multi-primitive composition patterns and integration scenarios."""

from __future__ import annotations

import torch

from autofield import branch, mux, nbr, rep, maxhood, sumhood
from autofield.dsl import field
from conftest import field_zeros
from tests.aggregate.support import ROUNDS, flags, values


class TestRepBranchMuxComposition:
    def test_mux_rep_branch_composition(self, line_ctx):
        cond_mux = flags(True, True, False, False)
        cond_branch = flags(True, True, True, False)

        results = []
        for _ in range(ROUNDS):
            with line_ctx.round():
                val = mux(
                    cond_mux,
                    rep(
                        field.zeros(),
                        lambda outer_s: branch(
                            cond_branch,
                            lambda: sumhood(nbr(outer_s)) + 1.0,
                            lambda: maxhood(nbr(outer_s)) + 10.0,
                        ),
                    ),
                    rep(
                        field.zeros(),
                        lambda s: sumhood(nbr(s)) + 100.0,
                    ),
                )
            results.append(val.clone())

        assert torch.allclose(results[0], values(1.0, 1.0, 100.0, 100.0))
        assert torch.allclose(results[1], values(2.0, 3.0, 300.0, 200.0))
        assert torch.allclose(results[2], values(4.0, 5.0, 600.0, 400.0))

    def test_rep_branch_rep_composition(self, line_ctx):
        cond = flags(True, True, False, False)

        results = []
        for _ in range(ROUNDS):
            with line_ctx.round():
                val = rep(
                    field.zeros(),
                    lambda outer_s: branch(
                        cond,
                        lambda: rep(
                            field.zeros(),
                            lambda inner_s: (
                                sumhood(nbr(outer_s)) + sumhood(nbr(inner_s)) + 1.0
                            ),
                        ),
                        lambda: rep(
                            field.zeros(),
                            lambda inner_s: (
                                maxhood(nbr(outer_s)) + maxhood(nbr(inner_s)) + 10.0
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
                x = rep(field.zeros(), lambda s: s + 1.0)
                y = branch(
                    cond,
                    lambda: rep(field.zeros(), lambda s: sumhood(nbr(s)) + x),
                    lambda: rep(field.zeros(), lambda s: maxhood(nbr(s)) + x * 2),
                )
                z = rep(field.zeros(), lambda s: sumhood(nbr(s + y)))

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


class TestNestedRepComposition:
    def test_nested_rep_multiple_nbr(self, line_ctx):
        zero_field = field_zeros(line_ctx)
        results = []
        for _ in range(ROUNDS):
            with line_ctx.round():
                val = rep(
                    zero_field,
                    lambda outer_s: rep(
                        zero_field,
                        lambda inner_s: (
                            sumhood(nbr(outer_s)) + sumhood(nbr(inner_s)) + 1.0
                        ),
                    ),
                )
            results.append(val.clone())

        assert torch.allclose(results[0], values(1.0, 1.0, 1.0, 1.0))
        assert torch.allclose(results[1], values(3.0, 5.0, 5.0, 3.0))
        assert torch.allclose(results[2], values(11.0, 17.0, 17.0, 11.0))
