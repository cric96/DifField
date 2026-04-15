"""Tests for control flow primitives: branch and mux."""

from __future__ import annotations

import torch

from autofield import (
    AggregateContext,
    branch,
    mux,
    nbr,
    rep,
    minhood,
    maxhood,
    sumhood,
)
from autofield.dsl import field
from conftest import field_from_values, field_zeros
from tests.aggregate.support import (
    ROUNDS,
    SOFT_BRANCH_TAU,
    SOFT_BRANCH_MAX_ERROR,
    flags,
    values,
)


class TestBranch:
    def test_isolation(self, line_ctx):
        cond = flags(True, True, False, False)
        x = field_from_values(line_ctx, [1.0, 2.0, 10.0, 20.0])

        with line_ctx.round():
            result = branch(
                cond,
                lambda: sumhood(nbr(x)),
                lambda: sumhood(nbr(x)),
            )
        assert torch.allclose(result, values(2.0, 1.0, 20.0, 10.0))

    def test_different_nbr_in_branches(self, line_ctx):
        cond = flags(True, True, False, False)
        x = field_from_values(line_ctx, [1.0, 2.0, 30.0, 10.0])

        with line_ctx.round():
            result = branch(
                cond,
                lambda: sumhood(nbr(x)),
                lambda: minhood(nbr(x)),
            )
        assert torch.allclose(result, values(2.0, 1.0, 10.0, 30.0))

    def test_state_reset_on_switch(self, line_ctx):
        cond1 = flags(True, True, True, True)
        with line_ctx.round():
            branch(
                cond1,
                lambda: rep(field.zeros(), lambda s: s + 1, name="val"),
                lambda: rep(field.zeros(), lambda s: s + 1, name="val"),
                reset_states={"val": field.zeros()},
            )
        assert torch.allclose(
            line_ctx.state.get_state(name="val"), values(1.0, 1.0, 1.0, 1.0)
        )

        cond2 = flags(True, True, True, False)
        with line_ctx.round():
            branch(
                cond2,
                lambda: rep(field.zeros(), lambda s: s + 1, name="val"),
                lambda: rep(field.zeros(), lambda s: s + 1, name="val"),
                reset_states={"val": field.zeros()},
            )
        assert torch.allclose(
            line_ctx.state.get_state(name="val"), values(2.0, 2.0, 2.0, 1.0)
        )

    def test_rep_nbr_nested(self, line_ctx):
        zero_field = field_zeros(line_ctx)
        cond = flags(True, True, False, False)

        results = []
        for _ in range(ROUNDS):
            with line_ctx.round():
                val = branch(
                    cond,
                    lambda: rep(
                        zero_field,
                        lambda s: sumhood(nbr(s)) + 1.0,
                        name="true_val",
                    ),
                    lambda: rep(
                        zero_field,
                        lambda s: maxhood(nbr(s)) + 10.0,
                        name="false_val",
                    ),
                )
            results.append(val.clone())

        assert torch.allclose(results[0], values(1.0, 1.0, 10.0, 10.0))
        assert torch.allclose(results[1], values(2.0, 2.0, 20.0, 20.0))
        assert torch.allclose(results[2], values(3.0, 3.0, 30.0, 30.0))

    def test_soft_branch_nbr_approximates_hard_isolation(self, line_topology):
        edge_index, n = line_topology
        cond = flags(True, True, False, False)

        hard_ctx = AggregateContext(edge_index, n)
        hard_x = field_from_values(hard_ctx, [10.0, 20.0, 30.0, 40.0])
        with hard_ctx.round():
            hard = branch(
                cond,
                lambda: sumhood(nbr(hard_x)),
                lambda: maxhood(nbr(hard_x)),
                branch_name="soft_vs_hard",
                mode="hard",
            )

        soft_ctx = AggregateContext(edge_index, n)
        soft_x = field_from_values(soft_ctx, [10.0, 20.0, 30.0, 40.0])
        with soft_ctx.round():
            soft = branch(
                cond,
                lambda: sumhood(nbr(soft_x)),
                lambda: maxhood(nbr(soft_x)),
                branch_name="soft_vs_hard",
                mode="soft",
                tau=SOFT_BRANCH_TAU,
            )

        assert soft.shape == hard.shape
        assert torch.isfinite(soft).all()
        assert (soft - hard).abs().max() < SOFT_BRANCH_MAX_ERROR

    def test_state_reset_on_partition_switch(self, line_ctx):
        cond1 = flags(True, True, False, False)
        cond2 = flags(True, False, False, False)

        with line_ctx.round():
            branch(
                cond1,
                lambda: rep(field.zeros(), lambda s: s + 5.0, name="reset_rep"),
                lambda: rep(field.zeros(), lambda s: s + 1.0, name="reset_rep"),
                branch_name="switch_test",
                reset_states={"reset_rep": field.zeros()},
            )

        with line_ctx.round():
            result = branch(
                cond2,
                lambda: rep(field.zeros(), lambda s: s + 5.0, name="reset_rep"),
                lambda: rep(field.zeros(), lambda s: s + 1.0, name="reset_rep"),
                branch_name="switch_test",
                reset_states={"reset_rep": field.zeros()},
            )

        assert torch.allclose(result, values(10.0, 1.0, 2.0, 2.0))


class TestMux:
    def test_no_isolation(self, line_ctx):
        cond = flags(True, True, False, False)
        x = field_from_values(line_ctx, [1.0, 2.0, 10.0, 20.0])

        with line_ctx.round():
            result = mux(
                cond.float(),
                lambda: sumhood(nbr(x)),
                lambda: sumhood(nbr(x)),
            )
        assert torch.allclose(result, values(2.0, 11.0, 22.0, 10.0))

    def test_selection(self, line_ctx):
        cond = values(1.0, 1.0, 0.0, 0.0)
        if_true = field_from_values(line_ctx, [10.0, 20.0, 30.0, 40.0])
        if_false = field_from_values(line_ctx, [100.0, 200.0, 300.0, 400.0])

        with line_ctx.round():
            result = mux(cond, if_true, if_false)
        assert torch.allclose(result, values(10.0, 20.0, 300.0, 400.0))


class TestLocalAggregateRoundPattern:
    def test_full_local_aggregate_round_pattern(self, line_ctx):
        cond = flags(True, True, False, False)
        local_input = field_from_values(line_ctx, [1.0, 2.0, 3.0, 4.0])

        results = []
        for _ in range(ROUNDS):
            with line_ctx.round():
                val = rep(
                    field.zeros(),
                    lambda s: branch(
                        cond,
                        lambda: sumhood(nbr(s)) + local_input,
                        lambda: (
                            maxhood(nbr(s), fill_value=float("-inf")) + local_input * 2
                        ),
                        branch_name="local_round",
                    ),
                )
            results.append(val.clone())

        assert torch.allclose(results[0], values(1.0, 2.0, 6.0, 8.0))
        assert torch.allclose(results[1], values(3.0, 3.0, 14.0, 14.0))
        assert torch.allclose(results[2], values(4.0, 5.0, 20.0, 22.0))
