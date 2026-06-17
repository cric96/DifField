"""Tests for control flow constructs: branch and mux."""

from __future__ import annotations

import torch

from conftest import field_from_values
from diffield import AggregateContext
from diffield.dsl import (
    branch,
    field,
    gather_max,
    gather_sum,
    iterate,
    mux,
    scatter,
)
from tests.aggregate.support import ROUNDS, flags, values


class TestBranch:
    def test_isolates_communication_between_partitions(self, line_ctx):
        cond = flags(True, True, False, False)
        x = field_from_values(line_ctx, [1.0, 2.0, 10.0, 20.0])

        with line_ctx.round():
            result = branch(
                cond,
                lambda: gather_sum(scatter(x)),
                lambda: gather_sum(scatter(x)),
                branch_name="isolation_test",
            )

        assert torch.allclose(result, values(2.0, 1.0, 20.0, 10.0))

    def test_different_scatter_in_branches(self, line_ctx):
        cond = flags(True, True, False, False)
        x = field_from_values(line_ctx, [1.0, 2.0, 30.0, 10.0])

        with line_ctx.round():
            result = branch(
                cond,
                lambda: gather_sum(scatter(x)),
                lambda: gather_max(scatter(x)),
                branch_name="mixed_aggr_test",
            )

        assert torch.allclose(result, values(2.0, 1.0, 10.0, 30.0))

    def test_nested_branch_isolation(self, line_ctx):
        c1 = flags(True, True, True, False)
        c2 = flags(True, False, False, False)
        x = values(1.0, 10.0, 100.0, 1000.0)

        with line_ctx.round():
            res = branch(
                c1,
                lambda: branch(
                    c2,
                    lambda: gather_sum(scatter(x)),
                    lambda: gather_max(scatter(x)),
                    branch_name="inner",
                ),
                field.zeros,
                branch_name="outer",
            )

        assert torch.allclose(res, values(0.0, 100.0, 10.0, 0.0))

    def test_iterate_scatter_nested(self, line_ctx):
        cond = flags(True, True, False, False)

        results = []
        for _ in range(ROUNDS):
            with line_ctx.round():
                val = branch(
                    cond,
                    lambda: iterate(field.zeros(), lambda s: gather_sum(scatter(s)) + 1.0),
                    lambda: iterate(field.zeros(), lambda s: gather_max(scatter(s)) + 10.0),
                    branch_name="state_test",
                )
            results.append(val.clone())

        assert torch.allclose(results[0], values(1.0, 1.0, 10.0, 10.0))
        assert torch.allclose(results[1], values(2.0, 2.0, 20.0, 20.0))

    def test_soft_branch_scatter_approximates_hard_isolation(self, line_topology):
        edge_index, n = line_topology
        cond = flags(True, True, False, False)
        x = values(1.0, 2.0, 3.0, 4.0)

        hard_ctx = AggregateContext(edge_index, n)
        with hard_ctx.round():
            hard = branch(
                cond,
                lambda: gather_sum(scatter(x)),
                lambda: gather_sum(scatter(x)),
                mode="hard",
            )

        soft_ctx = AggregateContext(edge_index, n)
        with soft_ctx.round():
            soft = branch(
                cond,
                lambda: gather_sum(scatter(x)),
                lambda: gather_sum(scatter(x)),
                mode="soft",
                tau=10.0,
            )

        assert torch.allclose(soft, hard, atol=1e-1)

    def test_reset_states_on_partition_switch(self, line_ctx):
        # Round 1: [T, T, T, T] -> all in branch_true
        # Round 2: [T, T, F, F] -> nodes 2,3 switch to branch_false
        cond1 = flags(True, True, True, True)
        cond2 = flags(True, True, False, False)

        with line_ctx.round():
            branch(
                cond1,
                lambda: iterate(field.zeros(), lambda s: s + 5.0, name="reset_iterate"),
                lambda: iterate(field.zeros(), lambda s: s + 1.0, name="reset_iterate"),
                branch_name="switcher",
            )

        # Before switch, all should be 5.0
        assert torch.allclose(line_ctx.get_state(name="reset_iterate"), values(5.0, 5.0, 5.0, 5.0))

        with line_ctx.round():
            branch(
                cond2,
                lambda: iterate(field.zeros(), lambda s: s + 5.0, name="reset_iterate"),
                lambda: iterate(field.zeros(), lambda s: s + 1.0, name="reset_iterate"),
                branch_name="switcher",
                reset_states={"reset_iterate": field.zeros()},
            )

        # After switch:
        # Nodes 0,1: stayed T -> 5.0 + 5.0 = 10.0
        # Nodes 2,3: switched T->F -> reset to 0.0, then +1.0 = 1.0
        expected = values(10.0, 10.0, 1.0, 1.0)
        assert torch.allclose(line_ctx.get_state(name="reset_iterate"), expected)


class TestMux:
    def test_pointwise_selection(self, line_ctx):
        cond = flags(True, True, False, False)
        x = field_from_values(line_ctx, [1.0, 2.0, 10.0, 20.0])

        with line_ctx.round():
            res = mux(cond, x, x * 10)

        assert torch.allclose(res, values(1.0, 2.0, 100.0, 200.0))

    def test_lazy_evaluation(self, line_ctx):
        cond = flags(True, True, False, False)
        if_true = field_from_values(line_ctx, [10.0, 20.0, 30.0, 40.0])
        if_false = field_from_values(line_ctx, [100.0, 200.0, 300.0, 400.0])

        # Counters to verify both branches are actually called (mux is pointwise but
        # currently evaluates both branches completely)
        self.true_called = 0
        self.false_called = 0

        def get_true():
            self.true_called += 1
            return if_true

        def get_false():
            self.false_called += 1
            return if_false

        with line_ctx.round():
            res = mux(cond, get_true, get_false)

        assert self.true_called == 1
        assert self.false_called == 1
        assert torch.allclose(res, values(10.0, 20.0, 300.0, 400.0))

    def test_local_mux_inside_round(self, line_ctx):
        source = flags(True, False, False, False)
        local_input = field_from_values(line_ctx, [1.0, 2.0, 3.0, 4.0])

        with line_ctx.round():
            res = mux(source, field.zeros(), local_input)

        assert torch.allclose(res, values(0.0, 2.0, 3.0, 4.0))
