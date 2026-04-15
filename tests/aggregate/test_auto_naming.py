"""Tests for auto-naming behavior in DSL primitives."""

from __future__ import annotations

import torch

from autofield import branch, nbr, rep, sumhood, maxhood
from autofield.dsl import field
from conftest import field_from_values, field_ones, field_of
from tests.aggregate.support import ROUNDS, flags, values


class TestAutoNamingRep:
    def test_rep_without_name_accumulates(self, line_ctx):
        results = []
        for _ in range(ROUNDS):
            with line_ctx.round():
                val = rep(field.zeros(), lambda s: s + 1)
            results.append(val[0].item())
        assert results == [1.0, 2.0, 3.0]

    def test_rep_same_position_same_name_in_branch(self, line_ctx):
        cond = flags(True, True, False, False)

        results = []
        for _ in range(ROUNDS):
            with line_ctx.round():
                val = branch(
                    cond,
                    lambda: rep(field.zeros(), lambda s: sumhood(nbr(s)) + 1.0),
                    lambda: rep(field.zeros(), lambda s: sumhood(nbr(s)) + 10.0),
                )
            results.append(val.clone())

        assert torch.allclose(results[0], values(1.0, 1.0, 10.0, 10.0))
        assert torch.allclose(results[1], values(2.0, 2.0, 20.0, 20.0))
        assert torch.allclose(results[2], values(3.0, 3.0, 30.0, 30.0))

    def test_rep_different_positions_different_names(self, line_ctx):
        with line_ctx.round():
            a = rep(field.zeros(), lambda s: s + 1.0)
            b = rep(field.zeros(), lambda s: s + 10.0)
        assert torch.allclose(a, field_ones(line_ctx))
        assert torch.allclose(b, field_of(line_ctx, 10.0))

    def test_explicit_name_overrides_auto(self, line_ctx):
        with line_ctx.round():
            rep(field.zeros(), lambda s: s + 1, name="my_state")
        assert line_ctx.get_state(name="my_state") is not None
        assert torch.allclose(
            line_ctx.get_state(name="my_state"), field_ones(line_ctx)
        )


class TestAutoNamingNbr:
    def test_two_nbr_same_level_different_tags(self, triangle_ctx):
        x = field_from_values(triangle_ctx, [1.0, 2.0, 3.0])
        with triangle_ctx.round() as round_ctx:
            a = sumhood(nbr(x))
            b = maxhood(nbr(x))
        assert torch.allclose(a, values(5.0, 4.0, 3.0))
        assert torch.allclose(b, values(3.0, 3.0, 2.0))
        tags = list(round_ctx.exports.keys())
        assert len(tags) == 2
        assert tags[0] != tags[1]
