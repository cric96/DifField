"""Alignment: each occurrence of a construct gets its own store slot.

Every test here pins a defect that the previous source-position naming scheme
produced, so that the library composes: using the same block twice, or the same
reduction twice, must not alias their state.
"""

from __future__ import annotations

import subprocess
import sys
import textwrap

import pytest
import torch

from diffield import (
    AggregateContext,
    branch,
    broadcast,
    collect_cast,
    gather,
    gather_min,
    gradient,
    iterate,
    scatter,
)
from diffield.core import AlignmentError, aggregate
from diffield.dsl import field


def _line_graph(num_nodes: int) -> tuple[torch.Tensor, int]:
    src: list[int] = []
    tgt: list[int] = []
    for node in range(num_nodes - 1):
        src += [node, node + 1]
        tgt += [node + 1, node]
    return torch.tensor([src, tgt]), num_nodes


class TestOccurrenceIdentity:
    def test_two_gradients_do_not_share_state(self):
        edge_index, n = _line_graph(6)
        ctx = AggregateContext(edge_index, n)
        left = torch.tensor([1.0, 0, 0, 0, 0, 0])
        right = torch.tensor([0.0, 0, 0, 0, 0, 1.0])

        for _ in range(3):
            with ctx.round():
                from_left = gradient(left)
                from_right = gradient(right)

        inf = float("inf")
        assert from_left.tolist() == [0.0, 1.0, 2.0, inf, inf, inf]
        assert from_right.tolist() == [inf, inf, inf, 2.0, 1.0, 0.0]
        assert len(ctx._ctx.state.keys()) == 2

    def test_sugar_reductions_get_distinct_tags(self):
        edge_index, n = _line_graph(3)
        ctx = AggregateContext(edge_index, n)
        with ctx.round() as round_ctx:
            gather_min(scatter(torch.tensor([1.0, 2.0, 3.0])))
            gather_min(scatter(torch.tensor([10.0, 20.0, 30.0])))
        assert len(round_ctx.exports) == 2

    def test_repeated_call_on_one_source_line_is_distinct(self):
        edge_index, n = _line_graph(3)
        ctx = AggregateContext(edge_index, n)
        with ctx.round():
            values = [iterate(field.zeros(), lambda s: s + 1.0) for _ in range(3)]
        assert len(ctx._ctx.state.keys()) == 3
        assert [v[0].item() for v in values] == [1.0, 1.0, 1.0]

    def test_nested_iterate_nests_in_the_path(self):
        edge_index, n = _line_graph(3)
        ctx = AggregateContext(edge_index, n)
        with ctx.round():
            iterate(
                field.zeros(),
                lambda outer: outer + iterate(field.zeros(), lambda inner: inner + 1.0),
            )
        keys = ctx._ctx.state.keys()
        assert "/it#0" in keys
        assert "/it#0/it#0" in keys

    def test_user_blocks_compose_via_the_aggregate_decorator(self):
        @aggregate
        def counter_pair() -> torch.Tensor:
            return iterate(field.zeros(), lambda s: s + 1.0)

        edge_index, n = _line_graph(3)
        ctx = AggregateContext(edge_index, n)
        with ctx.round():
            counter_pair()
            counter_pair()
        assert sorted(ctx._ctx.state.keys()) == [
            "/counter_pair#0/it#0",
            "/counter_pair#1/it#0",
        ]


class TestBranchAlignment:
    def test_partitions_get_disjoint_slots(self):
        edge_index, n = _line_graph(4)
        ctx = AggregateContext(edge_index, n)
        cond = torch.tensor([1.0, 1.0, 0.0, 0.0])
        with ctx.round():
            branch(
                cond,
                lambda: iterate(field.zeros(), lambda s: s + 5.0, name="s"),
                lambda: iterate(field.zeros(), lambda s: s + 1.0, name="s"),
                branch_name="split",
            )
        assert sorted(ctx._ctx.state.keys()) == [
            "/br:split/F#0/it:s",
            "/br:split/T#0/it:s",
        ]

    def test_label_used_in_both_partitions_is_ambiguous(self):
        edge_index, n = _line_graph(4)
        ctx = AggregateContext(edge_index, n)
        cond = torch.tensor([1.0, 1.0, 0.0, 0.0])
        with ctx.round():
            branch(
                cond,
                lambda: iterate(field.zeros(), lambda s: s + 5.0, name="s"),
                lambda: iterate(field.zeros(), lambda s: s + 1.0, name="s"),
            )
        with pytest.raises(AlignmentError):
            ctx._ctx.state.get_state(name="s")

    def test_reentering_a_partition_reads_the_initializer(self):
        edge_index, n = _line_graph(4)
        ctx = AggregateContext(edge_index, n)

        def run(cond):
            with ctx.round():
                return branch(
                    cond,
                    lambda: iterate(field.zeros(), lambda s: s + 5.0, name="s"),
                    lambda: iterate(field.zeros(), lambda s: s + 1.0, name="s"),
                )

        run(torch.tensor([1.0, 1.0, 1.0, 1.0]))
        # Nodes 2,3 move to the false partition and must not carry the value the
        # vectorised evaluation computed for them while they were misaligned.
        assert run(torch.tensor([1.0, 1.0, 0.0, 0.0])).tolist() == [
            10.0,
            10.0,
            1.0,
            1.0,
        ]

    @pytest.mark.parametrize("mode", ["hard", "soft"])
    def test_branch_contains_broadcast(self, mode):
        """broadcast is built on gather, so a branch restricts it in both modes."""
        edge_index, n = _line_graph(4)
        ctx = AggregateContext(edge_index, n)
        cond = torch.tensor([1.0, 1.0, 0.0, 0.0])
        root = torch.tensor([False, False, False, True])
        payload = torch.tensor([0.0, 0.0, 0.0, 9.0])

        for _ in range(8):
            with ctx.round():
                branch(
                    cond,
                    lambda: broadcast(root, payload, name="bc"),
                    lambda: field.zeros(),
                    mode=mode,
                )

        # The root sits in the false partition, so its payload must not reach
        # nodes 0 and 1 inside the true partition.
        carried = ctx._ctx.state.get_state(name="bc")[:, 1]
        assert carried[0].item() == pytest.approx(0.0, abs=1e-3)
        assert carried[1].item() == pytest.approx(0.0, abs=1e-3)

    @pytest.mark.parametrize("mode", ["hard", "soft"])
    def test_branch_contains_collect_cast(self, mode):
        edge_index, n = _line_graph(4)
        ctx = AggregateContext(edge_index, n)
        cond = torch.tensor([1.0, 1.0, 0.0, 0.0])
        potential = torch.tensor([0.0, 1.0, 2.0, 3.0])
        local = torch.ones(4)

        for _ in range(8):
            with ctx.round():
                branch(
                    cond,
                    lambda: collect_cast(
                        potential, local, torch.tensor(0.0), torch.add, name="cc"
                    ),
                    lambda: field.zeros(),
                    mode=mode,
                )

        # Only nodes 0 and 1 are in the partition, so the sink collects 2, not 4.
        collected = ctx._ctx.state.get_state(name="cc")
        assert collected[0].item() == pytest.approx(2.0, abs=1e-1)


class TestStability:
    def test_keys_are_identical_across_processes(self):
        program = textwrap.dedent(
            """
            import torch
            from diffield import AggregateContext, gradient, iterate
            from diffield.dsl import field

            ctx = AggregateContext(torch.tensor([[0, 1], [1, 0]]), 2)
            with ctx.round():
                iterate(field.zeros(), lambda s: s + 1.0)
                gradient(torch.tensor([1.0, 0.0]), name="d")
            print(",".join(sorted(ctx._ctx.state.keys())))
            """
        )
        runs = {
            subprocess.run(
                [sys.executable, "-c", program],
                capture_output=True,
                text=True,
                check=True,
            ).stdout.strip()
            for _ in range(2)
        }
        assert len(runs) == 1
        assert runs.pop() == "/gradient:d/it#0,/it#0"

    def test_same_program_at_two_call_sites_shares_keys(self):
        """Identity follows the evaluation tree, not the source position."""
        edge_index, n = _line_graph(3)

        def written_here(ctx):
            with ctx.round():
                iterate(field.zeros(), lambda s: s + 1.0)
            return ctx._ctx.state.keys()

        def written_over_there(ctx):
            with ctx.round():
                iterate(field.zeros(), lambda s: s + 1.0)
            return ctx._ctx.state.keys()

        assert written_here(AggregateContext(edge_index, n)) == written_over_there(
            AggregateContext(edge_index, n)
        )


class TestBroadcastMask:
    def test_float_and_bool_masks_agree(self):
        edge_index, n = _line_graph(3)
        payload = torch.tensor([7.0, 8.0, 9.0])

        results = []
        for mask in (torch.tensor([1.0, 0.0, 0.0]), torch.tensor([True, False, False])):
            ctx = AggregateContext(edge_index, n)
            for _ in range(4):
                with ctx.round():
                    out = broadcast(mask, payload)
            results.append(out.tolist())

        assert results[0] == results[1] == [7.0, 7.0, 7.0]


class TestGatherFillValue:
    def test_sum_honours_fill_value_for_isolated_nodes(self):
        ctx = AggregateContext(torch.tensor([[0], [1]]), 4)
        with ctx.round():
            out = gather(
                scatter(torch.tensor([1.0, 2.0, 3.0, 4.0])),
                aggr="sum",
                fill_value=-99.0,
            )
        assert out.tolist() == [-99.0, 1.0, -99.0, -99.0]
