"""The same program, run centrally and device by device, must agree.

These pin the claim the DSL makes: a field-calculus program compiles to local
message passing.  The decentralised side is driven by Mesa, which knows nothing
about field calculus, so agreement is evidence about the semantics rather than
about a shared implementation.
"""

from __future__ import annotations

import pytest
import torch

pytest.importorskip("mesa", reason="needs the 'decentralized' extra")

from channel.core import channel_body

from diffield import (
    GridScenario,
    SimulationEngine,
    branch,
    gradient,
)
from diffield.decentralized import run_decentralized
from diffield.decentralized.topology import in_edges_of
from diffield.dsl import field

ROWS = 6
COLS = 6
ROUNDS = 14
TOLERANCE = 1.0


def run_centrally(scenario, program, signals, rounds):
    """Per-round global fields from the batched engine."""
    engine = SimulationEngine.from_scenario(scenario)
    runtime = engine.init_runtime(signals=signals)
    with torch.no_grad():
        return [
            engine.step(runtime=runtime, program=program).detach().cpu().clone()
            for _ in range(rounds)
        ]


def assert_same_field(central: torch.Tensor, local: torch.Tensor, round_idx: int):
    """Exact agreement, with ``nan`` never passing as a match."""
    agree = (central == local) | (torch.isnan(central) & torch.isnan(local))
    assert agree.all(), (
        f"round {round_idx}: {int((~agree).sum())} node(s) differ\n"
        f"  central: {central[~agree][:8]}\n"
        f"  local:   {local[~agree][:8]}"
    )


# ----------------------------------------------------------------------
# programs under test
# ----------------------------------------------------------------------
def gradient_program(runtime):
    return gradient(runtime.signals["source"], name="dist")


def channel_program(runtime):
    signals = runtime.signals
    return channel_body(
        signals["source"], signals["dest"], TOLERANCE, signals["noise"]
    )


def obstacle_channel_program(runtime):
    signals = runtime.signals
    return branch(
        ~signals["obstacle"],
        lambda: channel_body(
            signals["source"], signals["dest"], TOLERANCE, signals["noise"]
        ),
        lambda: field.of(0.0),
        branch_name="obstacle",
    )


@pytest.fixture
def grid():
    return GridScenario(ROWS, COLS, connectivity=8)


@pytest.fixture
def channel_signals(grid):
    return {
        "source": grid.marker(ROWS // 2, 1),
        "dest": grid.marker(ROWS // 2, COLS - 2),
        "obstacle": grid.mask_from_predicate(
            lambda row, col: col == COLS // 2 and row < ROWS - 2
        ),
        "noise": torch.zeros(grid.num_nodes),
    }


# ----------------------------------------------------------------------
# equivalence
# ----------------------------------------------------------------------
def test_gradient_matches_round_by_round(grid):
    signals = {"source": grid.marker(0, 0)}
    central = run_centrally(grid, gradient_program, signals, ROUNDS)
    local = run_decentralized(
        scenario=grid, program=gradient_program, signals=signals, rounds=ROUNDS
    )

    for idx, (expected, actual) in enumerate(zip(central, local.fields)):
        assert_same_field(expected, actual, idx)


def test_channel_matches_round_by_round(grid, channel_signals):
    central = run_centrally(grid, channel_program, channel_signals, ROUNDS)
    local = run_decentralized(
        scenario=grid,
        program=channel_program,
        signals=channel_signals,
        rounds=ROUNDS,
    )

    for idx, (expected, actual) in enumerate(zip(central, local.fields)):
        assert_same_field(expected, actual, idx)


def test_obstacle_branch_matches_round_by_round(grid, channel_signals):
    """Domain restriction survives the move to one device at a time.

    ``branch`` keeps a link only when both endpoints are in the partition, so a
    device can only mask its own in-edges because its neighbours' obstacle flag
    travels with the message.
    """
    central = run_centrally(grid, obstacle_channel_program, channel_signals, ROUNDS)
    local = run_decentralized(
        scenario=grid,
        program=obstacle_channel_program,
        signals=channel_signals,
        rounds=ROUNDS,
    )

    for idx, (expected, actual) in enumerate(zip(central, local.fields)):
        assert_same_field(expected, actual, idx)


def test_activation_order_cannot_matter(grid, channel_signals):
    """Two seeds shuffle nothing that the synchronous barrier lets through."""
    first = run_decentralized(
        scenario=grid,
        program=obstacle_channel_program,
        signals=channel_signals,
        rounds=ROUNDS,
        seed=1,
    )
    second = run_decentralized(
        scenario=grid,
        program=obstacle_channel_program,
        signals=channel_signals,
        rounds=ROUNDS,
        seed=2,
    )
    assert torch.equal(first.final, second.final)


def test_async_converges_to_the_same_fixed_point(grid, channel_signals):
    """Without a barrier the transient differs, but the fixed point does not."""
    central = run_centrally(grid, obstacle_channel_program, channel_signals, ROUNDS)
    local = run_decentralized(
        scenario=grid,
        program=obstacle_channel_program,
        signals=channel_signals,
        rounds=ROUNDS * 6,
        mode="async",
        activation_prob=0.6,
        seed=3,
    )

    assert_same_field(central[-1], local.final, len(local) - 1)


# ----------------------------------------------------------------------
# the local view itself
# ----------------------------------------------------------------------
def test_device_only_sees_its_own_neighbourhood(grid):
    """A device's signals are its star graph, not the network."""
    signals = {"source": grid.marker(0, 0)}
    local = run_decentralized(
        scenario=grid, program=gradient_program, signals=signals, rounds=1
    )

    centre = (ROWS // 2) * COLS + COLS // 2
    runtime = local.runtimes[centre]
    neighbor_ids, _ = in_edges_of(centre, grid.edge_index, grid.edge_weight)

    assert runtime.neighbor_ids == neighbor_ids
    assert len(neighbor_ids) < grid.num_nodes
    # Row 0 is the device, rows 1..k its neighbours -- nothing else is visible.
    assert runtime.signals["source"].shape == (len(neighbor_ids) + 1,)


def test_local_view_preserves_mask_dtype(grid, channel_signals):
    """A boolean sensor stays boolean locally, so ``~mask`` keeps its meaning."""
    local = run_decentralized(
        scenario=grid,
        program=obstacle_channel_program,
        signals=channel_signals,
        rounds=1,
    )
    obstacle = local.runtimes[0].signals["obstacle"]
    assert obstacle.dtype == channel_signals["obstacle"].dtype


def test_self_loops_are_rejected():
    """Better an explicit refusal than a silently different topology."""
    scenario = GridScenario(3, 3, connectivity=4, self_loops=True)
    with pytest.raises(NotImplementedError, match="self-loops"):
        run_decentralized(
            scenario=scenario,
            program=gradient_program,
            signals={"source": scenario.marker(0, 0)},
            rounds=1,
        )
