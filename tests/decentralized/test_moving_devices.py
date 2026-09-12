"""Devices that move, and then slow to a halt.

Topology is no longer a fixed backdrop here: links form and break under the
devices as they drift, so a device's neighbour set changes beneath it between
rounds.  Two things are checked -- that the decentralised run still tracks the
batched one exactly while the graph is churning, and that once the devices stop
the field settles on the true shortest-path metric of wherever they ended up.
"""

from __future__ import annotations

import pytest
import torch

pytest.importorskip("mesa", reason="needs the 'decentralized' extra")

import networkx as nx

from diffield import (
    EventSchedule,
    ScheduledEvent,
    SimulationEngine,
    SpatialScenario,
    gradient,
    iterate,
)
from diffield.decentralized import run_decentralized
from diffield.decentralized.runtime import _resize_states
from diffield.dsl import field

NUM_DEVICES = 40
EDGE_RADIUS = 0.30
SOURCE = 0

DRIFT_ROUNDS = 12
SETTLE_ROUNDS = 18
TOTAL_ROUNDS = DRIFT_ROUNDS + SETTLE_ROUNDS

FAST_SPEED = 0.02
SLOW_SPEED = 0.0008
#: Below this the devices are parked, so the topology stops changing.
STOPPED = 1e-6


@pytest.fixture(scope="module")
def layout():
    generator = torch.Generator().manual_seed(11)
    positions = torch.rand(NUM_DEVICES, 2, generator=generator)
    directions = torch.rand(NUM_DEVICES, 2, generator=generator) - 0.5
    return positions, directions


def make_scenario(positions):
    return SpatialScenario(
        positions=positions.clone(),
        edge_radius=EDGE_RADIUS,
        edge_weight_mode="distance",
    )


def program(runtime):
    return gradient(runtime.signals["source"], name="dist")


def counting_program(runtime):
    """Neighbour exchange plus memory a device can only hold itself.

    The gradient keeps messages flowing so links still matter, while the round
    counter reads its *own* previous value -- the one row a rebuilt star graph
    has to carry across, and which a distance field alone never reads.
    """
    gradient(runtime.signals["source"], name="dist")
    return iterate(field.zeros(), lambda rounds: rounds + 1.0, name="rounds")


def drift_schedule(rounds: int, decay: float = 1.0) -> EventSchedule:
    """Move every device each round, damping the motion by *decay*.

    With ``decay < 1`` the devices slow down and eventually park, which is what
    lets the final assertion be exact rather than approximate.
    """

    def move(runtime):
        velocities = runtime.metadata["velocities"]
        if float(velocities.abs().max()) < STOPPED:
            return
        runtime.scenario.step_positions(velocities)
        runtime.metadata["velocities"] = (
            velocities * decay if velocities.abs().max() * decay >= STOPPED
            else torch.zeros_like(velocities)
        )

    return EventSchedule(
        [ScheduledEvent(round_idx=idx, callback=move, name="move") for idx in range(rounds)]
    )


def run_centrally(scenario, signals, velocities, rounds, schedule, body=program):
    engine = SimulationEngine.from_scenario(scenario)
    runtime = engine.init_runtime(
        signals=signals, metadata={"velocities": velocities.clone()}
    )
    with torch.no_grad():
        return [
            engine.step(runtime=runtime, program=body, schedule=schedule)
            .detach()
            .cpu()
            .clone()
            for _ in range(rounds)
        ]


def assert_same_field(central, local, round_idx):
    agree = (central == local) | (torch.isnan(central) & torch.isnan(local))
    assert agree.all(), (
        f"round {round_idx}: {int((~agree).sum())} node(s) differ\n"
        f"  central: {central[~agree][:8]}\n  local:   {local[~agree][:8]}"
    )


def true_distance_field(scenario, source: int) -> torch.Tensor:
    """Ground truth: weighted shortest paths on the graph, computed outside diffield."""
    graph = nx.Graph()
    graph.add_nodes_from(range(scenario.num_nodes))
    edges = scenario.edge_index.cpu().t().tolist()
    weights = scenario.edge_weight.detach().cpu().tolist()
    for (source_node, target_node), weight in zip(edges, weights):
        if source_node != target_node:
            graph.add_edge(int(source_node), int(target_node), weight=weight)

    lengths = nx.single_source_dijkstra_path_length(graph, source, weight="weight")
    field = torch.full((scenario.num_nodes,), float("inf"))
    for node, length in lengths.items():
        field[node] = length
    return field


# ----------------------------------------------------------------------
def test_moving_devices_match_round_by_round(layout):
    """The graph churns under the devices and the two runs still agree exactly."""
    positions, directions = layout
    velocities = directions * FAST_SPEED
    schedule = drift_schedule(TOTAL_ROUNDS)

    central_scenario = make_scenario(positions)
    central = run_centrally(
        central_scenario,
        {"source": central_scenario.marker(SOURCE)},
        velocities,
        TOTAL_ROUNDS,
        schedule,
    )

    local_scenario = make_scenario(positions)
    local = run_decentralized(
        scenario=local_scenario,
        program=program,
        signals={"source": local_scenario.marker(SOURCE)},
        rounds=TOTAL_ROUNDS,
        metadata={"velocities": velocities.clone()},
        schedule=schedule,
    )

    # Without this the test could pass on a graph that never actually moved.
    assert len(local.retopologized_rounds) > TOTAL_ROUNDS // 2
    assert torch.equal(central_scenario.edge_index, local_scenario.edge_index)

    for idx, (expected, actual) in enumerate(zip(central, local.fields)):
        assert_same_field(expected, actual, idx)


def test_slow_drift_keeps_links_but_restretches_them(layout):
    """Devices can move without anyone gaining or losing a neighbour.

    That path keeps each device's star graph and only re-measures its links,
    so it is worth pinning separately from a full rebuild.
    """
    positions, directions = layout
    velocities = directions * SLOW_SPEED
    schedule = drift_schedule(TOTAL_ROUNDS)

    central_scenario = make_scenario(positions)
    central = run_centrally(
        central_scenario,
        {"source": central_scenario.marker(SOURCE)},
        velocities,
        TOTAL_ROUNDS,
        schedule,
    )

    local_scenario = make_scenario(positions)
    local = run_decentralized(
        scenario=local_scenario,
        program=program,
        signals={"source": local_scenario.marker(SOURCE)},
        rounds=TOTAL_ROUNDS,
        metadata={"velocities": velocities.clone()},
        schedule=schedule,
    )

    # Most rounds move the devices without rewiring anything.
    assert 0 < len(local.retopologized_rounds) < TOTAL_ROUNDS // 2

    for idx, (expected, actual) in enumerate(zip(central, local.fields)):
        assert_same_field(expected, actual, idx)


def test_field_settles_on_the_true_metric_once_devices_stop(layout):
    """Adaptation: devices drift, slow to a halt, and the field re-converges.

    The reference is a Dijkstra computed by NetworkX on the final positions --
    outside diffield entirely -- so this pins that the decentralised system
    tracks a moving world rather than merely agreeing with itself.
    """
    positions, directions = layout
    velocities = directions * FAST_SPEED
    schedule = drift_schedule(TOTAL_ROUNDS, decay=0.5)

    scenario = make_scenario(positions)
    result = run_decentralized(
        scenario=scenario,
        program=program,
        signals={"source": scenario.marker(SOURCE)},
        rounds=TOTAL_ROUNDS,
        metadata={"velocities": velocities.clone()},
        schedule=schedule,
    )

    assert result.retopologized_rounds, "devices never moved far enough to rewire"
    # The devices parked well before the end, leaving rounds to re-converge.
    assert max(result.retopologized_rounds) < TOTAL_ROUNDS - 6

    expected = true_distance_field(scenario, SOURCE)
    actual = result.final

    reachable = torch.isfinite(expected)
    assert torch.equal(torch.isfinite(actual), reachable)
    assert torch.allclose(actual[reachable], expected[reachable], atol=1e-5)


def test_moving_devices_still_converge_without_a_barrier(layout):
    """Asynchronous devices on a moving graph reach the same resting field."""
    positions, directions = layout
    velocities = directions * FAST_SPEED

    scenario = make_scenario(positions)
    result = run_decentralized(
        scenario=scenario,
        program=program,
        signals={"source": scenario.marker(SOURCE)},
        rounds=TOTAL_ROUNDS * 2,
        mode="async",
        activation_prob=0.6,
        seed=5,
        metadata={"velocities": velocities.clone()},
        schedule=drift_schedule(TOTAL_ROUNDS * 2, decay=0.5),
    )

    expected = true_distance_field(scenario, SOURCE)
    reachable = torch.isfinite(expected)
    assert torch.equal(torch.isfinite(result.final), reachable)
    assert torch.allclose(result.final[reachable], expected[reachable], atol=1e-5)


def test_device_memory_survives_rewiring(layout):
    """A rebuilt star graph must not cost a device what it already knew.

    Rewiring re-creates every state slot at the new degree.  Row 0 is the only
    row that is genuinely the device's own, so it has to be carried across; a
    distance field would not notice, because it reads its neighbours' rows and
    never its own.
    """
    positions, directions = layout
    velocities = directions * FAST_SPEED
    schedule = drift_schedule(TOTAL_ROUNDS)

    central_scenario = make_scenario(positions)
    central = run_centrally(
        central_scenario,
        {"source": central_scenario.marker(SOURCE)},
        velocities,
        TOTAL_ROUNDS,
        schedule,
        body=counting_program,
    )

    local_scenario = make_scenario(positions)
    local = run_decentralized(
        scenario=local_scenario,
        program=counting_program,
        signals={"source": local_scenario.marker(SOURCE)},
        rounds=TOTAL_ROUNDS,
        metadata={"velocities": velocities.clone()},
        schedule=schedule,
    )

    assert len(local.retopologized_rounds) > TOTAL_ROUNDS // 2

    for idx, (expected, actual) in enumerate(zip(central, local.fields)):
        assert_same_field(expected, actual, idx)

    # Every device ran every round and remembers all of them, rewiring or not.
    assert torch.equal(
        local.final, torch.full((NUM_DEVICES,), float(TOTAL_ROUNDS))
    )


def test_resized_slots_keep_the_device_and_forget_everyone_else():
    """The contract a rewire relies on, pinned directly.

    End to end this is hard to observe: every device publishes every round, so
    the neighbour rows are overwritten by injection before anyone reads them.
    It matters the moment a message can go missing, so pin it here.
    """
    carried = {"/it:counter": torch.tensor(7.0)}
    initializers = {"/it:counter": torch.zeros(3)}

    resized = _resize_states(carried, initializers, num_nodes=5)

    slot = resized["/it:counter"]
    assert slot.shape == (5,)
    assert slot[0] == 7.0, "row 0 is the device's own value"
    assert torch.equal(slot[1:], torch.zeros(4)), (
        "a neighbour not yet heard from reads as the initialiser"
    )


def test_resized_slots_keep_payload_shape():
    """Non-scalar state -- a gradient_cast payload -- resizes on its first axis."""
    carried = {"/gc": torch.tensor([1.0, 2.0, 3.0])}
    initializers = {"/gc": torch.zeros(2, 3)}

    slot = _resize_states(carried, initializers, num_nodes=4)["/gc"]

    assert slot.shape == (4, 3)
    assert torch.equal(slot[0], torch.tensor([1.0, 2.0, 3.0]))
