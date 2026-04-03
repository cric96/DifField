"""Tests for runtime, scenarios, recording, and exported utilities."""

from __future__ import annotations

import torch

from aggregate_gnn import (
    EventSchedule,
    FullyConnectedScenario,
    GridScenario,
    ScheduledEvent,
    SimulationEngine,
    SnapshotRecorder,
    build_spatial_graph,
    gradient,
    mux,
    nbr,
    rep,
)
from aggregate_gnn.dsl import field
from aggregate_gnn.utils import get_device, get_grid_distances


class TestSimulationFramework:
    def test_event_schedule_moves_source(self):
        scenario = GridScenario(3, 3, connectivity=4)
        engine = SimulationEngine.from_scenario(scenario)

        source = scenario.marker(0, 0)

        def move_to_center(runtime):
            runtime.signals["source"] = scenario.marker(1, 1)
            runtime.metadata["source_pos"] = (1, 1)

        schedule = EventSchedule([ScheduledEvent(round_idx=2, callback=move_to_center, name="move_center")])
        recorder = SnapshotRecorder(state_fields=["dist"], capture_output=True, record_rounds={1, 2, 5})

        def program(runtime):
            src = runtime.signals["source"]
            return rep("dist", float("inf"), lambda d: mux(src, field.of(0.0), nbr(d + 1.0, aggr="min")))

        output, runtime = engine.run(
            rounds=6,
            program=program,
            signals={"source": source},
            metadata={"source_pos": (0, 0)},
            schedule=schedule,
            recorder=recorder,
        )

        center_idx = scenario.pos_to_idx(1, 1)
        assert runtime.metadata["source_pos"] == (1, 1)
        assert recorder.records[2]["output"][center_idx].item() == 0.0
        assert output[center_idx].item() == 0.0

    def test_step_api_and_recording(self):
        scenario = GridScenario(2, 2, connectivity=4)
        engine = SimulationEngine.from_scenario(scenario)
        runtime = engine.init_runtime(signals={"source": scenario.marker(0, 0)})
        recorder = SnapshotRecorder(state_fields=["dist"], capture_output=True, record_rounds={0, 1})

        def program(rt):
            src = rt.signals["source"]
            return rep("dist", float("inf"), lambda d: mux(src, field.of(0.0), nbr(d + 1.0, aggr="min")))

        out0 = engine.step(runtime=runtime, program=program, recorder=recorder)
        out1 = engine.step(runtime=runtime, program=program, recorder=recorder)

        assert runtime.round_idx == 2
        assert 0 in recorder.records and 1 in recorder.records
        assert torch.allclose(out0, recorder.records[0]["output"])
        assert torch.allclose(out1, recorder.records[1]["output"])


def test_event_schedule_applies_callbacks_in_insertion_order():
    runtime = SimulationEngine.from_scenario(GridScenario(1, 1)).init_runtime(signals={}, metadata={"events": []})
    schedule = EventSchedule()
    schedule.add(ScheduledEvent(round_idx=0, callback=lambda rt: rt.metadata["events"].append("first"), name="first"))
    schedule.add(ScheduledEvent(round_idx=0, callback=lambda rt: rt.metadata["events"].append("second"), name="second"))

    schedule.apply(0, runtime)

    assert runtime.metadata["events"] == ["first", "second"]


def test_snapshot_recorder_filters_state_and_export_fields():
    scenario = GridScenario(1, 2, connectivity=4)
    engine = SimulationEngine.from_scenario(scenario)
    recorder = SnapshotRecorder(
        state_fields=["counter"],
        export_fields=["counter_msg"],
        capture_output=True,
        record_rounds={0},
    )

    def program(_runtime):
        counter = rep("counter", 0.0, lambda s: s + 1)
        return nbr(counter, aggr="sum", tag="counter_msg")

    output, _ = engine.run(rounds=1, program=program, signals={}, recorder=recorder)
    record = recorder.records[0]

    assert set(record) == {"counter", "counter_msg", "output"}
    assert torch.allclose(record["counter"], torch.ones(scenario.num_nodes))
    assert torch.allclose(record["counter_msg"], torch.ones(scenario.num_nodes))
    assert torch.allclose(record["output"], output)
    assert recorder.timings[0] >= 0.0


def test_grid_scenario_helper_methods_cover_positions_and_masks():
    scenario = GridScenario(2, 3, connectivity=4)

    assert torch.allclose(scenario.zeros(), torch.zeros(6))
    assert torch.allclose(scenario.full(2.5), torch.full((6,), 2.5))
    assert scenario.pos_to_idx(1, 2) == 5
    assert scenario.idx_to_pos(4) == (1, 1)

    marker = scenario.marker(1, 2, value=3.0)
    assert marker[5].item() == 3.0

    markers = scenario.markers([(0, 1), (1, 0)], value=4.0)
    assert markers.tolist() == [0.0, 4.0, 0.0, 4.0, 0.0, 0.0]

    mask = scenario.mask_from_positions([(0, 0), (1, 2)])
    assert mask.tolist() == [True, False, False, False, False, True]

    diag_mask = scenario.mask_from_predicate(lambda row_idx, col_idx: row_idx == col_idx)
    assert diag_mask.tolist() == [True, False, False, False, True, False]


def test_fully_connected_scenario_sync_context_sets_complete_topology():
    scenario = FullyConnectedScenario(3, self_loops=False)
    engine = SimulationEngine.from_scenario(scenario)

    edges = {tuple(edge) for edge in engine.ctx._ctx.edge_index.t().tolist()}
    assert edges == {(0, 1), (0, 2), (1, 0), (1, 2), (2, 0), (2, 1)}
    assert torch.allclose(engine.ctx._ctx.edge_weight, torch.ones(6))

    scenario.sync_context(engine.ctx._ctx)
    assert torch.allclose(engine.ctx._ctx.edge_weight, torch.ones(6))


def test_grid_scenario_custom_edge_weight_supports_gradient_backward():
    scale = torch.tensor(1.5, requires_grad=True)
    scenario = GridScenario(1, 3, connectivity=4)
    scenario.set_edge_weight(scale)
    engine = SimulationEngine.from_scenario(scenario)

    source = scenario.marker(0, 0)

    def program(_runtime):
        return gradient(source, name="weighted")

    output, _ = engine.run(rounds=4, program=program, signals={"source": source})

    loss = output[output.isfinite()].sum()
    loss.backward()

    assert scale.grad is not None
    assert abs(scale.grad.item() - 3.0) < 1e-6


def test_build_spatial_graph_handles_empty_positions():
    positions = torch.zeros((0, 2), dtype=torch.float32)
    edge_index, edge_weight = build_spatial_graph(positions, edge_radius=0.1)
    assert edge_index.shape == (2, 0)
    assert edge_weight.shape == (0,)


def test_get_grid_distances_supports_four_and_eight_connectivity():
    manhattan = get_grid_distances(3, 3, 0, 0, connectivity=4)
    chebyshev = get_grid_distances(3, 3, 0, 0, connectivity=8)

    assert manhattan[-1].item() == 4.0
    assert chebyshev[-1].item() == 2.0


def test_get_device_respects_explicit_cpu_choice():
    assert str(get_device("cpu")) == "cpu"