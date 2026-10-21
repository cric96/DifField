"""Invariants of the runtime-adaptation experiments behind RQ2."""

import json

import pytest
import torch

from examples.seams.boids.adaptation import EVENTS, events, perturbed_rollout
from examples.seams.boids.adaptation import configuration as boids_configuration
from examples.seams.boids.devices import batched_rollout, device_rollout
from examples.seams.boids.dynamics import Boids, swarm, teacher
from examples.seams.boids.learning import GraphBoids
from examples.seams.channel_adaptation import campaign, configuration, run_seed
from examples.shared.artifacts import read_json


@pytest.fixture(autouse=True)
def single_thread():
    old = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(old)


def test_channel_adaptation_records_the_sources_of_the_channel_program(tmp_path, monkeypatch):
    class StopError(Exception):
        pass

    def stop(*args, **kwargs):
        raise StopError

    monkeypatch.setattr("examples.seams.channel_adaptation.run_seed", stop)

    with pytest.raises(StopError):
        campaign(tmp_path, "smoke")

    sources = read_json(tmp_path / "manifest.json")["software"]["source_sha256"]
    assert "examples/channel/core.py" in sources


def test_channel_converges_to_the_shortest_path_oracle_before_any_event():
    config = configuration("smoke")
    config["seeds"] = [0]
    run, _ = run_seed(config, 0)
    before = config["events"]["obstacle"] - 1
    assert run["curves"]["quality"][before] >= 0.99
    assert run["curves"]["stale"][before] == 0.0


def test_devices_under_the_obstacle_leave_the_channel_and_the_frozen_plan_breaks():
    config = configuration("smoke")
    run, snapshot = run_seed(config, 1)
    for frame in snapshot["snapshots"].values():
        assert not bool((frame["channel"] & frame["failed"]).any())
    relocate = config["events"]["relocate"]
    assert all(q == 0.0 for q in run["curves"]["frozen_quality"][relocate:])
    assert run["events"]["relocate"]["recovery"]["rounds"] is not None


def test_boids_events_are_deterministic_change_size_and_rollouts_stay_finite():
    config = boids_configuration("smoke")
    campaign = {"data_seed": 7, "nodes": 8}
    for size in (8, 16):
        p, v = swarm(size, torch.Generator().manual_seed(size))
        first, second = events(config, campaign, size, 0), events(config, campaign, size, 0)
        crashed = first["crash"](p, v)
        assert torch.equal(crashed[0], second["crash"](p, v)[0])
        assert len(crashed[0]) == size - round(0.25 * size)
        joined = first["join"](*crashed)
        assert torch.equal(joined[0], second["join"](*crashed)[0])
        assert len(joined[0]) == size
        schedule = {config["events"][name]: first[name] for name in ("split", "crash", "join")}
        for model in (None, Boids(), GraphBoids(8)):
            status, curves, (p_end, v_end) = perturbed_rollout(
                model, p, v, config["rounds"], schedule
            )
            assert status == "complete"
            assert len(curves["spread"]) == config["rounds"]
            assert curves["nodes"][config["events"]["crash"]] == size - round(0.25 * size)
            assert torch.isfinite(p_end).all() and torch.isfinite(v_end).all()


def test_boids_on_synchronous_devices_reproduce_the_batched_rollout_through_events():
    config = boids_configuration("smoke")
    p, v = swarm(16, torch.Generator().manual_seed(0))
    perturb = events(config, {"data_seed": 7, "nodes": 16}, 16, 0)
    schedule = {config["events"][name]: perturb[name] for name in ("split", "crash", "join")}
    model = teacher()
    _, batched, (p_batched, _) = perturbed_rollout(model, p, v, config["rounds"], schedule)
    status, devices, (p_devices, _) = device_rollout(model, p, v, config["rounds"], schedule)
    assert status == "complete"
    assert all(devices[key] == batched[key] for key in batched)
    assert torch.equal(p_devices, p_batched)


def test_asynchronous_devices_are_reproducible_and_skip_rounds():
    p, v = swarm(12, torch.Generator().manual_seed(1))
    runs = [device_rollout(Boids(), p, v, 20, {}, activation=0.5, seed=3) for _ in range(2)]
    (status, curves, (p_end, _)), (_, _, (p_again, _)) = runs
    assert status == "complete"
    assert torch.equal(p_end, p_again)
    assert curves["activated"][0] == 1.0
    assert 0.2 < sum(curves["activated"][1:]) / 19 < 0.8


def test_asynchronous_devices_repeat_their_activations_through_events_for_a_seed():
    config = boids_configuration("smoke")
    p, v = swarm(16, torch.Generator().manual_seed(0))
    perturb = events(config, {"data_seed": 7, "nodes": 16}, 16, 0)
    schedule = {config["events"][name]: perturb[name] for name in ("split", "crash", "join")}

    status, curves, (p_end, _) = device_rollout(
        teacher(), p, v, config["rounds"], schedule, activation=0.5, seed=3
    )

    # Recorded values: the seed fixes every activation, before and after the population changes.
    assert status == "complete"
    assert curves["nodes"][19:21] == [16, 12]
    assert sum(curves["activated"]) == pytest.approx(21.16666668653488, abs=1e-12)
    assert float(p_end.sum()) == pytest.approx(15.441049575805664, abs=1e-6)


def test_batched_asynchronous_rollout_matches_the_one_on_devices():
    p, v = swarm(12, torch.Generator().manual_seed(1))

    _, _, (p_devices, _) = device_rollout(teacher(), p, v, 20, {}, activation=0.5, seed=3)
    batched = batched_rollout(teacher(), p, v, 20, 0.5, seed=3)

    torch.testing.assert_close(batched[-1], p_devices)


def test_device_summary_reports_panels_whose_runs_all_diverged(tmp_path):
    from examples.seams.boids.devices import configuration, summarize, write_report

    config = configuration("smoke")
    events_ok = {name: {"recovery": {"rounds": 3}, "peak_change": 0.1} for name in EVENTS}
    complete = {"status": "complete", "episode": "e0", "equal_to_batched": True,
                "metrics": {**events_ok, "lcc_final": 1.0},
                "curves": {"spread": [0.2, 0.3], "activated": [1.0, 1.0]}}  # fmt: skip
    diverged = {"status": "diverged", "episode": "e0", "metrics": {}, "curves": {}}
    rows = [
        {**complete, "method": "parametric", "executor": "sync"},
        {**diverged, "method": "parametric", "executor": "async"},
        {**diverged, "method": "oracle", "executor": "async"},
    ]

    summary = summarize(rows, config)
    write_report(tmp_path, config, summary)

    json.dumps(summary, allow_nan=False)
    assert summary["parametric/async"]["lcc_final"] is None
    assert summary["parametric/sync"]["deviation_from_teacher_async"] is None
