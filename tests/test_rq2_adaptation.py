"""Invariants of the runtime-adaptation experiments behind RQ2."""

import pytest
import torch

from diffield.sim import normalize_vectors
from examples.seams.boids_adaptation import configuration as boids_configuration
from examples.seams.boids_adaptation import events, perturbed_rollout
from examples.seams.boids_devices import device_rollout, teacher_program
from examples.seams.boids_dynamics import SPEED, Boids
from examples.seams.boids_learning import GraphBoids
from examples.seams.channel_adaptation import configuration, run_seed


@pytest.fixture(autouse=True)
def single_thread():
    old = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(old)


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
        generator = torch.Generator().manual_seed(size)
        p = torch.rand(size, 2, generator=generator)
        v = normalize_vectors(torch.randn(p.shape, generator=generator)) * SPEED
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


def swarm(size, seed=0):
    generator = torch.Generator().manual_seed(seed)
    p = torch.rand(size, 2, generator=generator)
    return p, normalize_vectors(torch.randn(p.shape, generator=generator)) * SPEED


def test_boids_on_synchronous_devices_reproduce_the_batched_rollout_through_events():
    config = boids_configuration("smoke")
    p, v = swarm(16)
    perturb = events(config, {"data_seed": 7, "nodes": 16}, 16, 0)
    schedule = {config["events"][name]: perturb[name] for name in ("split", "crash", "join")}
    model = teacher_program()
    _, batched, (p_batched, _) = perturbed_rollout(model, p, v, config["rounds"], schedule)
    status, devices, (p_devices, _) = device_rollout(model, p, v, config["rounds"], schedule)
    assert status == "complete"
    assert all(devices[key] == batched[key] for key in batched)
    assert torch.equal(p_devices, p_batched)


def test_asynchronous_devices_are_reproducible_and_skip_rounds():
    p, v = swarm(12, seed=1)
    runs = [device_rollout(Boids(), p, v, 20, {}, activation=0.5, seed=3) for _ in range(2)]
    (status, curves, (p_end, _)), (_, _, (p_again, _)) = runs
    assert status == "complete"
    assert torch.equal(p_end, p_again)
    assert curves["activated"][0] == 1.0
    assert 0.2 < sum(curves["activated"][1:]) / 19 < 0.8
