"""Check open-world physics, local communication and coherent flock behaviour."""

import torch

from examples.seams.boids_flocking import (
    CoherentBoids, FlockingConfig, dense_step, diagnostics, initial_state, radius_edges, simulate,
)


def test_dense_and_dsl_agree_with_isolation_and_coincident_nodes():
    torch.set_num_threads(1)
    p = torch.tensor([[0.1, 0.1], [0.1, 0.1], [0.13, 0.12], [2.0, 3.0]])
    v = torch.tensor([[0.16, 0.0], [0.0, 0.16], [0.12, 0.09], [-0.16, 0.0]])
    model = CoherentBoids()
    expected = dense_step(p, v)
    actual = model.step(p, v)
    for a, b in zip(actual, expected, strict=True):
        torch.testing.assert_close(a, b, atol=1e-7, rtol=1e-5)


def test_isolated_boid_cruises_without_attraction_to_origin_or_box_bounce():
    config = FlockingConfig()
    model = CoherentBoids(config)
    p = torch.tensor([[20.0, -10.0]])
    v = torch.tensor([[config.cruise_speed, 0.0]])
    actual_p, actual_v = model.step(p, v)
    torch.testing.assert_close(actual_v, v)
    torch.testing.assert_close(actual_p, p + v * config.dt)


def test_translation_and_permutation_preserve_local_dynamics():
    p, v, _ = initial_state("merge", nodes=24)
    model = CoherentBoids()
    a, av = model.step(p, v)
    offset = torch.tensor([3.0, -2.0])
    order = torch.arange(len(p) - 1, -1, -1)
    b, bv = model.step(p[order] + offset, v[order])
    torch.testing.assert_close(a[order] + offset, b, atol=5e-7, rtol=1e-5)
    torch.testing.assert_close(av[order], bv, atol=1e-6, rtol=1e-5)


def test_turn_and_speed_limits_hold_under_large_disturbance():
    config = FlockingConfig()
    model = CoherentBoids(config)
    p, v, _ = initial_state("flock", nodes=24)
    external = torch.full_like(p, 1000.0)
    next_p, next_v = model.step(p, v, external=external)
    angle = torch.atan2(v[:, 0] * next_v[:, 1] - v[:, 1] * next_v[:, 0], (v * next_v).sum(-1))
    assert float(angle.abs().max()) <= config.max_turn_rate * config.dt + 1e-6
    assert float(next_v.norm(dim=-1).min()) >= config.min_speed - 1e-6
    assert float(next_v.norm(dim=-1).max()) <= config.max_speed + 1e-6
    torch.testing.assert_close(next_p - p, next_v * config.dt, atol=1e-7, rtol=1e-5)


def test_local_coefficients_are_differentiable():
    p, v, _ = initial_state("flock", nodes=24)
    model = CoherentBoids().double()
    p, v = p.double(), v.double()
    edges = radius_edges(p, model.config)
    _, predicted = model.step(p, v, edges)
    loss = predicted[:, 1].square().sum()
    loss.backward()
    assert torch.isfinite(model.log_weights.grad).all()
    assert (model.log_weights.grad.abs() > 1e-10).all()


def test_merging_flocks_align_without_stopping():
    torch.set_num_threads(1)
    replay = simulate("merge", seconds=12, nodes=96, seed=3)
    result = diagnostics(replay)
    assert result["finite"]
    assert result["coherence"][-1] > 0.95
    assert result["coherence"][-1] > result["coherence"][0] + 0.1
    assert result["min_speed"] >= replay["config"]["min_speed"] - 1e-6
    assert result["max_turn_rate"] <= replay["config"]["max_turn_rate"] + 1e-4
    assert result["median_nearest_distance"][-1] > 0.02
