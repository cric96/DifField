"""Regression tests for learnable aggregate boids training modes."""

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from examples.boids.learnable import LearnableAggregateBoids


def _make_model(mode: str) -> LearnableAggregateBoids:
    torch.manual_seed(0)
    positions0 = torch.rand(8, 2)
    return LearnableAggregateBoids(
        positions0=positions0,
        radius=0.3,
        sep=0.08,
        dt=1.0,
        mode=mode,
        init_connectivity="hybrid",
        init_k_neighbors=4,
        init_min_degree=2,
    )


def _make_frozen_speed_model() -> LearnableAggregateBoids:
    torch.manual_seed(0)
    positions0 = torch.rand(8, 2)
    return LearnableAggregateBoids(
        positions0=positions0,
        radius=0.3,
        sep=0.08,
        dt=1.0,
        mode="weights",
        init_connectivity="hybrid",
        init_k_neighbors=4,
        init_min_degree=2,
        init_max_speed_target=0.014,
        train_max_speed=False,
    )


def test_weights_mode_velocity_update_ignores_attention_parameters():
    model = _make_model("weights")
    pos_before, vel_before, final_before = model.rollout(rounds=4)

    with torch.no_grad():
        for param in model.align_aggr.parameters():
            param.fill_(25.0)
        for param in model.cohesion_aggr.parameters():
            param.fill_(-25.0)

    pos_after, vel_after, final_after = model.rollout(rounds=4)
    assert torch.allclose(pos_before, pos_after, atol=1e-6)
    assert torch.allclose(vel_before, vel_after, atol=1e-6)
    assert torch.allclose(final_before, final_after, atol=1e-6)


def test_rollout_records_speed_cap_diagnostics():
    model = _make_model("weights")
    _, _, _ = model.rollout(rounds=5)

    mean_pre_clip_speed = model.last_rollout_speed_health["mean_pre_clip_speed"]
    mean_cap_fraction = model.last_rollout_speed_health["mean_cap_fraction"]

    assert torch.isfinite(torch.tensor(mean_pre_clip_speed))
    assert torch.isfinite(torch.tensor(mean_cap_fraction))
    assert mean_pre_clip_speed >= 0.0
    assert 0.0 <= mean_cap_fraction <= 1.0


def test_frozen_speed_weights_mode_excludes_max_speed_parameter():
    model = _make_frozen_speed_model()

    params = model.trainable_parameters()

    assert all(param is not model.max_speed_raw for param in params)
    assert abs(float(model.max_speed.item()) - 0.014) < 1e-4