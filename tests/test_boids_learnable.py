"""Regression tests for learnable aggregate boids training modes."""

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from examples.boids.config import ModelSpec, SimulationSpec, TeacherDynamics
from examples.boids.learnable import LearnableAggregateBoids
from examples.boids.model import teacher_rollout_from_specs


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


def _softplus_inverse(value: float) -> torch.Tensor:
    adjusted = max(value - 1e-4, 1e-6)
    return torch.log(torch.expm1(torch.tensor(adjusted, dtype=torch.float32)))


def _bounded_sigmoid_inverse(value: float, low: float, high: float) -> torch.Tensor:
    scaled = (value - low) / (high - low)
    scaled = min(max(float(scaled), 1e-4), 1.0 - 1e-4)
    return torch.logit(torch.tensor(scaled, dtype=torch.float32))


def _teacher_specs(model: LearnableAggregateBoids, rounds: int) -> tuple[SimulationSpec, TeacherDynamics, ModelSpec]:
    simulation = SimulationSpec(
        num_nodes=model.positions0.shape[0],
        rounds=rounds,
        radius=model.radius,
        sep=model.sep,
        dt=model.dt,
        init_velocity_scale=0.01,
        device=torch.device("cpu"),
    )
    teacher = TeacherDynamics(
        w_sep=1.4,
        w_align=0.8,
        w_cohesion=0.6,
        damping=0.96,
        max_speed=0.014,
    )
    model_spec = ModelSpec(
        mode="weights",
        init_connectivity=model.init_connectivity,
        init_k_neighbors=model.init_k_neighbors,
        init_min_degree=model.init_min_degree,
        init_damping_target=teacher.damping,
        init_max_speed_target=teacher.max_speed,
        max_speed_min=model.max_speed_min,
        max_speed_max=model.max_speed_max,
        train_max_speed=True,
    )
    return simulation, teacher, model_spec


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


def test_rollout_keeps_fixed_topology_metrics_constant():
    model = _make_model("weights")

    model.rollout(rounds=6)

    assert model.last_rollout_graph_health["mean_num_edges"] == model.last_init_graph_stats["num_edges"]
    assert model.last_rollout_graph_health["mean_min_degree"] == model.last_init_graph_stats["min_degree"]
    assert model.last_rollout_graph_health["max_num_components"] == model.last_init_graph_stats["num_components"]


def test_truncated_rollout_backpropagates_finite_gradients():
    model = _make_model("weights")
    velocities0 = (torch.rand_like(model.positions0) - 0.5) * 0.01

    pred_pos_seq, pred_vel_seq, _ = model.rollout(rounds=6, velocities0=velocities0, trunc_window=2)
    loss = pred_pos_seq.square().mean() + pred_vel_seq.square().mean()
    loss.backward()

    for param in model.trainable_parameters():
        assert param.grad is not None
        assert torch.isfinite(param.grad).all()


def test_weights_rollout_matches_teacher_when_parameters_match():
    model = _make_model("weights")
    simulation, teacher, model_spec = _teacher_specs(model, rounds=5)
    velocities0 = (torch.rand_like(model.positions0) - 0.5) * 0.01

    with torch.no_grad():
        model.w_sep_raw.copy_(_softplus_inverse(teacher.w_sep))
        model.w_align_raw.copy_(_softplus_inverse(teacher.w_align))
        model.w_cohesion_raw.copy_(_softplus_inverse(teacher.w_cohesion))
        model.damping_raw.copy_(torch.logit(torch.tensor(teacher.damping, dtype=torch.float32)))
        model.max_speed_raw.copy_(_bounded_sigmoid_inverse(teacher.max_speed, model.max_speed_min, model.max_speed_max))

    teacher_pos_seq, teacher_vel_seq = teacher_rollout_from_specs(
        positions0=model.positions0,
        velocities0=velocities0,
        rounds=simulation.rounds,
        simulation=simulation,
        teacher=teacher,
        model=model_spec,
    )
    pred_pos_seq, pred_vel_seq, _ = model.rollout(rounds=simulation.rounds, velocities0=velocities0)

    assert torch.allclose(pred_pos_seq, teacher_pos_seq, atol=1e-5)
    assert torch.allclose(pred_vel_seq, teacher_vel_seq, atol=1e-5)