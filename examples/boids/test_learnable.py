"""Tests specific to the learnable boids example family.

These tests target the post-rename layout where aggregate logic lives in logics.py.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import torch

from examples.boids.logics import teacher_rollout_from_specs
from examples.boids.cli import DEFAULTS
from examples.boids.config import ModelSpec, SimulationSpec, TeacherDynamics, build_learnable_spec
from examples.boids.core import sample_initial_boids_state
from examples.boids.evaluation_utils import evaluate_seed
from examples.boids.learnable import curriculum_horizon_with_fraction, make_initial_conditions, scheduled_learning_rate, teacher_forced_step_losses
from examples.boids.losses import close_pair_distance_loss, trajectory_loss_components
from examples.boids.model import LearnableAggregateBoids
from examples.boids.replay import build_supervision_traces, load_boids_trace, save_boids_trace, teacher_trace_from_specs, trace_file_path
from examples.boids.reporting import extract_learned_parameters


def _make_learnable_args(**overrides: object) -> SimpleNamespace:
    values = {
        "num_nodes": DEFAULTS["num_nodes"],
        "rounds": DEFAULTS["rounds"],
        "epochs": DEFAULTS["epochs"],
        "radius": 0.23,
        "init_connectivity": "hybrid",
        "init_k_neighbors": 8,
        "init_min_degree": 2,
        "sep": 0.06,
        "dt": 1.0,
        "init_velocity_scale": 0.014,
        "teacher_w_sep": DEFAULTS["teacher_w_sep"],
        "teacher_w_align": DEFAULTS["teacher_w_align"],
        "teacher_w_cohesion": DEFAULTS["teacher_w_cohesion"],
        "init_w_sep_target": DEFAULTS["init_w_sep_target"],
        "init_w_align_target": DEFAULTS["init_w_align_target"],
        "init_w_cohesion_target": DEFAULTS["init_w_cohesion_target"],
        "teacher_damping": DEFAULTS["teacher_damping"],
        "teacher_max_speed": DEFAULTS["teacher_max_speed"],
        "init_damping_target": DEFAULTS["init_damping_target"],
        "init_max_speed_target": DEFAULTS["init_max_speed_target"],
        "max_speed_min": 0.004,
        "max_speed_max": 0.06,
        "seed": 5,
        "lr": DEFAULTS["lr"],
        "supervision_mode": DEFAULTS["supervision_mode"],
        "replay_trace_dir": "",
        "save_replay_traces": False,
        "curriculum_min_horizon": None,
        "curriculum_max_horizon": None,
        "curriculum_ramp_fraction": DEFAULTS["curriculum_ramp_fraction"],
        "final_lr_ratio": DEFAULTS["final_lr_ratio"],
        "trunc_window": None,
        "num_initial_conditions": DEFAULTS["num_initial_conditions"],
        "velocity_loss_weight": DEFAULTS["velocity_loss_weight"],
        "separation_loss_weight": DEFAULTS["separation_loss_weight"],
        "print_every": DEFAULTS["print_every"],
        "record_every": DEFAULTS["record_every"],
        "checkpoint_every_epochs": DEFAULTS["checkpoint_every_epochs"],
        "eval_seeds": DEFAULTS["eval_seeds"],
        "eval_every": DEFAULTS["eval_every"],
        "highlight_node": 0,
        "viz_prefix": "generated/boids/learnable",
        "gif_fps": 8,
        "no_viz": False,
        "no_gif": False,
        "hide_links": False,
        "links_alpha": 0.15,
        "links_width": 0.6,
        "device": "cpu",
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def _make_model() -> LearnableAggregateBoids:
    torch.manual_seed(0)
    positions0 = torch.rand(8, 2)
    return LearnableAggregateBoids(
        positions0=positions0,
        radius=0.3,
        sep=0.08,
        dt=1.0,
        init_connectivity="hybrid",
        init_k_neighbors=4,
        init_min_degree=2,
    )


def _make_small_boids_model() -> LearnableAggregateBoids:
    torch.manual_seed(0)
    positions0 = torch.rand(6, 2)
    return LearnableAggregateBoids(
        positions0=positions0,
        radius=0.35,
        sep=0.08,
        dt=1.0,
        init_connectivity="hybrid",
        init_k_neighbors=4,
        init_min_degree=2,
        init_damping_target=0.96,
        init_max_speed_target=0.014,
    )


def _make_frozen_speed_model() -> LearnableAggregateBoids:
    torch.manual_seed(0)
    positions0 = torch.rand(8, 2)
    return LearnableAggregateBoids(
        positions0=positions0,
        radius=0.3,
        sep=0.08,
        dt=1.0,
        init_connectivity="hybrid",
        init_k_neighbors=4,
        init_min_degree=2,
        init_max_speed_target=0.014,
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
        init_connectivity=model.init_connectivity,
        init_k_neighbors=model.init_k_neighbors,
        init_min_degree=model.init_min_degree,
        init_w_sep_target=float(model.w_sep.item()),
        init_w_align_target=float(model.w_align.item()),
        init_w_cohesion_target=float(model.w_cohesion.item()),
        init_damping_target=teacher.damping,
        init_max_speed_target=teacher.max_speed,
        max_speed_min=model.max_speed_min,
        max_speed_max=model.max_speed_max,
    )
    return simulation, teacher, model_spec


def _boids_teacher_targets(
    model: LearnableAggregateBoids,
    *,
    rounds: int,
    velocities0: torch.Tensor,
) -> tuple[SimulationSpec, torch.Tensor, torch.Tensor]:
    simulation, teacher, model_spec = _teacher_specs(model, rounds)
    teacher_pos_seq, teacher_vel_seq = teacher_rollout_from_specs(
        positions0=model.positions0,
        velocities0=velocities0,
        rounds=simulation.rounds,
        simulation=simulation,
        teacher=teacher,
        model=model_spec,
    )
    return simulation, teacher_pos_seq, teacher_vel_seq


def _assert_finite_gradients(params: list[torch.nn.Parameter]) -> None:
    for param in params:
        assert param.grad is not None
        assert torch.isfinite(param.grad).all()


def test_trainable_parameters_include_only_scalar_dynamics_weights():
    model = _make_model()

    params = model.trainable_parameters()

    assert [id(param) for param in params] == [
        id(model.w_sep_raw),
        id(model.w_align_raw),
        id(model.w_cohesion_raw),
        id(model.damping_raw),
    ]


def test_rollout_records_speed_cap_diagnostics():
    model = _make_model()
    _, _, _ = model.rollout(rounds=5)

    mean_pre_clip_speed = model.last_rollout_speed_health["mean_pre_clip_speed"]
    mean_cap_fraction = model.last_rollout_speed_health["mean_cap_fraction"]

    assert torch.isfinite(torch.tensor(mean_pre_clip_speed))
    assert torch.isfinite(torch.tensor(mean_cap_fraction))
    assert mean_pre_clip_speed >= 0.0
    assert 0.0 <= mean_cap_fraction <= 1.0
    assert mean_pre_clip_speed > 0.0


def test_frozen_speed_weights_mode_excludes_max_speed_parameter():
    model = _make_frozen_speed_model()

    params = model.trainable_parameters()

    assert all(param is not model.max_speed_raw for param in params)
    assert abs(float(model.max_speed.item()) - 0.014) < 1e-4


def test_explicit_init_weight_targets_are_respected():
    torch.manual_seed(0)
    positions0 = torch.rand(8, 2)
    model = LearnableAggregateBoids(
        positions0=positions0,
        radius=0.3,
        sep=0.08,
        dt=1.0,
        init_w_sep_target=0.10,
        init_w_align_target=2.40,
        init_w_cohesion_target=0.08,
        init_damping_target=0.55,
        init_max_speed_target=0.05,
    )

    assert abs(float(model.w_sep.item()) - 0.10) < 1e-4
    assert abs(float(model.w_align.item()) - 2.40) < 1e-4
    assert abs(float(model.w_cohesion.item()) - 0.08) < 1e-4
    assert abs(float(model.damping.item()) - 0.55) < 1e-4
    assert abs(float(model.max_speed.item()) - 0.05) < 1e-4


def test_rollout_reports_dynamic_topology_metrics():
    model = _make_model()

    model.rollout(rounds=6)

    assert torch.isfinite(torch.tensor(model.last_rollout_graph_health["mean_num_edges"]))
    assert torch.isfinite(torch.tensor(model.last_rollout_graph_health["mean_min_degree"]))
    assert torch.isfinite(torch.tensor(model.last_rollout_graph_health["max_num_components"]))
    assert model.last_rollout_graph_health["mean_num_edges"] > 0.0
    assert model.last_rollout_graph_health["mean_min_degree"] >= 0.0
    assert model.last_rollout_graph_health["max_num_components"] >= 1.0


def test_sample_initial_boids_state_matches_simple_speed_norms():
    positions0, velocities0 = sample_initial_boids_state(
        10,
        seed=7,
        velocity_scale=0.014,
        device=torch.device("cpu"),
    )

    assert positions0.shape == (10, 2)
    assert velocities0.shape == (10, 2)
    assert torch.allclose(velocities0.norm(dim=1), torch.full((10,), 0.014), atol=1e-6)


def test_extract_learned_parameters_returns_demo_weights_only():
    history = {
        "w_sep": [1.0],
        "w_align": [0.7],
        "w_cohesion": [0.6],
        "damping": [0.95],
        "max_speed": [0.014],
    }

    learned_params = extract_learned_parameters(history)

    assert learned_params == {
        "w_sep": 1.0,
        "w_align": 0.7,
        "w_cohesion": 0.6,
        "damping": 0.95,
        "max_speed": 0.014,
    }


def test_evaluate_seed_reports_requested_horizon_and_per_step_loss():
    model = _make_model()
    simulation, teacher, model_spec = _teacher_specs(model, rounds=6)

    metrics = evaluate_seed(
        model,
        seed=101,
        simulation=simulation,
        teacher=teacher,
        model_spec=model_spec,
        velocity_loss_weight=1.0,
        rounds=4,
    )

    assert metrics["horizon"] == 4.0
    assert metrics["per_step_loss"] == metrics["total_loss"] / 4.0
    assert metrics["center_error"] >= 0.0


def test_learnable_defaults_use_fixed_horizon_and_keep_validation_enabled():
    args = _make_learnable_args()

    spec = build_learnable_spec(
        args,
        run_name="demo_defaults",
        run_dir=Path("generated/tests/demo_defaults"),
        viz_prefix="generated/tests/demo_defaults/learnable",
    )

    assert spec.simulation.num_nodes == 24
    assert spec.simulation.rounds == 24
    assert spec.training.epochs == 40
    assert spec.training.min_horizon == spec.simulation.rounds
    assert spec.training.max_horizon == spec.simulation.rounds
    assert spec.training.trunc_window == spec.simulation.rounds
    assert spec.training.num_initial_conditions == 1
    assert spec.evaluation.seeds == [101]
    assert spec.evaluation.every == 5
    assert spec.model.init_max_speed_target == spec.teacher.max_speed


def test_learnable_defaults_respect_explicit_overrides():
    args = _make_learnable_args(
        num_nodes=12,
        rounds=10,
        epochs=12,
        lr=0.03,
        eval_seeds="205,207",
        eval_every=2,
    )

    spec = build_learnable_spec(
        args,
        run_name="demo_overrides",
        run_dir=Path("generated/tests/demo_overrides"),
        viz_prefix="generated/tests/demo_overrides/learnable",
    )

    assert spec.simulation.num_nodes == 12
    assert spec.simulation.rounds == 10
    assert spec.training.epochs == 12
    assert abs(spec.training.lr - 0.03) < 1e-12
    assert spec.training.min_horizon == 10
    assert spec.training.max_horizon == 10
    assert spec.evaluation.seeds == [205, 207]
    assert spec.evaluation.every == 2


def test_replay_spec_materializes_trace_directory_when_enabled(tmp_path: Path):
    args = _make_learnable_args(
        supervision_mode="replay",
        replay_trace_dir=str(tmp_path),
        num_initial_conditions=2,
        curriculum_min_horizon=3,
        curriculum_max_horizon=8,
        trunc_window=4,
    )

    spec = build_learnable_spec(
        args,
        run_name="demo_replay",
        run_dir=tmp_path / "run",
        viz_prefix="generated/tests/demo_replay/learnable",
    )

    assert spec.training.supervision_mode == "replay"
    assert spec.training.replay_trace_dir == tmp_path
    assert spec.training.num_initial_conditions == 2
    assert spec.training.min_horizon == 3
    assert spec.training.max_horizon == 8
    assert spec.training.trunc_window == 4


def test_boids_trace_roundtrip_preserves_teacher_targets(tmp_path: Path):
    model = _make_model()
    simulation, teacher, model_spec = _teacher_specs(model, rounds=5)
    velocities0 = (torch.rand_like(model.positions0) - 0.5) * 0.01

    trace = teacher_trace_from_specs(
        seed=101,
        positions0=model.positions0,
        velocities0=velocities0,
        simulation=simulation,
        teacher=teacher,
        model=model_spec,
    )
    path = trace_file_path(tmp_path, seed=101, rounds=simulation.rounds)
    save_boids_trace(trace, path)
    loaded = load_boids_trace(path, device=torch.device("cpu"), expected_metadata=trace.metadata)

    assert loaded.metadata == trace.metadata
    assert torch.allclose(loaded.positions0, trace.positions0)
    assert torch.allclose(loaded.velocities0, trace.velocities0)
    assert torch.allclose(loaded.pos_seq, trace.pos_seq)
    assert torch.allclose(loaded.vel_seq, trace.vel_seq)
    assert torch.allclose(loaded.preclip_vel_seq, trace.preclip_vel_seq)


def test_build_supervision_traces_replay_mode_reuses_materialized_files(tmp_path: Path):
    args = _make_learnable_args(
        supervision_mode="replay",
        replay_trace_dir=str(tmp_path),
        num_initial_conditions=2,
    )
    spec = build_learnable_spec(
        args,
        run_name="demo_replay_files",
        run_dir=tmp_path / "run",
        viz_prefix="generated/tests/demo_replay_files/learnable",
    )

    initial_conditions = make_initial_conditions(spec)
    traces = build_supervision_traces(initial_conditions, spec=spec)

    assert len(traces) == 2
    for idx, trace in enumerate(traces):
        seed = spec.seed + idx * 1337
        path = trace_file_path(tmp_path, seed=seed, rounds=spec.simulation.rounds)
        assert path.exists()
        assert trace.metadata["seed"] == seed
        assert trace.rounds == spec.simulation.rounds


def test_alignment_cohesion_demo_task_reduces_teacher_forced_loss():
    positions0, velocities0 = sample_initial_boids_state(
        10,
        seed=9,
        velocity_scale=0.014,
        device=torch.device("cpu"),
    )
    model = LearnableAggregateBoids(
        positions0=positions0,
        radius=0.23,
        sep=0.06,
        dt=1.0,
        init_connectivity="hybrid",
        init_k_neighbors=6,
        init_min_degree=2,
        init_w_sep_target=0.02,
        init_w_align_target=0.08,
        init_w_cohesion_target=1.10,
        init_damping_target=0.72,
        init_max_speed_target=0.014,
    )
    simulation = SimulationSpec(
        num_nodes=10,
        rounds=6,
        radius=0.23,
        sep=0.06,
        dt=1.0,
        init_velocity_scale=0.014,
        device=torch.device("cpu"),
    )
    teacher = TeacherDynamics(
        w_sep=0.05,
        w_align=0.90,
        w_cohesion=0.35,
        damping=0.94,
        max_speed=0.014,
    )
    model_spec = ModelSpec(
        init_connectivity="hybrid",
        init_k_neighbors=6,
        init_min_degree=2,
        init_w_sep_target=0.02,
        init_w_align_target=0.08,
        init_w_cohesion_target=1.10,
        init_damping_target=0.72,
        init_max_speed_target=0.014,
        max_speed_min=model.max_speed_min,
        max_speed_max=model.max_speed_max,
    )
    trace = teacher_trace_from_specs(
        seed=9,
        positions0=positions0,
        velocities0=velocities0,
        simulation=simulation,
        teacher=teacher,
        model=model_spec,
    )

    optimizer = torch.optim.Adam(model.trainable_parameters(), lr=0.05)
    initial_loss, _, _, _ = teacher_forced_step_losses(
        model,
        trace=trace,
        velocity_loss_weight=1.0,
        sep=simulation.sep,
    )

    for _ in range(12):
        optimizer.zero_grad()
        loss, _, _, _ = teacher_forced_step_losses(
            model,
            trace=trace,
            velocity_loss_weight=1.0,
            sep=simulation.sep,
        )
        loss.backward()
        optimizer.step()

    final_loss, _, _, _ = teacher_forced_step_losses(
        model,
        trace=trace,
        velocity_loss_weight=1.0,
        sep=simulation.sep,
    )

    assert final_loss.item() < initial_loss.item()


def test_curriculum_horizon_with_fraction_slows_ramp_for_long_runs():
    assert curriculum_horizon_with_fraction(0, 500, 3, 30, ramp_fraction=0.85) == 3
    assert curriculum_horizon_with_fraction(300, 500, 3, 30, ramp_fraction=0.85) < 30
    assert curriculum_horizon_with_fraction(425, 500, 3, 30, ramp_fraction=0.85) == 30


def test_scheduled_learning_rate_decays_after_curriculum():
    base_lr = 0.01
    early = scheduled_learning_rate(base_lr, 100, 500, ramp_fraction=0.85, final_lr_ratio=0.1)
    late = scheduled_learning_rate(base_lr, 450, 500, ramp_fraction=0.85, final_lr_ratio=0.1)
    final = scheduled_learning_rate(base_lr, 499, 500, ramp_fraction=0.85, final_lr_ratio=0.1)

    assert abs(early - base_lr) < 1e-12
    assert late < base_lr
    assert abs(final - base_lr * 0.1) < 1e-9


def test_close_pair_distance_loss_is_zero_for_identical_sequences():
    positions = torch.tensor(
        [
            [[0.0, 0.0], [0.05, 0.0], [0.2, 0.0]],
            [[0.0, 0.0], [0.06, 0.0], [0.2, 0.0]],
        ],
        dtype=torch.float32,
    )

    loss = close_pair_distance_loss(positions, positions, sep=0.06)

    assert torch.allclose(loss, torch.tensor(0.0), atol=1e-8)


def test_close_pair_distance_loss_penalizes_near_field_mismatch():
    teacher = torch.tensor(
        [
            [[0.0, 0.0], [0.05, 0.0], [0.2, 0.0]],
        ],
        dtype=torch.float32,
    )
    pred = teacher.clone()
    pred[0, 1, 0] = 0.10

    loss = close_pair_distance_loss(pred, teacher, sep=0.06)

    assert float(loss.item()) > 0.0


def test_teacher_forced_step_losses_vanish_when_model_matches_teacher():
    model = _make_model()
    simulation, teacher, model_spec = _teacher_specs(model, rounds=4)
    velocities0 = (torch.rand_like(model.positions0) - 0.5) * 0.01

    with torch.no_grad():
        model.w_sep_raw.copy_(_softplus_inverse(teacher.w_sep))
        model.w_align_raw.copy_(_softplus_inverse(teacher.w_align))
        model.w_cohesion_raw.copy_(_softplus_inverse(teacher.w_cohesion))
        model.damping_raw.copy_(torch.logit(torch.tensor(teacher.damping, dtype=torch.float32)))
        model.max_speed_raw.copy_(
            _bounded_sigmoid_inverse(teacher.max_speed, model.max_speed_min, model.max_speed_max),
        )

    trace = teacher_trace_from_specs(
        seed=17,
        positions0=model.positions0,
        velocities0=velocities0,
        simulation=simulation,
        teacher=teacher,
        model=model_spec,
    )
    total, pos, vel, sep = teacher_forced_step_losses(
        model,
        trace=trace,
        velocity_loss_weight=1.0,
        sep=simulation.sep,
    )

    assert torch.allclose(total, torch.tensor(0.0), atol=1e-10)
    assert torch.allclose(pos, torch.tensor(0.0), atol=1e-10)
    assert torch.allclose(vel, torch.tensor(0.0), atol=1e-10)
    assert torch.allclose(sep, torch.tensor(0.0), atol=1e-10)


def test_truncated_rollout_backpropagates_finite_gradients():
    model = _make_model()
    velocities0 = (torch.rand_like(model.positions0) - 0.5) * 0.01

    pred_pos_seq, pred_vel_seq, _ = model.rollout(rounds=6, velocities0=velocities0, trunc_window=2)
    loss = pred_pos_seq.square().mean() + pred_vel_seq.square().mean()
    loss.backward()

    _assert_finite_gradients(model.trainable_parameters())


def test_weights_rollout_matches_teacher_when_parameters_match():
    model = _make_model()
    simulation, teacher, model_spec = _teacher_specs(model, rounds=5)
    velocities0 = (torch.rand_like(model.positions0) - 0.5) * 0.01

    with torch.no_grad():
        model.w_sep_raw.copy_(_softplus_inverse(teacher.w_sep))
        model.w_align_raw.copy_(_softplus_inverse(teacher.w_align))
        model.w_cohesion_raw.copy_(_softplus_inverse(teacher.w_cohesion))
        model.damping_raw.copy_(torch.logit(torch.tensor(teacher.damping, dtype=torch.float32)))
        model.max_speed_raw.copy_(
            _bounded_sigmoid_inverse(teacher.max_speed, model.max_speed_min, model.max_speed_max),
        )

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


def test_rollout_truncation_matches_full_when_window_covers_horizon():
    model = _make_small_boids_model()
    velocities0 = torch.tensor(
        [
            [0.010, 0.000],
            [0.000, 0.008],
            [-0.006, 0.004],
            [0.003, -0.007],
            [0.005, 0.005],
            [-0.004, -0.003],
        ],
        dtype=torch.float32,
    )

    full_pos, full_vel, full_final = model.rollout(rounds=4, velocities0=velocities0, trunc_window=None)
    trunc_pos, trunc_vel, trunc_final = model.rollout(rounds=4, velocities0=velocities0, trunc_window=4)

    assert torch.allclose(trunc_pos, full_pos, atol=1e-6)
    assert torch.allclose(trunc_vel, full_vel, atol=1e-6)
    assert torch.allclose(trunc_final, full_final, atol=1e-6)


def test_truncated_training_reduces_teacher_loss():
    torch.manual_seed(0)
    model = _make_small_boids_model()
    velocities0 = torch.tensor(
        [
            [0.010, 0.000],
            [0.000, 0.008],
            [-0.006, 0.004],
            [0.003, -0.007],
            [0.005, 0.005],
            [-0.004, -0.003],
        ],
        dtype=torch.float32,
    )
    simulation, teacher_pos_seq, teacher_vel_seq = _boids_teacher_targets(model, rounds=4, velocities0=velocities0)
    optimizer = torch.optim.Adam(model.trainable_parameters(), lr=0.05)

    losses = []
    for _ in range(10):
        optimizer.zero_grad()
        pred_pos_seq, pred_vel_seq, _ = model.rollout(
            rounds=simulation.rounds,
            velocities0=velocities0,
            trunc_window=2,
        )
        total_loss, _, _ = trajectory_loss_components(
            pred_pos_seq,
            pred_vel_seq,
            teacher_pos_seq,
            teacher_vel_seq,
            velocity_weight=1.0,
        )
        total_loss.backward()
        _assert_finite_gradients(model.trainable_parameters())
        optimizer.step()
        losses.append(float(total_loss.item()))

    assert losses[-1] < losses[0]