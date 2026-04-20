"""Domain configuration objects for learnable boids workflows."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from ..cli import DEFAULTS
from autofield.utils import get_device

try:
    from ..shared.training import parse_int_csv
except ImportError:
    from shared.training import parse_int_csv


@dataclass(frozen=True)
class SimulationSpec:
    num_nodes: int
    rounds: int
    radius: float
    sep: float
    dt: float
    init_velocity_scale: float
    damping: float
    max_speed: float
    device: torch.device


@dataclass(frozen=True)
class TeacherDynamics:
    w_sep: float
    w_align: float
    w_cohesion: float


@dataclass(frozen=True)
class ModelSpec:
    init_connectivity: str
    init_k_neighbors: int
    init_min_degree: int
    init_w_sep_target: float
    init_w_align_target: float
    init_w_cohesion_target: float


@dataclass(frozen=True)
class TrainingSpec:
    epochs: int
    lr: float
    min_horizon: int
    max_horizon: int
    curriculum_ramp_fraction: float
    final_lr_ratio: float
    trunc_window: int
    num_initial_conditions: int
    velocity_loss_weight: float
    separation_loss_weight: float
    print_every: int
    checkpoint_every_epochs: int
    supervision_mode: str
    replay_trace_dir: Path | None
    save_replay_traces: bool


@dataclass(frozen=True)
class EvaluationSpec:
    seeds: list[int]
    every: int


@dataclass(frozen=True)
class VisualizationSpec:
    enabled: bool
    gif_enabled: bool
    show_links: bool
    links_alpha: float
    links_width: float
    gif_fps: int
    highlight_node: int
    record_every: int
    viz_prefix: str


@dataclass(frozen=True)
class LearnableBoidsSpec:
    seed: int
    run_name: str
    run_dir: Path
    simulation: SimulationSpec
    teacher: TeacherDynamics
    model: ModelSpec
    training: TrainingSpec
    evaluation: EvaluationSpec
    visualization: VisualizationSpec


def build_learnable_spec(
    args: Any,
    run_name: str,
    run_dir: Path,
    viz_prefix: str,
) -> LearnableBoidsSpec:
    num_nodes = int(args.num_nodes)
    rounds = int(args.rounds)
    epochs = int(args.epochs)
    teacher_w_sep = float(args.teacher_w_sep)
    teacher_w_align = float(args.teacher_w_align)
    teacher_w_cohesion = float(args.teacher_w_cohesion)
    damping = float(args.damping)
    max_speed = float(args.max_speed)
    init_w_sep_target = float(args.init_w_sep_target)
    init_w_align_target = float(args.init_w_align_target)
    init_w_cohesion_target = float(args.init_w_cohesion_target)
    lr = float(args.lr)
    min_horizon = (
        rounds
        if args.curriculum_min_horizon is None
        else max(1, min(int(args.curriculum_min_horizon), rounds))
    )
    requested_max_horizon = (
        rounds
        if args.curriculum_max_horizon is None
        else max(1, min(int(args.curriculum_max_horizon), rounds))
    )
    max_horizon = max(min_horizon, requested_max_horizon)
    trunc_window = (
        rounds
        if args.trunc_window is None
        else max(1, min(int(args.trunc_window), rounds))
    )
    num_initial_conditions = max(1, int(args.num_initial_conditions))
    velocity_loss_weight = float(args.velocity_loss_weight)
    separation_loss_weight = float(args.separation_loss_weight)
    print_every = int(args.print_every)
    record_every = int(args.record_every)
    eval_every = int(args.eval_every)
    eval_seed_csv = str(args.eval_seeds).strip() or DEFAULTS["eval_seeds"]
    checkpoint_every_epochs = max(1, int(args.checkpoint_every_epochs))
    curriculum_ramp_fraction = min(max(float(args.curriculum_ramp_fraction), 0.0), 1.0)
    final_lr_ratio = max(0.0, float(args.final_lr_ratio))
    supervision_mode = str(args.supervision_mode)
    replay_trace_dir_arg = str(args.replay_trace_dir).strip()
    replay_trace_dir: Path | None = None
    if replay_trace_dir_arg:
        replay_trace_dir = Path(replay_trace_dir_arg)
    elif supervision_mode == "replay" or bool(args.save_replay_traces):
        replay_trace_dir = run_dir / "replay_traces"

    sim_spec = SimulationSpec(
        num_nodes=num_nodes,
        rounds=rounds,
        radius=args.radius,
        sep=args.sep,
        dt=args.dt,
        init_velocity_scale=args.init_velocity_scale,
        damping=damping,
        max_speed=max_speed,
        device=get_device(args.device),
    )
    return LearnableBoidsSpec(
        seed=args.seed,
        run_name=run_name,
        run_dir=run_dir,
        simulation=sim_spec,
        teacher=TeacherDynamics(
            w_sep=teacher_w_sep,
            w_align=teacher_w_align,
            w_cohesion=teacher_w_cohesion,
        ),
        model=ModelSpec(
            init_connectivity=args.init_connectivity,
            init_k_neighbors=args.init_k_neighbors,
            init_min_degree=args.init_min_degree,
            init_w_sep_target=max(0.0, init_w_sep_target),
            init_w_align_target=max(0.0, init_w_align_target),
            init_w_cohesion_target=max(0.0, init_w_cohesion_target),
        ),
        training=TrainingSpec(
            epochs=epochs,
            lr=lr,
            min_horizon=min_horizon,
            max_horizon=max_horizon,
            curriculum_ramp_fraction=curriculum_ramp_fraction,
            final_lr_ratio=final_lr_ratio,
            trunc_window=trunc_window,
            num_initial_conditions=num_initial_conditions,
            velocity_loss_weight=max(0.0, velocity_loss_weight),
            separation_loss_weight=max(0.0, separation_loss_weight),
            print_every=print_every,
            checkpoint_every_epochs=checkpoint_every_epochs,
            supervision_mode=supervision_mode,
            replay_trace_dir=replay_trace_dir,
            save_replay_traces=bool(args.save_replay_traces),
        ),
        evaluation=EvaluationSpec(
            seeds=parse_int_csv(eval_seed_csv),
            every=eval_every,
        ),
        visualization=VisualizationSpec(
            enabled=not args.no_viz,
            gif_enabled=not args.no_gif,
            show_links=not args.hide_links,
            links_alpha=args.links_alpha,
            links_width=args.links_width,
            gif_fps=max(1, args.gif_fps),
            highlight_node=args.highlight_node,
            record_every=record_every,
            viz_prefix=viz_prefix,
        ),
    )
