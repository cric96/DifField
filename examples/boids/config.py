"""Domain configuration objects for learnable boids workflows."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from shared.training import parse_int_csv


@dataclass(frozen=True)
class SimulationSpec:
    num_nodes: int
    rounds: int
    radius: float
    sep: float
    dt: float


@dataclass(frozen=True)
class TeacherDynamics:
    w_sep: float
    w_align: float
    w_cohesion: float
    damping: float
    max_speed: float


@dataclass(frozen=True)
class ModelSpec:
    mode: str
    init_connectivity: str
    init_k_neighbors: int
    init_min_degree: int
    init_damping_target: float
    init_max_speed_target: float
    max_speed_min: float
    max_speed_max: float
    train_max_speed: bool


@dataclass(frozen=True)
class TrainingSpec:
    epochs: int
    lr: float
    traj_w: float
    cohesion_reg: float
    alignment_reg: float
    speed_reg: float
    accel_reg: float
    print_every: int
    checkpoint_every_epochs: int


@dataclass(frozen=True)
class EvaluationSpec:
    seeds: list[int]
    every: int


@dataclass(frozen=True)
class VisualizationSpec:
    enabled: bool
    gif_enabled: bool
    compare_panel_enabled: bool
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


def build_learnable_spec(args: Any, *, run_name: str, run_dir: Path, viz_prefix: str) -> LearnableBoidsSpec:
    train_max_speed = not (args.mode == "weights" and not args.train_max_speed_in_weights)
    init_max_speed_target = args.teacher_max_speed if not train_max_speed else args.init_max_speed_target
    return LearnableBoidsSpec(
        seed=args.seed,
        run_name=run_name,
        run_dir=run_dir,
        simulation=SimulationSpec(
            num_nodes=args.num_nodes,
            rounds=args.rounds,
            radius=args.radius,
            sep=args.sep,
            dt=args.dt,
        ),
        teacher=TeacherDynamics(
            w_sep=args.teacher_w_sep,
            w_align=args.teacher_w_align,
            w_cohesion=args.teacher_w_cohesion,
            damping=args.teacher_damping,
            max_speed=args.teacher_max_speed,
        ),
        model=ModelSpec(
            mode=args.mode,
            init_connectivity=args.init_connectivity,
            init_k_neighbors=args.init_k_neighbors,
            init_min_degree=args.init_min_degree,
            init_damping_target=args.init_damping_target,
            init_max_speed_target=init_max_speed_target,
            max_speed_min=args.max_speed_min,
            max_speed_max=args.max_speed_max,
            train_max_speed=train_max_speed,
        ),
        training=TrainingSpec(
            epochs=args.epochs,
            lr=args.lr,
            traj_w=args.traj_w,
            cohesion_reg=args.cohesion_reg,
            alignment_reg=args.alignment_reg,
            speed_reg=args.speed_reg,
            accel_reg=args.accel_reg,
            print_every=args.print_every,
            checkpoint_every_epochs=args.checkpoint_every_epochs,
        ),
        evaluation=EvaluationSpec(
            seeds=parse_int_csv(args.eval_seeds),
            every=args.eval_every,
        ),
        visualization=VisualizationSpec(
            enabled=not args.no_viz,
            gif_enabled=not args.no_gif,
            compare_panel_enabled=not args.no_compare_panel,
            show_links=not args.hide_links,
            links_alpha=args.links_alpha,
            links_width=args.links_width,
            gif_fps=max(1, args.gif_fps),
            highlight_node=args.highlight_node,
            record_every=args.record_every,
            viz_prefix=viz_prefix,
        ),
    )
