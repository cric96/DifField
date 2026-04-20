"""CLI definitions for learnable boids experiments."""

from __future__ import annotations

import argparse

DEFAULTS = {
    "num_nodes": 24,
    "rounds": 24,
    "epochs": 40,
    "supervision_mode": "teacher",
    "teacher_w_sep": 0.05,
    "teacher_w_align": 0.90,
    "teacher_w_cohesion": 0.35,
    "damping": 0.94,
    "max_speed": 0.014,
    "init_w_sep_target": 0.02,
    "init_w_align_target": 0.08,
    "init_w_cohesion_target": 1.10,
    "lr": 0.05,
    "curriculum_ramp_fraction": 1.0,
    "final_lr_ratio": 1.0,
    "num_initial_conditions": 1,
    "velocity_loss_weight": 1.0,
    "separation_loss_weight": 0.0,
    "print_every": 5,
    "record_every": 2,
    "checkpoint_every_epochs": 40,
    "eval_seeds": "101",
    "eval_every": 5,
}

HISTORY_KEYS = [
    "epoch",
    "horizon",
    "total",
    "per_step_loss",
    "objective_loss",
    "pos_loss",
    "vel_loss",
    "sep_focus_loss",
    "center_error",
    "w_sep",
    "w_sep_abs_error",
    "w_sep_rel_error",
    "w_align",
    "w_align_abs_error",
    "w_align_rel_error",
    "w_cohesion",
    "w_cohesion_abs_error",
    "w_cohesion_rel_error",
    "grad_norm",
    "lr",
    "cap_fraction",
    "pre_clip_speed",
    "val_curriculum_horizon",
    "val_curriculum_total_loss",
    "val_curriculum_pos_loss",
    "val_curriculum_vel_loss",
    "val_curriculum_per_step_loss",
    "val_curriculum_center_error",
    "val_full_horizon",
    "val_full_total_loss",
    "val_full_pos_loss",
    "val_full_vel_loss",
    "val_full_per_step_loss",
    "val_full_center_error",
]


def parse_learnable_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Aggregate boids demo with learnable dynamics")
    parser.add_argument("--num-nodes", type=int, default=DEFAULTS["num_nodes"])
    parser.add_argument("--rounds", type=int, default=DEFAULTS["rounds"])
    parser.add_argument("--epochs", type=int, default=DEFAULTS["epochs"])
    parser.add_argument("--radius", type=float, default=0.23)
    parser.add_argument("--init-connectivity", choices=["radius", "knn", "hybrid"], default="hybrid")
    parser.add_argument("--init-k-neighbors", type=int, default=8)
    parser.add_argument("--init-min-degree", type=int, default=2)
    parser.add_argument("--sep", type=float, default=0.06)
    parser.add_argument("--dt", type=float, default=1.0)
    parser.add_argument("--init-velocity-scale", type=float, default=0.014, help="Uniform initial velocity scale for sampled initial conditions")
    parser.add_argument("--teacher-w-sep", type=float, default=DEFAULTS["teacher_w_sep"], help="Teacher separation weight")
    parser.add_argument("--teacher-w-align", type=float, default=DEFAULTS["teacher_w_align"], help="Teacher alignment weight")
    parser.add_argument("--teacher-w-cohesion", type=float, default=DEFAULTS["teacher_w_cohesion"], help="Teacher cohesion weight")
    parser.add_argument("--init-w-sep-target", type=float, default=DEFAULTS["init_w_sep_target"], help="Initial learnable separation weight target")
    parser.add_argument("--init-w-align-target", type=float, default=DEFAULTS["init_w_align_target"], help="Initial learnable alignment weight target")
    parser.add_argument("--init-w-cohesion-target", type=float, default=DEFAULTS["init_w_cohesion_target"], help="Initial learnable cohesion weight target")
    parser.add_argument("--damping", type=float, default=DEFAULTS["damping"], help="Fixed damping factor in (0,1)")
    parser.add_argument("--max-speed", type=float, default=DEFAULTS["max_speed"], help="Fixed speed cap > 0")
    parser.add_argument("--seed", type=int, default=5)
    parser.add_argument("--lr", type=float, default=DEFAULTS["lr"])
    parser.add_argument("--out-dir", type=str, default="generated/results")
    parser.add_argument("--run-name", type=str, default="")
    parser.add_argument("--supervision-mode", choices=["teacher", "replay"], default=DEFAULTS["supervision_mode"], help="Use online teacher traces or persisted replay traces")
    parser.add_argument("--replay-trace-dir", type=str, default="", help="Directory containing or receiving persisted replay traces")
    parser.add_argument("--save-replay-traces", action="store_true", help="Persist teacher traces while training in teacher mode")
    parser.add_argument("--curriculum-min-horizon", type=int, default=None, help="Minimum teacher-forced horizon; defaults to full rounds")
    parser.add_argument("--curriculum-max-horizon", type=int, default=None, help="Maximum teacher-forced horizon; defaults to full rounds")
    parser.add_argument("--curriculum-ramp-fraction", type=float, default=DEFAULTS["curriculum_ramp_fraction"])
    parser.add_argument("--final-lr-ratio", type=float, default=DEFAULTS["final_lr_ratio"])
    parser.add_argument("--trunc-window", type=int, default=None, help="Detach recurrent state every N rounds; defaults to full rounds")
    parser.add_argument("--num-initial-conditions", type=int, default=DEFAULTS["num_initial_conditions"])
    parser.add_argument("--velocity-loss-weight", type=float, default=DEFAULTS["velocity_loss_weight"])
    parser.add_argument("--separation-loss-weight", type=float, default=DEFAULTS["separation_loss_weight"])
    parser.add_argument("--print-every", type=int, default=DEFAULTS["print_every"])
    parser.add_argument("--record-every", type=int, default=DEFAULTS["record_every"])
    parser.add_argument("--checkpoint-every-epochs", type=int, default=DEFAULTS["checkpoint_every_epochs"])
    parser.add_argument("--eval-seeds", type=str, default=DEFAULTS["eval_seeds"], help="Comma-separated held-out seeds")
    parser.add_argument("--eval-every", type=int, default=DEFAULTS["eval_every"])
    parser.add_argument("--highlight-node", type=int, default=0)
    parser.add_argument("--viz-prefix", type=str, default="generated/boids/learnable")
    parser.add_argument("--gif-fps", type=int, default=8)
    parser.add_argument("--no-viz", action="store_true", help="Disable figure export")
    parser.add_argument("--no-gif", action="store_true", help="Disable gif export")
    parser.add_argument("--hide-links", action="store_true", help="Do not draw graph links in visual outputs")
    parser.add_argument("--links-alpha", type=float, default=0.15)
    parser.add_argument("--links-width", type=float, default=0.6)
    parser.add_argument("--device", type=str, default="", help="Device (cuda/cpu) [auto if empty]")
    return parser.parse_args()
