"""CLI definitions for learnable boids experiments."""

from __future__ import annotations

import argparse

HISTORY_KEYS = [
    "epoch",
    "horizon",
    "total",
    "pos_loss",
    "vel_loss",
    "center_error",
    "w_sep",
    "w_align",
    "w_cohesion",
    "damping",
    "max_speed",
    "tau_align",
    "tau_cohesion",
    "grad_norm",
    "cap_fraction",
    "pre_clip_speed",
    "val_total_loss",
    "val_pos_loss",
    "val_vel_loss",
    "val_center_error",
]


def parse_learnable_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Aggregate boids with learnable dynamics")
    parser.add_argument("--num-nodes", type=int, default=60)
    parser.add_argument("--rounds", type=int, default=80)
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--radius", type=float, default=0.23)
    parser.add_argument("--init-connectivity", choices=["radius", "knn", "hybrid"], default="hybrid")
    parser.add_argument("--init-k-neighbors", type=int, default=8)
    parser.add_argument("--init-min-degree", type=int, default=2)
    parser.add_argument("--sep", type=float, default=0.06)
    parser.add_argument("--dt", type=float, default=1.0)
    parser.add_argument("--init-velocity-scale", type=float, default=0.014, help="Uniform initial velocity scale for sampled initial conditions")
    parser.add_argument("--teacher-w-sep", type=float, default=1.0, help="Teacher separation weight")
    parser.add_argument("--teacher-w-align", type=float, default=0.7, help="Teacher alignment weight")
    parser.add_argument("--teacher-w-cohesion", type=float, default=0.6, help="Teacher cohesion weight")
    parser.add_argument("--teacher-damping", type=float, default=0.95)
    parser.add_argument("--teacher-max-speed", type=float, default=0.014)
    parser.add_argument("--init-damping-target", type=float, default=0.94, help="Initial learnable damping value in (0,1)")
    parser.add_argument("--init-max-speed-target", type=float, default=0.03, help="Initial learnable speed cap > 0")
    parser.add_argument("--max-speed-min", type=float, default=0.004, help="Lower bound for learnable speed cap")
    parser.add_argument("--max-speed-max", type=float, default=0.06, help="Upper bound for learnable speed cap")
    parser.add_argument(
        "--train-max-speed-in-weights",
        action="store_true",
        help="When mode=weights, also optimize max_speed instead of keeping it fixed at the teacher cap",
    )
    parser.add_argument("--seed", type=int, default=5)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--mode", choices=["weights", "attention", "joint"], default="joint")
    parser.add_argument("--out-dir", type=str, default="generated/results")
    parser.add_argument("--run-name", type=str, default="")
    parser.add_argument("--curriculum-min-horizon", type=int, default=3)
    parser.add_argument("--curriculum-max-horizon", type=int, default=80)
    parser.add_argument("--trunc-window", type=int, default=8)
    parser.add_argument("--num-initial-conditions", type=int, default=6)
    parser.add_argument("--velocity-loss-weight", type=float, default=10.0)
    parser.add_argument("--print-every", type=int, default=20)
    parser.add_argument("--record-every", type=int, default=5)
    parser.add_argument("--checkpoint-every-epochs", type=int, default=20)
    parser.add_argument("--eval-seeds", type=str, default="", help="Comma-separated held-out seeds")
    parser.add_argument("--eval-every", type=int, default=20)
    parser.add_argument("--highlight-node", type=int, default=0)
    parser.add_argument("--viz-prefix", type=str, default="generated/boids/learnable")
    parser.add_argument("--gif-fps", type=int, default=8)
    parser.add_argument("--no-viz", action="store_true", help="Disable figure export")
    parser.add_argument("--no-gif", action="store_true", help="Disable gif export")
    parser.add_argument("--hide-links", action="store_true", help="Do not draw graph links in visual outputs")
    parser.add_argument("--links-alpha", type=float, default=0.15)
    parser.add_argument("--links-width", type=float, default=0.6)
    parser.add_argument("--no-compare-panel", action="store_true", help="Disable predicted-vs-teacher side-by-side panel")
    parser.add_argument("--device", type=str, default="", help="Device (cuda/cpu) [auto if empty]")
    return parser.parse_args()
