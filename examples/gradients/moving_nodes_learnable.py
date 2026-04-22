#!/usr/bin/env python3
"""Learnable moving-node DSL gradient example."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "examples"))

import torch
from diffield.utils import get_device

try:
    from .domain.specs import MovingGradientSpec
    from .training.moving_workflow import MovingGradientWorkflow
except ImportError:
    from gradients.domain.specs import MovingGradientSpec
    from gradients.training.moving_workflow import MovingGradientWorkflow


def parse_args():
    parser = argparse.ArgumentParser(description="Learnable moving-node DSL gradient")
    parser.add_argument("--num-nodes", type=int, default=24)
    parser.add_argument("--rounds", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--radius", type=float, default=0.32)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--source", type=int, default=0)
    parser.add_argument("--target", type=int, default=10)
    parser.add_argument("--learn", choices=["motion", "ac", "both"], default="both")
    parser.add_argument(
        "--device", type=str, default="", help="Device (cuda/cpu) [auto if empty]"
    )
    parser.add_argument(
        "--viz-prefix", type=str, default="generated/gradient_moving_learnable"
    )
    parser.add_argument("--gif-fps", type=int, default=10)
    parser.add_argument("--no-viz", action="store_true", help="Disable figure export")
    parser.add_argument("--no-gif", action="store_true", help="Disable gif export")
    return parser.parse_args()


def build_spec(args) -> MovingGradientSpec:
    return MovingGradientSpec(
        num_nodes=args.num_nodes,
        rounds=args.rounds,
        epochs=args.epochs,
        radius=args.radius,
        seed=args.seed,
        lr=args.lr,
        source=args.source,
        target=args.target,
        learn=args.learn,
    )


def main():
    args = parse_args()
    device = get_device(args.device)
    spec = build_spec(args)
    workflow = MovingGradientWorkflow(spec)
    workflow.run(
        device=device,
        gif=not args.no_gif,
        viz=not args.no_viz,
        viz_prefix=args.viz_prefix,
        gif_fps=args.gif_fps,
    )


if __name__ == "__main__":
    main()
