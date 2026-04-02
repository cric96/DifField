#!/usr/bin/env python3
"""Gradient example with a learnable hop weight."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "examples"))

import torch
from aggregate_gnn.utils import get_device

try:
    from .specs import GridSpec, LearnableGradientSpec, TrainingSpec
    from .workflow import LearnableGradientWorkflow
except ImportError:
    from specs import GridSpec, LearnableGradientSpec, TrainingSpec
    from workflow import LearnableGradientWorkflow


def parse_args():
    parser = argparse.ArgumentParser(description="Learnable Gradient AC program")
    parser.add_argument("--rows", type=int, default=5, help="Grid rows")
    parser.add_argument("--cols", type=int, default=5, help="Grid cols")
    parser.add_argument("--epochs", type=int, default=200, help="Training epochs")
    parser.add_argument("--lr", type=float, default=0.05, help="Learning rate")
    parser.add_argument("--initial-weight", type=float, default=3.0, help="Initial w value")
    parser.add_argument("--seed", type=int, default=7, help="Random seed")
    parser.add_argument("--device", type=str, default="", help="Device (cuda/cpu) [auto if empty]")
    parser.add_argument("--viz-prefix", type=str, default="generated/gradient_learnable")
    parser.add_argument("--gif-fps", type=int, default=10)
    parser.add_argument("--no-viz", action="store_true", help="Disable figure export")
    parser.add_argument("--no-gif", action="store_true", help="Disable gif export")
    return parser.parse_args()


def build_spec(args) -> LearnableGradientSpec:
    return LearnableGradientSpec(
        grid=GridSpec(rows=args.rows, cols=args.cols, connectivity=4),
        rounds=0,
        training=TrainingSpec(epochs=args.epochs, lr=args.lr),
        initial_weight=args.initial_weight,
    )


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    device = get_device(args.device)
    spec = build_spec(args)
    workflow = LearnableGradientWorkflow(spec)
    _, _, target, pred = workflow.run(
        device=device,
        gif=not args.no_gif,
        viz=not args.no_viz,
        viz_prefix=args.viz_prefix,
        gif_fps=args.gif_fps,
    )

    print()
    print("Predicted distances:")
    print(pred.view(args.rows, args.cols).detach().cpu().numpy())
    print()
    print("Target (Manhattan):")
    print(target.view(args.rows, args.cols).detach().cpu().numpy())


if __name__ == "__main__":
    main()