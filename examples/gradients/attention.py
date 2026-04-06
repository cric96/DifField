#!/usr/bin/env python3
"""Gradient example with a learnable attention-based aggregator."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "examples"))

import torch
from autofield.utils import get_device

try:
    from .domain.specs import AttentionGradientSpec, GridSpec, TrainingSpec
    from .training.attention_workflow import AttentionGradientWorkflow
except ImportError:
    from gradients.domain.specs import AttentionGradientSpec, GridSpec, TrainingSpec
    from gradients.training.attention_workflow import AttentionGradientWorkflow


def parse_args():
    parser = argparse.ArgumentParser(description="Attention Gradient AC program")
    parser.add_argument("--rows", type=int, default=5, help="Grid rows")
    parser.add_argument("--cols", type=int, default=5, help="Grid cols")
    parser.add_argument("--epochs", type=int, default=50, help="Training epochs")
    parser.add_argument("--lr", type=float, default=0.01, help="Learning rate")
    parser.add_argument("--seed", type=int, default=7, help="Random seed")
    parser.add_argument(
        "--device", type=str, default="", help="Device (cuda/cpu) [auto if empty]"
    )
    parser.add_argument(
        "--viz-prefix", type=str, default="generated/gradient_attention"
    )
    parser.add_argument("--gif-fps", type=int, default=10)
    parser.add_argument("--no-viz", action="store_true", help="Disable figure export")
    parser.add_argument("--no-gif", action="store_true", help="Disable gif export")
    return parser.parse_args()


def build_spec(args) -> AttentionGradientSpec:
    return AttentionGradientSpec(
        grid=GridSpec(rows=args.rows, cols=args.cols, connectivity=4),
        rounds=0,
        training=TrainingSpec(epochs=args.epochs, lr=args.lr),
    )


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    device = get_device(args.device)
    spec = build_spec(args)
    workflow = AttentionGradientWorkflow(spec)
    _, pred = workflow.run(
        device=device,
        gif=not args.no_gif,
        viz=not args.no_viz,
        viz_prefix=args.viz_prefix,
        gif_fps=args.gif_fps,
    )
    print()
    print("Predicted:")
    print(pred.view(args.rows, args.cols).detach().cpu().numpy().round(2))


if __name__ == "__main__":
    main()
