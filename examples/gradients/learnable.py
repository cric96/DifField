#!/usr/bin/env python3
"""Gradient example with a learnable hop weight."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

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
    spec = build_spec(args)
    workflow = LearnableGradientWorkflow(spec)
    _, _, target, pred = workflow.run()

    print()
    print("Predicted distances:")
    print(pred.view(args.rows, args.cols).numpy())
    print()
    print("Target (Manhattan):")
    print(target.view(args.rows, args.cols).numpy())


if __name__ == "__main__":
    main()