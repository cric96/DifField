#!/usr/bin/env python3
"""Gradient example with a learnable attention-based aggregator."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

try:
    from .specs import AttentionGradientSpec, GridSpec, TrainingSpec
    from .workflow import AttentionGradientWorkflow
except ImportError:
    from specs import AttentionGradientSpec, GridSpec, TrainingSpec
    from workflow import AttentionGradientWorkflow


def parse_args():
    parser = argparse.ArgumentParser(description="Attention Gradient AC program")
    parser.add_argument("--rows", type=int, default=5, help="Grid rows")
    parser.add_argument("--cols", type=int, default=5, help="Grid cols")
    parser.add_argument("--epochs", type=int, default=50, help="Training epochs")
    parser.add_argument("--lr", type=float, default=0.01, help="Learning rate")
    return parser.parse_args()


def build_spec(args) -> AttentionGradientSpec:
    return AttentionGradientSpec(
        grid=GridSpec(rows=args.rows, cols=args.cols, connectivity=4),
        rounds=0,
        training=TrainingSpec(epochs=args.epochs, lr=args.lr),
    )


def main():
    args = parse_args()
    spec = build_spec(args)
    workflow = AttentionGradientWorkflow(spec)
    _, pred = workflow.run()
    print()
    print("Predicted:")
    print(pred.view(args.rows, args.cols).numpy().round(2))


if __name__ == "__main__":
    main()