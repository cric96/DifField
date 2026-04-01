#!/usr/bin/env python3
"""Learnable moving-node DSL gradient example."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

try:
    from .specs import MovingGradientSpec
    from .workflow import MovingGradientWorkflow
except ImportError:
    from specs import MovingGradientSpec
    from workflow import MovingGradientWorkflow


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
    spec = build_spec(args)
    workflow = MovingGradientWorkflow(spec)
    workflow.run()


if __name__ == "__main__":
    main()