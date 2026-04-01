#!/usr/bin/env python3
"""Channel with obstacles on the original small grid scenario."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "examples"))

try:
    from .core import channel_body
    from .specs import ChannelProgramSpec, GridSpec, SmallChannelSpec
    from .workflow import SmallChannelWorkflow
except ImportError:
    from core import channel_body
    from specs import ChannelProgramSpec, GridSpec, SmallChannelSpec
    from workflow import SmallChannelWorkflow


def parse_args():
    parser = argparse.ArgumentParser(description="AC Channel with obstacles")
    parser.add_argument("--rows", type=int, default=15, help="Grid rows")
    parser.add_argument("--cols", type=int, default=15, help="Grid cols")
    parser.add_argument("--rounds", type=int, default=100, help="Number of compute rounds")
    parser.add_argument("--noise-scale", type=float, default=0.01, help="Hop cost noise")
    parser.add_argument("--tolerance", type=float, default=0.5, help="Path tolerance")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    return parser.parse_args()


def enforce_small_scale(args) -> None:
    max_rows, max_cols = 20, 20
    if args.rows > max_rows or args.cols > max_cols:
        print(
            "examples/channel/small.py is the small-scale demo. Falling back to 15x15. "
            "Use examples/channel/large.py for larger grids."
        )
        args.rows = 15
        args.cols = 15

def build_scenario(args):
    """Backward-compatibility shim; scenario creation now lives in SmallChannelWorkflow."""
    spec = SmallChannelSpec(
        grid=GridSpec(rows=args.rows, cols=args.cols),
        program=ChannelProgramSpec(rounds=args.rounds, tolerance=args.tolerance),
        noise_scale=args.noise_scale,
        seed=args.seed,
    )
    return SmallChannelWorkflow(spec)._build_scenario()


def main():
    args = parse_args()
    enforce_small_scale(args)
    spec = SmallChannelSpec(
        grid=GridSpec(rows=args.rows, cols=args.cols),
        program=ChannelProgramSpec(rounds=args.rounds, tolerance=args.tolerance),
        noise_scale=args.noise_scale,
        seed=args.seed,
    )
    SmallChannelWorkflow(spec).run()


if __name__ == "__main__":
    main()