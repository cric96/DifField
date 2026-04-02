#!/usr/bin/env python3
"""Large-scale channel example with maze-like obstacles."""

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
    from .specs import ChannelProgramSpec, GridSpec, LargeChannelSpec
    from .workflow import LargeChannelWorkflow
except ImportError:
    from specs import ChannelProgramSpec, GridSpec, LargeChannelSpec
    from workflow import LargeChannelWorkflow


def parse_args():
    parser = argparse.ArgumentParser(description="Large-scale AC Channel")
    parser.add_argument("--rows", type=int, default=50, help="Grid rows")
    parser.add_argument("--cols", type=int, default=80, help="Grid cols")
    parser.add_argument("--rounds", type=int, default=300, help="Number of compute rounds")
    parser.add_argument("--tolerance", type=float, default=0.5, help="Path tolerance")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--device", type=str, default="", help="Device (cuda/cpu) [auto if empty]")
    parser.add_argument("--viz-prefix", type=str, default="examples/channel_large")
    parser.add_argument("--gif-fps", type=int, default=10)
    parser.add_argument("--no-viz", action="store_true", help="Disable visualization")
    parser.add_argument("--no-gif", action="store_true", help="Disable GIF generation")
    return parser.parse_args()


def build_spec(args):
    return LargeChannelSpec(
        grid=GridSpec(rows=args.rows, cols=args.cols),
        program=ChannelProgramSpec(rounds=args.rounds, tolerance=args.tolerance),
    )


def main():
    args = parse_args()
    device = get_device(args.device)
    spec = build_spec(args)
    workflow = LargeChannelWorkflow(spec)
    workflow.run(
        device=device,
        gif=not args.no_gif,
        viz=not args.no_viz,
        viz_prefix=args.viz_prefix,
        gif_fps=args.gif_fps,
    )


if __name__ == "__main__":
    main()