#!/usr/bin/env python3
"""Channel with obstacles on the original small grid scenario."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "examples"))

from diffield.utils import get_device  # noqa: E402

try:
    from .specs import ChannelProgramSpec, GridSpec, SmallChannelSpec
    from .workflow import SmallChannelWorkflow
except ImportError:
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
    parser.add_argument("--device", type=str, default="", help="Device (cuda/cpu) [auto if empty]")
    parser.add_argument("--viz-prefix", type=str, default="generated/channel_small")
    parser.add_argument("--gif-fps", type=int, default=10)
    parser.add_argument("--no-viz", action="store_true", help="Disable visualization")
    parser.add_argument("--no-gif", action="store_true", help="Disable GIF generation")
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

def build_spec(args):
    """Backward-compatibility shim; scenario creation now lives in SmallChannelWorkflow."""
    spec = SmallChannelSpec(
        grid=GridSpec(rows=args.rows, cols=args.cols),
        program=ChannelProgramSpec(rounds=args.rounds, tolerance=args.tolerance),
        noise_scale=args.noise_scale,
        seed=args.seed,
    )
    return spec


def main():
    args = parse_args()
    enforce_small_scale(args)
    spec = build_spec(args)
    device = get_device(args.device)
    workflow = SmallChannelWorkflow(spec)
    workflow.run(
        device=device,
        gif=not args.no_gif,
        viz=not args.no_viz,
        viz_prefix=args.viz_prefix,
        gif_fps=args.gif_fps,
    )


if __name__ == "__main__":
    main()
