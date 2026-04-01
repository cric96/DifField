#!/usr/bin/env python3
"""Large-scale channel example with maze-like obstacles."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "examples"))

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
    return parser.parse_args()


def main():
    args = parse_args()
    spec = LargeChannelSpec(
        grid=GridSpec(rows=args.rows, cols=args.cols),
        program=ChannelProgramSpec(rounds=args.rounds, tolerance=args.tolerance),
    )
    LargeChannelWorkflow(spec).run()


if __name__ == "__main__":
    main()