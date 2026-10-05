#!/usr/bin/env python3
"""Run the current campaigns with separate artifacts and resumable budgets."""

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", choices=("space-fluid", "boids"), default="space-fluid")
    parser.add_argument("--profile", choices=("smoke", "compact-cpu", "paper-cpu"), default="smoke")
    parser.add_argument(
        "--stage", choices=("all", "train", "insights", "evaluate", "report"), default="all"
    )
    parser.add_argument("--out", type=Path)
    parser.add_argument("--budget-seconds", type=float, default=14400)
    args = parser.parse_args()
    command = [
        sys.executable,
        "-m",
        "examples.seams",
        args.suite,
        "--profile",
        args.profile,
        "--stage",
        args.stage,
        "--budget-seconds",
        str(args.budget_seconds),
    ]
    if args.out is not None:
        command += ["--out", str(args.out)]
    return subprocess.run(command, cwd=ROOT, check=False).returncode  # noqa: S603 -- fixed module, validated CLI choices


if __name__ == "__main__":
    raise SystemExit(main())
