#!/usr/bin/env python3
"""Render saved Space-Fluid results without training or changing the protocol."""

# Imports follow repository path setup for direct script execution.
# ruff: noqa: PLC0415

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    from examples.seams.__main__ import (
        main as campaign,
    )
    from examples.seams.artifacts import (
        read_json,
    )

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    manifest = read_json(args.out / "manifest.json")
    return campaign(
        [
            "space-fluid",
            "--profile",
            manifest["config"]["profile"],
            "--stage",
            "report",
            "--out",
            str(args.out),
        ]
    )


if __name__ == "__main__":
    raise SystemExit(main())
