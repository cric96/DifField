#!/usr/bin/env python3
"""Check the full Space-Fluid inventory; missing jobs are failures, not omissions."""

# Imports follow repository path setup for direct script execution.
# ruff: noqa: PLC0415

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    from examples.seams.artifacts import (
        read_json,
    )
    from examples.seams.space_fluid.campaign import (
        progress,
    )
    from examples.seams.space_fluid.config import (
        RegionConfig,
    )

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    manifest = read_json(args.out / "manifest.json")
    if manifest.get("experiment") != "space-fluid":
        parser.error("Use a Space-Fluid run directory; Boids reports its own completeness")
    state = progress(args.out, RegionConfig(**manifest["config"]))
    print(f"{state['status']}: {state['evaluation']}")
    return 0 if state["status"] == "complete" else 2


if __name__ == "__main__":
    raise SystemExit(main())
