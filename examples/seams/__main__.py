"""Run reproducible Space-Fluid or standalone Boids experiments."""

import argparse
import os
from dataclasses import replace
from pathlib import Path


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "command",
        choices=(
            "space-fluid",
            "space-fluid-clusters",
            "space-fluid-scenarios",
            "space-fluid-summary",
            "boids",
        ),
    )
    parser.add_argument("--profile", choices=("smoke", "compact-cpu", "paper-cpu"), default="smoke")
    parser.add_argument(
        "--stage", choices=("all", "train", "insights", "evaluate", "report"), default="all"
    )
    parser.add_argument("--out", type=Path)
    parser.add_argument("--source-run", type=Path, help="Optional historical Boids source")
    parser.add_argument("--budget-seconds", type=float, default=14400)
    parser.add_argument("--device-policy", choices=("cpu", "hybrid"), default="cpu")
    parser.add_argument(
        "--device", choices=("cpu", "cuda"), default="cpu", help="Space-Fluid training device"
    )
    args = parser.parse_args(argv)
    if args.device_policy != "cpu":
        parser.error("Boids and Space-Fluid campaigns use CPU")
    if args.device == "cuda":
        if args.command not in ("space-fluid", "space-fluid-clusters", "space-fluid-scenarios"):
            parser.error("--device cuda is supported by the space-fluid campaigns only")
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    if args.command == "boids":
        from .boids import campaign  # noqa: PLC0415 -- load only the requested experiment

        if args.stage != "all":
            parser.error("Boids retains its existing all-stage campaign command")
        out = args.out or Path("generated/seams") / f"boids-{args.profile}"
        return campaign(args.source_run, out, args.profile, args.budget_seconds)
    if args.command == "space-fluid-summary":
        from .space_fluid.summary import summary  # noqa: PLC0415

        return summary(args.out or Path("generated/seams/comparison"))
    if args.command == "space-fluid-scenarios":
        from .space_fluid.scenarios import campaign as scenarios  # noqa: PLC0415

        out = args.out or Path("generated/seams") / f"space-fluid-scenarios-{args.profile}"
        return scenarios(out, args.profile, args.budget_seconds, device=args.device)
    if args.command == "space-fluid-clusters":
        from .space_fluid.clusters import campaign as clusters  # noqa: PLC0415

        out = args.out or Path("generated/seams") / f"space-fluid-clusters-{args.profile}"
        return clusters(out, args.profile, args.budget_seconds, device=args.device)
    from .space_fluid.campaign import campaign  # noqa: PLC0415
    from .space_fluid.config import protocol  # noqa: PLC0415

    if args.source_run is not None:
        parser.error("Space-Fluid resumes directly from --out; --source-run is for Boids")
    out = args.out or Path("generated/seams") / f"space-fluid-{args.profile}"
    config = replace(protocol(args.profile), device=args.device)
    return campaign(out, config, args.stage, args.budget_seconds)


if __name__ == "__main__":
    raise SystemExit(main())
