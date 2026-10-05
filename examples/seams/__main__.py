"""Run reproducible Space-Fluid or standalone Boids experiments."""

import argparse
from pathlib import Path


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "command",
        choices=(
            "space-fluid",
            "space-fluid-clusters",
            "space-fluid-hotspot",
            "space-fluid-scenarios",
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
    args = parser.parse_args(argv)
    if args.device_policy != "cpu":
        parser.error("Boids and Space-Fluid campaigns use CPU")
    if args.command == "boids":
        from .boids import campaign  # noqa: PLC0415 -- load only the requested experiment

        if args.stage != "all":
            parser.error("Boids retains its existing all-stage campaign command")
        out = args.out or Path("generated/seams") / f"boids-{args.profile}"
        return campaign(args.source_run, out, args.profile, args.budget_seconds)
    if args.command == "space-fluid-scenarios":
        from .space_fluid.scenarios import campaign as scenarios  # noqa: PLC0415

        out = args.out or Path("generated/seams") / f"space-fluid-scenarios-{args.profile}"
        return scenarios(out, args.profile, args.budget_seconds)
    if args.command == "space-fluid-hotspot":
        from .space_fluid.hotspot import campaign as hotspot  # noqa: PLC0415

        out = args.out or Path("generated/seams") / f"space-fluid-hotspot-{args.profile}"
        return hotspot(out, args.profile, args.budget_seconds)
    if args.command == "space-fluid-clusters":
        from .space_fluid.clusters import campaign as clusters  # noqa: PLC0415

        out = args.out or Path("generated/seams") / f"space-fluid-clusters-{args.profile}"
        return clusters(out, args.profile, args.budget_seconds)
    from .space_fluid.campaign import campaign  # noqa: PLC0415
    from .space_fluid.config import protocol  # noqa: PLC0415

    if args.source_run is not None:
        parser.error("Space-Fluid resumes directly from --out; --source-run is for Boids")
    out = args.out or Path("generated/seams") / f"space-fluid-{args.profile}"
    return campaign(out, protocol(args.profile), args.stage, args.budget_seconds)


if __name__ == "__main__":
    raise SystemExit(main())
