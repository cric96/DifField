"""Reusable subprocess runner for learnable boids experiments."""

from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class LearnableRunOptions:
    python: str
    root: Path
    epochs: int
    rounds: int
    num_nodes: int
    eval_seeds: str
    eval_every: int
    skip_viz: bool


@dataclass(frozen=True)
class RunOutcome:
    run_name: str
    run_dir: Path
    returncode: int
    stdout: str
    stderr: str
    summary: dict[str, object] | None

    @property
    def ok(self) -> bool:
        return self.returncode == 0 and self.summary is not None


def build_learnable_command(
    *,
    options: LearnableRunOptions,
    run_name: str,
    mode: str,
    lr: float,
    seed: int | None = None,
    reg_scale: float | None = None,
) -> list[str]:
    cmd = [
        options.python,
        "examples/boids/learnable.py",
        "--mode",
        mode,
        "--epochs",
        str(options.epochs),
        "--rounds",
        str(options.rounds),
        "--num-nodes",
        str(options.num_nodes),
        "--lr",
        str(lr),
        "--eval-seeds",
        options.eval_seeds,
        "--eval-every",
        str(options.eval_every),
        "--print-every",
        "99999",
        "--checkpoint-every-epochs",
        str(max(10, options.epochs)),
        "--run-name",
        run_name,
        "--out-dir",
        str(options.root),
    ]

    if seed is not None:
        cmd.extend(["--seed", str(seed)])

    if reg_scale is not None:
        cmd.extend(
            [
                "--cohesion-reg",
                str(0.04 * reg_scale),
                "--alignment-reg",
                str(0.03 * reg_scale),
                "--speed-reg",
                str(0.01 * reg_scale),
                "--accel-reg",
                str(0.01 * reg_scale),
            ]
        )

    if options.skip_viz:
        cmd.extend(["--no-viz", "--no-gif", "--no-compare-panel"])
    return cmd


def run_learnable_subprocess(
    *,
    options: LearnableRunOptions,
    run_name: str,
    mode: str,
    lr: float,
    seed: int | None = None,
    reg_scale: float | None = None,
) -> RunOutcome:
    run_dir = options.root / run_name
    cmd = build_learnable_command(
        options=options,
        run_name=run_name,
        mode=mode,
        lr=lr,
        seed=seed,
        reg_scale=reg_scale,
    )
    proc = subprocess.run(cmd, capture_output=True, text=True)
    summary_path = run_dir / "summary.json"
    summary: dict[str, object] | None = None
    if proc.returncode == 0 and summary_path.exists():
        summary = json.loads(summary_path.read_text(encoding="utf-8"))

    return RunOutcome(
        run_name=run_name,
        run_dir=run_dir,
        returncode=proc.returncode,
        stdout=proc.stdout,
        stderr=proc.stderr,
        summary=summary,
    )
