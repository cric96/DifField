"""Subprocess orchestration for boids experiments."""

from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class LearnableRunOptions:
    """Configuration for a learnable boids run subprocess."""

    python: str
    root: Path
    epochs: int
    rounds: int
    num_nodes: int
    eval_seeds: str
    eval_every: int
    skip_viz: bool
    supervision_mode: str = "teacher"
    replay_trace_dir: str = ""
    save_replay_traces: bool = False
    device: str = ""


@dataclass(frozen=True)
class RunOutcome:
    """Outcome of a learnable boids run subprocess."""

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
    lr: float,
    seed: int | None = None,
    velocity_loss_weight: float | None = None,
) -> list[str]:
    """Construct the command line arguments for learnable.py."""
    cmd = [
        options.python,
        "examples/boids/learnable.py",
        "--epochs",
        str(options.epochs),
        "--rounds",
        str(options.rounds),
        "--num-nodes",
        str(options.num_nodes),
        "--lr",
        f"{lr:.4f}",
        "--out-dir",
        str(options.root),
        "--run-name",
        run_name,
        "--eval-seeds",
        options.eval_seeds,
        "--eval-every",
        str(options.eval_every),
        "--supervision-mode",
        options.supervision_mode,
    ]

    if options.replay_trace_dir:
        cmd.extend(["--replay-trace-dir", options.replay_trace_dir])

    if options.save_replay_traces:
        cmd.append("--save-replay-traces")

    if options.device:
        cmd.extend(["--device", options.device])

    if seed is not None:
        cmd.extend(["--seed", str(seed)])

    if velocity_loss_weight is not None:
        cmd.extend(["--velocity-loss-weight", str(velocity_loss_weight)])

    if options.skip_viz:
        cmd.extend(["--no-viz", "--no-gif"])
    return cmd


def run_learnable_subprocess(
    *,
    options: LearnableRunOptions,
    run_name: str,
    lr: float,
    seed: int | None = None,
    velocity_loss_weight: float | None = None,
) -> RunOutcome:
    """Run learnable.py as a subprocess and capture results."""
    run_dir = options.root / run_name
    cmd = build_learnable_command(
        options=options,
        run_name=run_name,
        lr=lr,
        seed=seed,
        velocity_loss_weight=velocity_loss_weight,
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
