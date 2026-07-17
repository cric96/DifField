#!/usr/bin/env python3
"""Head-to-head learning baseline: DIFFIELD vs. a standard PyG message-passing GNN.

This script answers the reviewer request for a baseline on the *learning* part of
the paper. Both models are trained with the **identical** teacher-generated boids
traces, teacher-forced loss, optimizer schedule, radius topology, and integration
step. The only thing that differs is the interaction function:

  * DIFFIELD (``LearnableBoids``): a field program whose neighbor aggregation is
    written by hand; learning fits 3 interpretable physical weights.
  * GNN baseline (``MessagePassingBoids``): a standard message-passing GNN that
    *learns* the interaction function (edge MLP -> permutation-invariant
    aggregation -> node MLP -> acceleration). It reuses DIFFIELD's integration
    (damping, speed clip, box bounce), so the comparison isolates exactly the
    part DIFFIELD hand-codes.

Statistics
----------
One independent training run is performed per ``--seed`` (each seed is a distinct
teacher trace = an independent draw of the learning problem). Every model is then
evaluated on the *same* held-out eval seed. Metrics are aggregated across seeds
and reported as mean with a 95% confidence interval (Student-t).

Outputs (under ``generated/boids-comparison/`` by default)
----------------------------------------------------------
  - ``comparison.json`` / ``comparison.csv`` -- metrics table (mean, ci, per-seed)
  - ``learning_curves.png``      -- train loss vs. epoch, mean +/- 95% CI band
  - ``accuracy_bars.png``        -- train/held-out position error with CI bars
  - ``rollout_error_curve.png``  -- per-round position error (compounding) +/- CI
  - ``param_efficiency.png``     -- accuracy vs. #parameters (efficiency frontier)
  - ``param_recovery.png``       -- DIFFIELD recovered weights vs. teacher (+/- CI)
  - ``diffield_rollout.gif`` / ``gnn_rollout.gif`` -- free-running rollout of the
    best model of each kind, side-by-side with the teacher reference
  - ``*_trajectories.png``       -- static trajectory comparison vs. teacher

Run:
    uv run python examples/boids-evaluation/compare_gnn.py
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "examples"))

# Reuse the existing DIFFIELD learning pipeline verbatim so the comparison is fair.
from main import (  # noqa: E402
    LearnableBoids,
    generate_or_load_trace,
    teacher_forced_step_losses,
    trajectory_mse_loss,
)
from shared.metrics import aggregate as agg  # noqa: E402
from shared.metrics import aggregate_curve as agg_curve  # noqa: E402
from shared.plotting import (  # noqa: E402
    FIG_WIDTH_1COL,
    FIG_WIDTH_2COL,
    band,
    color_of,
    export_moving_gif,
    marker_of,
    panel_label,
    plot_node_trajectories,
)
from shared.plotting import savefig as save_figure  # noqa: E402
from shared.plotting.style import INK, MUTED  # noqa: E402

from diffield.sim import (  # noqa: E402
    SpatialScenario,
    bounce_in_box,
    limit_speed,
)

try:
    from torch_geometric.nn import MessagePassing
except Exception as exc:  # pragma: no cover - import guard
    raise RuntimeError(
        "PyTorch Geometric is required for the GNN baseline. "
        "Install `torch-geometric` in the active environment."
    ) from exc

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from shared.plotting import apply_paper_style

    apply_paper_style()
except ImportError:
    plt = None


# Maps this script's model labels onto the paper-wide role palette (style.py):
# DIFFIELD is the interpretable field program, GNN is the black-box baseline.
MODEL_ROLE = {"DIFFIELD": "diffield", "GNN": "neural"}
WEIGHT_NAMES = ("w_sep", "w_align", "w_cohesion")


# ═════════════════════════════════════════════════════════════════════════════
# GNN BASELINE: standard message passing with shared physics integration
# ═════════════════════════════════════════════════════════════════════════════


class BoidsMessagePassing(MessagePassing):
    """Edge-conditioned message passing that predicts a per-node acceleration.

    Edge features are exactly the geometric quantities the boids rule reads
    (relative position, relative velocity, distance), so the GNN has access to
    the same information as DIFFIELD but must *learn* how to combine them.
    """

    def __init__(self, hidden: int = 64, aggr: str = "mean") -> None:
        super().__init__(aggr=aggr)
        self.edge_mlp = nn.Sequential(
            nn.Linear(5, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
        )
        self.node_mlp = nn.Sequential(
            nn.Linear(hidden + 2, hidden),
            nn.SiLU(),
            nn.Linear(hidden, 2),
        )

    def forward(
        self, pos: torch.Tensor, vel: torch.Tensor, edge_index: torch.Tensor
    ) -> torch.Tensor:
        agg = self.propagate(edge_index, pos=pos, vel=vel)
        return self.node_mlp(torch.cat([vel, agg], dim=-1))

    def message(
        self,
        pos_i: torch.Tensor,
        pos_j: torch.Tensor,
        vel_i: torch.Tensor,
        vel_j: torch.Tensor,
    ) -> torch.Tensor:
        rel_pos = pos_j - pos_i
        rel_vel = vel_j - vel_i
        dist = rel_pos.norm(dim=-1, keepdim=True)
        feat = torch.cat([rel_pos, rel_vel, dist], dim=-1)
        return self.edge_mlp(feat)


class MessagePassingBoids(nn.Module):
    """GNN baseline with the *same* step/rollout interface as ``LearnableBoids``.

    Drops directly into ``teacher_forced_step_losses`` and the rollout-based
    evaluation, so DIFFIELD and the GNN are trained and measured identically.
    """

    def __init__(
        self,
        *,
        radius: float,
        dt: float,
        damping: float,
        max_speed: float,
        hidden: int = 64,
        aggr: str = "mean",
    ) -> None:
        super().__init__()
        self.radius = radius
        self.dt = dt
        self.damping = damping
        self.max_speed = max_speed
        self.net = BoidsMessagePassing(hidden=hidden, aggr=aggr)

    def trainable_parameters(self) -> list[nn.Parameter]:
        return [p for p in self.parameters() if p.requires_grad]

    def step(
        self,
        positions: torch.Tensor,
        velocities: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        scenario = SpatialScenario(
            positions=positions,
            edge_radius=self.radius,
            device=positions.device,
        )
        edge_index = scenario.edge_index
        acc = self.net(scenario.positions, velocities, edge_index)

        pre_clip = self.damping * velocities + self.dt * acc
        pre_clip = torch.nan_to_num(pre_clip, nan=0.0, posinf=0.0, neginf=0.0)
        clipped = limit_speed(pre_clip, self.max_speed)
        clipped = torch.nan_to_num(clipped, nan=0.0, posinf=0.0, neginf=0.0)

        new_pos = scenario.positions + self.dt * clipped
        new_pos, vel_bounced = bounce_in_box(new_pos, clipped)
        return new_pos, vel_bounced, pre_clip

    @torch.no_grad()
    def rollout(
        self,
        rounds: int,
        *,
        positions0: torch.Tensor,
        velocities0: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, list[torch.Tensor]]:
        positions = positions0.clone()
        velocities = velocities0.clone()
        positions_seq: list[torch.Tensor] = []
        velocities_seq: list[torch.Tensor] = []
        edge_index_seq: list[torch.Tensor] = []

        for _ in range(rounds):
            scenario = SpatialScenario(
                positions=positions,
                edge_radius=self.radius,
                device=positions.device,
            )
            edge_index_seq.append(scenario.edge_index.clone())
            pred_pos, pred_vel, _ = self.step(positions, velocities)
            positions_seq.append(pred_pos)
            velocities_seq.append(pred_vel)
            positions = pred_pos
            velocities = pred_vel

        return (
            torch.stack(positions_seq, dim=0),
            torch.stack(velocities_seq, dim=0),
            edge_index_seq,
        )


# ═════════════════════════════════════════════════════════════════════════════
# TRAINING / EVALUATION (identical protocol for both models)
# ═════════════════════════════════════════════════════════════════════════════


@dataclass
class RunResult:
    """Metrics from a single training run (one seed) of one model."""

    seed: int
    loss_curve: list[float]
    train: dict[str, float]
    eval: dict[str, float]
    eval_round_err: list[float]
    learned_params: dict[str, float] | None
    train_time_s: float


@dataclass
class ModelReport:
    """All runs for one model, plus the best model kept for rendering."""

    label: str
    num_params: int
    runs: list[RunResult] = field(default_factory=list)
    best_model: nn.Module | None = None
    best_eval_err: float = math.inf


def count_trainable_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def rollout_accuracy(
    model: nn.Module,
    *,
    rounds: int,
    positions0: torch.Tensor,
    velocities0: torch.Tensor,
    teacher_pos_seq: torch.Tensor,
    teacher_vel_seq: torch.Tensor,
) -> dict:
    """Free-running (no teacher forcing) rollout accuracy vs. a teacher trace."""
    with torch.no_grad():
        pred_pos, pred_vel, _ = model.rollout(
            rounds, positions0=positions0, velocities0=velocities0
        )
    per_round_err = (teacher_pos_seq - pred_pos).norm(dim=-1).mean(dim=1)
    traj_mse, pos_mse, vel_mse = trajectory_mse_loss(
        pred_pos, pred_vel, teacher_pos_seq, teacher_vel_seq
    )
    return {
        "pos_error": float(per_round_err.mean().item()),
        "traj_mse": float(traj_mse.item()),
        "pos_mse": float(pos_mse.item()),
        "vel_mse": float(vel_mse.item()),
        "round_err": [float(v) for v in per_round_err.tolist()],
    }


def train_run(
    model: nn.Module,
    *,
    label: str,
    seed: int,
    trace,
    eval_trace,
    epochs: int,
    lr: float,
    velocity_loss_weight: float,
    sep: float,
    rounds: int,
    device: torch.device,
) -> RunResult:
    """Train one model on a single teacher trace with the shared protocol."""
    optimizer = torch.optim.Adam(model.trainable_parameters(), lr=lr)
    loss_curve: list[float] = []

    t0 = time.perf_counter()
    for _ in range(epochs):
        optimizer.zero_grad()
        total, _, _, _ = teacher_forced_step_losses(
            model,
            positions0=trace.positions0,
            velocities0=trace.velocities0,
            teacher_pos_seq=trace.pos_seq,
            teacher_vel_seq=trace.vel_seq,
            teacher_preclip_seq=trace.preclip_seq,
            velocity_loss_weight=velocity_loss_weight,
            sep=sep,
        )
        total.backward()
        torch.nn.utils.clip_grad_norm_(model.trainable_parameters(), 5.0)
        optimizer.step()
        loss_curve.append(float(total.item()))
    train_time = time.perf_counter() - t0

    train_acc = rollout_accuracy(
        model, rounds=rounds,
        positions0=trace.positions0, velocities0=trace.velocities0,
        teacher_pos_seq=trace.pos_seq, teacher_vel_seq=trace.vel_seq,
    )
    eval_acc = rollout_accuracy(
        model, rounds=rounds,
        positions0=eval_trace.positions0, velocities0=eval_trace.velocities0,
        teacher_pos_seq=eval_trace.pos_seq, teacher_vel_seq=eval_trace.vel_seq,
    )

    learned = None
    if isinstance(model, LearnableBoids):
        learned = {name: float(getattr(model, name).item()) for name in WEIGHT_NAMES}

    print(
        f"  [{label}] seed={seed:>3}  final_loss={loss_curve[-1]:.2e}  "
        f"train_err={train_acc['pos_error']:.4f}  eval_err={eval_acc['pos_error']:.4f}  "
        f"({train_time:.1f}s)"
    )
    return RunResult(
        seed=seed,
        loss_curve=loss_curve,
        train={k: v for k, v in train_acc.items() if k != "round_err"},
        eval={k: v for k, v in eval_acc.items() if k != "round_err"},
        eval_round_err=eval_acc["round_err"],
        learned_params=learned,
        train_time_s=train_time,
    )


def build_model(label: str, args: argparse.Namespace, device: torch.device) -> nn.Module:
    if label == "DIFFIELD":
        return LearnableBoids(
            radius=args.radius, sep=args.sep, dt=args.dt,
            damping=args.damping, max_speed=args.max_speed,
        ).to(device)
    return MessagePassingBoids(
        radius=args.radius, dt=args.dt, damping=args.damping,
        max_speed=args.max_speed, hidden=args.gnn_hidden, aggr=args.gnn_aggr,
    ).to(device)


# ═════════════════════════════════════════════════════════════════════════════
# PLOTTING
# ═════════════════════════════════════════════════════════════════════════════


FIGSIZE_1COL = (FIG_WIDTH_1COL, 2.6)
WEIGHT_TEX = {
    "w_sep": "$w_{\\mathrm{sep}}$",
    "w_align": "$w_{\\mathrm{align}}$",
    "w_cohesion": "$w_{\\mathrm{coh}}$",
}


def _draw_learning_curves(
    ax, reports: list[ModelReport], *, legend: bool = True, legend_loc: str = "best"
) -> None:
    for rep in reports:
        means, cis = agg_curve([r.loss_curve for r in rep.runs])
        x = list(range(1, len(means) + 1))
        band(
            ax, x, means, cis,
            role=MODEL_ROLE[rep.label],
            label=f"{rep.label} ({rep.num_params:,} params)",
        )
    ax.set_yscale("log")
    ax.set_xlabel("epoch")
    ax.set_ylabel("teacher-forced loss")
    if legend:
        ax.legend(loc=legend_loc)
    ax.grid(True, which="both", alpha=0.4)


def _draw_rollout_error(ax, reports: list[ModelReport], *, legend: bool = True) -> None:
    for rep in reports:
        means, cis = agg_curve([r.eval_round_err for r in rep.runs])
        x = list(range(1, len(means) + 1))
        band(ax, x, means, cis, role=MODEL_ROLE[rep.label], label=rep.label)
    ax.set_xlabel("rollout round")
    ax.set_ylabel("position error vs. teacher")
    if legend:
        ax.legend(loc="upper left")


def plot_learning_curves(reports: list[ModelReport], out_path: Path) -> None:
    if plt is None:
        return
    fig, ax = plt.subplots(figsize=FIGSIZE_1COL)
    _draw_learning_curves(ax, reports)
    save_figure(fig, out_path)


def plot_rollout_error_curve(reports: list[ModelReport], out_path: Path) -> None:
    if plt is None:
        return
    fig, ax = plt.subplots(figsize=FIGSIZE_1COL)
    _draw_rollout_error(ax, reports)
    save_figure(fig, out_path)


def plot_comparison_overview(reports: list[ModelReport], out_path: Path) -> None:
    """Two-panel headline: (a) training loss, (b) compounding rollout error."""
    if plt is None:
        return
    fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(FIG_WIDTH_2COL, 2.6))
    _draw_learning_curves(ax_a, reports, legend_loc="upper right")
    panel_label(ax_a, "a", loc="lower left")
    _draw_rollout_error(ax_b, reports, legend=False)
    panel_label(ax_b, "b")
    save_figure(fig, out_path)


def plot_accuracy_bars(reports: list[ModelReport], out_path: Path) -> None:
    if plt is None:
        return
    width = 0.38
    err_kw = {"elinewidth": 1.0, "capsize": 2.5}
    fig, ax = plt.subplots(figsize=FIGSIZE_1COL)
    for i, rep in enumerate(reports):
        color = color_of(MODEL_ROLE[rep.label])
        train = agg([run.train["pos_error"] for run in rep.runs])
        eval_ = agg([run.eval["pos_error"] for run in rep.runs])
        ax.bar(
            i - width / 2, train["mean"], width, yerr=train["ci"],
            color=color, alpha=0.45, error_kw=err_kw,
        )
        ax.bar(
            i + width / 2, eval_["mean"], width, yerr=eval_["ci"],
            color=color, error_kw=err_kw,
        )
    ax.set_xticks(range(len(reports)))
    ax.set_xticklabels([f"{r.label}\n{r.num_params:,} params" for r in reports])
    ax.set_ylabel("position error\n(free-running rollout)")
    handles = [
        plt.Rectangle((0, 0), 1, 1, color=MUTED, alpha=0.45, label="train trace"),
        plt.Rectangle((0, 0), 1, 1, color=MUTED, label="held-out eval"),
    ]
    ax.legend(handles=handles, loc="upper left")
    ax.grid(True, axis="y", alpha=0.5)
    ax.grid(False, axis="x")
    save_figure(fig, out_path)


def plot_param_efficiency(reports: list[ModelReport], out_path: Path) -> None:
    if plt is None:
        return
    fig, ax = plt.subplots(figsize=FIGSIZE_1COL)
    biggest = max(rep.num_params for rep in reports)
    for rep in reports:
        role = MODEL_ROLE[rep.label]
        stats = agg([run.eval["pos_error"] for run in rep.runs])
        ax.errorbar(
            rep.num_params, stats["mean"], yerr=stats["ci"],
            fmt=marker_of(role), markersize=8, capsize=3,
            color=color_of(role), label=rep.label,
        )
        on_right = rep.num_params < biggest
        ax.annotate(
            f"{rep.label}\n{rep.num_params:,} params",
            (rep.num_params, stats["mean"]),
            textcoords="offset points",
            xytext=(9, 0) if on_right else (-9, 0),
            ha="left" if on_right else "right",
            va="center", fontsize=9, color=INK,
        )
    ax.set_xscale("log")
    ax.set_xlabel("trainable parameters")
    ax.set_ylabel("held-out position error")
    ax.grid(True, which="both", alpha=0.4)
    ax.margins(x=0.4, y=0.3)
    ax.set_ylim(bottom=0)
    save_figure(fig, out_path)


def plot_param_recovery(
    report: ModelReport, teacher: dict[str, float], out_path: Path
) -> None:
    if plt is None:
        return
    learned = [run.learned_params for run in report.runs if run.learned_params]
    if not learned:
        return
    means = [agg([lp[name] for lp in learned])["mean"] for name in WEIGHT_NAMES]
    cis = [agg([lp[name] for lp in learned])["ci"] for name in WEIGHT_NAMES]
    teacher_vals = [teacher[name] for name in WEIGHT_NAMES]

    x = range(len(WEIGHT_NAMES))
    fig, ax = plt.subplots(figsize=FIGSIZE_1COL)
    ax.bar(
        x, means, 0.5, yerr=cis,
        error_kw={"elinewidth": 1.0, "capsize": 2.5},
        color=color_of("diffield"), label="learned",
    )
    ax.scatter(
        list(x), teacher_vals, color=INK, marker="_", s=520,
        linewidths=2.0, label="teacher", zorder=3,
    )
    ax.set_xticks(list(x))
    ax.set_xticklabels([WEIGHT_TEX[name] for name in WEIGHT_NAMES])
    ax.set_ylabel("weight value")
    ax.legend(loc="upper left")
    ax.grid(True, axis="y", alpha=0.5)
    ax.grid(False, axis="x")
    save_figure(fig, out_path)


def render_rollout_artifacts(
    report: ModelReport,
    *,
    eval_trace,
    rounds: int,
    out_dir: Path,
    highlight_node: int,
    fps: int,
    show_links: bool,
    links_alpha: float,
    links_width: float,
) -> None:
    """Render a GIF + static trajectory plot of the best model vs. the teacher."""
    if report.best_model is None:
        return
    with torch.no_grad():
        pred_pos, pred_vel, edge_seq = report.best_model.rollout(
            rounds,
            positions0=eval_trace.positions0,
            velocities0=eval_trace.velocities0,
        )

    tag = report.label.lower()
    export_moving_gif(
        positions_by_round={r: pred_pos[r] for r in range(rounds)},
        values_by_round={r: pred_vel[r].norm(dim=1) for r in range(rounds)},
        edge_index_by_round={r: edge_seq[r] for r in range(rounds)},
        source_idx=highlight_node,
        output_path=str(out_dir / f"{tag}_rollout.gif"),
        title=f"{report.label} free-running rollout vs. teacher",
        fps=fps,
        show_links=show_links,
        links_alpha=links_alpha,
        links_width=links_width,
        show_source=False,
        phantom_pos_seq=eval_trace.pos_seq,
    )
    plot_node_trajectories(
        positions_over_time=[pred_pos[r] for r in range(rounds)],
        source_idx=highlight_node,
        output_path=str(out_dir / f"{tag}_trajectories.png"),
        show_source=False,
        phantom_pos_seq=eval_trace.pos_seq,
    )


# ═════════════════════════════════════════════════════════════════════════════
# PERSISTENCE
# ═════════════════════════════════════════════════════════════════════════════


def build_summary(reports: list[ModelReport], args: argparse.Namespace) -> dict:
    teacher = {
        "w_sep": args.teacher_w_sep,
        "w_align": args.teacher_w_align,
        "w_cohesion": args.teacher_w_cohesion,
    }
    out: dict = {
        "config": {
            "seeds": args.seed_list,
            "eval_seed": args.eval_seed,
            "epochs": args.epochs,
            "rounds": args.rounds,
            "num_nodes": args.num_nodes,
            "radius": args.radius,
            "teacher_params": teacher,
        },
        "results": [],
    }
    for rep in reports:
        loss_mean, loss_ci = agg_curve([r.loss_curve for r in rep.runs])
        rerr_mean, rerr_ci = agg_curve([r.eval_round_err for r in rep.runs])
        entry = {
            "label": rep.label,
            "num_params": rep.num_params,
            "train_time_s": agg([r.train_time_s for r in rep.runs]),
            "train_pos_error": agg([r.train["pos_error"] for r in rep.runs]),
            "eval_pos_error": agg([r.eval["pos_error"] for r in rep.runs]),
            "eval_traj_mse": agg([r.eval["traj_mse"] for r in rep.runs]),
            # per-epoch / per-round series (mean + 95% CI) so the charts can
            # be regenerated/restyled later without retraining
            "loss_curve": {"mean": list(loss_mean), "ci": list(loss_ci)},
            "eval_round_err": {"mean": list(rerr_mean), "ci": list(rerr_ci)},
            "per_seed": [
                {
                    "seed": r.seed,
                    "train_pos_error": r.train["pos_error"],
                    "eval_pos_error": r.eval["pos_error"],
                    "eval_traj_mse": r.eval["traj_mse"],
                    "learned_params": r.learned_params,
                }
                for r in rep.runs
            ],
        }
        if any(r.learned_params for r in rep.runs):
            learned = [r.learned_params for r in rep.runs if r.learned_params]
            entry["recovered_params"] = {
                name: agg([lp[name] for lp in learned]) for name in WEIGHT_NAMES
            }
        out["results"].append(entry)
    return out


def write_csv(summary: dict, csv_path: Path) -> None:
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "model", "num_params",
            "eval_pos_error_mean", "eval_pos_error_ci",
            "train_pos_error_mean", "train_pos_error_ci",
            "eval_traj_mse_mean", "eval_traj_mse_ci",
            "train_time_s_mean",
        ])
        for r in summary["results"]:
            writer.writerow([
                r["label"], r["num_params"],
                f"{r['eval_pos_error']['mean']:.6f}", f"{r['eval_pos_error']['ci']:.6f}",
                f"{r['train_pos_error']['mean']:.6f}", f"{r['train_pos_error']['ci']:.6f}",
                f"{r['eval_traj_mse']['mean']:.6e}", f"{r['eval_traj_mse']['ci']:.6e}",
                f"{r['train_time_s']['mean']:.3f}",
            ])


# ═════════════════════════════════════════════════════════════════════════════
# MAIN
# ═════════════════════════════════════════════════════════════════════════════


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="DIFFIELD vs. GNN learning baseline")
    p.add_argument("--seeds", type=str, default="1,2,3,4,5,6,7,8")
    p.add_argument("--eval-seed", type=int, default=101)
    p.add_argument("--epochs", type=int, default=80)
    p.add_argument("--rounds", type=int, default=24)
    p.add_argument("--num-nodes", type=int, default=50)
    p.add_argument("--diffield-lr", type=float, default=0.05)
    p.add_argument("--gnn-lr", type=float, default=1e-3)
    p.add_argument("--gnn-hidden", type=int, default=64)
    p.add_argument("--gnn-aggr", type=str, default="mean", choices=["mean", "add", "max"])
    # Physics / teacher params (kept identical to main.py defaults).
    p.add_argument("--radius", type=float, default=0.23)
    p.add_argument("--sep", type=float, default=0.06)
    p.add_argument("--dt", type=float, default=1.0)
    p.add_argument("--damping", type=float, default=0.94)
    p.add_argument("--max-speed", type=float, default=0.014)
    p.add_argument("--velocity-scale", type=float, default=0.014)
    p.add_argument("--teacher-w-sep", type=float, default=0.05)
    p.add_argument("--teacher-w-align", type=float, default=0.90)
    p.add_argument("--teacher-w-cohesion", type=float, default=0.35)
    p.add_argument("--velocity-loss-weight", type=float, default=1.0)
    p.add_argument("--highlight-node", type=int, default=0)
    p.add_argument("--gif-fps", type=int, default=8)
    p.add_argument("--hide-links", action="store_true")
    p.add_argument("--links-alpha", type=float, default=0.15)
    p.add_argument("--links-width", type=float, default=0.6)
    p.add_argument("--no-render", action="store_true", help="Skip GIF/trajectory rendering")
    p.add_argument("--out-dir", type=str, default="generated/boids-comparison")
    p.add_argument("--device", type=str, default="")
    return p.parse_args()


def _device(s: str) -> torch.device:
    if s:
        return torch.device(s)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def parse_int_csv(text: str) -> list[int]:
    return [int(t.strip()) for t in text.split(",") if t.strip()]


def lr_for(label: str, args: argparse.Namespace) -> float:
    return args.diffield_lr if label == "DIFFIELD" else args.gnn_lr


def main() -> None:
    args = parse_args()
    device = _device(args.device)
    seeds = parse_int_csv(args.seeds)
    args.seed_list = seeds

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    trace_dir = out_dir / "replay-traces"

    print("=== DIFFIELD vs. GNN baseline (identical teacher traces) ===")
    print(f"  device={device}  train_seeds={seeds}  eval_seed={args.eval_seed}")
    print(f"  epochs={args.epochs}  rounds={args.rounds}  nodes={args.num_nodes}")

    common_trace_kwargs = {
        "num_nodes": args.num_nodes,
        "rounds": args.rounds,
        "radius": args.radius,
        "sep": args.sep,
        "dt": args.dt,
        "damping": args.damping,
        "max_speed": args.max_speed,
        "w_sep": args.teacher_w_sep,
        "w_align": args.teacher_w_align,
        "w_cohesion": args.teacher_w_cohesion,
        "velocity_scale": args.velocity_scale,
        "trace_dir": trace_dir,
        "device": device,
    }
    eval_trace = generate_or_load_trace(seed=args.eval_seed, **common_trace_kwargs)

    reports = {label: ModelReport(label=label, num_params=0) for label in ("DIFFIELD", "GNN")}

    for seed in seeds:
        trace = generate_or_load_trace(seed=seed, **common_trace_kwargs)
        for label in ("DIFFIELD", "GNN"):
            torch.manual_seed(seed)
            model = build_model(label, args, device)
            reports[label].num_params = count_trainable_params(model)
            run = train_run(
                model, label=label, seed=seed, trace=trace, eval_trace=eval_trace,
                epochs=args.epochs, lr=lr_for(label, args),
                velocity_loss_weight=args.velocity_loss_weight, sep=args.sep,
                rounds=args.rounds, device=device,
            )
            reports[label].runs.append(run)
            if run.eval["pos_error"] < reports[label].best_eval_err:
                reports[label].best_eval_err = run.eval["pos_error"]
                reports[label].best_model = model

    report_list = [reports["DIFFIELD"], reports["GNN"]]

    # --- Persist metrics ---
    summary = build_summary(report_list, args)
    with (out_dir / "comparison.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    write_csv(summary, out_dir / "comparison.csv")

    # --- Plots ---
    plot_learning_curves(report_list, out_dir / "learning_curves.png")
    plot_accuracy_bars(report_list, out_dir / "accuracy_bars.png")
    plot_rollout_error_curve(report_list, out_dir / "rollout_error_curve.png")
    plot_comparison_overview(report_list, out_dir / "comparison_overview.png")
    plot_param_efficiency(report_list, out_dir / "param_efficiency.png")
    teacher = summary["config"]["teacher_params"]
    plot_param_recovery(reports["DIFFIELD"], teacher, out_dir / "param_recovery.png")

    # --- GIFs + trajectory renders ---
    if not args.no_render:
        for rep in report_list:
            render_rollout_artifacts(
                rep, eval_trace=eval_trace, rounds=args.rounds, out_dir=out_dir,
                highlight_node=args.highlight_node, fps=args.gif_fps,
                show_links=not args.hide_links,
                links_alpha=args.links_alpha, links_width=args.links_width,
            )

    # --- Console summary ---
    print("\n=== Summary (mean +/- 95% CI over seeds) ===")
    header = f"{'model':>10} | {'#params':>8} | {'eval pos_err':>20} | {'eval traj_mse':>22}"
    print(header)
    print("-" * len(header))
    for r in summary["results"]:
        pe, tm = r["eval_pos_error"], r["eval_traj_mse"]
        print(
            f"{r['label']:>10} | {r['num_params']:>8,} | "
            f"{pe['mean']:>11.5f} +/- {pe['ci']:<6.5f} | "
            f"{tm['mean']:>12.4e} +/- {tm['ci']:<8.2e}"
        )
    if "recovered_params" in summary["results"][0]:
        rp = summary["results"][0]["recovered_params"]
        print("\nDIFFIELD recovered weights (teacher in parens):")
        for name in WEIGHT_NAMES:
            m, c, tv = rp[name]["mean"], rp[name]["ci"], teacher[name]
            print(f"  {name:>12} = {m:.4f} +/- {c:.4f}  ({tv})")
    print(f"\nArtifacts saved to {out_dir}")


if __name__ == "__main__":
    main()
