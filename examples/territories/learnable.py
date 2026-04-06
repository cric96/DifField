#!/usr/bin/env python3
"""Entry point for learnable multi-sink territories experiments."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "examples") not in sys.path:
    sys.path.insert(0, str(ROOT / "examples"))

import torch

from autofield.utils import get_device
from examples.shared.diagnostics import save_history_csv, save_summary_csv
from examples.shared.experiment import flatten_summary_for_csv
from examples.shared.training import parse_int_csv

try:
    from .reporting import TerritoriesSummaryBuilder, compute_parameter_recovery_metrics, extract_teacher_parameters_from_spec
    from .specs import (
        GridSpec,
        LearnableTerritoriesSpec,
        TerritoryEvaluationSpec,
        TerritoryModelSpec,
        TerritoryProgramSpec,
        TerritoryTeacherSpec,
        TerritoryTrainingSpec,
    )
    from .workflow import LearnableTerritoriesWorkflow
except ImportError:
    from reporting import TerritoriesSummaryBuilder, compute_parameter_recovery_metrics, extract_teacher_parameters_from_spec
    from specs import (
        GridSpec,
        LearnableTerritoriesSpec,
        TerritoryEvaluationSpec,
        TerritoryModelSpec,
        TerritoryProgramSpec,
        TerritoryTeacherSpec,
        TerritoryTrainingSpec,
    )
    from workflow import LearnableTerritoriesWorkflow


DEFAULTS = {
    "rows": 18,
    "cols": 18,
    "num_sinks": 4,
    "scenario_preset": "asymmetric_canyon",
    "rounds": 0,
    "epochs": 120,
    "lr": 0.03,
    "risk_loss_weight": 0.5,
    "seed": 11,
    "eval_seeds": "101,103,107",
    "eval_every": 5,
    "print_every": 10,
    "clip_grad_norm": 5.0,
    "teacher_range_weight": 1.0,
    "teacher_risk_weight": 1.8,
    "teacher_assignment_tau": 0.25,
    "init_range_weight_target": 0.55,
    "init_risk_weight_target": 0.20,
    "init_assignment_tau_target": 0.90,
    "init_surcharge_weight_target": 0.08,
    "hidden_dim": 32,
    "mode": "scalars",
    "out_dir": "generated/territories/results",
    "viz_prefix": "",
}


def parse_sink_positions(text: str) -> tuple[tuple[int, int], ...] | None:
    cleaned = text.strip()
    if not cleaned:
        return None
    positions: list[tuple[int, int]] = []
    for item in cleaned.split(","):
        row_text, sep, col_text = item.strip().partition(":")
        if not sep:
            raise ValueError(f"Invalid sink position '{item}'. Expected row:col entries separated by commas")
        positions.append((int(row_text.strip()), int(col_text.strip())))
    return tuple(positions)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Learnable multi-sink territories with final-only supervision")
    parser.add_argument("--rows", type=int, default=DEFAULTS["rows"])
    parser.add_argument("--cols", type=int, default=DEFAULTS["cols"])
    parser.add_argument("--connectivity", type=int, default=4, choices=[4, 8])
    parser.add_argument("--num-sinks", type=int, default=None)
    parser.add_argument(
        "--scenario-preset",
        type=str,
        default=DEFAULTS["scenario_preset"],
        choices=["balanced", "asymmetric_canyon"],
    )
    parser.add_argument(
        "--sink-positions",
        type=str,
        default="",
        help="Comma-separated row:col positions that override the preset, e.g. 2:2,6:3,11:14",
    )
    parser.add_argument("--rounds", type=int, default=DEFAULTS["rounds"], help="Compute rounds (0 = auto)")
    parser.add_argument("--epochs", type=int, default=DEFAULTS["epochs"])
    parser.add_argument("--lr", type=float, default=DEFAULTS["lr"])
    parser.add_argument("--risk-loss-weight", type=float, default=DEFAULTS["risk_loss_weight"])
    parser.add_argument("--seed", type=int, default=DEFAULTS["seed"], help="Training layout seed")
    parser.add_argument("--eval-seeds", type=str, default=DEFAULTS["eval_seeds"])
    parser.add_argument("--eval-every", type=int, default=DEFAULTS["eval_every"])
    parser.add_argument("--print-every", type=int, default=DEFAULTS["print_every"])
    parser.add_argument("--clip-grad-norm", type=float, default=DEFAULTS["clip_grad_norm"])
    parser.add_argument("--teacher-range-weight", type=float, default=DEFAULTS["teacher_range_weight"])
    parser.add_argument("--teacher-risk-weight", type=float, default=DEFAULTS["teacher_risk_weight"])
    parser.add_argument("--teacher-assignment-tau", type=float, default=DEFAULTS["teacher_assignment_tau"])
    parser.add_argument("--init-range-weight-target", type=float, default=DEFAULTS["init_range_weight_target"])
    parser.add_argument("--init-risk-weight-target", type=float, default=DEFAULTS["init_risk_weight_target"])
    parser.add_argument("--init-assignment-tau-target", type=float, default=DEFAULTS["init_assignment_tau_target"])
    parser.add_argument("--init-surcharge-weight-target", type=float, default=DEFAULTS["init_surcharge_weight_target"])
    parser.add_argument("--hidden-dim", type=int, default=DEFAULTS["hidden_dim"])
    parser.add_argument("--mode", type=str, default=DEFAULTS["mode"], choices=["scalars", "hybrid"])
    parser.add_argument("--out-dir", type=str, default=DEFAULTS["out_dir"])
    parser.add_argument("--run-name", type=str, default="")
    parser.add_argument("--viz-prefix", type=str, default=DEFAULTS["viz_prefix"])
    parser.add_argument("--no-viz", action="store_true")
    parser.add_argument("--device", type=str, default="")
    return parser.parse_args()


def build_spec(args: argparse.Namespace) -> LearnableTerritoriesSpec:
    sink_positions = parse_sink_positions(args.sink_positions)
    if sink_positions is not None:
        if args.num_sinks is not None and args.num_sinks != len(sink_positions):
            raise ValueError("--num-sinks must match the number of explicit --sink-positions entries")
        num_sinks = len(sink_positions)
    else:
        num_sinks = DEFAULTS["num_sinks"] if args.num_sinks is None else args.num_sinks

    return LearnableTerritoriesSpec(
        grid=GridSpec(rows=args.rows, cols=args.cols, connectivity=args.connectivity),
        program=TerritoryProgramSpec(
            rounds=args.rounds,
            num_sinks=num_sinks,
            scenario_preset=args.scenario_preset,
            sink_positions=sink_positions,
        ),
        teacher=TerritoryTeacherSpec(
            range_weight=args.teacher_range_weight,
            risk_weight=args.teacher_risk_weight,
            assignment_tau=args.teacher_assignment_tau,
        ),
        model=TerritoryModelSpec(
            mode=args.mode,
            hidden_dim=args.hidden_dim,
            init_range_weight_target=args.init_range_weight_target,
            init_risk_weight_target=args.init_risk_weight_target,
            init_assignment_tau_target=args.init_assignment_tau_target,
            init_surcharge_weight_target=args.init_surcharge_weight_target,
        ),
        training=TerritoryTrainingSpec(
            epochs=args.epochs,
            lr=args.lr,
            risk_loss_weight=args.risk_loss_weight,
            train_seed=args.seed,
            eval_every=args.eval_every,
            print_every=args.print_every,
            clip_grad_norm=args.clip_grad_norm,
        ),
        evaluation=TerritoryEvaluationSpec(eval_seeds=tuple(parse_int_csv(args.eval_seeds))),
    )


def _prepare_run(args: argparse.Namespace) -> tuple[Path, str, str]:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = args.run_name.strip() or f"territories_{args.mode}_seed{args.seed}_{timestamp}"
    run_dir = Path(args.out_dir) / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    viz_prefix = args.viz_prefix.strip() or str(run_dir / "territories")
    return run_dir, run_name, viz_prefix


def main() -> None:
    args = parse_args()
    spec = build_spec(args)
    workflow = LearnableTerritoriesWorkflow(spec)
    device = get_device(args.device)
    run_dir, run_name, viz_prefix = _prepare_run(args)
    _, history, summary, evaluation = workflow.run(device=device, viz=not args.no_viz, viz_prefix=viz_prefix)

    teacher_params = extract_teacher_parameters_from_spec(spec)
    learned_params = {
        "range_weight": float(summary["learned_range_weight"]),
        "risk_weight": float(summary["learned_risk_weight"]),
        "assignment_tau": float(summary["learned_assignment_tau"]),
    }
    recovery = compute_parameter_recovery_metrics(teacher_params, learned_params)
    nested_summary = TerritoriesSummaryBuilder(
        run_name=run_name,
        spec=spec,
        history=history,
        flat_summary=summary,
        evaluation=evaluation,
        recovery=recovery,
    ).build()

    save_history_csv(history, run_dir / "history.csv")
    save_summary_csv(flatten_summary_for_csv(nested_summary), run_dir / "summary.csv")
    with (run_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(nested_summary, handle, indent=2)
    with (run_dir / "args.json").open("w", encoding="utf-8") as handle:
        json.dump(vars(args), handle, indent=2)

    print()
    print("=== Final summary ===")
    for key, value in summary.items():
        if isinstance(value, float):
            print(f"{key}: {value:.6f}")
        else:
            print(f"{key}: {value}")


if __name__ == "__main__":
    main()