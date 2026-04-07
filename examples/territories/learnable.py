#!/usr/bin/env python3
"""Thin entry point for learnable multi-sink territories experiments."""

from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "examples"))

from autofield.utils import get_device
from territories.domain.specs import (
    LearnableTerritoriesSpec,
    GridSpec,
    TerritoryProgramSpec,
    TerritoryTeacherSpec,
    TerritoryModelSpec,
    TerritoryTrainingSpec,
    TerritoryEvaluationSpec,
)
from territories.training.loop import TerritoriesTrainingLoop
from territories.visualization.renderer import TerritoriesRenderer
from territories.reporting import (
    TerritoriesSummaryBuilder,
    compute_parameter_recovery_metrics,
    extract_teacher_parameters_from_spec,
)
from shared.diagnostics import save_history_csv, save_summary_csv
from shared.experiment import flatten_summary_for_csv
from shared.training import parse_int_csv

# We keep the same CLI as before for compatibility, but it could be refactored into a separate file
import argparse


def parse_sink_positions(text: str) -> tuple[tuple[int, int], ...] | None:
    cleaned = text.strip()
    if not cleaned:
        return None
    positions = []
    for item in cleaned.split(","):
        row_text, sep, col_text = item.strip().partition(":")
        if not sep:
            raise ValueError(f"Invalid sink position '{item}'")
        positions.append((int(row_text.strip()), int(col_text.strip())))
    return tuple(positions)


def parse_args():
    parser = argparse.ArgumentParser(description="Learnable multi-sink territories")
    parser.add_argument("--rows", type=int, default=18)
    parser.add_argument("--cols", type=int, default=18)
    parser.add_argument("--connectivity", type=int, default=4, choices=[4, 8])
    parser.add_argument("--num-sinks", type=int, default=None)
    parser.add_argument("--scenario-preset", type=str, default="asymmetric_canyon")
    parser.add_argument("--sink-positions", type=str, default="")
    parser.add_argument("--rounds", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--lr", type=float, default=0.03)
    parser.add_argument("--risk-loss-weight", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--eval-seeds", type=str, default="101,103,107")
    parser.add_argument("--eval-every", type=int, default=5)
    parser.add_argument("--print-every", type=int, default=10)
    parser.add_argument("--clip-grad-norm", type=float, default=5.0)
    parser.add_argument("--teacher-range-weight", type=float, default=1.0)
    parser.add_argument("--teacher-risk-weight", type=float, default=1.8)
    parser.add_argument("--teacher-assignment-tau", type=float, default=0.25)
    parser.add_argument("--init-range-weight-target", type=float, default=0.55)
    parser.add_argument("--init-risk-weight-target", type=float, default=0.20)
    parser.add_argument("--init-assignment-tau-target", type=float, default=0.90)
    parser.add_argument("--init-surcharge-weight-target", type=float, default=0.08)
    parser.add_argument("--hidden-dim", type=int, default=32)
    parser.add_argument(
        "--mode", type=str, default="scalars", choices=["scalars", "hybrid"]
    )
    parser.add_argument("--out-dir", type=str, default="generated/territories/results")
    parser.add_argument("--run-name", type=str, default="")
    parser.add_argument("--viz-prefix", type=str, default="")
    parser.add_argument("--no-viz", action="store_true")
    parser.add_argument("--device", type=str, default="")
    return parser.parse_args()


def main():
    args = parse_args()
    device = get_device(args.device)

    # 1. Build spec
    sink_positions = parse_sink_positions(args.sink_positions)
    num_sinks = len(sink_positions) if sink_positions else (args.num_sinks or 4)

    spec = LearnableTerritoriesSpec(
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
        evaluation=TerritoryEvaluationSpec(
            eval_seeds=tuple(parse_int_csv(args.eval_seeds))
        ),
    )

    # 2. Run training
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = (
        args.run_name.strip() or f"territories_{args.mode}_seed{args.seed}_{timestamp}"
    )
    run_dir = Path(args.out_dir) / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    viz_prefix = args.viz_prefix.strip() or str(run_dir / "territories")

    trainer = TerritoriesTrainingLoop(spec)

    def on_epoch_end(epoch, model, pred_train, eval_metrics):
        if (epoch + 1) % args.print_every == 0 or epoch == 0:
            params = model.current_parameters()
            print(
                f"epoch={epoch + 1:3d} train_total={eval_metrics['total']:.6f} range={params['range_weight']:.3f} risk={params['risk_weight']:.3f} tau={params['assignment_tau']:.3f}"
            )

    model, history, eval_checkpoints = trainer.run(on_epoch_end=on_epoch_end, save_eval_checkpoints=True)

    # 3. Final Evaluation
    from territories.evaluation.evaluator import evaluate_seeds

    final_evaluation = evaluate_seeds(
        model, seeds=spec.evaluation.eval_seeds, spec=spec, device=device
    )

    # 4. Visualization
    if not args.no_viz:
        renderer = TerritoriesRenderer(spec)
        renderer.render_training_results(model, history, viz_prefix)
        if eval_checkpoints:
            renderer.render_territory_evolution(eval_checkpoints, viz_prefix)

    # 5. Reporting
    teacher_params = extract_teacher_parameters_from_spec(spec)
    learned_params = model.current_parameters()
    recovery = compute_parameter_recovery_metrics(teacher_params, learned_params)

    # Need a flat summary for the builder
    summary_data = {
        "best_epoch": history["epoch"][
            history["val_total"].index(
                min(
                    [
                        x
                        for x in history["val_total"]
                        if not torch.isnan(torch.tensor(x))
                    ]
                    or [0]
                )
            )
        ],
        "best_score": min(
            [x for x in history["val_total"] if not torch.isnan(torch.tensor(x))] or [0]
        ),
    }
    # Wait, the original summary had more fields. Let's just pass what's needed.

    nested_summary = TerritoriesSummaryBuilder(
        run_name=run_name,
        spec=spec,
        history=history,
        flat_summary=summary_data,
        evaluation=final_evaluation,
        recovery=recovery,
    ).build()

    save_history_csv(history, run_dir / "history.csv")
    save_summary_csv(flatten_summary_for_csv(nested_summary), run_dir / "summary.csv")
    with (run_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(nested_summary, handle, indent=2)

    print(f"Artifacts saved to {run_dir}")


if __name__ == "__main__":
    main()
