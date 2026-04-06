"""Workflow orchestration for learnable territories experiments."""

from __future__ import annotations

from pathlib import Path

import torch

from autofield import GridScenario, SnapshotRecorder
from examples.shared.diagnostics import save_history_csv, save_summary_csv
from examples.shared.history import MetricHistory
from examples.shared.plotting import to_grid
from examples.shared.training import grad_norm

try:
    import matplotlib.pyplot as plt
    from matplotlib import colors as mcolors
except ImportError:
    plt = None
    mcolors = None

try:
    from .core import TerritoryLayout, TerritoryOutputs, auto_rounds, decode_territory_output
    from .evaluation_utils import (
        build_layout_from_spec,
        build_scenario_from_spec,
        build_teacher_model_from_spec,
        evaluate_seeds,
        rollout_summary,
        summary_metrics,
    )
    from .models import LearnableTerritoryModel
    from .specs import LearnableTerritoriesSpec
except ImportError:
    from core import TerritoryLayout, TerritoryOutputs, auto_rounds, decode_territory_output
    from evaluation_utils import (
        build_layout_from_spec,
        build_scenario_from_spec,
        build_teacher_model_from_spec,
        evaluate_seeds,
        rollout_summary,
        summary_metrics,
    )
    from models import LearnableTerritoryModel
    from specs import LearnableTerritoriesSpec


class LearnableTerritoriesWorkflow:
    def __init__(self, spec: LearnableTerritoriesSpec):
        self.spec = spec

    def run(
        self,
        *,
        device: torch.device | None = None,
        viz: bool = True,
        viz_prefix: str = "generated/territories/learnable",
    ) -> tuple[
        LearnableTerritoryModel,
        dict[str, list[float]],
        dict[str, float | int | str],
        dict[str, object],
    ]:
        scenario = self._build_scenario(device=device)
        train_layout = self._build_layout(scenario, seed=self.spec.training.train_seed)
        teacher = self._build_teacher_model().to(scenario.device)
        model = LearnableTerritoryModel.from_spec(self.spec.model).to(scenario.device)
        optimizer = torch.optim.Adam(model.trainable_parameters(), lr=self.spec.training.lr)
        history = MetricHistory.from_keys(
            [
                "epoch",
                "total",
                "load_loss",
                "risk_loss",
                "owner_agreement",
                "conservation_error",
                "grad_norm",
                "range_weight",
                "risk_weight",
                "assignment_tau",
                "surcharge_weight",
                "val_total",
                "val_load_loss",
                "val_risk_loss",
                "val_owner_agreement",
                "val_conservation_error",
            ]
        )

        print("=== Learnable Multi-Sink Territories ===")
        print(
            f"Grid: {self.spec.grid.rows}x{self.spec.grid.cols}  "
            f"Sinks: {train_layout.num_sinks}  Rounds: {self._rounds()}  Scenario: {train_layout.scenario_preset}"
        )
        print(f"Mode: {self.spec.model.mode}  Train layout seed: {self.spec.training.train_seed}")
        print("Aggregate core:")
        print("  pot = gradient(is_sink, weight = range_weight * nbr_range() + risk_weight * risk + surcharge)")
        print("  owner_xy = gradient_cast(is_sink, sink_xy, id, weight = same_cost)")
        print("  load = collect_cast(pot, demand, 0, add)")
        print("Loss: final-round sink load/risk summaries only")
        print()

        with torch.no_grad():
            target_train = self._rollout_summary(scenario, train_layout, teacher)
        validation_layout = None
        validation_target = None
        validation_seed = None
        validation_checkpoints: list[tuple[int, TerritoryOutputs]] = []
        if viz:
            validation_seed = self._validation_viz_seed()
            validation_layout = self._build_layout(scenario, seed=validation_seed)
            with torch.no_grad():
                validation_target = self._rollout_summary(scenario, validation_layout, teacher)

        best_score = float("inf")
        best_epoch = 0
        best_state = {key: value.detach().clone() for key, value in model.state_dict().items()}

        for epoch in range(self.spec.training.epochs):
            optimizer.zero_grad()
            pred_train = self._rollout_summary(scenario, train_layout, model)
            train_metrics = self._summary_metrics(pred_train, target_train)
            train_metrics["total"].backward()
            trainable_params = model.trainable_parameters()
            epoch_grad_norm = grad_norm(trainable_params)
            torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=self.spec.training.clip_grad_norm)
            optimizer.step()

            if (epoch + 1) % self.spec.training.eval_every == 0 or epoch + 1 == self.spec.training.epochs:
                eval_report = self._evaluate(model, device=scenario.device)
                eval_metrics = eval_report["mean"]
                if validation_layout is not None:
                    with torch.no_grad():
                        validation_checkpoints.append(
                            (
                                epoch + 1,
                                self._rollout_summary(scenario, validation_layout, model),
                            )
                        )
            else:
                eval_metrics = {
                    "total": float("nan"),
                    "load_loss": float("nan"),
                    "risk_loss": float("nan"),
                    "owner_agreement": float("nan"),
                    "conservation_error": float("nan"),
                }

            score = float(eval_metrics["total"])
            if not torch.isfinite(torch.tensor(score)):
                score = float(train_metrics["total"].detach().item())
            if score < best_score:
                best_score = score
                best_epoch = epoch + 1
                best_state = {key: value.detach().clone() for key, value in model.state_dict().items()}

            params = model.current_parameters()
            history.append(
                epoch=float(epoch + 1),
                total=float(train_metrics["total"].detach().item()),
                load_loss=float(train_metrics["load_loss"].detach().item()),
                risk_loss=float(train_metrics["risk_loss"].detach().item()),
                owner_agreement=float(train_metrics["owner_agreement"].detach().item()),
                conservation_error=float(train_metrics["conservation_error"].detach().item()),
                grad_norm=epoch_grad_norm,
                range_weight=params["range_weight"],
                risk_weight=params["risk_weight"],
                assignment_tau=params["assignment_tau"],
                surcharge_weight=params["surcharge_weight"],
                val_total=float(eval_metrics["total"]),
                val_load_loss=float(eval_metrics["load_loss"]),
                val_risk_loss=float(eval_metrics["risk_loss"]),
                val_owner_agreement=float(eval_metrics["owner_agreement"]),
                val_conservation_error=float(eval_metrics["conservation_error"]),
            )

            if epoch == 0 or (epoch + 1) % self.spec.training.print_every == 0 or epoch + 1 == self.spec.training.epochs:
                print(
                    f"epoch={epoch + 1:3d} "
                    f"train_total={float(train_metrics['total'].detach().item()):.6f} "
                    f"train_owner={float(train_metrics['owner_agreement'].detach().item()):.3f} "
                    f"range={params['range_weight']:.3f} "
                    f"risk={params['risk_weight']:.3f} "
                    f"tau={params['assignment_tau']:.3f} "
                    f"val_total={float(eval_metrics['total']):.6f}"
                )

        model.load_state_dict(best_state)
        history_dict = history.to_dict()
        final_train = self._rollout_summary(scenario, train_layout, model)
        final_train_metrics = self._summary_metrics(final_train, target_train)
        final_evaluation = self._evaluate(model, device=scenario.device)
        final_eval_mean = final_evaluation["mean"]
        final_eval_std = final_evaluation["std"]
        final_eval_seed = (
            int(final_evaluation["per_seed"][0]["seed"])
            if final_evaluation["per_seed"]
            else self.spec.training.train_seed
        )
        final_eval_layout = self._build_layout(scenario, seed=final_eval_seed)
        with torch.no_grad():
            final_eval_target = self._rollout_summary(scenario, final_eval_layout, teacher)
            final_eval_pred = self._rollout_summary(scenario, final_eval_layout, model)
        final_eval_metrics = self._summary_metrics(final_eval_pred, final_eval_target)

        summary = {
            "train_seed": self.spec.training.train_seed,
            "eval_seed": final_eval_seed,
            "num_eval_seeds": len(self.spec.evaluation.eval_seeds),
            "mode": self.spec.model.mode,
            "scenario_preset": train_layout.scenario_preset,
            "num_sinks": train_layout.num_sinks,
            "teacher_range_weight": self.spec.teacher.range_weight,
            "teacher_risk_weight": self.spec.teacher.risk_weight,
            "teacher_assignment_tau": self.spec.teacher.assignment_tau,
            "learned_range_weight": model.current_parameters()["range_weight"],
            "learned_risk_weight": model.current_parameters()["risk_weight"],
            "learned_assignment_tau": model.current_parameters()["assignment_tau"],
            "learned_surcharge_weight": model.current_parameters()["surcharge_weight"],
            "best_epoch": best_epoch,
            "best_score": best_score,
            "train_total": float(final_train_metrics["total"].detach().item()),
            "train_load_loss": float(final_train_metrics["load_loss"].detach().item()),
            "train_risk_loss": float(final_train_metrics["risk_loss"].detach().item()),
            "train_owner_agreement": float(final_train_metrics["owner_agreement"].detach().item()),
            "train_conservation_error": float(final_train_metrics["conservation_error"].detach().item()),
            "eval_total": float(final_eval_mean["total"]),
            "eval_total_std": float(final_eval_std["total"]),
            "eval_load_loss": float(final_eval_mean["load_loss"]),
            "eval_risk_loss": float(final_eval_mean["risk_loss"]),
            "eval_owner_agreement": float(final_eval_mean["owner_agreement"]),
            "eval_owner_agreement_std": float(final_eval_std["owner_agreement"]),
            "eval_conservation_error": float(final_eval_mean["conservation_error"]),
            "eval_conservation_error_std": float(final_eval_std["conservation_error"]),
            "viz_eval_total": float(final_eval_metrics["total"].detach().item()),
            "viz_eval_owner_agreement": float(final_eval_metrics["owner_agreement"].detach().item()),
            "viz_eval_conservation_error": float(final_eval_metrics["conservation_error"].detach().item()),
        }

        save_history_csv(history_dict, f"{viz_prefix}_history.csv")
        save_summary_csv(summary, f"{viz_prefix}_summary.csv")
        if viz:
            self._render_loss_curves(
                history_dict,
                output_path=f"{viz_prefix}_loss.png",
                title="Territories training and validation losses",
            )
            self._render_summary_panel(
                layout=train_layout,
                teacher_output=target_train,
                learned_output=final_train,
                output_path=f"{viz_prefix}_train.png",
                title="Training layout",
            )
            self._render_summary_panel(
                layout=final_eval_layout,
                teacher_output=final_eval_target,
                learned_output=final_eval_pred,
                output_path=f"{viz_prefix}_eval_seed{final_eval_seed}.png",
                title=f"Held-out layout (seed={final_eval_seed})",
            )
            if validation_layout is not None and validation_target is not None and validation_seed is not None:
                self._render_validation_progress_panel(
                    layout=validation_layout,
                    teacher_output=validation_target,
                    checkpoints=validation_checkpoints,
                    output_path=f"{viz_prefix}_validation_progress.png",
                    title=f"Validation progression across checkpoints (seed={validation_seed})",
                )
        return model, history_dict, summary, final_evaluation

    def _rounds(self) -> int:
        return auto_rounds(self.spec.grid.rows, self.spec.grid.cols, self.spec.program.rounds)

    def _build_scenario(self, device: torch.device | None = None) -> GridScenario:
        return build_scenario_from_spec(self.spec, device=device)

    def _build_layout(self, scenario: GridScenario, *, seed: int) -> TerritoryLayout:
        return build_layout_from_spec(self.spec, scenario, seed=seed)

    def _build_teacher_model(self) -> LearnableTerritoryModel:
        return build_teacher_model_from_spec(self.spec)

    def _validation_viz_seed(self) -> int:
        if self.spec.evaluation.eval_seeds:
            return int(self.spec.evaluation.eval_seeds[0])
        return self.spec.training.train_seed

    def _rollout_summary(
        self,
        scenario: GridScenario,
        layout: TerritoryLayout,
        model: LearnableTerritoryModel,
    ) -> TerritoryOutputs:
        return rollout_summary(
            self.spec,
            scenario,
            layout,
            model,
            name_prefix=f"territories_{self.spec.model.mode}",
        )

    def _summary_metrics(
        self,
        pred: TerritoryOutputs,
        target: TerritoryOutputs,
    ) -> dict[str, torch.Tensor]:
        return summary_metrics(pred, target, risk_loss_weight=self.spec.training.risk_loss_weight)

    @torch.no_grad()
    def _evaluate(self, model: LearnableTerritoryModel, *, device: torch.device) -> dict[str, object]:
        return evaluate_seeds(
            model,
            seeds=self.spec.evaluation.eval_seeds,
            spec=self.spec,
            device=device,
        )

    def _snapshot_rounds(self, rounds: int, max_snapshots: int = 5) -> list[int]:
        if rounds <= 1:
            return [0]
        anchors = {0, rounds - 1}
        if max_snapshots >= 3:
            anchors.add(int(round((rounds - 1) * 0.25)))
        if max_snapshots >= 4:
            anchors.add(int(round((rounds - 1) * 0.50)))
        if max_snapshots >= 5:
            anchors.add(int(round((rounds - 1) * 0.75)))
        return sorted(max(0, min(rounds - 1, step)) for step in anchors)

    def _record_evolution_snapshots(
        self,
        scenario: GridScenario,
        layout: TerritoryLayout,
        model: LearnableTerritoryModel,
        *,
        name_prefix: str,
    ) -> dict[int, TerritoryOutputs]:
        rounds = self._rounds()
        recorder = SnapshotRecorder(record_rounds=set(self._snapshot_rounds(rounds)))
        rollout_summary(
            self.spec,
            scenario,
            layout,
            model,
            name_prefix=name_prefix,
            recorder=recorder,
        )
        return {
            round_idx: decode_territory_output(
                payload["output"],
                layout=layout,
            )
            for round_idx, payload in sorted(recorder.records.items())
            if "output" in payload
        }

    def _owner_style(
        self,
        num_sinks: int,
    ) -> tuple[object, object, list[int], list[str]]:
        if plt is None or mcolors is None:
            raise RuntimeError("matplotlib is required for owner styling")
        cmap_name = "tab20" if num_sinks <= 20 else "gist_ncar"
        sink_cmap = plt.get_cmap(cmap_name, max(num_sinks, 1))
        owner_colors = ["#6C7A89"] + [mcolors.to_hex(sink_cmap(index)) for index in range(num_sinks)]
        owner_cmap = mcolors.ListedColormap(owner_colors)
        owner_norm = mcolors.BoundaryNorm(
            boundaries=[-1.5] + [index - 0.5 for index in range(num_sinks)] + [num_sinks - 0.5],
            ncolors=owner_cmap.N,
        )
        ticks = list(range(-1, num_sinks))
        labels = ["unassigned"] + [f"S{index}" for index in range(num_sinks)]
        return owner_cmap, owner_norm, ticks, labels

    def _select_validation_checkpoints(
        self,
        checkpoints: list[tuple[int, TerritoryOutputs]],
        *,
        max_frames: int = 6,
    ) -> list[tuple[int, TerritoryOutputs]]:
        if len(checkpoints) <= max_frames:
            return checkpoints
        indices = torch.linspace(0, len(checkpoints) - 1, steps=max_frames)
        chosen = sorted({int(round(float(index))) for index in indices.tolist()})
        return [checkpoints[index] for index in chosen]

    def _finite_series(self, xs: list[float], ys: list[float]) -> tuple[list[float], list[float]]:
        plot_x: list[float] = []
        plot_y: list[float] = []
        for x_value, y_value in zip(xs, ys):
            if torch.isfinite(torch.tensor(y_value)):
                plot_x.append(float(x_value))
                plot_y.append(float(y_value))
        return plot_x, plot_y

    def _render_loss_curves(self, history: dict[str, list[float]], *, output_path: str, title: str) -> None:
        if plt is None:
            print("matplotlib not available; skipping territories loss curves")
            return

        epochs = history["epoch"]
        fig, axes = plt.subplots(2, 1, figsize=(10.0, 7.4), sharex=True)

        axes[0].plot(epochs, history["total"], label="train total", linewidth=2.0, color="#1F4E5F")
        val_x, val_y = self._finite_series(epochs, history["val_total"])
        if val_x:
            axes[0].plot(val_x, val_y, label="eval total", linewidth=1.8, linestyle=":", color="#C06C84")
        axes[0].set_ylabel("loss")
        axes[0].set_title("Final-summary objective")
        axes[0].grid(alpha=0.25)
        axes[0].legend(loc="best")

        component_specs = [
            ("load_loss", "train load", "#355C7D", "-"),
            ("risk_loss", "train risk", "#6C5B7B", "-"),
            ("val_load_loss", "eval load", "#355C7D", "--"),
            ("val_risk_loss", "eval risk", "#6C5B7B", "--"),
        ]
        for key, label, color, linestyle in component_specs:
            curve_x, curve_y = self._finite_series(epochs, history[key])
            if curve_x:
                axes[1].plot(curve_x, curve_y, label=label, color=color, linestyle=linestyle, linewidth=1.8)
        axes[1].set_xlabel("epoch")
        axes[1].set_ylabel("component loss")
        axes[1].set_title("Load and risk components")
        axes[1].grid(alpha=0.25)
        axes[1].legend(loc="best")

        fig.suptitle(title)
        fig.tight_layout()
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=160)
        plt.close(fig)

    def _render_validation_progress_panel(
        self,
        *,
        layout: TerritoryLayout,
        teacher_output: TerritoryOutputs,
        checkpoints: list[tuple[int, TerritoryOutputs]],
        output_path: str,
        title: str,
    ) -> None:
        if plt is None or mcolors is None:
            print("matplotlib not available; skipping territories validation progression panel")
            return
        selected = self._select_validation_checkpoints(checkpoints)
        if not selected:
            print("no validation checkpoints available; skipping territories validation progression panel")
            return

        rows = self.spec.grid.rows
        cols = self.spec.grid.cols
        owner_cmap, owner_norm, owner_ticks, owner_labels = self._owner_style(layout.num_sinks)
        agreement_cmap = mcolors.ListedColormap(["#B23A48", "#DCE8C8"])
        agreement_norm = mcolors.BoundaryNorm(boundaries=[-0.5, 0.5, 1.5], ncolors=agreement_cmap.N)
        fig, axes = plt.subplots(2, len(selected), figsize=(3.4 * len(selected), 6.1), constrained_layout=True)
        if len(selected) == 1:
            axes = axes.reshape(2, 1)

        owner_image = None
        agreement_image = None
        for column, (epoch, predicted_output) in enumerate(selected):
            metrics = self._summary_metrics(predicted_output, teacher_output)
            owner_grid = to_grid(predicted_output.hard_owner.float(), rows, cols, replace_inf=False)
            agreement_grid = to_grid(
                (predicted_output.hard_owner == teacher_output.hard_owner).float(),
                rows,
                cols,
                replace_inf=False,
            )

            owner_axis = axes[0, column]
            agreement_axis = axes[1, column]
            owner_image = owner_axis.imshow(owner_grid, cmap=owner_cmap, norm=owner_norm, interpolation="nearest")
            agreement_image = agreement_axis.imshow(
                agreement_grid,
                cmap=agreement_cmap,
                norm=agreement_norm,
                interpolation="nearest",
            )
            self._mark_sinks(owner_axis, layout.sink_positions)
            self._mark_sinks(agreement_axis, layout.sink_positions)
            owner_axis.set_title(
                f"epoch = {epoch}\nval = {float(metrics['total'].detach().item()):.3f}",
            )
            owner_axis.set_xticks([])
            owner_axis.set_yticks([])
            agreement_axis.set_xticks([])
            agreement_axis.set_yticks([])
            if column == 0:
                owner_axis.set_ylabel("pred owner")
                agreement_axis.set_ylabel("owner match")

        if owner_image is not None:
            owner_colorbar = fig.colorbar(
                owner_image,
                ax=axes[0, :],
                location="right",
                shrink=0.86,
                pad=0.02,
                ticks=owner_ticks,
            )
            owner_colorbar.set_ticklabels(owner_labels)
            owner_colorbar.set_label("predicted sink")
        if agreement_image is not None:
            agreement_colorbar = fig.colorbar(
                agreement_image,
                ax=axes[1, :],
                location="right",
                shrink=0.86,
                pad=0.02,
                ticks=[0.0, 1.0],
            )
            agreement_colorbar.set_ticklabels(["mismatch", "match"])
            agreement_colorbar.set_label("vs teacher")

        fig.suptitle(title)
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=160)
        plt.close(fig)

    def _render_evolution_panel(
        self,
        *,
        layout: TerritoryLayout,
        snapshots: dict[int, TerritoryOutputs],
        output_path: str,
        title: str,
    ) -> None:
        if plt is None:
            print("matplotlib not available; skipping territories evolution panel")
            return
        if not snapshots:
            print("no territories snapshots available; skipping evolution panel")
            return

        rows = self.spec.grid.rows
        cols = self.spec.grid.cols
        steps = sorted(snapshots.keys())
        fig, axes = plt.subplots(2, len(steps), figsize=(3.2 * len(steps), 6.0), constrained_layout=True)
        if len(steps) == 1:
            axes = axes.reshape(2, 1)

        owner_cmap, owner_norm, owner_ticks, owner_labels = self._owner_style(layout.num_sinks)
        owner_image = None
        potential_image = None
        for column, step in enumerate(steps):
            snapshot = snapshots[step]
            owner_grid = to_grid(snapshot.hard_owner.float(), rows, cols, replace_inf=False)
            potential_grid = to_grid(snapshot.potential, rows, cols, replace_inf=False)

            owner_axis = axes[0, column]
            potential_axis = axes[1, column]

            owner_image = owner_axis.imshow(owner_grid, cmap=owner_cmap, norm=owner_norm, interpolation="nearest")
            potential_image = potential_axis.imshow(potential_grid, cmap="viridis", interpolation="nearest")

            self._mark_sinks(owner_axis, layout.sink_positions)
            self._mark_sinks(potential_axis, layout.sink_positions)
            owner_axis.set_title(f"t = {step + 1}")
            owner_axis.set_xticks([])
            owner_axis.set_yticks([])
            potential_axis.set_xticks([])
            potential_axis.set_yticks([])
            if column == 0:
                owner_axis.set_ylabel("hard owner")
                potential_axis.set_ylabel("potential")

        if owner_image is not None:
            owner_colorbar = fig.colorbar(
                owner_image,
                ax=axes[0, :],
                location="right",
                shrink=0.86,
                pad=0.02,
                ticks=owner_ticks,
            )
            owner_colorbar.set_ticklabels(owner_labels)
            owner_colorbar.set_label("hard owner")
        if potential_image is not None:
            potential_colorbar = fig.colorbar(
                potential_image,
                ax=axes[1, :],
                location="right",
                shrink=0.86,
                pad=0.02,
            )
            potential_colorbar.set_label("distance to nearest sink")

        fig.suptitle(title)
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=160)
        plt.close(fig)

    def _render_summary_panel(
        self,
        *,
        layout: TerritoryLayout,
        teacher_output: TerritoryOutputs,
        learned_output: TerritoryOutputs,
        output_path: str,
        title: str,
    ) -> None:
        if plt is None:
            print("matplotlib not available; skipping territory summary panel")
            return

        rows = self.spec.grid.rows
        cols = self.spec.grid.cols
        fig, axes = plt.subplots(2, 3, figsize=(15.0, 8.8))
        owner_cmap, owner_norm, owner_ticks, owner_labels = self._owner_style(layout.num_sinks)

        demand_grid = to_grid(layout.demand, rows, cols, replace_inf=False)
        risk_grid = to_grid(layout.risk, rows, cols, replace_inf=False)
        teacher_owner = to_grid(teacher_output.hard_owner.float(), rows, cols, replace_inf=False)
        learned_owner = to_grid(learned_output.hard_owner.float(), rows, cols, replace_inf=False)

        scalar_panels = [
            (axes[0, 0], demand_grid, "Demand field", "cividis"),
            (axes[0, 1], risk_grid, "Risk field", "magma"),
        ]
        for axis, grid, panel_title, cmap in scalar_panels:
            image = axis.imshow(grid, cmap=cmap, interpolation="nearest")
            self._mark_sinks(axis, layout.sink_positions)
            axis.set_title(panel_title)
            axis.set_xticks([])
            axis.set_yticks([])
            fig.colorbar(image, ax=axis, fraction=0.046, pad=0.04)

        teacher_image = axes[0, 2].imshow(teacher_owner, cmap=owner_cmap, norm=owner_norm, interpolation="nearest")
        self._mark_sinks(axes[0, 2], layout.sink_positions)
        axes[0, 2].set_title("Teacher hard territories")
        axes[0, 2].set_xticks([])
        axes[0, 2].set_yticks([])
        teacher_colorbar = fig.colorbar(teacher_image, ax=axes[0, 2], fraction=0.046, pad=0.04, ticks=owner_ticks)
        teacher_colorbar.set_ticklabels(owner_labels)

        learned_image = axes[1, 0].imshow(learned_owner, cmap=owner_cmap, norm=owner_norm, interpolation="nearest")
        self._mark_sinks(axes[1, 0], layout.sink_positions)
        axes[1, 0].set_title("Learned hard territories")
        axes[1, 0].set_xticks([])
        axes[1, 0].set_yticks([])
        learned_colorbar = fig.colorbar(learned_image, ax=axes[1, 0], fraction=0.046, pad=0.04, ticks=owner_ticks)
        learned_colorbar.set_ticklabels(owner_labels)

        self._plot_sink_bars(
            axes[1, 1],
            teacher_output.sink_loads,
            learned_output.sink_loads,
            title="Final sink loads",
            ylabel="load",
        )
        self._plot_sink_bars(
            axes[1, 2],
            teacher_output.sink_risks,
            learned_output.sink_risks,
            title="Final sink risk totals",
            ylabel="risk-weighted load",
        )
        fig.suptitle(title)
        fig.tight_layout()
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=160)
        plt.close(fig)

    def _mark_sinks(self, axis, sink_positions: tuple[tuple[int, int], ...]) -> None:
        for sink_idx, (row, col) in enumerate(sink_positions):
            axis.plot(col, row, marker="o", markersize=8, color="white", markeredgecolor="black", markeredgewidth=0.8)
            axis.text(col + 0.2, row + 0.2, f"S{sink_idx}", color="white", fontsize=9, weight="bold")

    def _plot_sink_bars(self, axis, teacher_values: torch.Tensor, learned_values: torch.Tensor, *, title: str, ylabel: str) -> None:
        sink_ids = list(range(teacher_values.shape[0]))
        teacher_cpu = teacher_values.detach().cpu().numpy()
        learned_cpu = learned_values.detach().cpu().numpy()
        width = 0.35
        axis.bar([value - width / 2 for value in sink_ids], teacher_cpu, width=width, label="teacher", color="#355C7D")
        axis.bar([value + width / 2 for value in sink_ids], learned_cpu, width=width, label="learned", color="#C06C84")
        axis.set_title(title)
        axis.set_xlabel("sink")
        axis.set_ylabel(ylabel)
        axis.set_xticks(sink_ids)
        axis.grid(axis="y", alpha=0.25)
        axis.legend(loc="best")