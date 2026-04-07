"""Rendering utilities for territory experiments: loss curves and summary panels."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING
import torch

from ..domain.factory import build_layout, build_scenario_from_spec
from ..domain.program import rollout_summary
from ..training.metrics import summary_metrics

try:
    import matplotlib.pyplot as plt
    from matplotlib import colors as mcolors
except ImportError:
    plt = None
    mcolors = None

if TYPE_CHECKING:
    from ..domain.specs import LearnableTerritoriesSpec
    from ..domain.layout import TerritoryLayout, TerritoryOutputs


class TerritoriesRenderer:
    """Orchestrates visual output generation for territory experiments."""

    def __init__(self, spec: "LearnableTerritoriesSpec"):
        self.spec = spec

    def render_training_results(
        self,
        model,
        history: dict[str, list[float]],
        viz_prefix: str,
    ):
        """Generate loss curves and final result panels."""
        if plt is None:
            return

        self._render_loss_curves(
            history,
            output_path=f"{viz_prefix}_loss.png",
            title="Territories training and validation losses",
        )

        # We need a teacher model to compare against
        from ..domain.factory import build_teacher_model_from_spec

        scenario = build_scenario_from_spec(self.spec)
        teacher = build_teacher_model_from_spec(self.spec).to(scenario.device)

        # 1. Training layout summary
        train_layout = build_layout(
            scenario, seed=self.spec.training.train_seed, **self._program_args()
        )
        with torch.no_grad():
            teacher_train = rollout_summary(
                self.spec, scenario, train_layout, teacher, name_prefix="teacher_viz"
            )
            model_train = rollout_summary(
                self.spec, scenario, train_layout, model, name_prefix="model_viz"
            )

        self._render_summary_panel(
            layout=train_layout,
            teacher_output=teacher_train,
            learned_output=model_train,
            output_path=f"{viz_prefix}_train.png",
            title="Training layout",
        )

        # 2. Evaluation layout summary (first seed)
        if self.spec.evaluation.eval_seeds:
            eval_seed = self.spec.evaluation.eval_seeds[0]
            eval_layout = build_layout(scenario, seed=eval_seed, **self._program_args())
            with torch.no_grad():
                teacher_eval = rollout_summary(
                    self.spec,
                    scenario,
                    eval_layout,
                    teacher,
                    name_prefix="teacher_eval_viz",
                )
                model_eval = rollout_summary(
                    self.spec,
                    scenario,
                    eval_layout,
                    model,
                    name_prefix="model_eval_viz",
                )

            self._render_summary_panel(
                layout=eval_layout,
                teacher_output=teacher_eval,
                learned_output=model_eval,
                output_path=f"{viz_prefix}_eval_seed{eval_seed}.png",
                title=f"Held-out layout (seed={eval_seed})",
            )

    def _program_args(self) -> dict:
        return {
            "num_sinks": self.spec.program.num_sinks,
            "scenario_preset": self.spec.program.scenario_preset,
            "sink_positions": self.spec.program.sink_positions,
        }

    def _render_loss_curves(
        self, history: dict[str, list[float]], *, output_path: str, title: str
    ) -> None:
        epochs = history["epoch"]
        fig, axes = plt.subplots(2, 1, figsize=(10.0, 7.4), sharex=True)

        axes[0].plot(
            epochs,
            history["total"],
            label="train total",
            linewidth=2.0,
            color="#1F4E5F",
        )
        val_x, val_y = self._finite_series(epochs, history["val_total"])
        if val_x:
            axes[0].plot(
                val_x,
                val_y,
                label="eval total",
                linewidth=1.8,
                linestyle=":",
                color="#C06C84",
            )
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
                axes[1].plot(
                    curve_x,
                    curve_y,
                    label=label,
                    color=color,
                    linestyle=linestyle,
                    linewidth=1.8,
                )
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

    def _finite_series(
        self, xs: list[float], ys: list[float]
    ) -> tuple[list[float], list[float]]:
        plot_x: list[float] = []
        plot_y: list[float] = []
        for x_value, y_value in zip(xs, ys):
            if torch.isfinite(torch.tensor(y_value)):
                plot_x.append(float(x_value))
                plot_y.append(float(y_value))
        return plot_x, plot_y

    def _owner_style(self, num_sinks: int):
        cmap_name = "tab20" if num_sinks <= 20 else "gist_ncar"
        sink_cmap = plt.get_cmap(cmap_name, max(num_sinks, 1))
        owner_colors = ["#6C7A89"] + [
            mcolors.to_hex(sink_cmap(index)) for index in range(num_sinks)
        ]
        owner_cmap = mcolors.ListedColormap(owner_colors)
        owner_norm = mcolors.BoundaryNorm(
            boundaries=[-1.5]
            + [index - 0.5 for index in range(num_sinks)]
            + [num_sinks - 0.5],
            ncolors=owner_cmap.N,
        )
        ticks = list(range(-1, num_sinks))
        labels = ["unassigned"] + [f"S{index}" for index in range(num_sinks)]
        return owner_cmap, owner_norm, ticks, labels

    def _render_summary_panel(
        self,
        *,
        layout: "TerritoryLayout",
        teacher_output: "TerritoryOutputs",
        learned_output: "TerritoryOutputs",
        output_path: str,
        title: str,
    ) -> None:
        try:
            from ...shared.plotting import to_grid
        except ImportError:
            from shared.plotting import to_grid

        rows, cols = self.spec.grid.rows, self.spec.grid.cols
        fig, axes = plt.subplots(2, 3, figsize=(15.0, 8.8))
        owner_cmap, owner_norm, owner_ticks, owner_labels = self._owner_style(
            layout.num_sinks
        )

        demand_grid = to_grid(layout.demand, rows, cols, replace_inf=False)
        risk_grid = to_grid(layout.risk, rows, cols, replace_inf=False)
        teacher_owner = to_grid(
            teacher_output.hard_owner.float(), rows, cols, replace_inf=False
        )
        learned_owner = to_grid(
            learned_output.hard_owner.float(), rows, cols, replace_inf=False
        )

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

        teacher_image = axes[0, 2].imshow(
            teacher_owner, cmap=owner_cmap, norm=owner_norm, interpolation="nearest"
        )
        self._mark_sinks(axes[0, 2], layout.sink_positions)
        axes[0, 2].set_title("Teacher hard territories")
        axes[0, 2].set_xticks([])
        axes[0, 2].set_yticks([])
        teacher_colorbar = fig.colorbar(
            teacher_image, ax=axes[0, 2], fraction=0.046, pad=0.04, ticks=owner_ticks
        )
        teacher_colorbar.set_ticklabels(owner_labels)

        learned_image = axes[1, 0].imshow(
            learned_owner, cmap=owner_cmap, norm=owner_norm, interpolation="nearest"
        )
        self._mark_sinks(axes[1, 0], layout.sink_positions)
        axes[1, 0].set_title("Learned hard territories")
        axes[1, 0].set_xticks([])
        axes[1, 0].set_yticks([])
        learned_colorbar = fig.colorbar(
            learned_image, ax=axes[1, 0], fraction=0.046, pad=0.04, ticks=owner_ticks
        )
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

    def _mark_sinks(self, axis, sink_positions) -> None:
        for sink_idx, (row, col) in enumerate(sink_positions):
            axis.plot(
                col,
                row,
                marker="o",
                markersize=8,
                color="white",
                markeredgecolor="black",
                markeredgewidth=0.8,
            )
            axis.text(
                col + 0.2,
                row + 0.2,
                f"S{sink_idx}",
                color="white",
                fontsize=9,
                weight="bold",
            )

    def render_territory_evolution(
        self,
        checkpoints: list[dict],
        viz_prefix: str,
    ) -> None:
        if plt is None or not checkpoints:
            return

        from ..domain.factory import build_scenario_from_spec, build_teacher_model_from_spec

        scenario = build_scenario_from_spec(self.spec)
        teacher = build_teacher_model_from_spec(self.spec).to(scenario.device)

        if self.spec.evaluation.eval_seeds:
            eval_seed = self.spec.evaluation.eval_seeds[0]
        else:
            eval_seed = self.spec.training.train_seed

        eval_layout = build_layout(scenario, seed=eval_seed, **self._program_args())

        with torch.no_grad():
            teacher_output = rollout_summary(
                self.spec, scenario, eval_layout, teacher, name_prefix="teacher_evo_viz"
            )

        epochs = [cp["epoch"] for cp in checkpoints]
        num_epochs = len(epochs)

        max_cols = min(num_epochs, 5)
        num_row_groups = (num_epochs + max_cols - 1) // max_cols

        fig_height = 4.5 * num_row_groups + 1.5
        fig_width = 3.8 * max_cols + 1.0
        fig, axes = plt.subplots(num_row_groups, max_cols, figsize=(fig_width, fig_height))

        if num_row_groups == 1 and max_cols == 1:
            axes = [[axes]]
        elif num_row_groups == 1:
            axes = [axes]
        elif max_cols == 1:
            axes = [[ax] for ax in axes]

        owner_cmap, owner_norm, _, _ = self._owner_style(eval_layout.num_sinks)

        try:
            from ...shared.plotting import to_grid
        except ImportError:
            from shared.plotting import to_grid

        rows, cols = self.spec.grid.rows, self.spec.grid.cols
        teacher_owner = to_grid(teacher_output.hard_owner.float(), rows, cols, replace_inf=False)

        for idx, cp in enumerate(checkpoints):
            row = idx // max_cols
            col = idx % max_cols
            ax = axes[row][col]

            from ..model.territory_model import LearnableTerritoryModel
            model = LearnableTerritoryModel.from_spec(self.spec.model).to(scenario.device)
            model.load_state_dict(cp["state_dict"])

            with torch.no_grad():
                model_output = rollout_summary(
                    self.spec, scenario, eval_layout, model, name_prefix=f"evo_viz_{idx}"
                )

            learned_owner = to_grid(model_output.hard_owner.float(), rows, cols, replace_inf=False)

            ax.imshow(teacher_owner, cmap=owner_cmap, norm=owner_norm, interpolation="nearest", alpha=0.35)
            ax.imshow(learned_owner, cmap=owner_cmap, norm=owner_norm, interpolation="nearest", alpha=0.65)
            self._mark_sinks(ax, eval_layout.sink_positions)
            ax.set_title(f"Epoch {epochs[idx]}", fontsize=10)
            ax.set_xticks([])
            ax.set_yticks([])

        for idx in range(num_epochs, num_row_groups * max_cols):
            row = idx // max_cols
            col = idx % max_cols
            axes[row][col].axis("off")

        fig.suptitle(f"Territory evolution over time (eval seed={eval_seed})", fontsize=13)
        fig.tight_layout()
        Path(viz_prefix).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(f"{viz_prefix}_evolution.png", dpi=150)
        plt.close(fig)

    def _plot_sink_bars(
        self, axis, teacher_values, learned_values, *, title: str, ylabel: str
    ) -> None:
        sink_ids = list(range(teacher_values.shape[0]))
        teacher_cpu = teacher_values.detach().cpu().numpy()
        learned_cpu = learned_values.detach().cpu().numpy()
        width = 0.35
        axis.bar(
            [v - width / 2 for v in sink_ids],
            teacher_cpu,
            width=width,
            label="teacher",
            color="#355C7D",
        )
        axis.bar(
            [v + width / 2 for v in sink_ids],
            learned_cpu,
            width=width,
            label="learned",
            color="#C06C84",
        )
        axis.set_title(title)
        axis.set_xlabel("sink")
        axis.set_ylabel(ylabel)
        axis.set_xticks(sink_ids)
        axis.grid(axis="y", alpha=0.25)
        axis.legend(loc="best")
