#!/usr/bin/env python3
"""Cross-scenario comparison: one paper-ready overview figure per mode.

Reads the per-scenario ``summary.json`` files that ``main.py`` writes under
``<out-root>/vmas-<scenario>-<mode>/`` and produces the headline figure plus
two composite overviews (SHAC final behaviour and imitation parameter
recovery / depth expressivity, across all five scenarios at once) and a single
master CSV table, so the whole pipeline's results can be read/compared at a
glance instead of scenario-by-scenario.

Run after ``main.py --mode shac`` and ``main.py --mode imitation`` have
produced their per-scenario summaries:
    uv run python examples/vmas_diffield/summarize.py
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "examples"))

from shared.plotting import apply_paper_style, color_of, label_of, panel_label  # noqa: E402
from shared.plotting import savefig as _savefig  # noqa: E402
from shared.plotting.style import MUTED  # noqa: E402
from vmas_diffield.vmas_env import SCENARIO_SPEC  # noqa: E402

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    apply_paper_style()
except ImportError:
    plt = None

SCENARIOS = list(SCENARIO_SPEC)
POLICY_ORDER = (
    "expert", "parametric", "hybrid", "hybrid_res", "leader",
    "neural", "neural_d2", "neural_d3",
)


def _load(out_root: Path, scenario: str, mode: str) -> dict | None:
    path = out_root / f"vmas-{scenario}-{mode}" / "summary.json"
    if not path.exists():
        return None
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def _present_policies(summaries: dict[str, dict | None]) -> list[str]:
    seen = {k for s in summaries.values() if s for k in s["results"]}
    return [k for k in POLICY_ORDER if k in seen]


def plot_shac_overview(out_root: Path, out_path: Path) -> None:
    summaries = {sc: _load(out_root, sc, "shac") for sc in SCENARIOS}
    if plt is None or not any(summaries.values()):
        print("no SHAC summaries found; skipping overview")
        return
    kinds = _present_policies(summaries)
    fig, axes = plt.subplots(2, len(SCENARIOS), figsize=(3.4 * len(SCENARIOS), 6.4), squeeze=False)
    for col, scenario in enumerate(SCENARIOS):
        summ = summaries[scenario]
        ax_r, ax_p = axes[0][col], axes[1][col]
        if summ is None:
            ax_r.axis("off")
            ax_p.axis("off")
            ax_r.set_title(f"{scenario}\n(no data)", fontsize=9.5, fontweight="normal", color=MUTED)
            continue
        present = [k for k in kinds if k in summ["results"]]
        rewards = [summ["results"][k]["final_reward"] for k in present]
        primaries = [summ["results"][k]["final_primary"] for k in present]
        colors = [color_of(k) for k in present]
        ax_r.bar(
            range(len(present)),
            [r["mean"] for r in rewards],
            yerr=[r["ci"] for r in rewards],
            capsize=3,
            color=colors,
            width=0.6,
            error_kw={"linewidth": 1.0, "ecolor": "#52514e"},
        )
        ax_r.set_xticks([])
        ax_r.set_title(scenario, fontsize=9.5, fontweight="normal", color=MUTED)
        if col == 0:
            ax_r.set_ylabel("final reward (↑)")
        ax_r.grid(True, axis="y", alpha=0.5)
        ax_r.grid(False, axis="x")

        metric_name = summ["primary_metric"]
        ax_p.bar(
            range(len(present)),
            [p["mean"] for p in primaries],
            yerr=[p["ci"] for p in primaries],
            capsize=3,
            color=colors,
            width=0.6,
            error_kw={"linewidth": 1.0, "ecolor": "#52514e"},
        )
        ax_p.set_xticks([])
        ax_p.set_xlabel(f"{metric_name} (↑)")
        if col == 0:
            ax_p.set_ylabel("final primary metric")
        ax_p.grid(True, axis="y", alpha=0.5)
        ax_p.grid(False, axis="x")

    handles = [plt.Rectangle((0, 0), 1, 1, color=color_of(k), label=label_of(k)) for k in kinds]
    fig.legend(handles=handles, loc="upper center", ncol=len(kinds), bbox_to_anchor=(0.5, 1.02))
    _savefig(fig, out_path)


def plot_headline(out_root: Path, out_path: Path) -> None:
    """The cross-scenario "money figure": is a field program >= the GNN, everywhere?

    (a) Final primary metric per scenario, normalized within each scenario to the
    best policy's mean (=1.0), so four metrics with wildly different absolute
    scales (order ~0.9 vs coverage ~0.005) read on one axis. (b) The same
    normalized score (mean across scenarios) against trainable-parameter count
    (log): the interpretable field programs should sit top-left -- matching or
    beating the GNN with orders of magnitude fewer parameters. ``leader`` is a
    flocking-only specialist, so it appears in (a) but is excluded from (b)'s
    cross-scenario mean.
    """
    summaries = {sc: _load(out_root, sc, "shac") for sc in SCENARIOS}
    if plt is None or not any(summaries.values()):
        print("no SHAC summaries found; skipping headline")
        return
    kinds = _present_policies(summaries)
    fig = plt.figure(figsize=(10.0, 3.9))
    gs = fig.add_gridspec(1, 2, width_ratios=[2.1, 1.0], wspace=0.30)
    ax_a = fig.add_subplot(gs[0])
    ax_b = fig.add_subplot(gs[1])

    n_k = len(kinds)
    width = 0.8 / n_k
    norm_scores: dict[str, list[float]] = {k: [] for k in kinds}
    xticks, xlabels = [], []
    for i, scenario in enumerate(SCENARIOS):
        summ = summaries[scenario]
        if summ is None:
            continue
        present = [k for k in kinds if k in summ["results"]]
        best = max(summ["results"][k]["final_primary"]["mean"] for k in present)
        if abs(best) < 1e-12:
            best = 1.0
        for j, k in enumerate(kinds):
            if k not in summ["results"]:
                continue
            r = summ["results"][k]["final_primary"]
            x = i + (j - (n_k - 1) / 2) * width
            ax_a.bar(
                x, r["mean"] / best, width * 0.92, yerr=r["ci"] / best, capsize=2.5,
                color=color_of(k), error_kw={"linewidth": 1.0, "ecolor": "#52514e"},
            )
            if k != "leader":
                norm_scores[k].append(r["mean"] / best)
        xticks.append(i)
        xlabels.append(f"{scenario}\n({summ['primary_metric']})")
    ax_a.set_xticks(xticks)
    ax_a.set_xticklabels(xlabels, fontsize=8.5)
    ax_a.set_ylabel("final primary metric\n(normalized to scenario best = 1)")
    ax_a.axhline(1.0, color="#9a9890", linestyle=":", linewidth=1.0)
    ax_a.grid(True, axis="y", alpha=0.5)
    ax_a.grid(False, axis="x")
    panel_label(ax_a, "a")

    _headline_params_panel(ax_b, summaries, kinds, norm_scores)

    handles = [plt.Rectangle((0, 0), 1, 1, color=color_of(k), label=label_of(k)) for k in kinds]
    fig.legend(handles=handles, loc="upper center", ncol=len(kinds), bbox_to_anchor=(0.5, 1.04))
    _savefig(fig, out_path)


def _headline_params_panel(ax_b, summaries, kinds, norm_scores) -> None:
    """Panel (b): mean normalized score vs trainable-parameter count (log x).

    ``expert`` trains zero parameters, which a log axis cannot place — it is
    drawn as a dashed reference line instead (the "no learning at all" floor
    every learned policy should clear).
    """
    for k in kinds:
        if k == "leader" or not norm_scores[k]:
            continue
        params = [
            s["results"][k]["n_params"]
            for s in (summaries[sc] for sc in SCENARIOS)
            if s and k in s["results"]
        ]
        p_mean = sum(params) / len(params)
        s_mean = sum(norm_scores[k]) / len(norm_scores[k])
        if p_mean <= 0:
            ax_b.axhline(s_mean, color=color_of(k), linestyle="--", linewidth=1.4)
            ax_b.annotate(
                "hand program (0 params)", (0.04, s_mean), xycoords=("axes fraction", "data"),
                textcoords="offset points", xytext=(0, 4), fontsize=8, color=MUTED,
            )
            continue
        ax_b.scatter(p_mean, s_mean, s=95, color=color_of(k), zorder=3)
        ax_b.annotate(
            f"{round(p_mean):,}", (p_mean, s_mean), textcoords="offset points",
            xytext=(0, -13), ha="center", fontsize=8, color=MUTED,
        )
    ax_b.set_xscale("log")
    ax_b.margins(0.2)
    ax_b.set_xlabel("trainable parameters")
    ax_b.set_ylabel("mean normalized primary (↑)")
    ax_b.grid(True, which="both", alpha=0.4)
    panel_label(ax_b, "b")


def plot_imitation_overview(out_root: Path, out_path: Path) -> None:
    """Row (a): final imitation loss per student, log scale — the depth/
    expressivity result (program students fit the expert exactly; GNN students
    plateau). Row (b): weight-recovery MAE, only for students that actually
    recover program weights (bars for the others would fake perfect recovery).
    """
    summaries = {sc: _load(out_root, sc, "imitation") for sc in SCENARIOS}
    if plt is None or not any(summaries.values()):
        print("no imitation summaries found; skipping overview")
        return
    kinds = _present_policies(summaries)
    loss_floor = 1e-6
    fig, axes = plt.subplots(2, len(SCENARIOS), figsize=(3.4 * len(SCENARIOS), 6.4), squeeze=False)
    for col, scenario in enumerate(SCENARIOS):
        summ = summaries[scenario]
        ax_loss, ax_mae = axes[0][col], axes[1][col]
        if summ is None:
            ax_loss.axis("off")
            ax_mae.axis("off")
            ax_loss.set_title(
                f"{scenario}\n(no data)", fontsize=9.5, fontweight="normal", color=MUTED
            )
            continue
        present = [k for k in kinds if k in summ["results"]]
        colors = [color_of(k) for k in present]

        raw = [summ["results"][k]["final_loss"]["mean"] for k in present]
        losses = [max(v, loss_floor) for v in raw]
        cis = [summ["results"][k]["final_loss"]["ci"] for k in present]
        yerr = [[min(c, v * 0.99) for c, v in zip(cis, losses, strict=True)], cis]
        ax_loss.bar(
            range(len(present)), losses, yerr=yerr, color=colors, width=0.6,
            error_kw={"elinewidth": 1.0, "capsize": 2.0},
        )
        for i, v in enumerate(raw):
            if v < loss_floor:
                ax_loss.annotate(
                    "≈0", (i, loss_floor), textcoords="offset points", xytext=(0, 3),
                    ha="center", fontsize=8, color=MUTED,
                )
        ax_loss.set_yscale("log")
        ax_loss.set_xticks([])
        ax_loss.set_title(scenario, fontsize=9.5, fontweight="normal", color=MUTED)
        ax_loss.grid(True, axis="y", which="both", alpha=0.5)
        ax_loss.grid(False, axis="x")

        rec_kinds = [k for k in present if summ["results"][k]["recovered"] is not None]
        expert = summ["expert"]
        maes = []
        for k in rec_kinds:
            rec = summ["results"][k]["recovered"]
            errs = [abs(rec[t]["mean"] - expert[t]) for t in expert]
            maes.append(sum(errs) / len(errs))
        ax_mae.bar(
            [present.index(k) for k in rec_kinds], maes,
            color=[color_of(k) for k in rec_kinds], width=0.6,
        )
        for k, v in zip(rec_kinds, maes, strict=True):
            if v < 5e-3:
                ax_mae.annotate(
                    "≈0", (present.index(k), v), textcoords="offset points", xytext=(0, 3),
                    ha="center", fontsize=8, color=MUTED,
                )
        ax_mae.set_xlim(ax_loss.get_xlim())
        ax_mae.set_xticks([])
        ax_mae.grid(True, axis="y", alpha=0.5)
        ax_mae.grid(False, axis="x")

        if col == 0:
            ax_loss.set_ylabel("final imitation loss (↓)")
            ax_mae.set_ylabel("|recovered - expert θ| MAE (↓)")
            panel_label(ax_loss, "a")
            panel_label(ax_mae, "b")

    handles = [plt.Rectangle((0, 0), 1, 1, color=color_of(k), label=label_of(k)) for k in kinds]
    fig.legend(handles=handles, loc="upper center", ncol=len(kinds), bbox_to_anchor=(0.5, 1.03))
    _savefig(fig, out_path)


def write_master_csv(out_root: Path, out_path: Path) -> None:
    rows = []
    for scenario in SCENARIOS:
        spec = SCENARIO_SPEC[scenario]
        shac = _load(out_root, scenario, "shac")
        imit = _load(out_root, scenario, "imitation")
        for kind in POLICY_ORDER:
            row = {
                "scenario": scenario,
                "policy": kind,
                "shac_final_reward_mean": "",
                "shac_final_reward_ci": "",
                "shac_primary_metric": spec.primary_metric,
                "shac_final_primary_mean": "",
                "shac_final_primary_ci": "",
                "imit_final_loss_mean": "",
                "imit_final_loss_ci": "",
                "imit_recovery_mae": "",
            }
            if shac and kind in shac["results"]:
                r = shac["results"][kind]
                row["shac_final_reward_mean"] = f"{r['final_reward']['mean']:.4f}"
                row["shac_final_reward_ci"] = f"{r['final_reward']['ci']:.4f}"
                row["shac_final_primary_mean"] = f"{r['final_primary']['mean']:.4f}"
                row["shac_final_primary_ci"] = f"{r['final_primary']['ci']:.4f}"
            if imit and kind in imit["results"]:
                ri = imit["results"][kind]
                row["imit_final_loss_mean"] = f"{ri['final_loss']['mean']:.5f}"
                row["imit_final_loss_ci"] = f"{ri['final_loss']['ci']:.5f}"
                if ri["recovered"] is not None:
                    expert = imit["expert"]
                    errs = [abs(ri["recovered"][t]["mean"] - expert[t]) for t in expert]
                    row["imit_recovery_mae"] = f"{sum(errs) / len(errs):.4f}"
            if any(
                v != ""
                for k, v in row.items()
                if k not in ("scenario", "policy", "shac_primary_metric")
            ):
                rows.append(row)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else [])
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved {out_path}")


def main() -> None:
    p = argparse.ArgumentParser(description="Cross-scenario SHAC/imitation comparison")
    p.add_argument("--out-root", type=str, default="generated")
    args = p.parse_args()
    out_root = Path(args.out_root)
    comp_dir = out_root / "vmas-comparison"
    plot_headline(out_root, comp_dir / "headline.png")
    plot_shac_overview(out_root, comp_dir / "shac_overview.png")
    plot_imitation_overview(out_root, comp_dir / "imitation_overview.png")
    write_master_csv(out_root, comp_dir / "master_comparison.csv")


if __name__ == "__main__":
    main()
