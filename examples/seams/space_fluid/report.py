"""Paired complete-panel inference, fixed visual episodes and honest partial reports."""

import csv
from collections import defaultdict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.colors import Normalize

from ..artifacts import checksum, json_write, read_json
from ..metrics import interval
from .config import FIXED, LEARNERS, TRAIN_FAMILIES
from .program import ELECTED, LEADER, SAMPLE

GROUP = ("panel", "executor", "family", "nodes", "layout", "condition")
METRICS = (
    "nrmse",
    "regions",
    "homogeneity",
    "region_mean_std",
    "mean_region_std",
    "mean_region_size",
    "collected_fraction",
    "fragmentation",
    "assignment_stability",
    "message_bytes",
)
COLORS = {
    "fixed-spatial": "#808080",
    "fixed-combined": "#202020",
    "fixed-value": "#c49b2a",
    "fixed-variance": "#8c6d1f",
    "parametric": "#147d92",
    "hybrid": "#d46532",
    "kmeans": "#769751",
    "gnn": "#5b4fb3",
}
# Generalisation panels: (title, row filter); every row is at the main tradeoff.
PANELS = (
    (
        "In-dist.",
        lambda r: (
            r["panel"] == "main" and r["condition"] == "clean" and r["family"] in TRAIN_FAMILIES
        ),
    ),
    (
        "Held-out shape",
        lambda r: (
            r["panel"] == "main" and r["condition"] == "clean" and r["family"] not in TRAIN_FAMILIES
        ),
    ),
    ("Faults", lambda r: r["panel"] == "main" and r["condition"] != "clean"),
    ("Uneven layout", lambda r: r["panel"] == "topology"),
    ("Larger N", lambda r: r["panel"] == "transfer"),
    ("Async", lambda r: r["panel"] == "distributed" and r["executor"].startswith("async")),
)


def main_methods(config):
    lam = f"-l{config.main_lambda:g}"
    return (
        *(f"fixed-{m}" for m in FIXED),
        *(f"{m}{lam}" for m in LEARNERS),
        *(f"kmeans-k{k}" for k in config.k_values),
    )


def method_color(method):
    return COLORS.get(method, COLORS.get(method.split("-")[0], "gray"))


def export(fig, out, name):
    directory = out / "figures"
    directory.mkdir(parents=True, exist_ok=True)
    fig.savefig(directory / f"{name}.png", dpi=160, bbox_inches="tight")
    fig.savefig(directory / f"{name}.pdf", bbox_inches="tight")
    plt.close(fig)


def quality_figure(out, summaries):
    fig, axes = plt.subplots(2, 3, figsize=(12, 7), layout="constrained")
    families = ("constant", "gaussian", "mixture", "ellipse", "ring", "front")
    for ax, family in zip(axes.flat, families, strict=True):
        selected = [
            r
            for r in summaries
            if r["panel"] == "main"
            and r["condition"] == "clean"
            and r["family"] == family
            and r["metric"] in ("nrmse", "regions")
        ]
        table = {(r["method"], r["metric"]): r for r in selected}
        for method in sorted({r["method"] for r in selected}):
            x, y = table[method, "regions"], table[method, "nrmse"]
            if x["mean"] is None or y["mean"] is None:
                continue
            ax.scatter(
                x["mean"],
                y["mean"],
                s=35,
                color=method_color(method),
                marker="x" if method.startswith("kmeans") else "o",
                label=method,
            )
            # Percentile intervals need not contain the point estimate in a small sample.
            if x["ci95"]:
                ax.plot(x["ci95"], [y["mean"]] * 2, color=method_color(method), lw=1)
            if y["ci95"]:
                ax.plot([x["mean"]] * 2, y["ci95"], color=method_color(method), lw=1)
        ax.set(
            title=family + (" (held out shape)" if family in ("ring", "front") else ""),
            xlabel="Mean assigned regions",
            ylabel="NRMSE (training scale)",
        )
        ax.grid(alpha=0.15)
    handles, labels = axes.flat[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="outside lower center", ncol=5, fontsize=7)
    export(fig, out, "error-compression")


def learning_figure(out):
    paths = sorted((out / "checkpoints").glob("*/seed*/training.json"))
    if not paths:
        return
    fig, ax = plt.subplots(figsize=(8, 4), layout="constrained")
    for path in paths:
        history = read_json(path)["history"]
        history = [r for r in history if "validation_hard" in r]
        method = path.parent.parent.name
        ax.plot(
            [r["step"] for r in history],
            [r["validation_hard"] for r in history],
            color=method_color(method),
            alpha=0.6,
            label=f"{method}/{path.parent.name}",
        )
    ax.set(xlabel="Adam update", ylabel="Hard validation objective")
    ax.grid(alpha=0.15)
    ax.legend(fontsize=6, ncol=3)
    export(fig, out, "hard-validation")


def recovery_figure(out, names):
    """Show identical Gaussian episodes under each event, with a common service scale."""
    conditions = ("link_loss", "node_stop", "partition")
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.6), layout="constrained", sharey=True)
    any_data = False
    for ax, condition in zip(axes.flat, conditions, strict=True):
        for name in names:
            paths = sorted(
                (out / "episodes/main/batched" / name).glob(
                    f"seed*/gaussian-*-jittered-{condition}.json"
                )
            )
            records = [read_json(path) for path in paths]
            curves = [r["curves"]["nrmse"] for r in records if r.get("status") == "complete"]
            if not curves:
                continue
            values = np.array(curves)
            any_data = True
            ax.plot(values.mean(0), label=name, color=method_color(name))
            ax.axvspan(values.shape[1] // 3, 2 * values.shape[1] // 3, alpha=0.025, color="black")
        ax.axhline(0.2, color="gray", ls=":", lw=1)
        ax.set(title=condition, xlabel="Round", ylabel="Mean NRMSE")
        ax.grid(alpha=0.15)
    if any_data:
        axes.flat[0].legend(fontsize=7)
        export(fig, out, "service-recovery")
    else:
        plt.close(fig)


def animate(out, family, budget, names):  # noqa: PLR0915 -- shared scales and persistent animation artists
    payloads = []
    for name in names:
        paths = sorted(
            (out / "replays/main/batched" / name / "seed0").glob(f"{family}-0-*-clean.pt")
        )
        if not paths:
            return False
        payloads.append(torch.load(paths[0], weights_only=True))
    episode = payloads[0]["episode"]
    for item in payloads[1:]:
        if not torch.equal(item["episode"]["truth"], episode["truth"]):
            raise ValueError("Animation methods must share the exact episode")
    path = out / "animations" / f"{family}.gif"
    sources = [
        checksum(p)
        for name in names
        for p in sorted(
            (out / "replays/main/batched" / name / "seed0").glob(f"{family}-0-*-clean.pt")
        )
    ]
    record = path.with_suffix(".json")
    if path.exists() and record.exists() and read_json(record)["source_sha256"] == sources:
        return True
    if budget is not None:
        budget.check()
    path.parent.mkdir(parents=True, exist_ok=True)
    positions = episode["positions"].numpy()
    truth = episode["truth"].numpy()
    frames, nodes = truth.shape
    predictions = [p["trace"]["fields"][..., SAMPLE].numpy() for p in payloads]
    scale = payloads[0]["normalization"]["scale"]
    value_norm = Normalize(
        min(float(truth.min()), *(float(p.min()) for p in predictions)),
        max(float(truth.max()), *(float(p.max()) for p in predictions)),
    )
    errors = [np.abs(p - truth) / scale for p in predictions]
    error_norm = Normalize(0, max(1e-6, *(float(e.max()) for e in errors)))
    fig, axes = plt.subplots(len(names), 4, figsize=(11, 2.3 * len(names)), layout="constrained")
    artists = []
    for row, (name, data) in enumerate(zip(names, payloads, strict=True)):
        fields = data["trace"]["fields"]
        collections = []
        for col, title in enumerate(
            ("Phenomenon", "Regions / leaders", "Reconstruction", "Absolute error / sigma")
        ):
            ax = axes[row, col]
            norm = (
                Normalize(0, max(nodes - 1, 1))
                if col == 1
                else error_norm
                if col == 3
                else value_norm
            )
            scatter = ax.scatter(
                *positions.T,
                c=np.zeros(nodes),
                s=max(8, 500 / nodes),
                cmap="turbo" if col == 1 else "magma" if col == 3 else "viridis",
                norm=norm,
            )
            ax.set(xlim=(-0.03, 1.03), ylim=(-0.03, 1.03), aspect="equal", xticks=[], yticks=[])
            if row == 0:
                ax.set_title(title, fontsize=10)
            if col == 0:
                ax.set_ylabel(name, fontsize=9)
            collections.append(scatter)
        leaders = axes[row, 1].scatter(
            [], [], marker="*", s=55, facecolors="none", edgecolors="black", linewidths=0.8
        )
        artists.append((fields, collections, leaders))
    fig.colorbar(
        artists[0][1][0], ax=list(axes[:, 0]) + list(axes[:, 2]), shrink=0.4, label="Field value"
    )
    fig.colorbar(
        artists[0][1][3], ax=list(axes[:, 3]), shrink=0.4, label="Normalized absolute error"
    )
    title = fig.suptitle("")

    def draw(t):
        title.set_text(f"{family} | shared trajectory | round {t + 1}/{frames}")
        for row, (fields, collections, leaders) in enumerate(artists):
            arrays = (truth[t], fields[t, :, LEADER].numpy(), predictions[row][t], errors[row][t])
            for artist, values in zip(collections, arrays, strict=True):
                artist.set_array(values)
            leaders.set_offsets(positions[fields[t, :, ELECTED].numpy() > 0.5])
        return []

    draw(frames // 2)
    fig.savefig(path.with_suffix(".png"), dpi=120)
    animation = FuncAnimation(fig, draw, frames=frames, interval=125, blit=False)
    temporary = path.with_name(path.stem + ".tmp.gif")
    animation.save(temporary, writer=PillowWriter(fps=8), dpi=85)
    temporary.replace(path)
    plt.close(fig)
    json_write(
        record,
        {
            "source_sha256": sources,
            "frames": frames,
            "fps": 8,
            "common_value_scale": [value_norm.vmin, value_norm.vmax],
            "common_error_scale": [error_norm.vmin, error_norm.vmax],
            "methods": names,
        },
    )
    return True


def report(out, config, *, render=True, budget=None):  # noqa: PLR0912, PLR0915 -- complete and partial artifacts
    from .campaign import (  # noqa: PLC0415 -- shared inventory
        evaluation_jobs,
        progress,
        result_path,
    )

    state = progress(out, config)
    groups, rows = defaultdict(list), []
    for spec, method, seed in evaluation_jobs(config):
        path = result_path(out, spec, method, seed)
        row = read_json(path) if path.exists() else None
        if row is None or row.get("status") != "complete":
            row = {
                **spec,
                "method": method["id"],
                "seed": seed,
                "episode": f"{spec['family']}-{spec['index']}",
                "metrics": {},
            }
        else:
            rows.append({k: v for k, v in row.items() if k != "curves"})
        # Identical logical episode IDs for complete and missing cells.
        row = {**row, "episode": f"{spec['family']}-{spec['index']}"}
        groups[tuple(spec[k] for k in GROUP)].append(row)
    summaries, paired = [], []
    for key, panel in groups.items():
        metadata = dict(zip(GROUP, key, strict=True))
        reference = [r for r in panel if r["method"] == "fixed-combined"]
        for name in sorted({r["method"] for r in panel}):
            selected = [r for r in panel if r["method"] == name]
            for metric in METRICS:
                summaries.append(
                    {**metadata, "method": name, "metric": metric, **interval(selected, metric)}
                )
                if metric in ("nrmse", "regions") and name != "fixed-combined":
                    paired.append(
                        {
                            **metadata,
                            "method": name,
                            "reference": "fixed-combined",
                            "metric": metric,
                            **interval(selected, metric, reference=reference),
                        }
                    )
    json_write(out / "results.json", rows)
    json_write(out / "summaries.json", summaries)
    json_write(out / "paired.json", paired)
    recoveries = defaultdict(lambda: defaultdict(int))
    for row in rows:
        key = (row["panel"], row["executor"], row["condition"], row["method"])
        for event in ("fault_recovery", "restoration_recovery"):
            recoveries[(*key, event)][row["metrics"][event]["status"]] += 1
    json_write(
        out / "recovery.json",
        [
            {
                **dict(
                    zip(("panel", "executor", "condition", "method", "event"), key, strict=True)
                ),
                "counts": dict(value),
            }
            for key, value in sorted(recoveries.items())
        ],
    )
    if rows:
        with (out / "results.csv").open("w", newline="") as stream:
            columns = [
                *GROUP,
                "method",
                "seed",
                "episode",
                *METRICS,
                "inference_seconds",
                "restoration_status",
                "restoration_rounds",
            ]
            writer = csv.DictWriter(stream, fieldnames=columns)
            writer.writeheader()
            for row in rows:
                flat = {key: row[key] for key in (*GROUP, "method", "seed", "episode")}
                flat.update(
                    {key: row["metrics"].get(key) for key in (*METRICS, "inference_seconds")}
                )
                recovery = row["metrics"]["restoration_recovery"]
                writer.writerow(
                    {
                        **flat,
                        "restoration_status": recovery["status"],
                        "restoration_rounds": recovery["rounds"],
                    }
                )
    training = sum(j["status"] == "complete" for j in state["training"])
    probabilities = "/".join(f"{p:g}" for p in config.activation_probabilities)
    evaluation = state["evaluation"]
    text = [
        "# Space-Fluid SCR learning",
        "",
        f"Protocol: **{config.profile}**. Status: **{state['status']}**.",
        "",
        f"Training/diagnostic jobs: {training}/{len(state['training'])}. "
        f"Evaluation jobs: {evaluation['complete']}/{evaluation['expected']}; "
        f"pending: {evaluation['pending']}; failed: {evaluation['failed']}.",
        "",
        "This report never treats a partial panel as a complete experiment. Missing seeds/episodes "
        "invalidate confidence intervals and panel means. Smoke uses one seed and "
        "cannot establish a learning advantage.",
        "",
        "The outcome of interest is lower error at comparable region counts, including "
        "held-out rings/fronts "
        "and distributed execution. A lower error with more regions alone is not "
        "evidence of a better tradeoff.",
        "",
        "Reproduce/resume from the repository root:",
        "",
        "```bash",
        f"python -m examples.seams space-fluid --profile {config.profile} --stage all --out {out}",
        "```",
        "",
        "## Measurement and limits",
        "",
        "NRMSE uses the training-only standard deviation, including on constant fields. "
        "Region count is the number "
        "of assigned leader IDs; active leader fraction is measured separately. "
        "Homogeneity is within-region truth "
        "variance divided by training variance. Fragmentation counts extra connected "
        "components per label on the current graph. "
        "Assignment stability compares device leader IDs between successive rounds. All "
        "rounds, including startup, are scored.",
        "",
        "Recovery requires NRMSE ≤ 0.2 for five consecutive rounds. `no_recovery` "
        "remains explicit; `attained` "
        "distinguishes cases whose pre-event baseline was already noncompliant. Event "
        "and restoration measurements are separate.",
        "",
        "Paired differences are formed before a two-way bootstrap of training seed and "
        "shared episode. "
        "Intervals are descriptive 95% percentile intervals, without multiplicity correction. "
        "Fixed and K-means references share identical data across seeds; seed "
        "uncertainty comes from learned methods.",
        "",
        "Elections are discrete. Training differentiates a relaxed forward (first admissible "
        "candidate in lexicographic order, probability-weighted collect) with "
        f"T={config.error_temperature:g} for the error and T={config.leader_temperature:g} "
        "for the leader term; all reported numbers use the hard program. "
        "`sensitivity.json` compares the surrogate derivative with hard central differences "
        "per weight; it does not claim an exact derivative or retraining uncertainty.",
        "",
        "Synchronous IDs and values must agree before results are accepted. "
        "Asynchronous schedules share one initial "
        f"execution, then activate with p={probabilities} "
        "and deliver only actual messages. No "
        "state resets occur at faults. "
        "Message costs count numeric bytes; serialization, headers and energy are "
        "excluded. K-means has instantaneous "
        "global observations, reconstruction by cluster means, and no comparable "
        "distributed byte count.",
        "",
        "Connected regions are a convergence objective. Dynamics and stale messages can "
        "fragment a label; "
        "the evaluation measures this rather than repairing outputs centrally. The "
        "cumulative-distance bound "
        "is not a physical reconstruction-error guarantee for a learned metric.",
        "",
    ]
    if rows:
        text += [
            "## Completed main clean panel",
            "",
            "Equal-weight episode means across all six families. Only methods with every "
            "planned seed and episode appear here; family-specific intervals are in "
            "summaries.json.",
            "",
            "| Method | NRMSE | Regions | Leader fraction |",
            "|---|---:|---:|---:|",
        ]
        for name in sorted({row["method"] for row in rows}):
            selected = [
                r
                for r in rows
                if r["panel"] == "main" and r["condition"] == "clean" and r["method"] == name
            ]
            if len(selected) != 6 * config.test_episodes * len(config.seeds):
                continue
            averages = [
                np.mean([r["metrics"][metric] for r in selected])
                for metric in ("nrmse", "regions", "leader_fraction")
            ]
            text.append(f"| {name} | {averages[0]:.4f} | {averages[1]:.2f} | {averages[2]:.4f} |")
        text.append("")
        counts = defaultdict(int)
        for row in rows:
            counts[row["metrics"]["restoration_recovery"]["status"]] += 1
        text += [
            "Restoration outcomes among completed jobs: "
            + ", ".join(f"{k}={v}" for k, v in sorted(counts.items()))
            + ".",
            "",
        ]
    if rows:
        text += generalisation_tables(rows, config)
    text += insight_summary(out)
    if (out / "timing.json").exists():
        t = read_json(out / "timing.json")
        text += [
            f"Pilot: batch forward/backward {t['batch_forward_backward_seconds']:.3f} s; "
            f"batched episode {t['batched_episode_seconds']:.3f} s; "
            f"synchronous devices {t['sync_episode_seconds']:.3f} s. "
            "See timing.json for dimensions and exclusions.",
            "",
        ]
    text += [
        "Protocol and scientific references: [suite "
        "documentation](../../../examples/seams/README.md).",
        "",
        "Raw per-round results: `episodes/`; selected spatial traces: `replays/`; "
        "checkpoints and gradient diagnostics: `checkpoints/`; software and data "
        "hashes: `manifest.json`.",
        "",
    ]
    (out / "REPORT.md").write_text("\n".join(text))
    if render:
        quality_figure(out, summaries)
        pareto_figure(out, rows, config)
        learning_figure(out)
        lam = f"-l{config.main_lambda:g}"
        names = ("fixed-combined", *(f"{m}{lam}" for m in LEARNERS))
        recovery_figure(out, names)
        visual = {family: animate(out, family, budget, names) for family in ("gaussian", "ring")}
        json_write(
            out / "visuals.json",
            {"animations": visual, "status": "complete" if all(visual.values()) else "incomplete"},
        )
    return state


def pooled(rows, name, keep, metric):
    return [
        {"seed": r["seed"], "episode": r["episode"], "metrics": {metric: r["metrics"].get(metric)}}
        for r in rows
        if r["method"] == name and keep(r)
    ]


def generalisation_tables(rows, config):
    """Main-tradeoff NRMSE/regions per panel and paired NRMSE difference to fixed-combined."""

    def cell(values, digits):
        return "—" if values["mean"] is None else f"{values['mean']:.{digits}f}"

    header = "| Method | " + " | ".join(t for t, _ in PANELS) + " |"
    rule = "|---|" + "---:|" * len(PANELS)
    tables = {key: [header, rule] for key in ("nrmse", "regions", "stability", "delta")}
    for name in main_methods(config):
        cells = {key: [] for key in tables}
        for _, keep in PANELS:
            selected = pooled(rows, name, keep, "nrmse")
            if not selected:
                for column in cells.values():
                    column.append("—")
                continue
            cells["nrmse"].append(cell(interval(selected, "nrmse"), 4))
            cells["regions"].append(
                cell(interval(pooled(rows, name, keep, "regions"), "regions"), 1)
            )
            stability = pooled(rows, name, keep, "assignment_stability")
            cells["stability"].append(cell(interval(stability, "assignment_stability"), 3))
            reference = pooled(rows, "fixed-combined", keep, "nrmse")
            delta = (
                interval(selected, "nrmse", reference=reference)
                if name != "fixed-combined"
                else None
            )
            if delta is None or delta["mean"] is None:
                cells["delta"].append("—")
            else:
                ci = delta["ci95"]
                cells["delta"].append(
                    f"{delta['mean']:+.4f}" + (f" [{ci[0]:+.3f}, {ci[1]:+.3f}]" if ci else "")
                )
        for key, table in tables.items():
            table.append(f"| {name} | " + " | ".join(cells[key]) + " |")
    return [
        f"## Generalisation at the main tradeoff (lambda={config.main_lambda:g})",
        "",
        "Episode means pooled over each panel. Held-out shapes are rings and fronts; "
        "faults pool "
        "link loss, paused nodes and partition; larger N transfers models trained at "
        f"N={config.nodes}. Async uses DeviceRuntime instances. "
        "'—' marks a method outside the panel; "
        "the GNN keeps one value per device, i.e. every device is a sampler.",
        "",
        "NRMSE (lower is better):",
        "",
        *tables["nrmse"],
        "",
        "Mean regions:",
        "",
        *tables["regions"],
        "",
        "Assignment stability (share of devices keeping their leader between rounds):",
        "",
        *tables["stability"],
        "",
        "Paired NRMSE difference to fixed-combined, mean [95% two-way bootstrap]; "
        "negative is better:",
        "",
        *tables["delta"],
        "",
    ]


def pareto_figure(out, rows, config):
    """In-distribution and held-out tradeoff: NRMSE against regions, main/clean panel."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), layout="constrained", sharey=True)
    any_data = False
    for ax, (title, keep) in zip(axes, PANELS[:2], strict=True):
        points = {}
        for name in sorted({r["method"] for r in rows}):
            error = interval(pooled(rows, name, keep, "nrmse"), "nrmse")
            regions = interval(pooled(rows, name, keep, "regions"), "regions")
            if error["n"] and error["mean"] is not None:
                points[name] = (regions["mean"], error["mean"])
        for name, (x, y) in points.items():
            any_data = True
            family = name.split("-l")[0] if "-l" in name else name
            if name.startswith("gnn"):
                ax.scatter(x, y, color=method_color(name), marker="D", label=f"{name} (per device)")
            elif not name.startswith(("kmeans", *(f"{m}-l" for m in LEARNERS))):
                ax.scatter(x, y, color=method_color(name), marker="s", label=name)
            elif name.startswith("kmeans"):
                ax.scatter(x, y, color=method_color(name), marker="x")
            else:
                ax.scatter(
                    x,
                    y,
                    color=method_color(family),
                    s=55 if name.endswith(f"l{config.main_lambda:g}") else 25,
                )
        for family in (*LEARNERS, "kmeans"):
            line = sorted(v for n, v in points.items() if n.startswith(f"{family}-"))
            if len(line) > 1:
                ax.plot(*zip(*line, strict=True), color=method_color(family), lw=1, label=family)
        ax.set(title=title, xlabel="Mean regions (samplers)", ylabel="NRMSE (training scale)")
        ax.set_xscale("log")
        ax.grid(alpha=0.15)
    if any_data:
        axes[0].legend(fontsize=7)
        export(fig, out, "pareto")
    else:
        plt.close(fig)


def insight_summary(out):
    path = out / "insights.json"
    if not path.exists():
        return []
    rows = read_json(path)["rows"]
    text = [
        "## Gradient insights",
        "",
        "Surrogate gradients on fixed Gaussian and ring test episodes (mean over seeds and the two "
        "episodes). Saliency ratios divide by the mean |dL/d observation|. The hard check perturbs "
        "the top 10% salient readings or the same number of random readings by 0.1 sigma and "
        "reports the absolute change of the hard loss. |dL/d X| is the gradient reaching the "
        "output of block X (S election confidence, G distance, C converge-cast, "
        "B broadcast), summed "
        "over rounds. S share is |grad strength| / (|grad strength| + |grad metric|). "
        "See figures/insights-*.png.",
        "",
        "| Method | Leader | Boundary | Interior | Hard: salient | Hard: random "
        "| S share | dL/dS | dL/dG | dL/dC | dL/dB |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]

    def mean(values):
        values = [v for v in values if v is not None]
        return f"{np.mean(values):.3g}" if values else "—"

    for method in sorted({r["method"] for r in rows}):
        selected = [r for r in rows if r["method"] == method]
        regional = [r for r in selected if "roles" in r]
        text.append(
            f"| {method} | "
            + " | ".join(
                [
                    mean([r["roles"]["leader"] for r in regional]),
                    mean([r["roles"]["boundary"] for r in regional]),
                    mean([r["roles"]["interior"] for r in regional]),
                    mean([r["perturbation_salient"] for r in selected]),
                    mean([r["perturbation_random"] for r in selected]),
                    mean(
                        [
                            r["norms"]["strength"] / (r["norms"]["strength"] + r["norms"]["metric"])
                            for r in regional
                        ]
                    ),
                    *(mean([r["blocks"][b] for r in regional]) for b in ("S", "G", "C", "B")),
                ]
            )
            + " |"
        )
    return [*text, ""]
