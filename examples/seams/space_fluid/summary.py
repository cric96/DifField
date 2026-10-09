"""One comparison across the three studies: reconstruction, static/moving phenomenon,
clustering. Reads ``<out>/main``, ``<out>/scenarios`` and ``<out>/clusters``."""

import itertools
import shutil

import matplotlib.pyplot as plt
import numpy as np

from ..artifacts import read_json
from ..metrics import interval
from . import clusters, scenarios
from .config import protocol
from .report import PANELS, export, method_color, pooled

LEARNED = ("parametric", "hybrid")
LABELS = {
    "fixed-combined": "initial configuration",
    "parametric": "parametric (6)",
    "hybrid": "hybrid (529)",
    "gnn": "GNN",
    "gnn-central": "central GNN",
    "kmeans": "K-means (true K)",
    "ward": "Ward (true K)",
}


def label(method):
    return LABELS.get(method, method)


def dots(ax, positions, stats, color, name):
    """Mean with its 95% interval; methods side by side."""
    means = [s["mean"] if s["mean"] is not None else np.nan for s in stats]
    low = [m - s["ci95"][0] if s["ci95"] else 0 for m, s in zip(means, stats, strict=True)]
    high = [s["ci95"][1] - m if s["ci95"] else 0 for m, s in zip(means, stats, strict=True)]
    ax.errorbar(positions, means, yerr=[low, high], fmt="o", ms=6, lw=1.5, capsize=3,
                color=color, label=name)  # fmt: skip


def offsets(count, width=0.6):
    return np.linspace(-width / 2, width / 2, count) if count > 1 else np.zeros(1)


def reconstruction(study, text):
    rows = read_json(study / "results.json")
    lam = f"-l{protocol('compact-cpu').main_lambda:g}"
    fig, (left, right) = plt.subplots(1, 2, figsize=(13, 4), layout="constrained")
    names = [f"{m}{lam}" for m in (*LEARNED, "gnn")]
    shift = offsets(len(names) + 1)[1:]  # the first slot holds the initial configuration
    text += [
        "## 1. Reconstruction (main study, lambda = 0.3)",
        "",
        "Paired NRMSE difference to the initial configuration (negative is better), "
        "regions in brackets.",
        "",
        "| Method | " + " | ".join(t for t, _ in PANELS) + " |",
        "|---|" + "---:|" * len(PANELS),
    ]
    for name, delta in zip(names, shift, strict=True):
        method = name.split("-l")[0]
        stats, regions, cells = [], [], []
        for _, keep in PANELS:
            selected = pooled(rows, name, keep, "nrmse")
            reference = pooled(rows, "fixed-combined", keep, "nrmse")
            stats.append(interval(selected, "nrmse", reference=reference))
            regions.append(interval(pooled(rows, name, keep, "regions"), "regions"))
            ci = stats[-1]["ci95"]
            cells.append(
                f"{stats[-1]['mean']:+.3f}"
                + (f" [{ci[0]:+.3f}, {ci[1]:+.3f}]" if ci else "")
                + f" ({regions[-1]['mean']:.0f})"
            )
        x = np.arange(len(PANELS)) + delta
        dots(left, x, stats, method_color(method), label(method))
        dots(right, x, regions, method_color(method), label(method))
        text.append(f"| {label(method)} | " + " | ".join(cells) + " |")
    reference = [
        interval(pooled(rows, "fixed-combined", keep, "regions"), "regions") for _, keep in PANELS
    ]
    dots(right, np.arange(len(PANELS)) + offsets(len(names) + 1)[0], reference,
         method_color("fixed-combined"),
         label("fixed-combined"))  # fmt: skip
    left.axhline(0, color="black", lw=0.8)
    left.set(ylabel="NRMSE - initial configuration", title="Reconstruction error (paired)")
    right.set(ylabel="mean regions (samplers)", yscale="log", title="Samplers used")
    for ax in (left, right):
        ax.set_xticks(range(len(PANELS)), [t for t, _ in PANELS])
        ax.grid(alpha=0.15, axis="y")
        ax.legend(fontsize=8, frameon=False)
    export(fig, study.parent, "compare-reconstruction")
    text += ["", "![](figures/compare-reconstruction.png)", ""]


def phenomenon(study, text):
    methods = ("fixed-combined", *LEARNED)
    cases = list(itertools.product(scenarios.SCENARIOS, scenarios.SCENARIOS))
    fig, (left, right) = plt.subplots(1, 2, figsize=(13, 4), layout="constrained")
    text += [
        "## 2. Static and moving phenomenon",
        "",
        "Hard objective on clean batched test episodes (lower is better), trained on one "
        "scenario and tested on both; recovery of the asynchronous runs after each fault.",
        "",
        "| Method | " + " | ".join(f"train {a} / test {b}" for a, b in cases) + " |",
        "|---|" + "---:|" * len(cases),
    ]
    for method, delta in zip(methods, offsets(len(methods)), strict=True):
        stats = []
        for trained, tested in cases:
            rows = scenarios.select(
                scenarios.rows_of(study, trained), method, panel="cross", test=tested
            )
            stats.append(interval(rows, "objective"))
        dots(left, np.arange(len(cases)) + delta, stats, method_color(method), label(method))
        text.append(f"| {label(method)} | " + " | ".join(f"{s['mean']:.3f}" for s in stats) + " |")
    left.set_xticks(range(len(cases)), [f"train {a}\ntest {b}" for a, b in cases])
    left.set(ylabel="objective", title="Objective (clean, batched)")
    faults = [c for c in scenarios.CONDITIONS if c != "clean"]
    columns = list(itertools.product(scenarios.SCENARIOS, faults))
    for method, delta in zip(methods, offsets(len(methods)), strict=True):
        shares = []
        for scenario, condition in columns:
            selected = scenarios.select(
                scenarios.rows_of(study, scenario), method, panel="faults", condition=condition
            )
            events = [r["metrics"]["fault_recovery"]["rounds"] for r in selected]
            shares.append(np.mean([e is not None for e in events]) if events else np.nan)
        right.bar(np.arange(len(columns)) + delta, shares, width=0.6 / len(methods),
                  color=method_color(method), label=label(method))  # fmt: skip
    right.set_xticks(range(len(columns)), [f"{s}\n{c}" for s, c in columns], fontsize=7)
    right.set(ylabel="share of episodes recovered", ylim=(0, 1.05),
              title="Recovery after faults (async devices)")  # fmt: skip
    for ax in (left, right):
        ax.grid(alpha=0.15, axis="y")
        ax.legend(fontsize=8, frameon=False)
    export(fig, study.parent, "compare-phenomenon")
    shutil.copy(
        study / "figures" / "scenarios-stabilization.png",
        study.parent / "figures" / "compare-stabilization.png",
    )
    text += [
        "",
        "![](figures/compare-phenomenon.png)",
        "",
        "Per-round state with a frozen and a moving phenomenon (clean episodes): the static "
        "rows settle to a fixed partition with as many leaders as regions.",
        "",
        "![](figures/compare-stabilization.png)",
        "",
    ]


def clustering(study, text):
    panels = [("static", "batched", "clean"), ("moving", "batched", "clean")] + [
        ("distributed", "async-0.5", c) for c in clusters.CONDITIONS
    ]
    titles = ["test static", "test moving", *(f"async {c}" for c in clusters.CONDITIONS)]
    fig, axes = plt.subplots(2, 1, figsize=(13, 7), layout="constrained", sharex=True)
    text += [
        "## 3. Clustering of the phenomenon into contiguous zones (ARI, higher is better)",
        "",
        "Every learned method minimises the same objective from the phenomenon "
        "(region-mean error + 0.3 x regions / N, no labels); the central GNN sees every "
        "device each round; K-means and Ward receive the true K.",
        "",
    ]
    for ax, trained in zip(axes, clusters.SCENARIOS, strict=True):
        rows = [read_json(p) for p in sorted((study / trained / "episodes").glob("*/seed*/*.json"))]
        methods = clusters.methods()
        text += [
            f"Trained on {trained} zones:",
            "",
            "| Method | " + " | ".join(titles) + " |",
            "|---|" + "---:|" * len(titles),
        ]
        for method, delta in zip(methods, offsets(len(methods), 0.75), strict=True):
            stats = []
            for panel, executor, condition in panels:
                selected = [
                    r
                    for r in rows
                    if r["method"] == method
                    and r["panel"] == panel
                    and r["executor"] == executor
                    and r["condition"] == condition
                ]
                stats.append(
                    interval(selected, "ari_final") if selected else {"mean": None, "ci95": None}
                )
            dots(ax, np.arange(len(panels)) + delta, stats, method_color(method), label(method))
            text.append(
                f"| {label(method)} | "
                + " | ".join("—" if s["mean"] is None else f"{s['mean']:.3f}" for s in stats)
                + " |"
            )
        text.append("")
        ax.set(ylabel="final ARI", title=f"Trained on {trained} zones", ylim=(0.4, 1.02))
        ax.grid(alpha=0.15, axis="y")
    axes[0].legend(fontsize=8, frameon=False, ncol=4)
    axes[-1].set_xticks(range(len(panels)), titles)
    export(fig, study.parent, "compare-clustering")
    text += ["![](figures/compare-clustering.png)", ""]


def matching(selection, method, spec):
    name, executor, condition = spec
    return [
        r
        for r in selection
        if r["method"] == method
        and r["panel"] == name
        and r["executor"] == executor
        and r["condition"] == condition
    ]


def cluster_counts(study, text):
    """Final regions and leaders against the true number of zones, and their rounds."""
    panels = [("static", "batched", "clean"), ("moving", "batched", "clean")] + [
        ("distributed", "async-0.5", c) for c in clusters.CONDITIONS
    ]
    titles = ["test static", "test moving", *(f"async {c}" for c in clusters.CONDITIONS)]
    methods = ("fixed-combined", *LEARNED, "gnn-central")
    rows = {
        trained: [
            read_json(p) for p in sorted((study / trained / "episodes").glob("*/seed*/*.json"))
        ]
        for trained in clusters.SCENARIOS
    }
    truth = rows["static"][0]["count"]

    text += [
        f"### Number of clusters: regions (elected leaders) against the true {truth} zones",
        "",
    ]
    fig, axes = plt.subplots(2, 2, figsize=(14, 7), layout="constrained", sharex=True)
    for row, trained in zip(axes, clusters.SCENARIOS, strict=True):
        text += [
            f"Trained on {trained} zones:",
            "",
            "| Method | " + " | ".join(titles) + " |",
            "|---|" + "---:|" * len(titles),
        ]
        for method, delta in zip(methods, offsets(len(methods)), strict=True):
            stats = {key: [] for key in ("clusters_final", "leaders_final")}
            for spec in panels:
                selected = matching(rows[trained], method, spec)
                for key, values in stats.items():
                    usable = [r for r in selected if key in r["metrics"]]
                    values.append(interval(usable, key) if usable else {"mean": None, "ci95": None})
            x = np.arange(len(panels)) + delta
            for ax, key in zip(row, stats, strict=True):
                dots(ax, x, stats[key], method_color(method), label(method))
            text.append(
                f"| {label(method)} | "
                + " | ".join(
                    "—"
                    if r["mean"] is None
                    else f"{r['mean']:.1f}" + ("" if q["mean"] is None else f" ({q['mean']:.1f})")
                    for r, q in zip(stats["clusters_final"], stats["leaders_final"], strict=True)
                )
                + " |"
            )
        text.append("")
        for ax, title in zip(row, ("regions", "elected leaders"), strict=True):
            ax.axhline(truth, color="black", ls="--", lw=1, label=f"true zones ({truth})")
            ax.set(title=f"Trained on {trained} zones: final {title}", ylabel=title)
            ax.grid(alpha=0.15, axis="y")
    axes[0, 0].legend(fontsize=8, frameon=False, ncol=2)
    for ax in axes[-1]:
        ax.set_xticks(range(len(panels)), titles, rotation=20)
    export(fig, study.parent, "compare-cluster-count")

    cluster_rounds(study, rows, methods, truth)
    text += [
        "Dashed: the true number of zones. A region whose leader is no longer elected makes "
        "regions exceed leaders; the central GNN has no leaders.",
        "",
        "![](figures/compare-cluster-count.png)",
        "",
        "Round by round on clean episodes, trained and tested on the same zones (at round "
        "0 every device leads itself; the axis is cut at four times the true count):",
        "",
        "![](figures/compare-cluster-rounds.png)",
        "",
    ]


def cluster_rounds(study, rows, methods, truth):
    cases = [
        ("static", ("static", "batched", "clean"), "test static, batched"),
        ("moving", ("moving", "batched", "clean"), "test moving, batched"),
        ("static", ("distributed", "async-0.5", "clean"), "static zones, async devices"),
        ("moving", ("distributed", "async-0.5", "clean"), "moving zones, async devices"),
    ]
    columns = (("ari", "ARI"), ("clusters", "regions"), ("leaders", "elected leaders"))
    fig, axes = plt.subplots(len(cases), len(columns), figsize=(13, 10), layout="constrained",
                             sharex=True)  # fmt: skip
    for index, (axrow, (trained, spec, title)) in enumerate(zip(axes, cases, strict=True)):
        for method in methods:
            selected = matching(rows[trained], method, spec)
            for ax, (key, _) in zip(axrow, columns, strict=True):
                curves = [r["curves"][key] for r in selected if key in r["curves"]]
                if curves:
                    ax.plot(np.mean(curves, 0), color=method_color(method), label=label(method))
        for ax, (key, name) in zip(axrow, columns, strict=True):
            if key != "ari":
                ax.axhline(truth, color="black", ls="--", lw=1)
            ax.set(title=name if index == 0 else None)
            ax.grid(alpha=0.15)
        axrow[0].set(ylabel=title)
        axrow[1].set(ylim=(0, 4 * truth))  # round 0: every device leads itself
        axrow[2].set(ylim=(0, 4 * truth))
    axes[0, 0].legend(fontsize=8, frameon=False)
    for ax in axes[-1]:
        ax.set(xlabel="Round")
    export(fig, study.parent, "compare-cluster-rounds")


def training(out, text):
    """Hard validation objective over training, every learned method of every study."""
    sources = [
        ("reconstruction", out / "main" / "checkpoints", lambda n: n.endswith("-l0.3")),
        ("static phenomenon", out / "scenarios" / "static" / "checkpoints", None),
        ("moving phenomenon", out / "scenarios" / "moving" / "checkpoints", None),
        ("clustering, static", out / "clusters" / "static" / "checkpoints", None),
        ("clustering, moving", out / "clusters" / "moving" / "checkpoints", None),
    ]
    fig, axes = plt.subplots(1, len(sources), figsize=(4 * len(sources), 3.4),
                             layout="constrained")  # fmt: skip
    for ax, (title, root, keep) in zip(axes, sources, strict=True):
        for directory in sorted(root.iterdir()):
            if keep is not None and not keep(directory.name):
                continue
            method = directory.name.split("-l")[0]
            histories = [
                [h for h in read_json(p)["history"] if "validation_hard" in h]
                for p in sorted(directory.glob("seed*/training.json"))
            ]
            steps = [h["step"] for h in histories[0]]
            values = np.array([[h["validation_hard"] for h in hist] for hist in histories])
            color = method_color(method)
            ax.fill_between(steps, values.min(0), values.max(0), color=color, alpha=0.15, lw=0)
            ax.plot(steps, values.mean(0), color=color, lw=2, label=label(method))
        ax.set(title=title, xlabel="Adam update", yscale="log")
        ax.grid(alpha=0.15)
        ax.legend(fontsize=7, frameon=False)
    axes[0].set(ylabel="hard validation objective")
    export(fig, out, "compare-training")
    text += [
        "## 4. Training",
        "",
        "Hard validation objective every 20 full-batch updates (mean over seeds, band "
        "min-max); the selected checkpoint is the minimum.",
        "",
        "![](figures/compare-training.png)",
        "",
    ]


def summary(out):
    text = [
        "# SEAMS comparison: learned SCR programs against fixed and neural references",
        "",
        "Three campaigns on the same code: [reconstruction](main/REPORT.md), "
        "[static and moving phenomenon](scenarios/REPORT.md), "
        "[clustering](clusters/REPORT.md). Intervals: two-way bootstrap over seeds "
        "and episodes. With full-batch training the parametric program is deterministic, "
        "so its seeds coincide and its intervals reflect episodes only.",
        "",
    ]
    reconstruction(out / "main", text)
    phenomenon(out / "scenarios", text)
    clustering(out / "clusters", text)
    cluster_counts(out / "clusters", text)
    training(out, text)
    (out / "SUMMARY.md").write_text("\n".join(text))
    print(f"Summary: {out / 'SUMMARY.md'}", flush=True)
    return 0
