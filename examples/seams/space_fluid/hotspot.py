"""Space-Fluid with a hotspot requirement: where gradients beat black-box search.

The reconstruction task of the main study with an alarm requirement: errors where
the truth exceeds THRESHOLD weigh up to 1 + gain times more (gain 0 is the main
study). The best program then wants fine regions on hotspots and coarse ones
elsewhere, which one global metric cannot express. The metric becomes a
piecewise-linear curve of the level with K knots (K = 1 is the main-study program,
6 parameters). Adam, random search and CEM tune the same SCR program with the same
measured compute; the neurosymbolic program and the GNN are the high-dimensional
ends.
"""

import itertools
from dataclasses import replace
from typing import NamedTuple

import matplotlib.pyplot as plt
import numpy as np
import torch

from ..artifacts import Budget, json_write, read_json
from ..metrics import interval
from ..randomness import seed_for
from .campaign import output_lock
from .config import TEST_FAMILIES, protocol
from .data import combine, make_episode, normalization, training_bank
from .execution import central, decentralized, equivalence
from .program import ELECTED, SAMPLE, make_program
from .report import export, method_color
from .training import load_program, objective, surrogate_objective, train_job


class Study(NamedTuple):
    gain: float  # extra weight of errors above THRESHOLD
    knots: tuple[int, ...]

    @property
    def name(self):
        return f"gain{self.gain:g}"


THRESHOLD, SHARPNESS = 0.5, 0.05  # alarm level, raw units
STUDIES = {
    "compact-cpu": [Study(g, (1, 2, 4, 8, 16, 32, 64)) for g in (0.0, 3.0, 9.0, 27.0)],
    "smoke": [Study(9.0, (1, 4))],
}
SEEDS = (0, 1)
LEARNERS = ("parametric", "search", "cem")
# Optimizer settings chosen on hard validation in a seed-0 pilot (gain 9, K = 1 and 16):
# Adam rate among 0.005/0.03/0.1 per method; a search box wide enough to hold the
# pilot Adam solutions (range weight 0.01-0.04 costs < 0.004 per link at radius 1);
# 528 candidates take the wall-clock of 400 Adam updates (~460 s against ~480 s).
# The GNN keeps its main-study rate.
RATES = {"parametric": 0.1, "neural": 0.03}
SEARCH_BOX = (4.0, 6.0)
CANDIDATES = 528
STYLE = {"parametric": "-o", "search": "--s", "cem": ":^"}
CURVE_KNOTS = 16


def importance(truth, gain):
    return 1 + gain * ((truth - THRESHOLD) / SHARPNESS).sigmoid()


def weighted(episode, gain):
    episode.importance = importance(episode.truth, gain)
    return episode


def runs(study):
    """(method, knots); neural and GNN at one knot, trained by Adam only."""
    return [*itertools.product(LEARNERS, study.knots), ("neural", 1), ("gnn", 1)]


def label(method, knots):
    return method if method in ("neural", "gnn") else f"{method}-k{knots}"


def dimension(model):
    """Tuned scalars: metric curve, strength weights, neural modulators (or the GNN)."""
    if not hasattr(model, "metric"):
        return sum(p.numel() for p in model.parameters())
    networks = sum(p.numel() for n, p in model.named_parameters() if "network" in n)
    return len(model.metric.weights) + len(model.strength.weights) + networks


def knot_levels(norm, knots):
    return norm["mean"] + norm["scale"] * torch.linspace(-1, 3, knots)


def settings(profile):
    config = replace(protocol(profile), search_box=SEARCH_BOX)
    if profile == "smoke":
        return config
    return replace(config, search_candidates=CANDIDATES, seeds=SEEDS)


def training(config, gain):
    bank = {s: [weighted(e, gain) for e in eps] for s, eps in training_bank(config).items()}
    return bank, normalization(bank["train"])


def test_episodes(config, gain):
    return [
        weighted(make_episode(config, "test", family, index), gain)
        for family in TEST_FAMILIES
        for index in range(config.test_episodes)
    ]


def train_one(out, config, bank, norm, method, knots, seed, budget):  # noqa: PLR0917
    directory = out / "checkpoints" / label(method, knots) / f"seed{seed}"
    if not (directory / "best.pt").exists():
        print(f"train {out.name} {label(method, knots)} seed={seed}", flush=True)
        config = replace(config, learning_rate=RATES.get(method, config.learning_rate))
        train_job(
            directory, config, bank, norm, method, seed, config.main_lambda, budget, knots=knots
        )


def episode_metrics(episode, trace, scale, penalty):
    error = trace.fields[..., SAMPLE] - episode.truth
    hot = episode.truth > THRESHOLD

    def nrmse(mask):
        mask = mask & episode.active
        return float(error[mask].square().mean().sqrt() / scale) if bool(mask.any()) else None

    return {
        "objective": float(objective(episode, trace, scale, penalty)),
        "hot_nrmse": nrmse(hot),
        "cold_nrmse": nrmse(~hot),
        "leaders": float(trace.fields[..., ELECTED][episode.active].mean()),
    }


@torch.no_grad()
def evaluate(out, config, study, norm, budget):
    episodes = test_episodes(config, study.gain)
    jobs = [(label(m, k), s) for (m, k), s in itertools.product(runs(study), config.seeds)]
    for name, seed in [*jobs, ("fixed-combined", config.seeds[0])]:
        path = out / "episodes" / name / f"seed{seed}.json"
        if path.exists():
            continue
        budget.check()
        print(f"evaluate {out.name} {name} seed={seed}", flush=True)
        if name == "fixed-combined":
            model, seconds = make_program("combined", **norm), 0.0
        else:
            directory = out / "checkpoints" / name / f"seed{seed}"
            model = load_program(directory / "best.pt")
            seconds = read_json(directory / "training.json")["seconds"]
        rows = [
            {
                "seed": seed,
                "episode": episode.key,
                "metrics": episode_metrics(
                    episode, central(episode, model), norm["scale"], config.main_lambda
                ),
            }
            for episode in episodes
        ]
        check = None
        if name in (f"parametric-k{study.knots[-1]}", "gnn") and seed == config.seeds[0]:
            episode = episodes[0]
            check = equivalence(
                central(episode, model),
                decentralized(episode, model, seed=seed_for(config.data_seed, episode.key)),
            )
            if not (check["identifiers_equal"] and check["values_close"]):
                raise AssertionError(f"Synchronous mismatch: {check}")
        json_write(
            path,
            {
                "method": name,
                "seed": seed,
                "parameters": dimension(model),
                "seconds": seconds,
                "rows": rows,
                "equivalence": check,
            },
        )


def gradient_at_start(config, bank, norm, knots):
    """Training gradient per knot at the common start, on validation batches."""
    model = make_program("parametric", **norm, knots=knots)
    episodes = bank["validation"]
    for offset in range(0, len(episodes), config.batch_size):
        episode = combine(episodes[offset : offset + config.batch_size])
        surrogate_objective(episode, model, norm["scale"], config.main_lambda, config).backward()
    return model.metric.log_weights.grad.view(knots, 3) / -len(episodes)  # descent direction


@torch.no_grad()
def leader_density(model, episodes, bins):
    """Fraction of elected devices per truth level, second half of each episode."""
    elected, levels = [], []
    for episode in episodes:
        trace = central(episode, model)
        late = slice(episode.rounds // 2, None)
        active = episode.active[late]
        elected.append(trace.fields[late, :, ELECTED][active])
        levels.append(episode.truth[late][active])
    elected, levels = torch.cat(elected), torch.cat(levels)
    index = torch.bucketize(levels, bins)
    return [
        float(elected[index == i].mean()) if bool((index == i).any()) else None
        for i in range(1, len(bins))
    ]


def order(study):
    """Report order: learners by knots, then the neural ends and the fixed program."""
    return [label(m, k) for m, k in runs(study)] + ["fixed-combined"]


def summary(out, study):
    files = [read_json(p) for p in sorted((out / "episodes").glob("*/seed*.json"))]
    rows = {f["method"]: [] for f in files}
    table = {}
    for name in [n for n in order(study) if n in rows]:
        selected = [f for f in files if f["method"] == name]
        rows[name] = [r for f in selected for r in f["rows"]]
        table[name] = {
            "parameters": selected[0]["parameters"],
            "seconds": float(np.mean([f["seconds"] for f in selected])),
            "seeds": len(selected),
            **{
                m: interval(rows[name], m)
                for m in ("objective", "hot_nrmse", "cold_nrmse", "leaders")
            },
        }
    first, last = study.knots[0], study.knots[-1]
    pairs = [(f"{m}-k{k}", f"parametric-k{k}") for k in study.knots for m in LEARNERS[1:]]
    pairs += [(f"parametric-k{k}", f"parametric-k{first}") for k in study.knots[1:]]
    pairs += [(name, f"parametric-k{first}") for name in ("neural", "gnn")]
    pairs += [(f"{m}-k{last}", f"{m}-k{first}") for m in LEARNERS[1:]]
    paired = {
        f"{a} - {b}": interval(rows[a], "objective", reference=rows[b])
        for a, b in pairs
        if rows.get(a) and rows.get(b)
    }
    checks = [f["equivalence"] for f in files if f["equivalence"]]
    return table, paired, checks


def value(stats):
    return stats["mean"] if stats["mean"] is not None else stats["finite_mean"]


def report(out, config, study, bank, norm):
    table, paired, checks = summary(out, study)
    if not table:
        return
    json_write(out / "summary.json", {"table": table, "paired": paired, "equivalence": checks})

    def cell(stats, digits=4):
        ci = stats["ci95"]
        text = f"{value(stats):.{digits}f}" if value(stats) is not None else "—"
        return text + (f" [{ci[0]:.{digits}f}, {ci[1]:.{digits}f}]" if ci else "")

    text = [
        f"# Space-Fluid hotspot, gain {study.gain:g}",
        "",
        f"Profile **{config.profile}**. Reconstruction + {config.main_lambda:g} x leaders, with "
        f"errors weighted 1 + {study.gain:g} * sigmoid((truth - {THRESHOLD:g}) / {SHARPNESS:g}). "
        "The metric is a piecewise-linear curve of the level with K knots (K = 1: the "
        f"main-study program). Adam trains for {config.updates} updates (rate "
        f"{RATES['parametric']:g}, neural {RATES['neural']:g}, GNN "
        f"{config.gnn_learning_rate:g}); random search and CEM (population 24, 6 elites) "
        f"evaluate {config.search_candidates} candidates on the hard validation objective in a "
        f"box of +-{config.search_box[0]:g} log metric weights and +-{config.search_box[1]:g} "
        "strength, about the same wall-clock. The GNN is its own sampler on every device: its "
        f"leader term is the constant {config.main_lambda:g}. Test: held-out families, clean. "
        "Intervals: two-way bootstrap over seeds and episodes.",
        "",
        "| Run | Parameters | Objective | NRMSE above | NRMSE below | Leaders | Seconds |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for name, row in table.items():
        metrics = " | ".join(
            cell(row[m]) for m in ("objective", "hot_nrmse", "cold_nrmse", "leaders")
        )
        text.append(f"| {name} | {row['parameters']} | {metrics} | {row['seconds']:.0f} |")
    text += [
        "",
        "Seconds were measured with 6 to 20 runs in parallel on 24 cores; equal compute "
        "is set by the budgets (400 updates, 528 candidates), timed in the seed-0 pilot.",
        "",
        "## Paired differences of the test objective",
        "",
        "Same seeds and episodes; negative means the first run is better.",
        "",
        "| Difference | Mean | 95% CI |",
        "|---|---:|---|",
    ]
    for pair, stats in paired.items():
        ci = stats["ci95"]
        interval_text = f"[{ci[0]:+.4f}, {ci[1]:+.4f}]" if ci else stats["ci_reason"]
        text.append(f"| {pair} | {stats['finite_mean']:+.4f} | {interval_text} |")
    passed = sum(c["identifiers_equal"] and c["values_close"] for c in checks)
    text += ["", f"Synchronous central/device equivalence: {passed}/{len(checks)} passed.", ""]
    (out / "REPORT.md").write_text("\n".join(text))
    fig, ax = plt.subplots(figsize=(6, 3.8), layout="constrained")
    scaling(ax, table, study)
    export(fig, out, "hotspot-scaling")
    figures(out, config, study, bank, norm)


def scaling(ax, table, study):
    """Hard test objective against the number of tuned scalars."""
    for method in LEARNERS:
        names = [f"{method}-k{k}" for k in study.knots if f"{method}-k{k}" in table]
        if not names:
            continue
        y = np.array([value(table[n]["objective"]) for n in names])
        ci = np.array(
            [table[n]["objective"]["ci95"] or [v, v] for n, v in zip(names, y, strict=True)]
        )
        ax.errorbar(
            [table[n]["parameters"] for n in names],
            y,
            yerr=np.abs(ci.T - y),
            fmt=STYLE[method],
            color=method_color(method),
            capsize=3,
            label="Adam (parametric)" if method == "parametric" else method,
        )
    for name, marker in (("neural", "*"), ("gnn", "P")):
        if name in table:
            ax.scatter(
                table[name]["parameters"],
                value(table[name]["objective"]),
                marker=marker,
                s=140,
                color=method_color(name),
                label=f"Adam ({name})",
                zorder=3,
            )
    if "fixed-combined" in table:
        ax.axhline(
            value(table["fixed-combined"]["objective"]),
            color=method_color("fixed-combined"),
            ls=":",
            label="fixed combined",
        )
    ax.set(xscale="log", xlabel="Tuned scalars", ylabel="Hard weighted test objective")
    ax.set_title(f"gain {study.gain:g}")
    ax.grid(alpha=0.15)
    ax.legend(fontsize=7)


def figures(out, config, study, bank, norm):
    # Learned curves and the descent direction at the common start.
    knots = CURVE_KNOTS if CURVE_KNOTS in study.knots else study.knots[-1]
    levels = knot_levels(norm, knots).numpy()
    start = gradient_at_start(config, bank, norm, knots).numpy()
    titles = ("range weight", "difference weight", "range x difference weight")
    fig, axes = plt.subplots(2, 3, figsize=(12, 6), layout="constrained", sharex=True)
    for column, title in enumerate(titles):
        top, bottom = axes[0, column], axes[1, column]
        for method, seed in itertools.product(LEARNERS, config.seeds):
            path = out / "checkpoints" / f"{method}-k{knots}" / f"seed{seed}" / "best.pt"
            if not path.exists():
                continue
            weights = load_program(path).metric.weights.view(knots, 3)[:, column]
            top.plot(
                levels,
                weights.numpy(),
                STYLE[method],
                color=method_color(method),
                alpha=0.7,
                ms=3,
                label=method if seed == config.seeds[0] else None,
            )
        top.set(title=f"learned {title} (K={knots})", yscale="log")
        width = 0.8 * (levels[1] - levels[0]) if knots > 1 else 0.05
        bottom.bar(levels, start[:, column], width=width, color="C0")
        bottom.axhline(0, color="black", lw=0.5)
        bottom.set(title=f"-dL/d log w at the start: {title}", xlabel="level (raw units)")
        for ax in (top, bottom):
            ax.axvline(THRESHOLD, color="red", ls=":", lw=1)
    axes[0, 0].legend(fontsize=8)
    export(fig, out, "hotspot-curves")

    # Where samplers go: elected fraction per truth level, K = 1 against the curve K.
    bins = torch.tensor([0.0, 0.15, 0.25, 0.35, 0.45, 0.6, 0.8, 1.0, 3.0])
    episodes = test_episodes(config, study.gain)
    fig, ax = plt.subplots(figsize=(6, 3.6), layout="constrained")
    centres = ((bins[1:] + bins[:-1]) / 2).numpy()
    centres[-1] = 1.2
    for k, style in ((study.knots[0], "--o"), (knots, "-o")):
        path = out / "checkpoints" / f"parametric-k{k}" / f"seed{config.seeds[0]}" / "best.pt"
        if path.exists():
            density = leader_density(load_program(path), episodes, bins)
            values = [np.nan if v is None else v for v in density]
            ax.plot(centres, values, style, color=method_color("parametric"), label=f"K={k}")
    ax.axvline(THRESHOLD, color="red", ls=":", lw=1)
    ax.set(xlabel="truth level", ylabel="elected fraction", yscale="log")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.15)
    export(fig, out, "hotspot-leaders")


def overview(out, studies):
    """Every gain side by side: objective per learner and K, plus the neural ends."""
    tables = {
        s: read_json(out / s.name / "summary.json")["table"]
        for s in studies
        if (out / s.name / "summary.json").exists()
    }
    if not tables:
        return
    knots = sorted({k for s in tables for k in s.knots})
    text = [
        "# Space-Fluid hotspot: tuning the program as it grows",
        "",
        f"Mean hard weighted test objective (lower is better); errors above {THRESHOLD:g} "
        "weigh 1 + gain times more. Per-gain reports with intervals and paired differences: "
        "`gain*/REPORT.md`.",
        "",
        "| Gain | Method | " + " | ".join(f"K={k}" for k in knots) + " | neural | GNN | fixed |",
        "|---|---|" + "---:|" * (len(knots) + 3),
    ]
    for study, table in tables.items():
        for method in LEARNERS:
            cells = [table.get(f"{method}-k{k}") for k in knots]
            cells = [f"{value(c['objective']):.4f}" if c else "—" for c in cells]
            ends = ["", "", ""]
            if method == "parametric":
                ends = [
                    f"{value(table[n]['objective']):.4f}" if n in table else "—"
                    for n in ("neural", "gnn", "fixed-combined")
                ]
            text.append(f"| {study.gain:g} | {method} | " + " | ".join(cells + ends) + " |")
    (out / "REPORT.md").write_text("\n".join(text) + "\n")
    fig, axes = plt.subplots(
        1, len(tables), figsize=(4.2 * len(tables), 3.8), layout="constrained", squeeze=False
    )
    for ax, (study, table) in zip(axes[0], tables.items(), strict=True):
        scaling(ax, table, study)
        if ax is not axes[0, 0]:
            ax.get_legend().remove()
    export(fig, out, "hotspot-scaling")


def campaign(out, profile, seconds):
    config, studies = settings(profile), STUDIES[profile]
    with output_lock(out):
        torch.set_num_threads(config.threads)
        torch.use_deterministic_algorithms(True)
        budget = Budget(seconds)
        json_write(
            out / "manifest.json",
            {
                "profile": profile,
                "threshold": THRESHOLD,
                "sharpness": SHARPNESS,
                "studies": [s._asdict() for s in studies],
                "learning_rates": RATES,
                "config": config.to_dict(),
            },
        )
        try:
            for study in studies:
                bank, norm = training(config, study.gain)
                for (method, knots), seed in itertools.product(runs(study), config.seeds):
                    train_one(out / study.name, config, bank, norm, method, knots, seed, budget)
                evaluate(out / study.name, config, study, norm, budget)
        except TimeoutError:
            print(f"Incomplete, resumable: {out}", flush=True)
            return 2
        for study in studies:
            report(out / study.name, config, study, *training(config, study.gain))
        overview(out, studies)
    print(f"Artifacts: {out}", flush=True)
    return 0
