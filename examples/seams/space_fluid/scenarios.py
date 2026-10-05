"""Does learning improve the program's initial configuration, wherever it was learned?

Two training scenarios: a frozen phenomenon and a moving one. In each, Adam tunes the
six weights of the SCR program (metric and candidacy strength), starting from the
fixed-combined configuration, and a recurrent GNN is trained as the all-neural
reference. Every model is tested on both scenarios (batched, clean) and, in its own
scenario, on independent devices (asynchronous DeviceRuntime) under link loss, node
stops, a partition and permanent crashes. Objective: reconstruction error + lambda x
leader fraction, no hotspot weighting.
"""

import itertools
from dataclasses import replace

import matplotlib.pyplot as plt
import numpy as np
import torch

from ..artifacts import Budget, json_write, read_json
from ..metrics import interval
from ..randomness import seed_for
from .campaign import output_lock
from .config import TEST_FAMILIES, protocol
from .data import make_episode, normalization, training_bank
from .execution import central, decentralized, equivalence
from .metrics import summarize
from .program import make_program
from .report import export, method_color
from .training import load_program, objective, train_job

SCENARIOS = ("static", "moving")
LEARNERS = ("parametric", "gnn")
METHODS = ("fixed-combined", *LEARNERS)
CONDITIONS = ("clean", "link_loss", "node_stop", "partition", "crash")
FAULT_FAMILIES = ("gaussian", "ring")  # one training family, one held out
FAULT_EPISODES = 2
# Adam rate for the six weights: best mean hard validation among 0.005/0.03/0.1 in a
# seed-0 pilot over both scenarios. The GNN keeps its main-study rate.
RATE = 0.1
NAMES = {"fixed-combined": "initial configuration", "parametric": "Adam", "gnn": "GNN"}


def banks(config):
    """Training bank and its normalization per scenario."""
    result = {}
    for scenario in SCENARIOS:
        bank = training_bank(config, static=scenario == "static")
        result[scenario] = (bank, normalization(bank["train"]))
    return result


def train(out, config, scenario, bank, norm, *, budget, methods=LEARNERS, seeds=None):
    for method, seed in itertools.product(methods, seeds or config.seeds):
        directory = out / scenario / "checkpoints" / method / f"seed{seed}"
        if not (directory / "best.pt").exists():
            print(f"train {scenario} {method} seed={seed}", flush=True)
            train_job(
                directory,
                replace(config, learning_rate=RATE),  # the GNN reads gnn_learning_rate
                bank,
                norm,
                method,
                seed,
                config.main_lambda,
                budget,
            )


def specs(config, scenario):
    """Both test scenarios (batched, clean), faults in the training scenario, one sync check."""
    for test, family, index in itertools.product(
        SCENARIOS, TEST_FAMILIES, range(config.test_episodes)
    ):
        yield {"panel": "cross", "test": test, "family": family, "index": index,
               "condition": "clean", "executor": "batched"}  # fmt: skip
    for condition, family, index in itertools.product(
        CONDITIONS, FAULT_FAMILIES, range(FAULT_EPISODES)
    ):
        yield {"panel": "faults", "test": scenario, "family": family, "index": index,
               "condition": condition, "executor": "async-0.5"}  # fmt: skip
    yield {"panel": "equivalence", "test": scenario, "family": "gaussian", "index": 0,
           "condition": "partition", "executor": "sync"}  # fmt: skip


def spec_name(spec):
    return "-".join(str(spec[k]) for k in ("panel", "test", "family", "index", "condition"))


@torch.no_grad()
def evaluate(out, config, scenario, norms, budget, *, methods=METHODS, seeds=None):
    """``norms`` per scenario: errors use the test scenario's scale, so a column compares."""
    models = {}
    for spec, method, seed in itertools.product(
        specs(config, scenario), methods, seeds or config.seeds
    ):
        deterministic = method == "fixed-combined" or spec["executor"] == "sync"
        if deterministic and seed != config.seeds[0]:
            continue
        path = out / scenario / "episodes" / method / f"seed{seed}" / f"{spec_name(spec)}.json"
        if path.exists():
            continue
        budget.check()
        print(f"evaluate {scenario} {spec_name(spec)} {method} seed={seed}", flush=True)
        if (method, seed) not in models:
            models[method, seed] = (
                make_program("combined", **norms[scenario])
                if method == "fixed-combined"
                else load_program(
                    out / scenario / "checkpoints" / method / f"seed{seed}" / "best.pt"
                )
            )
        model = models[method, seed]
        episode = make_episode(
            config, "test", spec["family"], spec["index"], condition=spec["condition"],
            static=spec["test"] == "static",
        )  # fmt: skip
        check = None
        if spec["executor"] == "batched":
            trace = central(episode, model)
        else:
            trace = decentralized(
                episode,
                model,
                activation_probability=1.0 if spec["executor"] == "sync" else 0.5,
                seed=seed_for(config.data_seed, episode.key, "schedule"),
            )
            if spec["executor"] == "sync":
                check = equivalence(central(episode, model), trace)
                if not (check["identifiers_equal"] and check["values_close"]):
                    raise AssertionError(f"Synchronous mismatch: {check}")
        scale = norms[spec["test"]]["scale"]
        metrics, curves = summarize(episode, trace, scale, config)
        metrics["objective"] = float(objective(episode, trace, scale, config.main_lambda))
        json_write(
            path,
            {
                **spec,
                "method": method,
                "seed": seed,
                "episode": spec_name(spec),
                "metrics": metrics,
                "curves": {k: curves[k] for k in ("nrmse", "leader_fraction", "regions")},
                "equivalence": check,
            },
        )


def rows_of(out, scenario):
    rows = [read_json(p) for p in sorted((out / scenario / "episodes").glob("*/seed*/*.json"))]
    seeds = sorted({r["seed"] for r in rows if r["method"] in LEARNERS})
    # The initial configuration is deterministic: pair it with every learned seed.
    fixed = [r for r in rows if r["method"] == "fixed-combined"]
    return [r for r in rows if r["method"] != "fixed-combined"] + [
        {**r, "seed": s} for r in fixed for s in seeds
    ]


def select(rows, method, **keys):
    return [r for r in rows if r["method"] == method and all(r[k] == v for k, v in keys.items())]


def cell(stats, digits=3):
    if stats["mean"] is None:
        return "—"
    ci = stats["ci95"]
    return f"{stats['mean']:.{digits}f}" + (
        f" [{ci[0]:.{digits}f}, {ci[1]:.{digits}f}]" if ci else ""
    )


def report(out, config):
    text = [
        "# Space-Fluid learned in a static and in a moving world",
        "",
        f"Profile **{config.profile}**. Objective: reconstruction error / sigma^2 + "
        f"{config.main_lambda:g} x leader fraction. Adam tunes the six weights (rate {RATE:g}) "
        "from the initial configuration (fixed-combined); the GNN (rate "
        f"{config.gnn_learning_rate:g}) is the all-neural reference: every device is its own "
        f"sampler, so its leader term is the constant {config.main_lambda:g}. Seeds "
        f"{len(config.seeds)}; test families {', '.join(TEST_FAMILIES)} (ring and front held "
        "out), errors on the test scenario's training scale. Intervals: two-way bootstrap over "
        "seeds and episodes, paired on the same episodes.",
        "",
        "## Clean, batched: trained on one scenario, tested on both",
        "",
        "| Trained on | Method | Objective static | Objective moving | NRMSE static | "
        "NRMSE moving | Leaders static | Leaders moving |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    paired_rows = []
    for scenario in SCENARIOS:
        rows = rows_of(out, scenario)
        for method in METHODS:
            cells = []
            for metric in ("objective", "nrmse", "leader_fraction"):
                for test in SCENARIOS:
                    cells.append(
                        cell(interval(select(rows, method, panel="cross", test=test), metric))
                    )
            text.append(f"| {scenario} | {NAMES[method]} | " + " | ".join(cells) + " |")
        for test in SCENARIOS:
            reference = select(rows, "fixed-combined", panel="cross", test=test)
            learned = select(rows, "parametric", panel="cross", test=test)
            stats = interval(learned, "objective", reference=reference)
            paired_rows.append(f"| {scenario} | {test} | {cell(stats, 4)} |")
    text += [
        "",
        "## Improvement over the initial configuration (Adam - initial, objective)",
        "",
        "| Trained on | Tested on | Paired difference |",
        "|---|---|---:|",
        *paired_rows,
        "",
        "## Decentralized (async DeviceRuntime, p=0.5), trained and tested on the same scenario",
        "",
        "NRMSE before the fault (5 rounds), during it, over the last 10 rounds; recovery = "
        f"rounds until NRMSE <= {config.recovery_threshold:g} for {config.recovery_rounds} rounds "
        "after the fault (after restoration for temporary faults).",
        "",
        "| Scenario | Condition | Method | Objective | NRMSE before | during | final | Recovery |",
        "|---|---|---|---:|---:|---:|---:|---:|",
    ]
    for scenario, condition in itertools.product(SCENARIOS, CONDITIONS):
        rows = rows_of(out, scenario)
        for method in METHODS:
            selected = select(rows, method, panel="faults", condition=condition)
            if not selected:
                continue
            curves = np.array([r["curves"]["nrmse"] for r in selected])
            third = curves.shape[1] // 3
            event = (
                "restoration_recovery"
                if condition in ("node_stop", "partition", "link_loss")
                else "fault_recovery"
            )
            rounds = [r["metrics"][event]["rounds"] for r in selected]
            done = [x for x in rounds if x is not None]
            recovery = (
                "—"
                if condition == "clean"
                else f"{len(done)}/{len(rounds)}"
                + (f", median {np.median(done):.0f}" if done else "")
            )
            before, during = (
                curves[:, third - 5 : third].mean(),
                curves[:, third : 2 * third].mean(),
            )
            text.append(
                f"| {scenario} | {condition} | {NAMES[method]} | "
                f"{cell(interval(selected, 'objective'))} | {before:.3f} | {during:.3f} | "
                f"{curves[:, -10:].mean():.3f} | {recovery} |"
            )
    checks = [
        read_json(path)["equivalence"]
        for path in sorted(out.glob("*/episodes/*/seed*/equivalence-*.json"))
    ]
    passed = sum(c["identifiers_equal"] and c["values_close"] for c in checks)
    text += [
        "",
        f"Synchronous DeviceRuntime equals batched execution (partition episode, every method, "
        f"both scenarios): {passed}/{len(checks)}.",
        "",
    ]
    (out / "REPORT.md").write_text("\n".join(text))
    figures(out, config)


def figures(out, config):
    # Loss over time: hard validation objective, error and leaders, both learners.
    fig, axes = plt.subplots(2, 3, figsize=(13, 6.5), layout="constrained", sharex=True)
    for row, scenario in zip(axes, SCENARIOS, strict=True):
        for method, seed in itertools.product(LEARNERS, config.seeds):
            path = out / scenario / "checkpoints" / method / f"seed{seed}" / "training.json"
            if not path.exists():
                continue
            history = [h for h in read_json(path)["history"] if "validation_hard" in h]
            steps = [h["step"] for h in history]
            for ax, key in zip(
                row, ("validation_hard", "validation_error", "validation_leaders"), strict=True
            ):
                ax.plot(steps, [h[key] for h in history], color=method_color(method), alpha=0.75,
                        label=NAMES[method] if seed == config.seeds[0] else None)  # fmt: skip
        for ax, title in zip(row, ("objective", "error / sigma^2", "leader fraction"), strict=True):
            ax.set(title=f"{scenario}: {title}", yscale="log")
            ax.grid(alpha=0.15)
        row[0].legend(fontsize=8)
    for ax in axes[-1]:
        ax.set(xlabel="Adam update (hard validation every 20)")
    export(fig, out, "scenarios-loss")

    # NRMSE over rounds under each fault, in the training scenario.
    fig, axes = plt.subplots(
        2, len(CONDITIONS), figsize=(3.2 * len(CONDITIONS), 6), layout="constrained", sharey=True
    )
    for row, scenario in zip(axes, SCENARIOS, strict=True):
        rows = rows_of(out, scenario)
        for ax, condition in zip(row, CONDITIONS, strict=True):
            for method in METHODS:
                curves = [
                    r["curves"]["nrmse"]
                    for r in select(rows, method, panel="faults", condition=condition)
                ]
                if curves:
                    ax.plot(np.mean(curves, 0), color=method_color(method), label=NAMES[method])
            rounds = config.eval_rounds
            end = rounds if condition == "crash" else 2 * rounds // 3
            if condition != "clean":
                ax.axvspan(rounds // 3, end, alpha=0.06, color="black")
            ax.set(title=f"{scenario}: {condition}", xlabel="Round", ylabel="NRMSE", yscale="log")
            ax.grid(alpha=0.15)
        row[0].legend(fontsize=7)
    export(fig, out, "scenarios-faults")


def campaign(out, profile, seconds):
    config = protocol(profile)
    with output_lock(out):
        torch.set_num_threads(config.threads)
        torch.use_deterministic_algorithms(True)
        budget = Budget(seconds)
        prepared = banks(config)
        norms = {s: norm for s, (_, norm) in prepared.items()}
        json_write(
            out / "manifest.json",
            {"profile": profile, "rate": RATE, "normalization": norms, "config": config.to_dict()},
        )
        try:
            for scenario, (bank, norm) in prepared.items():
                train(out, config, scenario, bank, norm, budget=budget)
            for scenario in SCENARIOS:
                evaluate(out, config, scenario, norms, budget)
        except TimeoutError:
            print(f"Incomplete, resumable: {out}", flush=True)
            return 2
        report(out, config)
    print(f"Artifacts: {out}", flush=True)
    return 0
