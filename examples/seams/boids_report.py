"""Reconstructible Boids tables, paired intervals, scientific figures and fixed replays."""

import csv

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.animation import FFMpegWriter

from .artifacts import checksum, json_write, read_json
from .boids_dynamics import TEACHER
from .boids_learning import REGIMES, VARIANTS
from .metrics import interval

METHODS = ("oracle", "fixed", *VARIANTS)
COLORS = {
    "oracle": "#222222",
    "fixed": "#888888",
    "legacy": "#bd7b24",
    "parametric": "#16734a",
    "gnn32": "#3066be",
    "gnn64": "#9b438c",
}
LABELS = {
    "oracle": "Teacher replay",
    "fixed": "Fixed",
    "legacy": "Legacy",
    "parametric": "DIFFIELD (3)",
    "gnn32": "GNN32",
    "gnn64": "GNN64",
}


def csv_write(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = list(dict.fromkeys(k for row in rows for k in row))
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def export(fig, out, name):
    directory = out / "figures"
    directory.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    for suffix in ("png", "pdf"):
        fig.savefig(directory / f"{name}.{suffix}", dpi=180, bbox_inches="tight")
    plt.close(fig)


def complete_panel(rows, config, data):
    lookup = {(r["seed"], r["episode"]): r for r in rows}
    if len(lookup) != len(rows):
        raise ValueError("Duplicate Boids seed/episode")
    return [
        lookup.get(
            (seed, ep["key"]),
            {"seed": seed, "episode": ep["key"], "status": "missing", "metrics": {}},
        )
        for seed in config["seeds"]
        for ep in data["test"]
    ]


def figures(out, rows, summaries, paired, config):
    methods = config["methods"]
    fig, axes = plt.subplots(len(REGIMES), 2, figsize=(10, 3.5 * len(REGIMES)), squeeze=False)
    for i, regime in enumerate(REGIMES):
        for j, metric in enumerate(("position_rmse", "velocity_rmse")):
            ax = axes[i, j]
            for k, method in enumerate(methods):
                stats = next(
                    s
                    for s in summaries
                    if (s["regime"], s["method"], s["metric"]) == (regime, method, metric)
                )
                if stats["mean"] is None:
                    continue
                ax.bar(k, stats["mean"], color=COLORS[method])
                if stats["ci95"]:
                    lo, hi = stats["ci95"]
                    ax.plot([k, k], [lo, hi], color="black", lw=1.3)
            ax.set(
                xticks=range(len(methods)),
                xticklabels=[LABELS[m] for m in methods],
                ylabel=metric.replace("_", " "),
                title=regime,
            )
            ax.tick_params(axis="x", rotation=30, labelsize=8)
            ax.grid(axis="y", alpha=0.2)
    export(fig, out, "accuracy")

    fig, axes = plt.subplots(1, len(REGIMES), figsize=(7 * len(REGIMES), 4), squeeze=False)
    for ax, regime in zip(axes[0], REGIMES, strict=True):
        for method in methods:
            selected = [r for r in rows if r["regime"] == regime and r["method"] == method]
            if len(selected) != len(config["seeds"]) * config["test_episodes"] or any(
                r["status"] != "complete" for r in selected
            ):
                continue
            means, lows, highs = [], [], []
            for t in range(config["rounds"]):
                samples = [
                    {
                        "seed": r["seed"],
                        "episode": r["episode"],
                        "metrics": {"error": r["curves"]["position_rmse"][t]},
                    }
                    for r in selected
                ]
                stats = interval(samples, "error")
                means.append(stats["mean"])
                lows.append(stats["ci95"][0] if stats["ci95"] else np.nan)
                highs.append(stats["ci95"][1] if stats["ci95"] else np.nan)
            clock = np.arange(1, config["rounds"] + 1)
            ax.plot(clock, means, label=LABELS[method], color=COLORS[method])
            ax.fill_between(clock, lows, highs, color=COLORS[method], alpha=0.1)
        ax.set(title=regime, xlabel="Free-running round", ylabel="Position RMSE")
        ax.grid(alpha=0.2)
    axes[0, -1].legend(fontsize=8)
    export(fig, out, "rollout_error")

    fig, axes = plt.subplots(1, len(REGIMES), figsize=(7 * len(REGIMES), 4), squeeze=False)
    for ax, regime in zip(axes[0], REGIMES, strict=True):
        for variant in VARIANTS:
            histories = []
            for seed in config["seeds"]:
                path = out / "jobs" / regime / variant / f"seed{seed}" / "training.json"
                if path.exists():
                    histories.append(read_json(path)["history"])
            if not histories:
                continue
            updates = [h["update"] for h in histories[0] if h["validation_loss"] is not None]
            values = np.array(
                [
                    [h["validation_loss"] for h in history if h["validation_loss"] is not None]
                    for history in histories
                ]
            )
            if values.size:
                ax.plot(
                    updates,
                    np.maximum(values.mean(0), 1e-14),
                    color=COLORS[variant],
                    label=LABELS[variant],
                )
                ax.fill_between(
                    updates,
                    np.maximum(values.min(0), 1e-14),
                    np.maximum(values.max(0), 1e-14),
                    color=COLORS[variant],
                    alpha=0.13,
                )
        ax.set(title=regime, xlabel="Update", ylabel="Observed validation loss", yscale="log")
        ax.grid(alpha=0.2)
    axes[0, -1].legend(fontsize=8)
    fig.suptitle("Mean and seed range; identical observed validation objective")
    export(fig, out, "learning_curves")

    fig, axes = plt.subplots(len(REGIMES), 3, figsize=(11, 3.5 * len(REGIMES)), squeeze=False)
    for i, regime in enumerate(REGIMES):
        for j, name in enumerate(("Separation", "Alignment", "Cohesion")):
            ax = axes[i, j]
            for seed in config["seeds"]:
                path = out / "jobs" / regime / "parametric" / f"seed{seed}" / "training.json"
                if path.exists():
                    history = read_json(path)["history"]
                    ax.plot(
                        [h["update"] for h in history],
                        [h["weights"][j] for h in history],
                        alpha=0.6,
                    )
                    selected = read_json(path)["selected_update"]
                    if selected:
                        ax.scatter(
                            [selected], [history[selected - 1]["weights"][j]], s=12, color="black"
                        )
            ax.axhline(TEACHER[j], color="black", linestyle="--", label="Teacher")
            ax.set(title=f"{regime}: {name}", xlabel="Update", ylabel="Weight")
            ax.grid(alpha=0.2)
    axes[0, 0].legend(fontsize=8)
    export(fig, out, "parameter_recovery")

    fig, axes = plt.subplots(1, len(REGIMES), figsize=(7 * len(REGIMES), 4), squeeze=False)
    for ax, regime in zip(axes[0], REGIMES, strict=True):
        for variant in VARIANTS:
            selected = [r for r in rows if r["regime"] == regime and r["method"] == variant]
            stats = next(
                s
                for s in summaries
                if (s["regime"], s["method"], s["metric"]) == (regime, variant, "position_rmse")
            )
            if selected and stats["mean"] is not None:
                ax.scatter(selected[0]["parameters"], stats["mean"], color=COLORS[variant], s=65)
                ax.annotate(
                    LABELS[variant],
                    (selected[0]["parameters"], stats["mean"]),
                    xytext=(4, 5),
                    textcoords="offset points",
                    fontsize=8,
                )
        ax.set(title=regime, xlabel="Model parameters", ylabel="Position RMSE", xscale="log")
        ax.margins(0.25)
        ax.grid(alpha=0.2)
    export(fig, out, "parameter_efficiency")

    fig, axes = plt.subplots(1, len(REGIMES), figsize=(7 * len(REGIMES), 4), squeeze=False)
    for ax, regime in zip(axes[0], REGIMES, strict=True):
        contrasts = [p for p in paired if p["regime"] == regime and p["metric"] == "position_rmse"]
        for i, contrast in enumerate(contrasts):
            if contrast["mean"] is None:
                continue
            ax.scatter(contrast["mean"], i, color=COLORS["parametric"])
            if contrast["ci95"]:
                ax.plot(contrast["ci95"], [i, i], color=COLORS["parametric"])
        ax.axvline(0, color="black", lw=0.8)
        ax.set(
            yticks=range(len(contrasts)),
            yticklabels=[f"DIFFIELD − {LABELS[p['reference']]}" for p in contrasts],
            title=regime,
            xlabel="Paired difference in position RMSE (95% CI)",
        )
        ax.grid(alpha=0.2)
    export(fig, out, "paired_differences")


def videos(out, data, config, budget):
    directory = out / "videos"
    directory.mkdir(parents=True, exist_ok=True)
    if not FFMpegWriter.isAvailable():
        raise RuntimeError("ffmpeg is required for Boids replays")
    methods = ("teacher", *(m for m in config["methods"] if m != "oracle"))
    for regime in REGIMES:
        if budget:
            budget.check()
        metadata = directory / f"{regime}.json"
        if metadata.exists():
            continue
        sources = []
        for ep in data["test"][:3]:
            states = {"teacher": {"positions": ep["target_pos"], "velocities": ep["target_vel"]}}
            for method in methods[1:]:
                path = out / "replays" / regime / method / f"{ep['key']}.pt"
                states[method] = torch.load(path, weights_only=True)
                sources.append({"path": str(path.relative_to(out)), "sha256": checksum(path)})
            sources.append({"episode": ep["key"], "states": states})
        fig, axes = plt.subplots(2, 3, figsize=(11, 7), squeeze=False)
        for ax in axes.flat[len(methods) :]:
            ax.set_visible(False)
        path = directory / f"{regime}.mp4"
        temporary = directory / f"{regime}.part.mp4"
        writer = FFMpegWriter(fps=24, codec="libx264", extra_args=["-pix_fmt", "yuv420p"])
        with writer.saving(fig, str(temporary), dpi=110):
            for source in [s for s in sources if "states" in s]:
                states = source["states"]
                for t in range(config["rounds"]):
                    if budget:
                        budget.check()
                    for ax, method in zip(axes.flat[: len(methods)], methods, strict=True):
                        ax.clear()
                        p = states[method]["positions"][t].numpy()
                        v = states[method]["velocities"][t].numpy()
                        ax.quiver(
                            p[:, 0],
                            p[:, 1],
                            v[:, 0],
                            v[:, 1],
                            color=COLORS.get(method, "black"),
                            angles="xy",
                            scale_units="xy",
                            scale=0.35,
                            width=0.006,
                        )
                        ax.set(
                            xlim=(0, 1),
                            ylim=(0, 1),
                            aspect="equal",
                            xticks=[],
                            yticks=[],
                            title="Recorded teacher" if method == "teacher" else LABELS[method],
                        )
                    fig.suptitle(f"{regime} · seed 0 · {source['episode']} · round {t + 1}/24")
                    fig.tight_layout(rect=(0, 0, 1, 0.94))
                    for _ in range(6):  # Four simulated rounds/s, no interpolated states.
                        writer.grab_frame()
        plt.close(fig)
        temporary.replace(path)
        json_write(
            metadata,
            {
                "regime": regime,
                "seed": 0,
                "rounds": config["rounds"],
                "episodes": [ep["key"] for ep in data["test"][:3]],
                "fps": 24,
                "frames_per_round": 6,
                "frames": min(3, len(data["test"])) * config["rounds"] * 6,
                "inputs": [s for s in sources if "states" not in s],
                "sha256": checksum(path),
            },
        )
    (directory / "index.html").write_text(
        '<!doctype html><html lang="en"><meta charset="utf-8"><title>Boids: GNN Comparison</title>'
        "<style>body{font:17px system-ui;max-width:1100px;margin:32px auto}video{width:100%}</style>"
        "<h1>Boids: GNN Comparison</h1><p>Seed 0, first three test episodes, 24 rounds. "
        'Frozen weights, 4 rounds/s, no interpolation.</p><a href="../REPORT.md">Report</a>'
        + "".join(
            f'<h2>{regime}</h2><video controls src="{regime}.mp4"></video>' for regime in REGIMES
        )
        + "</html>"
    )


def report(out, data, config, budget, render=True):
    methods = config["methods"]
    rows = [
        read_json(p)
        for regime in REGIMES
        for p in sorted((out / "episodes" / regime).glob("*/*/*.json"))
    ]
    if any(r["method"] not in methods for r in rows):
        raise ValueError("Unexpected method in Boids comparison")
    if any(r["regime"] not in REGIMES for r in rows):
        raise ValueError("Unexpected supervision in observed episode directory")
    summaries, paired = [], []
    for regime in REGIMES:
        panels = {
            method: complete_panel(
                [r for r in rows if r["regime"] == regime and r["method"] == method], config, data
            )
            for method in methods
        }
        for method, panel in panels.items():
            for metric in ("position_rmse", "velocity_rmse"):
                summaries.append(
                    {
                        "regime": regime,
                        "method": method,
                        "metric": metric,
                        **interval(panel, metric),
                    }
                )
        for reference in (m for m in methods if m not in ("oracle", "parametric")):
            for metric in ("position_rmse", "velocity_rmse"):
                paired.append(
                    {
                        "regime": regime,
                        "method": "parametric",
                        "reference": reference,
                        "metric": metric,
                        **interval(panels["parametric"], metric, reference=panels[reference]),
                    }
                )
    json_write(out / "results.json", rows)
    json_write(out / "summary.json", summaries)
    json_write(out / "paired.json", paired)
    csv_write(
        out / "tables/episodes.csv",
        [
            {**{k: v for k, v in r.items() if k not in ("metrics", "curves")}, **r["metrics"]}
            for r in rows
        ],
    )
    csv_write(out / "tables/summary.csv", summaries)
    csv_write(out / "tables/paired.csv", paired)
    diagnostics = read_json(out / "diagnostics.json") if (out / "diagnostics.json").exists() else {}
    expected = len(REGIMES) * len(methods) * len(config["seeds"]) * config["test_episodes"]
    complete = len(rows) == expected and all(r["status"] == "complete" for r in rows)
    text = [
        "# Boids SEAMS: Learning and GNN Comparison",
        "",
        f"Results status: **{'complete' if complete else 'incomplete'}** ({len(rows)}/{expected} episodes).",
        "",
        "## Diagnosis",
        "",
        "SEAMS v1 uses a loss over 24-round free-running trajectories; the new training "
        "supervises one step at a time from observed positions and velocities. "
        "The control compares one-step and 24-step training with the "
        "same 100 updates (2 in the smoke test), Adam 0.03, minibatches, and normalized loss. "
        "Float64 precision is only a numerical control.",
        "",
    ]
    if diagnostics:
        text += [
            f"Saturated teacher updates: **{diagnostics['saturated_fraction']:.1%}**. "
            f"Maximum one-step errors: `{diagnostics['one_step_max_errors']}`.",
            "",
            "| Training horizon | Final weights | Observed validation loss |",
            "|---|---|---:|",
        ]
        for a in diagnostics["ablations"]:
            text.append(
                f"| {a['horizon']} | {', '.join(f'{w:.6f}' for w in a['weights'])} | {a['observed_validation_loss']:.6g} |"
            )
        text += [
            "",
            "Small numerical differences can amplify in free-running replays. "
            "The teacher replay is a sensitivity reference, not a lower bound on error.",
            "",
        ]
    text += [
        "## Protocol",
        "",
        f"Shared data: {config['train_episodes']} train, {config['validation_episodes']} validation, "
        f"{config['test_episodes']} test episodes; {len(config['seeds'])} seeds, 32 boids, 24 rounds. "
        f"{config['updates']} updates; tuning on seed 10001 and checkpoint selection by "
        "observed validation loss. No test-set tuning. CPU float32, one thread.",
        "",
        "All models learn only from observed positions and velocities. "
        "GNN32/GNN64 receive the "
        "same local information as the three-weight program, without precomputed Boids forces. "
        "Architecture: [PyG MessagePassing](https://pytorch-geometric.readthedocs.io/en/stable/generated/torch_geometric.nn.conv.MessagePassing.html), "
        "with an edge MLP, mean message aggregation, and a node MLP.",
        "",
        "## Free-Running Results",
        "",
        "| Model | Position RMSE | 95% CI | Velocity RMSE | Parameters |",
        "|---|---:|---|---:|---:|",
    ]
    for regime in REGIMES:
        for method in methods:
            stats = [s for s in summaries if s["regime"] == regime and s["method"] == method]
            p, v = stats
            fmt = lambda x: "incomplete" if x is None else f"{x:.6f}"
            ci = "N/A" if not p["ci95"] else f"[{p['ci95'][0]:.6f}, {p['ci95'][1]:.6f}]"
            selected = [r for r in rows if r["regime"] == regime and r["method"] == method]
            params = selected[0]["parameters"] if selected else "—"
            text.append(
                f"| {LABELS[method]} | {fmt(p['mean'])} | {ci} | {fmt(v['mean'])} | {params} |"
            )
    text += [
        "",
        "Intervals use a crossed seed–episode bootstrap with 2,000 resamples. "
        "Differences between methods are computed before bootstrapping. Seeds vary initialization "
        "and minibatches, using a shared training bank; copies of the fixed baseline are not "
        "independent replicates. Incomplete panels have no confidence interval. Fixed retains three "
        "coefficients but does not train them.",
        "",
        "## Paired Comparisons",
        "",
        "Difference DIFFIELD − reference: negative values favor DIFFIELD.",
        "",
        "| Reference | Δ Position RMSE | 95% CI | Outcome |",
        "|---|---:|---|---|",
    ]
    for contrast in paired:
        if contrast["metric"] != "position_rmse":
            continue
        ci = contrast["ci95"]
        verdict = (
            "inconclusive"
            if ci is None or ci[0] <= 0 <= ci[1]
            else ("DIFFIELD advantage" if ci[1] < 0 else "reference advantage")
        )
        text.append(
            f"| {LABELS[contrast['reference']]} | "
            f"{contrast['mean'] if contrast['mean'] is not None else 'N/A'} | {ci or 'N/A'} | {verdict} |"
        )
    text += [
        "",
        "## Weight Recovery and Cost",
        "",
        f"Teacher: `{TEACHER}`. Checkpoint selezionati:",
        "",
        "| Seed | Weights | Selected update | Training time (s) |",
        "|---:|---|---:|---:|",
    ]
    for regime in REGIMES:
        for seed in config["seeds"]:
            selected = next(
                (
                    r
                    for r in rows
                    if r["regime"] == regime and r["method"] == "parametric" and r["seed"] == seed
                ),
                None,
            )
            if selected:
                text.append(
                    f"| {seed} | {', '.join(f'{w:.6f}' for w in selected['weights'])} | "
                    f"{selected['selected_update']} | {selected['training_seconds']:.2f} |"
                )
    text += [
        "",
        "Training and inference times for all methods and parameter counts are in `tables/episodes.csv`; "
        "training times repeated across episodes refer to the same checkpoint and should not be summed."
        + (
            " Legacy is a historical checkpoint imported and reevaluated in the current environment."
            if "legacy" in methods
            else ""
        ),
        "",
        "## Limitations and Artifacts",
        "",
        "The teacher belongs to the three-weight program family: this comparison measures the benefit "
        "of that structure on this specific benchmark, not general superiority over GNNs. "
        "Transfer to more nodes and longer horizons are not included.",
        "",
        "[Replay of the first test episodes](videos/index.html). PNG/PDF figures are in `figures/`; "
        "tables are in `tables/`. Configuration, sources, input hashes, and checkpoints enable reproduction.",
        "",
    ]
    (out / "REPORT.md").write_text("\n".join(text))
    if render and complete:
        if budget:
            budget.check()
        figures(out, rows, summaries, paired, config)
        videos(out, data, config, budget)
        verification = []
        for regime in REGIMES:
            for method in methods:
                for ep in data["test"][:3]:
                    path = out / "replays" / regime / method / f"{ep['key']}.pt"
                    if not path.exists():
                        continue
                    replay = torch.load(path, weights_only=True)
                    row = next(
                        r
                        for r in rows
                        if (r["regime"], r["method"], r["seed"], r["episode"])
                        == (regime, method, 0, ep["key"])
                    )
                    for state, target, metric in (
                        ("positions", "target_pos", "position_rmse"),
                        ("velocities", "target_vel", "velocity_rmse"),
                    ):
                        value = float((replay[state] - ep[target]).square().mean().sqrt())
                        if abs(value - row["metrics"][metric]) > 1e-7:
                            raise ValueError("Replay metric differs from reported metric")
                    verification.append(
                        {
                            "path": str(path.relative_to(out)),
                            "sha256": checksum(path),
                            "metrics_match": True,
                        }
                    )
        json_write(
            out / "verification.json",
            {
                "replays": verification,
                "episode_rows": len(rows),
                "expected_rows": expected,
                "complete": complete,
            },
        )
    return complete
