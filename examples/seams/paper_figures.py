"""Paper figures for RQ1 (learning), RQ2 (runtime adaptation) and RQ3 (scalability).

Reads finished runs only: the channel adaptation campaign, the Boids campaign
with its adaptation and device stages, the Space-Fluid clustering episodes,
and the channel scaling benchmark. Bands are 95% percentile bootstraps over
runs. Spatial snapshots re-run one episode of a finished run deterministically
and check it against the stored results.
"""

import csv
import glob
from pathlib import Path

import numpy as np
import torch
from matplotlib.colors import to_rgba
from matplotlib.patches import Rectangle
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import adjusted_rand_score

from ..shared.artifacts import PROJECT, json_write, read_json
from ..shared.metrics import (
    bootstrap_band,
    interval,
    recovery,
    recovery_summary,
    relative_deviation,
)
from ..shared.plotting.common import LineCollection, plt, save_figure
from ..shared.plotting.moving import _get_identity_colors
from ..shared.plotting.style import (
    AQUA,
    BLUE,
    GREEN,
    INK,
    MAGENTA,
    MUTED,
    ORANGE,
    RED,
    VIOLET,
    YELLOW,
    apply_paper_style,
)
from .boids.devices import sync_async_rmse
from .boids.dynamics import FIXED, TEACHER
from .boids.learning import (
    TRACES,
    combine,
    free_rollout,
    job_path,
    load_model,
    make_model,
    objective,
    rmse,
    transitions,
)
from .channel_adaptation import run_seed
from .space_fluid import clusters
from .space_fluid.config import ASYNC, protocol
from .space_fluid.execution import central, on_devices
from .space_fluid.program import ELECTED
from .space_fluid.report import deterministic, with_every_seed
from .space_fluid.training import load_method

DARK_RED = "#8f2f1d"
STYLE = {  # role: (colour, linestyle, label)
    "diffield": (BLUE, "-", "DifField"),
    "initial": (MUTED, "--", "hand-tuned / initial"),
    "teacher": (INK, "-", "teacher"),
    "gnn32": (RED, ":", "GNN32"),
    "gnn64": (DARK_RED, "-.", "GNN64"),
    "frozen": (ORANGE, "--", "frozen plan (no adaptation)"),
    "diffield_async": (BLUE, (0, (3, 1.2)), "DifField, async devices"),
    "teacher_async": (INK, (0, (3, 1.2)), "teacher, async devices"),
}
BOIDS_ROLE = {"oracle": "teacher", "fixed": "initial", "parametric": "diffield",
              "gnn32": "gnn32", "gnn64": "gnn64"}  # fmt: skip
SCR_ROLE = {"parametric": "diffield", "fixed-combined": "initial"}
FONT = 7.5
# Snapshots: true zones in label order; regions matching no zone take the extra colours.
ZONES = (AQUA, YELLOW, VIOLET, MAGENTA)
EXTRA = (GREEN, ORANGE, RED, BLUE, "#b4b2aa", "#5d5b56", "#d8a7b9", "#9fc7ea")
IDLE = "#c9c7bf"


def style():
    apply_paper_style()
    plt.rcParams.update(
        {
            "font.size": FONT,
            "axes.labelsize": FONT,
            "axes.titlesize": FONT,
            "axes.titleweight": "normal",
            "xtick.labelsize": FONT - 0.5,
            "ytick.labelsize": FONT - 0.5,
            "legend.fontsize": FONT,
            "lines.linewidth": 1.1,
            "axes.linewidth": 0.6,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.01,
        }
    )


def draw(ax, x, curves, role, *, fill=True, **kwargs):
    colour, linestyle, label = STYLE[role]
    mean, low, high = bootstrap_band(curves)
    ax.plot(x, mean, color=colour, linestyle=linestyle, label=label, **kwargs)
    if fill and len(curves) > 1:
        ax.fill_between(x, low, high, color=colour, alpha=0.18, linewidth=0)


def events(ax, marks, top=1.02):
    for name, t in marks.items():
        ax.axvline(t, color=MUTED, linewidth=0.6, zorder=0)
        ax.text(t + 3, top, name, transform=ax.get_xaxis_transform(), fontsize=FONT - 1,
                color=MUTED, va="bottom", ha="left")  # fmt: skip


def panel(letter, text=""):
    """Panel label: the letter in italics, then plain text."""
    return rf"$\mathit{{({letter})}}$ {text}".rstrip()


def scr_curves(root, scenario="static", condition="crash", methods=tuple(SCR_ROLE),
               executor="async-0.5"):
    """ARI per round of the Space-Fluid clustering runs (runs x rounds).

    Full-batch training makes the parametric seeds identical, so duplicate runs
    are dropped: a band over copies would look narrower than the evidence.
    """
    result = {}
    for method in methods:
        name = f"distributed-{executor}-{condition}-*.json"
        pattern = f"{root}/{scenario}/episodes/{method}/seed*/{name}"
        rows = [read_json(Path(p))["curves"]["ari"] for p in sorted(glob.glob(pattern))]
        if not rows:
            raise FileNotFoundError(pattern)
        result[method] = np.unique(np.array(rows), axis=0)
    return result


def scr_numbers(scr, fault=None, restore=None, tolerance=0.05, consecutive=5):
    """ARI before the fault, at its trough and at the end; recovery measured after the trough.

    By default the fault comes at a third of the rounds and the restore at two thirds, where
    the episodes schedule them.
    """
    rounds = next(iter(scr.values())).shape[1]
    fault = rounds // 3 if fault is None else fault
    restore = 2 * rounds // 3 if restore is None else restore
    numbers = {}
    for method, ari in scr.items():
        rounds, after_restore = [], []
        for curve in ari:
            pre = float(curve[fault - 5 : fault].mean())
            trough = fault + int(curve[fault:restore].argmin())
            result = recovery([1 - a for a in curve], trough, 1 - (pre - tolerance), consecutive)
            if result["rounds"] is not None:
                rounds.append(trough - fault + result["rounds"])
            rest = recovery([1 - a for a in curve], restore, 1 - (pre - tolerance), consecutive)
            if rest["rounds"] is not None:
                after_restore.append(rest["rounds"])
        numbers[method] = {
            "runs": len(ari),
            "ari_before": float(ari[:, fault - 5 : fault].mean()),
            "ari_trough": float(ari[:, fault:restore].min(1).mean()),
            "ari_during": float(ari[:, fault:restore].mean()),
            "ari_final": float(ari[:, -10:].mean()),
            **recovery_summary(rounds),
            "rounds": rounds,
            "after_restore_rounds": after_restore,
            "after_restore_median": float(np.median(after_restore)) if after_restore else None,
        }
    return numbers


def rq2_figure(channel, boids, scr_root, out, *, size=64, scenario="static", condition="crash"):
    channel_runs = [read_json(p) for p in sorted((channel / "runs").glob("seed*.json"))]
    channel_config = read_json(channel / "summary.json")["config"]
    scr_async = scr_curves(scr_root, scenario, condition, executor="async-0.5")
    scr_sync = scr_curves(scr_root, scenario, condition, executor="sync")
    clean = scr_curves(scr_root, scenario, "clean", executor="async-0.5")
    transient = {
        c: scr_curves(scr_root, scenario, c, executor="async-0.5")
        for c in ("node_stop", "partition", "link_loss")
    }
    boids_rmse = sync_async_rmse(boids)

    fig, axes = plt.subplots(1, 4, figsize=(7.16, 1.28), constrained_layout=True,
                             gridspec_kw={"width_ratios": (1.02, 1, 1, 1)})  # fmt: skip

    # (a) Channel: runtime faults (obstacle, clear, relocate)
    ax = axes[0]
    x, start = np.arange(channel_config["rounds"]), channel_config["events"]["obstacle"]
    frozen = [r["curves"]["frozen_quality"][start:] for r in channel_runs]
    draw(ax, x[start:], frozen, "frozen")
    draw(ax, x, [r["curves"]["quality"] for r in channel_runs], "diffield")
    events(ax, channel_config["events"], top=0.98)
    ax.set(ylabel="channel F1", ylim=(-0.03, 1.03), xlim=(0, x[-1]))
    ax.set_xlabel("round", labelpad=1.0)
    ax.set_yticks([0, 0.5, 1], ["0", "0.5", "1"])
    ax.set_title(panel("a", "channel: faults"), loc="left", pad=11)
    ax.legend(
        handles=[
            plt.Line2D([], [], color=BLUE, linestyle="-", label="DifField"),
            plt.Line2D([], [], color=ORANGE, linestyle="--", label="frozen plan"),
        ],
        loc="upper center", bbox_to_anchor=(0.5, -0.46), ncol=2, fontsize=FONT - 1.5,
        handlelength=1.2, handletextpad=0.35, columnspacing=0.8, frameon=False,
    )

    # (b) Regions: permanent leader crash (sync vs async)
    ax = axes[1]
    x = np.arange(scr_async["parametric"].shape[1])
    for method, colour, prefix in (("parametric", BLUE, "DifField"),
                                   ("fixed-combined", MUTED, "initial")):
        mean_s, low_s, high_s = bootstrap_band(scr_sync[method])
        ax.plot(x, mean_s, color=colour, linestyle="-", linewidth=1.0, label=f"{prefix}, sync")
        ax.fill_between(x, low_s, high_s, color=colour, alpha=0.12, linewidth=0)
        mean_a, low_a, high_a = bootstrap_band(scr_async[method])
        ax.plot(x, mean_a, color=colour, linestyle=(0, (3, 1.2)), linewidth=1.0,
                label=f"{prefix}, async")
        ax.fill_between(x, low_a, high_a, color=colour, alpha=0.12, linewidth=0)
    rounds = len(x)
    fault, restore = rounds // 3, 2 * rounds // 3
    crash = scr_numbers(scr_async, restore=rounds)  # a crash is never restored
    learned, initial = crash["parametric"], crash["fixed-combined"]
    ax.axvspan(fault, x[-1], color=MUTED, alpha=0.07, linewidth=0)
    events(ax, {"leader crash": fault}, top=0.98)
    recovered = (f"recovered: {learned['recovered']}/{learned['runs']}\n"
                 f"(initial: {initial['recovered']}/{initial['runs']})")  # fmt: skip
    ax.text(x[-1], 0.05, recovered, fontsize=FONT - 2.0, color=INK,
            ha="right", va="bottom", linespacing=1.05)
    ax.set(ylabel="clustering ARI", ylim=(0, 1.03), xlim=(0, x[-1]))
    ax.set_xlabel("round", labelpad=1.0)
    ax.set_yticks([0, 0.5, 1], ["0", "0.5", "1"])
    ax.set_title(panel("b", "regions: leader crash"), loc="left", pad=11)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.46), ncol=2, fontsize=FONT - 1.5,
              handlelength=1.2, handletextpad=0.35, columnspacing=0.6, labelspacing=0.2,
              frameon=False)

    # (c) Regions: transient network and device faults (async)
    ax = axes[2]
    ax.axvspan(fault, restore, color=MUTED, alpha=0.11, linewidth=0)
    events(ax, {"fault": fault, "restored": restore}, top=0.98)
    fault_styles = (
        ("node_stop", "paused devices", GREEN),
        ("partition", "partition", VIOLET),
        ("link_loss", "link loss", BLUE),
    )
    for key, label, colour in fault_styles:
        mean, low, high = bootstrap_band(transient[key]["parametric"])
        ax.plot(x, mean, color=colour, linestyle="-", linewidth=1.0, label=label)
        ax.fill_between(x, low, high, color=colour, alpha=0.14, linewidth=0)
    init_all = np.concatenate([transient[k]["fixed-combined"] for k, _, _ in fault_styles], axis=0)
    mean_init, _, _ = bootstrap_band(init_all)
    ax.plot(x, mean_init, color=MUTED, linestyle="--", linewidth=0.9, label="initial (mean)")
    ax.set(ylabel="clustering ARI", ylim=(0, 1.03), xlim=(0, x[-1]))
    ax.set_xlabel("round", labelpad=1.0)
    ax.set_yticks([0, 0.5, 1], ["0", "0.5", "1"])
    ax.set_title(panel("c", "regions: transient faults"), loc="left", pad=11)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.46), ncol=2, fontsize=FONT - 1.5,
              handlelength=1.2, handletextpad=0.35, columnspacing=0.6, labelspacing=0.2,
              frameon=False)

    # (d) Boids: synchronous vs asynchronous devices (no faults)
    ax = axes[3]
    xb = np.arange(1, boids_rmse["diffield_sync"].shape[1] + 1)
    boids_series = (
        ("diffield_sync", "DifField, sync", BLUE, "-"),
        ("diffield_async", "DifField, async", BLUE, (0, (3, 1.2))),
        ("gnn64_sync", "GNN64, sync", DARK_RED, "-"),
        ("gnn64_async", "GNN64, async", DARK_RED, (0, (3, 1.2))),
    )
    for key, label, colour, ls in boids_series:
        mean, low, high = bootstrap_band(boids_rmse[key])
        ax.plot(xb, mean, color=colour, linestyle=ls, linewidth=1.0, label=label)
        if key.startswith("gnn64"):
            ax.fill_between(xb, low, high, color=colour, alpha=0.14, linewidth=0)
    peak = max(boids_rmse[k].mean(0).max() for k in ("diffield_sync", "diffield_async"))
    bound = r"DifField: $<10^{-3}$" if peak < 1e-3 else f"DifField: {peak:.3f}"
    ax.text(xb[-1] - 0.5, 0.006, bound, fontsize=FONT - 1.5, color=BLUE, ha="right", va="bottom")
    ax.set(ylabel="position RMSE", xlim=(1, xb[-1]), ylim=(-0.003, 0.095))
    ax.set_xlabel("round", labelpad=1.0)
    ax.set_xticks([1, 8, 16, 24])
    ax.set_yticks([0, 0.04, 0.08], ["0", "0.04", "0.08"])
    ax.set_title(panel("d", "Boids: sync vs async"), loc="left", pad=11)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.46), ncol=2, fontsize=FONT - 1.5,
              handlelength=1.2, handletextpad=0.35, columnspacing=0.6, labelspacing=0.2,
              frameon=False)

    save_figure(fig, out / "rq2-adaptation")
    deviations = sync_deviations(boids, scr_root, size, scenario)
    return {
        "scr": crash,
        "scr_sync": scr_numbers(scr_sync, restore=scr_sync["parametric"].shape[1]),
        "scr_clean": scr_numbers(clean),
        "scr_transient": {k: scr_numbers(v) for k, v in transient.items()},
        "boids_sync_async_rmse": {
            k: {"mean": float(v.mean()), "final": float(v[:, -1].mean())}
            for k, v in boids_rmse.items()
        },
        "sync_deviation": {k: {"mean": float(v.mean()), "episodes": len(v)}
                           for k, v in deviations.items()},
    }


def sync_deviations(boids, scr_root, size=64, scenario="static"):
    """Per episode, how far asynchronous execution drifts from synchronous execution of the
    same program: Boids spread (teacher and learned program) and regions ARI (all conditions)."""

    def spread(pattern):
        rows = (read_json(Path(p)) for p in sorted(glob.glob(str(pattern))))
        return {r["episode"]: r["curves"]["spread"] for r in rows if r["status"] == "complete"}

    devices = boids / "devices/episodes"
    pairs = {
        "teacher": (spread(devices / f"oracle-async/N{size}-*.json"),
                    spread(boids / f"adaptation/episodes/oracle/deterministic/N{size}-*.json")),
        "diffield": (spread(devices / f"parametric-async/N{size}-*.json"),
                     spread(devices / f"parametric-sync/N{size}-*.json")),
    }  # fmt: skip
    result = {}
    for name, (asynchronous, synchronous) in pairs.items():
        shared = sorted(set(asynchronous) & set(synchronous))
        result[f"Boids, {name}"] = np.array(
            [relative_deviation(asynchronous[k], synchronous[k]) for k in shared]
        )
    for method in ("fixed-combined", "parametric"):
        errors = []
        for path in sorted(glob.glob(
            f"{scr_root}/{scenario}/episodes/{method}/seed0/distributed-async-0.5-*.json"
        )):  # fmt: skip
            reference = Path(path.replace("distributed-async-0.5-", "distributed-sync-"))
            errors.append(relative_deviation(read_json(Path(path))["curves"]["ari"],
                                             read_json(reference)["curves"]["ari"]))  # fmt: skip
        result[f"regions, {SCR_ROLE[method]}"] = np.array(errors)
    return result


def boids_validation(boids, variant, validation):
    """Validation loss every few updates of each seed, from update 0, and the selected update."""
    runs = []
    for path in sorted(job_path(boids, variant, 0).parent.glob("seed*/training.json")):
        job = read_json(path)
        with torch.no_grad():  # update 0 is evaluated for selection but not kept in the history
            start = float(objective(make_model(variant, job["seed"]), validation))
        points = [(0, start)] + [
            (h["update"], h["validation_loss"])
            for h in job["history"]
            if h["validation_loss"] is not None
        ]
        weights = None
        if variant == "parametric":  # the program starts from the fixed weights
            weights = [FIXED] + [h["weights"] for h in job["history"]]
        runs.append({"points": points, "selected": job["selected_update"], "weights": weights})
    return runs


def trajectories_panel(ax, traj, colours, title, *, background="white"):
    """Fading trails per agent (start: faint dot, end: cross), as in the earlier Boids figure."""
    segments = len(traj) - 1
    for i, colour in enumerate(colours):
        points = traj[:, i, None, :]
        trail = np.concatenate([points[:-1], points[1:]], axis=1)
        ax.add_collection(
            LineCollection(trail, colors=[colour] * segments,
                           linewidths=np.linspace(0.3, 1.3, segments),
                           alpha=np.linspace(0.12, 0.85, segments))
        )  # fmt: skip
    ax.scatter(traj[0, :, 0], traj[0, :, 1], c=colours, s=3, alpha=0.35, linewidths=0)
    ax.scatter(traj[-1, :, 0], traj[-1, :, 1], c=colours, marker="x", s=9, linewidths=0.8)
    ax.set(xlim=(0, 1), ylim=(0, 1), xticks=[], yticks=[], aspect="equal", facecolor=background)
    ax.set_title(title, loc="left", pad=2)


def boids_row(axes, boids, *, episode=0, seed=0, gnn="gnn64", zoom=200):
    """Validation loss (a), learned weights relative to the teacher's (b), and free-running
    trajectories of the untrained and trained program, a GNN, and the teacher (c-f)."""
    data = torch.load(boids / TRACES, weights_only=True)
    validation = combine([transitions(e) for e in data["validation"]])
    ax, losses = axes[0], {}
    for variant in ("gnn32", "gnn64", "parametric"):
        runs = boids_validation(boids, variant, validation)
        x = np.array([u for u, _ in runs[0]["points"]])
        curves = np.array([[v for _, v in run["points"]] for run in runs])
        picks = [(run["selected"], dict(run["points"])[run["selected"]]) for run in runs]
        colour, linestyle, _ = STYLE[BOIDS_ROLE[variant]]
        ax.fill_between(x, curves.min(0), curves.max(0), color=colour, alpha=0.18, linewidth=0)
        ax.plot(x, np.median(curves, 0), color=colour, linestyle=linestyle)
        ax.scatter(*zip(*picks, strict=True), s=7, color=colour, edgecolors="white",
                   linewidths=0.4, zorder=3)  # fmt: skip
        losses[variant] = float(np.median(curves[:, -1]))
        if variant == "parametric":
            initial, weight_runs = float(curves[0, 0]), runs
    colour, linestyle, _ = STYLE["initial"]
    ax.axhline(initial, color=colour, linestyle=linestyle, linewidth=0.9, zorder=1)
    plateau = np.median([losses["gnn32"], losses["gnn64"]])
    ax.text(x[-1] - 8, 0.16, f"GNNs: {plateau:.2f}", fontsize=FONT - 1.5, color=DARK_RED,
            ha="right", va="top")  # fmt: skip
    loss_handles = [
        plt.Line2D([], [], color=STYLE[r][0], linestyle=STYLE[r][1], label=STYLE[r][2])
        for r in ("diffield", "gnn64", "gnn32")
    ] + [plt.Line2D([], [], color=MUTED, linestyle="--", label="untrained")]
    ax.legend(handles=loss_handles, loc="center right", bbox_to_anchor=(1.0, 0.44),
              fontsize=FONT - 1.5, handlelength=1.35, handletextpad=0.35,
              labelspacing=0.18, frameon=False, borderaxespad=0.2)
    ax.set(ylabel="one-step MSE loss", yscale="log", xlim=(0, x[-1]), ylim=(1e-13, 30))
    ax.set_xlabel("update", labelpad=1.0)
    ax.set_yticks([1e0, 1e-6, 1e-12])
    ax.set_xticks([0, 250, 500])
    ax.set_title(panel("a", "one-step loss"), loc="left", pad=2)

    ax = axes[1]
    for j, colour in enumerate((AQUA, VIOLET, YELLOW)):
        for run in weight_runs:
            ax.plot(range(len(run["weights"])), [w[j] / TEACHER[j] for w in run["weights"]],
                    color=colour, linewidth=0.8, alpha=0.7)  # fmt: skip
    ax.axhline(1, color=INK, linestyle="--", linewidth=0.8)
    ax.set(yscale="log", xlim=(0, zoom))
    ax.set_xlabel("update", labelpad=1.0)
    ax.set_ylabel("weight / teacher", labelpad=1.5)
    ax.set_xticks([0, 100, 200])
    ax.set_yticks([0.1, 0.3, 1, 3], ["0.1", "0.3", "1", "3"])
    weight_handles = [
        plt.Line2D([], [], color=c, label=n)
        for n, c in (("separation", AQUA), ("alignment", VIOLET), ("cohesion", YELLOW))
    ] + [plt.Line2D([], [], color=INK, linestyle="--", label="teacher")]
    ax.legend(handles=weight_handles, loc="lower right", fontsize=FONT - 1.5, handlelength=1.1,
              handletextpad=0.35, borderaxespad=0.2, labelspacing=0.18, frameon=False)
    ax.set_title(panel("b", "weights"), loc="left", pad=2)

    test = data["test"][episode]
    training = read_json(job_path(boids, "parametric", seed) / "training.json")
    history = {h["update"]: h["weights"] for h in training["history"]}
    rounds, colours = len(test["target_pos"]), _get_identity_colors(len(test["positions"]))
    errors = {}
    stages = (("untrained", FIXED), ("DifField", history[training["selected_update"]]))
    for k, (label, weights) in enumerate(stages):
        model = make_model("parametric", seed)
        with torch.no_grad():
            model.log_weights.copy_(torch.tensor(weights).log())
            positions, _ = free_rollout(model, test["positions"], test["velocities"], rounds)
        traj = torch.cat((test["positions"][None], positions)).numpy()
        trajectories_panel(axes[2 + k], traj, colours, panel("cd"[k], label))
        err = rmse(positions, test["target_pos"])
        errors[label] = err
        badge = f"RMSE {err:.2f}" if err >= 1e-3 else r"RMSE $<10^{-3}$"
        box_style = {"facecolor": "white", "alpha": 0.78, "edgecolor": "none", "pad": 0.8}
        axes[2 + k].text(0.96, 0.04, badge, transform=axes[2 + k].transAxes,
                         fontsize=FONT - 1.5, color=INK, ha="right", va="bottom", bbox=box_style)
    model = load_model(boids, gnn, seed)
    with torch.no_grad():
        positions, _ = free_rollout(model, test["positions"], test["velocities"], rounds)
    traj = torch.cat((test["positions"][None], positions)).numpy()
    trajectories_panel(axes[4], traj, colours, panel("e", STYLE[gnn][2]))
    err_gnn = rmse(positions, test["target_pos"])
    axes[4].text(0.96, 0.04, f"RMSE {err_gnn:.2f}", transform=axes[4].transAxes,
                 fontsize=FONT - 1.5, color=INK, ha="right", va="bottom", bbox=box_style)
    if len(axes) > 5:
        teacher = torch.cat((test["positions"][None], test["target_pos"])).numpy()
        trajectories_panel(axes[5], teacher, colours, panel("f", "teacher"), background="#f4f8ff")

    numbers = {"validation_loss": losses, "initial_weights_loss": initial, "position_rmse": errors}
    return numbers, []


def snapshot_axes(ax, title, box=(0.0, 1.0, 0.0, 1.0), margin=0.03):
    x0, x1, y0, y1 = box
    ax.set(xticks=[], yticks=[], aspect="equal", xlim=(x0 - margin, x1 + margin),
           ylim=(y0 - margin, y1 + margin))  # fmt: skip
    ax.set_title(title, fontsize=7, pad=2, linespacing=1.1)
    ax.grid(False)
    for spine in ax.spines.values():
        spine.set(visible=True, linewidth=0.5, color=MUTED)


def region_colours(labels, truth):
    """Each region takes the colour of the true zone it best overlaps; the others stay distinct."""
    regions, inverse = np.unique(labels, return_inverse=True)
    overlap = np.array(
        [[np.sum((inverse == r) & (truth == z)) for z in range(len(ZONES))]
         for r in range(len(regions))]
    )  # fmt: skip
    rows, cols = linear_sum_assignment(-overlap)
    colour = {r: ZONES[z] for r, z in zip(rows, cols, strict=True) if overlap[r, z] > 0}
    rest = sorted(set(range(len(regions))) - set(colour), key=lambda r: -overlap[r].sum())
    colour |= {r: EXTRA[k % len(EXTRA)] for k, r in enumerate(rest)}
    return np.array([colour[r] for r in inverse])


def regions_panel(ax, positions, labels, *, truth, elected, active, title):
    """Regions coloured against the true zones; leaders black-edged, crashed devices crossed."""
    colours, points = region_colours(labels[active], truth[active]), positions[active]
    ax.scatter(*points.T, c=colours, s=9, linewidths=0)
    heads = elected[active]
    ax.scatter(*points[heads].T, c=colours[heads], s=24, edgecolors=INK, linewidths=0.6, zorder=3)
    if not active.all():
        ax.scatter(*positions[~active].T, marker="x", c=MUTED, s=10, linewidths=0.6)
    snapshot_axes(ax, title)


def scr_program(root, scenario, method):
    norm = read_json(root / scenario / "manifest.json")["normalization"]
    return load_method(root / scenario, method, 0, norm)


@torch.no_grad()
def scr_snapshot_panels(
    axes, root, scenario="static", index=0, letters="abcd", *, zones=True, one_line=False
):
    """Field, true zones and final regions of the initial and learned programs (batched, hard)."""
    config = protocol(read_json(root / "manifest.json")["profile"])
    episode = clusters.cluster_episode(config, "test", index, moving=scenario == "moving")
    positions, truth = episode.positions.numpy(), clusters.truth_labels(episode)[-1]
    axes[0].scatter(*positions.T, c=episode.observations[-1].numpy(), cmap="viridis", s=9,
                    linewidths=0)  # fmt: skip
    snapshot_axes(axes[0], panel(letters[0], "sensed field"))
    if zones:
        axes[1].scatter(*positions.T, c=np.array(ZONES)[truth], s=9, linewidths=0)
        snapshot_axes(axes[1], panel(letters[1], "true zones"))
    else:
        axes, letters = [axes[0], None, *axes[1:]], f"{letters[0]} {letters[1:]}"
    shown = {"episode": episode.key, "round": episode.rounds - 1}
    panels = (("fixed-combined", panel(letters[2], "initial")),
              ("parametric", panel(letters[3], "learned")))  # fmt: skip
    for ax, (method, name) in zip(axes[2:], panels, strict=True):
        trace = central(episode, scr_program(root, scenario, method))
        labels = trace.leaders[-1].numpy()
        stored = root / scenario / "episodes" / method / "seed0"
        stored = read_json(stored / f"{scenario}-batched-clean-4-0.3-{index}.json")
        if stored["final_labels"] != labels.tolist():
            raise AssertionError(f"{method}: snapshot differs from the evaluated episode")
        ari, regions = float(adjusted_rand_score(truth, labels)), len(np.unique(labels))
        elected = trace.fields[-1, :, ELECTED].numpy() > 0.5
        title = (
            f"{name} ($K={regions}$)"
            if one_line
            else f"{name}\nARI {ari:.2f}, $K={regions}$"
        )
        regions_panel(ax, positions, labels, truth=truth, elected=elected,
                      active=np.ones(episode.nodes, dtype=bool),
                      title=title)  # fmt: skip
        shown[method] = {"ari": ari, "regions": regions, "leaders": int(elected.sum())}
    return shown


def rq1_scr_snapshot(root, out, *, scenario="static", index=0):
    fig, axes = plt.subplots(1, 4, figsize=(3.5, 1.15))
    shown = scr_snapshot_panels(axes, root, scenario, index)
    fig.subplots_adjust(left=0.005, right=0.995, bottom=0.01, top=0.74, wspace=0.07)
    save_figure(fig, out / "rq1-scr-snapshot", dpi=300)
    return shown


SCR_METHODS = (  # method, label, colour, given K
    ("fixed-combined", "initial program", MUTED, False),
    ("parametric", "DifField, learned", BLUE, False),
    ("gnn-central", "central GNN", DARK_RED, False),
    ("kmeans", "K-means", "#5d5b56", True),
    ("ward", "Ward", "#5d5b56", True),
)


def scr_panel_rows(root, method, panel, trained="static"):
    pattern = f"{root}/{trained}/episodes/{method}/seed*/{panel}-batched-clean-*.json"
    return [read_json(Path(p)) for p in sorted(glob.glob(pattern))]


def regions_panel_ari(ax, root, *, scenario="static", letter="f"):
    """Final ARI of every method on static and drifting zones, with the number of regions K."""
    numbers, height = {}, 0.34
    for y, (method, _, colour, given) in enumerate(SCR_METHODS):
        numbers[method] = {}
        for zone, offset, hatch in (("static", -height / 2, None), ("moving", height / 2, "///")):
            rows = scr_panel_rows(root, method, zone, scenario)
            if deterministic(method):  # one run per episode, met at every learned seed
                learned = scr_panel_rows(root, "parametric", zone, scenario)
                rows = [r for r in with_every_seed(rows + learned) if r["method"] == method]
            stats = interval(rows, "ari_final")
            mean = stats["finite_mean"]
            face = to_rgba(colour, 0.8) if hatch is None else "white"
            ax.barh(y + offset, mean, height, color=face, edgecolor=colour, hatch=hatch,
                    linewidth=0.6)  # fmt: skip
            if stats["ci95"]:
                low, high = stats["ci95"]
                ax.errorbar(mean, y + offset, xerr=[[mean - low], [high - mean]], color=INK,
                            linewidth=0.6, capsize=1.2)  # fmt: skip
            numbers[method][zone] = {"ari": mean, "ci95": stats["ci95"]}
        regions = np.mean([r["metrics"]["clusters_final"]
                           for r in scr_panel_rows(root, method, "static", scenario)])  # fmt: skip
        numbers[method]["regions"] = float(regions)
        ax.text(1.03, y, "$4^*$" if given else f"{regions:.1f}", transform=ax.get_yaxis_transform(),
                fontsize=FONT - 1, color=colour, va="center", ha="left")  # fmt: skip
    ax.text(1.03, -0.75, "$K$", transform=ax.get_yaxis_transform(), fontsize=FONT - 1,
            color=MUTED, va="center", ha="left")  # fmt: skip
    short = {"fixed-combined": "initial", "parametric": "DifField", "gnn-central": "central GNN"}
    ax.set_yticks(range(len(SCR_METHODS)), [short.get(m[0], m[1]) for m in SCR_METHODS],
                  fontsize=FONT - 1)  # fmt: skip
    ax.set(xlim=(0, 1), ylim=(len(SCR_METHODS) - 0.5, -0.5))
    ax.set_xlabel("clustering ARI", labelpad=1.5)
    ax.set_xticks([0, 0.5, 1])
    ax.tick_params(axis="y", length=0)
    ax.grid(axis="y", visible=False)
    ax.set_title(panel(letter, "regions"), loc="left", pad=2)
    zone_handles = [
        Rectangle((0, 0), 1, 1, facecolor=MUTED, edgecolor=MUTED, label="static"),
        Rectangle((0, 0), 1, 1, facecolor="white", edgecolor=MUTED, hatch="///",
                  label="drifting"),
    ]
    ax.legend(handles=zone_handles, loc="lower right", bbox_to_anchor=(1.0, 1.02), ncol=2,
              frameon=False, handlelength=0.95, handleheight=0.58, columnspacing=0.65,
              handletextpad=0.3, fontsize=FONT - 1.5, borderaxespad=0.0)
    return numbers


def rq1_figure(boids, root, out):
    """RQ1 in two rows: Boids (a-f, including teacher) and coordination regions (g-k)."""
    fig = plt.figure(figsize=(7.16, 2.28))
    top_widths = (1.12, 0.48, 1.0, 0.12, 0.9, 0.05, 0.9, 0.05, 0.9, 0.05, 0.9)
    top_grid = fig.add_gridspec(
        1, len(top_widths), width_ratios=top_widths, wspace=0,
        left=0.058, right=0.995, bottom=0.57, top=0.94,
    )
    top_axes = [fig.add_subplot(top_grid[0, k]) for k in (0, 2, 4, 6, 8, 10)]
    boids_numbers, _ = boids_row(top_axes, boids)

    bot_widths = (0.95, 0.06, 0.95, 0.06, 0.95, 0.06, 0.95, 0.52, 2.05)
    bot_grid = fig.add_gridspec(
        1, len(bot_widths), width_ratios=bot_widths, wspace=0,
        left=0.058, right=0.962, bottom=0.04, top=0.33,
    )
    bot_axes = [fig.add_subplot(bot_grid[0, k]) for k in (0, 2, 4, 6, 8)]
    scr_snapshot_panels(bot_axes[:4], root, letters="ghij", one_line=True)
    regions = regions_panel_ari(bot_axes[4], root, letter="k")
    save_figure(fig, out / "rq1", dpi=300)
    return {"boids": boids_numbers, "regions": regions}


def channel_snapshots(axes, channel):
    """Seed 0 before the obstacle, right after it appears, and once the channel has re-formed."""
    config = read_json(channel / "summary.json")["config"]
    stored = read_json(channel / "runs/seed0.json")
    start = config["events"]["obstacle"]
    rounds = (start - 1, start + 2, start + stored["events"]["obstacle"]["recovery"]["rounds"])
    run, snapshot = run_seed(config, 0, extra=rounds)
    if run["curves"]["quality"] != stored["curves"]["quality"]:
        raise AssertionError("Channel snapshot run differs from the stored seed 0")
    positions = snapshot["positions"].numpy()
    ends = [snapshot["source"], snapshot["dest"]]
    x0, x1, y0, y1 = config["obstacle"]
    for ax, t, label in zip(axes, rounds, ("before", "obstacle", "re-formed"), strict=True):
        state = snapshot["snapshots"][t]
        channel_on, failed = state["channel"].numpy(), state["failed"].numpy()
        if failed.any():
            ax.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0, facecolor=RED, alpha=0.1,
                                   edgecolor=RED, linewidth=0.6))  # fmt: skip
            ax.scatter(*positions[failed].T, marker="x", c=RED, s=3, linewidths=0.35, alpha=0.5)
        idle = ~channel_on & ~failed
        ax.scatter(*positions[idle].T, c=IDLE, s=1.2, linewidths=0)
        ax.scatter(*positions[channel_on].T, c=BLUE, s=3.5, linewidths=0)
        ax.scatter(*positions[ends].T, marker="*", c=INK, s=30, linewidths=0, zorder=3)
        snapshot_axes(ax, f"$t={t}$, {label}")
    return {"seed": 0, "rounds": list(rounds)}


def regions_snapshots(axes, root, scenario, index):
    """Asynchronous crash run of the learned program: before the crash, just after it, and once
    its regions have re-elected their leaders."""
    config = protocol(read_json(root / "manifest.json")["profile"])
    model = scr_program(root, scenario, "parametric")
    episode = clusters.leader_crash(config, index, model, ASYNC, moving=scenario == "moving")
    trace = on_devices(episode, model, ASYNC, config.data_seed)
    name = f"distributed-async-0.5-crash-4-0.3-{index}.json"
    ari = read_json(root / scenario / "episodes/parametric/seed0" / name)["curves"]["ari"]
    truth, fault = clusters.truth_labels(episode), episode.fault_at
    elected = trace.fields[..., ELECTED].numpy() > 0.5
    labels_t = trace.leaders.numpy()

    def settled(t):  # every region led by a live leader, and the zones recovered
        alive = episode.active[t].numpy()
        heads = set(np.flatnonzero(elected[t] & alive).tolist())
        return ari[t] >= 0.95 and set(labels_t[t][alive].tolist()) == heads

    back = next(t for t in range(fault + 1, episode.rounds - 3)
                if all(settled(u) for u in range(t, t + 3)))  # fmt: skip
    rounds = (fault - 1, fault + 2, back)
    positions = episode.positions.numpy()
    for ax, t, label in zip(axes, rounds, ("before", "crash", "re-elected"), strict=True):
        labels, active = trace.leaders[t].numpy(), episode.active[t].numpy()
        if not np.isclose(adjusted_rand_score(truth[t][active], labels[active]), ari[t]):
            raise AssertionError("Regions snapshot run differs from the stored episode")
        regions_panel(ax, positions, labels, truth=truth[t], active=active,
                      elected=trace.fields[t, :, ELECTED].numpy() > 0.5,
                      title=f"$t={t}$, {label}")  # fmt: skip
    return {"episode": episode.key, "seed": 0, "fault_at": fault, "rounds": list(rounds),
            "ari": [float(ari[t]) for t in rounds]}  # fmt: skip


@torch.no_grad()
def rq2_snapshots(channel, root, out, *, scenario="static", index=0):
    """Before/after views of the channel and the regions, each checked against its stored run."""
    fig = plt.figure(figsize=(7.16, 1.0))
    grid = fig.add_gridspec(1, 7, width_ratios=(1, 1, 1, 0.16, 1, 1, 1), wspace=0.06)
    axes = [fig.add_subplot(grid[0, k]) for k in (0, 1, 2, 4, 5, 6)]
    shown = {
        "channel": channel_snapshots(axes[:3], channel),
        "regions": regions_snapshots(axes[3:], root, scenario, index),
    }
    fig.subplots_adjust(left=0.02, right=0.997, bottom=0.01, top=0.9)
    for ax, label in ((axes[0], "channel"), (axes[3], "regions")):
        ax.set_ylabel(label, fontsize=FONT, color=INK, labelpad=2)
    save_figure(fig, out / "rq2-snapshots", dpi=300)
    return shown


def recovery_table(channel, numbers, out):
    """A compact LaTeX tabular with the recovery rounds behind the RQ2 figure."""
    events = read_json(channel / "summary.json")["events"]

    def rounds(entry):
        if entry["median_rounds"] is None:
            return "--"
        return f"{entry['median_rounds']:g} ({entry['max_rounds']})"

    lines = [
        r"\begin{tabular}{@{}llrr@{}}",
        r"\toprule",
        r"Program & Event & Median (max) rounds & Recovered \\",
        r"\midrule",
    ]
    for name, label in (("obstacle", "obstacle"), ("clear", "cleared"), ("relocate", "relocation")):
        e = events[name]
        total = e["recovered"] + e["censored"]
        lines.append(f"Channel & {label} & {rounds(e)} & {e['recovered']}/{total} \\\\")
    for sched, key in (("sync", "scr_sync"), ("async", "scr")):
        for method in ("parametric", "fixed-combined"):
            e = numbers[key][method]
            lines.append(
                f"SCR ({sched}), {STYLE[SCR_ROLE[method]][2]} & leader crash & {rounds(e)} & "
                f"{e['recovered']}/{e['runs']} \\\\"
            )
    for fault, label in (("node_stop", "paused devices"), ("partition", "partition"),
                         ("link_loss", "link loss")):
        for method in ("parametric", "fixed-combined"):
            e = numbers["scr_transient"][fault][method]
            lines.append(
                f"SCR (async), {STYLE[SCR_ROLE[method]][2]} & {label} & {rounds(e)} & "
                f"{e['recovered']}/{e['runs']} \\\\"
            )
    lines += [r"\bottomrule", r"\end{tabular}", ""]
    (out / "rq2-recovery.tex").write_text("\n".join(lines))


def recovery_rows(channel, numbers):
    """Per-run recovery rounds of every runtime fault on Channel and Coordination Regions."""
    runs = [read_json(p) for p in sorted((channel / "runs").glob("seed*.json"))]
    groups = {"channel": [], "crash": [], "transient": []}
    for event, label in (("obstacle", "obstacle"), ("clear", "obstacle removed"),
                         ("relocate", "destination moved")):  # fmt: skip
        values = [r["events"][event]["recovery"]["rounds"] for r in runs]
        recovered = [v for v in values if v is not None]
        groups["channel"].append((label, BLUE, [(recovered, len(values), "o")]))
    for sched, key in (("sync", "scr_sync"), ("async", "scr")):
        for method, name in (("parametric", "DifField"), ("fixed-combined", "initial")):
            entry = numbers[key][method]
            colour = STYLE[SCR_ROLE[method]][0]
            groups["crash"].append((f"{name}, {sched}", colour,
                                    [(entry["rounds"], entry["runs"], "o")]))  # fmt: skip
    for fault, title in (("node_stop", "paused"), ("partition", "partition"),
                         ("link_loss", "link loss")):
        for method, name in (("parametric", "DifField"), ("fixed-combined", "initial")):
            entry = numbers["scr_transient"][fault][method]
            colour = STYLE[SCR_ROLE[method]][0]
            groups["transient"].append((f"{title}, {name}", colour,
                                        [(entry["rounds"], entry["runs"], "o")]))  # fmt: skip
    return groups


def rq2_recovery_figure(channel, numbers_rq2, out):
    """Rounds to recover after each event (log scale): one dot per recovered run, a bar at the
    median, and the recovered runs out of all runs on the right."""
    groups = recovery_rows(channel, numbers_rq2)
    blocks = {
        panel("a", "channel"): [(label, colour, series[0])
                                for label, colour, series in groups["channel"]],
        panel("b", "regions: leader crash"): [(label, colour, series[0])
                                              for label, colour, series in groups["crash"]],
        panel("c", "regions: transient faults (async)"): [
            (label, colour, series[0]) for label, colour, series in groups["transient"]
        ],
    }  # fmt: skip
    heights = [len(rows) + 0.35 for rows in blocks.values()]
    fig, axes = plt.subplots(len(blocks), 1, figsize=(3.5, 2.38), sharex=True,
                             gridspec_kw={"height_ratios": heights, "hspace": 0.48})  # fmt: skip
    generator = np.random.default_rng(0)
    numbers = {}
    for ax, (title, rows) in zip(axes, blocks.items(), strict=True):
        for y, (label, colour, (values, total, _)) in enumerate(rows):
            numbers[f"{title}/{label}"] = {
                "recovered": len(values), "runs": total,
                "median": float(np.median(values)) if values else None}  # fmt: skip
            if values:
                v = np.maximum(np.asarray(values, dtype=float), 1.0)
                jitter = generator.uniform(-0.26, 0.26, len(v))
                ax.scatter(v, y + jitter, s=4, color=colour, alpha=0.45, linewidths=0, zorder=2)
                ax.plot([np.median(v)] * 2, [y - 0.36, y + 0.36], color=colour, linewidth=1.8,
                        solid_capstyle="butt", zorder=3)  # fmt: skip
            ax.text(1.02, y, f"{len(values)}/{total}", transform=ax.get_yaxis_transform(),
                    fontsize=FONT - 1, color=colour, va="center", ha="left")  # fmt: skip
        ax.set_yticks(range(len(rows)), [r[0] for r in rows], fontsize=FONT - 1)
        ax.tick_params(axis="y", length=0)
        ax.set_ylim(len(rows) - 0.5, -0.5)
        ax.grid(axis="y", visible=False)
        ax.set_title(title, fontsize=FONT - 0.5, loc="left", pad=2)
    axes[0].text(1.02, 1.04, "recovered", transform=axes[0].transAxes, fontsize=FONT - 1.5,
                 color=MUTED, va="bottom", ha="left")  # fmt: skip
    axes[-1].set_xscale("log")
    axes[-1].set_xticks([1, 3, 10, 30, 100, 300], ["≤1", "3", "10", "30", "100", "300"])
    axes[-1].set_xlim(0.8, 400)
    axes[-1].set_xlabel("rounds to recover")
    fig.subplots_adjust(left=0.31, right=0.83, top=0.94, bottom=0.11)
    save_figure(fig, out / "rq2-recovery", dpi=300)
    return numbers


def scaling_times(path):
    """Mean seconds of one channel rollout per (devices, k), as written by benchmark_scaling.py."""
    with path.open(encoding="utf-8") as stream:
        return {
            (int(row["num_nodes"]), int(row["k_neighbors"])): float(row["mean_time_seconds"])
            for row in csv.DictReader(stream)
        }


def rq3_figure(root, out, *, degree=32):
    """Rollout time on CPU and GPU at one degree (a), and the GPU speedup for every degree (b)."""
    cpu = scaling_times(root / "channel_scaling_multi_cpu/aggregated.csv")
    gpu = scaling_times(root / "channel_scaling_multi_cuda/aggregated.csv")
    nodes, degrees = sorted({n for n, _ in cpu}), sorted({k for _, k in cpu})
    fig, axes = plt.subplots(1, 2, figsize=(3.5, 1.5), constrained_layout=True)
    ax = axes[0]
    for label, times, colour, marker, linestyle in (
        ("CPU", cpu, ORANGE, "s", "--"),
        ("GPU", gpu, BLUE, "o", "-"),
    ):
        ax.plot(nodes, [times[n, degree] for n in nodes], color=colour, marker=marker,
                linestyle=linestyle, markersize=2.8, label=label)  # fmt: skip
    ax.set(yscale="log", ylabel="seconds per rollout")
    ax.set_yticks([0.2, 0.5, 1, 2], ["0.2", "0.5", "1", "2"])
    ax.set_title(panel("a", f"time, $k={degree}$"), loc="left")
    ax.legend(frameon=False, loc="upper left", handlelength=2.0)
    ax = axes[1]
    shades = plt.get_cmap("viridis")(np.linspace(0.05, 0.85, len(degrees)))
    for k, shade in zip(degrees, shades, strict=True):
        ax.plot(nodes, [cpu[n, k] / gpu[n, k] for n in nodes], color=shade, marker="o",
                markersize=2.2, label=f"$k={k}$")  # fmt: skip
    ax.axhline(1, color=MUTED, linewidth=0.6, linestyle=":")
    ax.set(ylabel="GPU speedup", ylim=(0, None))
    ax.set_title(panel("b", "GPU speedup"), loc="left")
    ax.legend(frameon=False, loc="upper left", fontsize=FONT - 1.5, handlelength=1.2,
              labelspacing=0.15)  # fmt: skip
    for ax in axes:
        ax.set_xscale("log")
        ax.set_xticks(nodes, [f"{n // 1000}k" for n in nodes])
        ax.minorticks_off()
        ax.set_xlabel("devices $N$")
    save_figure(fig, out / "rq3-scaling", dpi=300)
    speedup = {f"N{n}-k{k}": cpu[n, k] / gpu[n, k] for n in nodes for k in degrees}
    return {"speedup": speedup, "max_speedup": max(speedup.values()),
            "cpu_seconds": {f"N{n}": cpu[n, degree] for n in nodes},
            "gpu_seconds": {f"N{n}": gpu[n, degree] for n in nodes}}  # fmt: skip


def paper_figures(profile="paper-cpu", figures=None):
    style()
    generated = PROJECT / "generated/seams"
    channel = generated / f"channel-adaptation-{profile}"
    boids = generated / f"boids-{profile}"
    scr_root = generated / "comparison/clusters-sharp"
    out = figures or (
        PROJECT / "paper-2026-seams-diffield/figures"
        if profile == "paper-cpu"
        else generated / "figures-smoke"
    )
    out.mkdir(parents=True, exist_ok=True)
    numbers = rq2_figure(channel, boids, scr_root, out)
    numbers["rq1"] = rq1_figure(boids, scr_root, out)
    numbers["scr_snapshot"] = rq1_scr_snapshot(scr_root, out)
    numbers["rq2_snapshots"] = rq2_snapshots(channel, scr_root, out)
    recovery_table(channel, numbers, out)
    numbers["recovery"] = rq2_recovery_figure(channel, numbers, out)
    numbers["channel"] = read_json(channel / "summary.json")["events"]
    numbers["boids_adaptation"] = read_json(boids / "adaptation/summary.json")["summary"]
    numbers["boids_devices"] = read_json(boids / "devices/summary.json")["summary"]
    numbers["rq3"] = rq3_figure(PROJECT / "generated/results", out)
    json_write(generated / f"paper-numbers-{profile}.json", numbers)
    print(f"Paper figures: {out}", flush=True)
    return numbers
