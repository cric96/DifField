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
from matplotlib.patches import Rectangle
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import adjusted_rand_score

from ..shared.plotting.common import LineCollection, plt
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
from .artifacts import PROJECT, json_write, read_json
from .boids_dynamics import FIXED, TEACHER
from .boids_learning import combine, free_rollout, make_model, objective, transitions
from .channel_adaptation import run_seed
from .randomness import seed_for
from .space_fluid import clusters
from .space_fluid.execution import central, decentralized
from .space_fluid.metrics import recovery
from .space_fluid.program import ELECTED, make_program
from .space_fluid.training import load_program

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
LEGEND = ("diffield", "diffield_async", "teacher", "teacher_async", "gnn32", "gnn64",
          "initial", "frozen")  # fmt: skip
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


def band(curves, samples=2000, seed=0):
    """Mean and 95% percentile bootstrap over runs (rows)."""
    curves = np.asarray(curves, dtype=float)
    generator = np.random.default_rng(seed)
    draws = curves[generator.integers(len(curves), size=(samples, len(curves)))].mean(1)
    low, high = np.quantile(draws, [0.025, 0.975], axis=0)
    return curves.mean(0), low, high


def draw(ax, x, curves, role, *, fill=True, **kwargs):
    colour, linestyle, label = STYLE[role]
    mean, low, high = band(curves)
    ax.plot(x, mean, color=colour, linestyle=linestyle, label=label, **kwargs)
    if fill and len(curves) > 1:
        ax.fill_between(x, low, high, color=colour, alpha=0.18, linewidth=0)


def events(ax, marks, top=1.02):
    for name, t in marks.items():
        ax.axvline(t, color=MUTED, linewidth=0.6, zorder=0)
        ax.text(t + 3, top, name, transform=ax.get_xaxis_transform(), fontsize=FONT - 1,
                color=MUTED, va="bottom", ha="left")  # fmt: skip


def tag(ax, label):
    ax.text(-0.02, 1.0, f"({label})", transform=ax.transAxes, fontsize=FONT + 0.5,
            fontweight="semibold", ha="right", va="bottom", color=INK)  # fmt: skip


def scr_curves(root, scenario="static", condition="crash", methods=tuple(SCR_ROLE)):
    """ARI per round of the asynchronous Space-Fluid clustering runs (runs x rounds).

    Full-batch training makes the parametric seeds identical, so duplicate runs
    are dropped: a band over copies would look narrower than the evidence.
    """
    result = {}
    for method in methods:
        name = f"distributed-async-0.5-{condition}-*.json"
        pattern = f"{root}/{scenario}/episodes/{method}/seed*/{name}"
        rows = [read_json(Path(p))["curves"]["ari"] for p in sorted(glob.glob(pattern))]
        if not rows:
            raise FileNotFoundError(pattern)
        result[method] = np.unique(np.array(rows), axis=0)
    return result


def scr_numbers(scr, fault=32, tolerance=0.05, consecutive=5):
    """ARI before the fault, at its trough and at the end; recovery measured after the trough."""
    numbers = {}
    for method, ari in scr.items():
        rounds = []
        for curve in ari:
            pre = float(curve[fault - 5 : fault].mean())
            trough = fault + int(curve[fault:].argmin())
            result = recovery([1 - a for a in curve], trough, 1 - (pre - tolerance), consecutive)
            if result["rounds"] is not None:
                rounds.append(trough - fault + result["rounds"])
        numbers[method] = {
            "runs": len(ari),
            "ari_before": float(ari[:, fault - 5 : fault].mean()),
            "ari_trough": float(ari[:, fault:].min(1).mean()),
            "ari_final": float(ari[:, -10:].mean()),
            "recovered": len(rounds),
            "median_rounds": float(np.median(rounds)) if rounds else None,
            "max_rounds": max(rounds) if rounds else None,
        }
    return numbers


def boids_adaptation_curves(run, size, key="spread"):
    by_method = {}
    for path in sorted((run / "adaptation/episodes").glob(f"*/*/N{size}-*.json")):
        row = read_json(path)
        if row["status"] == "complete":
            by_method.setdefault(row["method"], []).append(row["curves"][key])
    return {m: np.array(c) for m, c in by_method.items()}


def boids_device_curves(run, size, executor="async", key="spread"):
    """Curves of the device stage: the learned program and the teacher on independent agents."""
    curves = {}
    for method in ("parametric", "oracle"):
        paths = sorted((run / "devices/episodes" / f"{method}-{executor}").glob(f"N{size}-*.json"))
        rows = [read_json(path) for path in paths]
        curves[method] = np.array([r["curves"][key] for r in rows if r["status"] == "complete"])
    return curves


def rq2_figure(channel, boids, scr_root, out, *, size=64, scenario="static", condition="crash"):
    channel_runs = [read_json(p) for p in sorted((channel / "runs").glob("seed*.json"))]
    channel_config = read_json(channel / "summary.json")["config"]
    boids_config = read_json(boids / "adaptation/summary.json")["config"]
    swarm = boids_adaptation_curves(boids, size)
    devices = boids_device_curves(boids, size)
    scr = scr_curves(scr_root, scenario, condition)
    clean = scr_curves(scr_root, scenario, "clean")

    fig, axes = plt.subplots(1, 3, figsize=(7.16, 2.1), constrained_layout=True)
    ax = axes[0]
    x, start = np.arange(channel_config["rounds"]), channel_config["events"]["obstacle"]
    frozen = [r["curves"]["frozen_quality"][start:] for r in channel_runs]
    draw(ax, x[start:], frozen, "frozen")
    draw(ax, x, [r["curves"]["quality"] for r in channel_runs], "diffield")
    events(ax, channel_config["events"])
    ax.set(xlabel="round", ylabel="channel quality", ylim=(-0.03, 1.03), xlim=(0, x[-1]))
    tag(ax, "a")

    ax = axes[1]
    x = np.arange(1, boids_config["rounds"] + 1)
    for method in ("gnn32", "gnn64", "parametric", "oracle"):
        draw(ax, x, swarm[method], BOIDS_ROLE[method], fill=method in ("parametric", "gnn64"))
    for method, role in (("oracle", "teacher_async"), ("parametric", "diffield_async")):
        draw(ax, x, devices[method], role, fill=False, linewidth=0.9)
    events(ax, boids_config["events"])
    ax.set(xlabel="round", ylabel=f"swarm spread ($N={size}$)", xlim=(0, x[-1]))
    tag(ax, "b")

    ax = axes[2]
    x = np.arange(scr["parametric"].shape[1])
    for method in ("fixed-combined", "parametric"):
        colour, _, _ = STYLE[SCR_ROLE[method]]
        mean, _, _ = band(clean[method])
        ax.plot(x, mean, color=colour, linestyle=(0, (1, 1.5)), linewidth=0.8, alpha=0.8)
        draw(ax, x, scr[method], SCR_ROLE[method])
    fault = 32
    ax.axvspan(fault, x[-1], color=MUTED, alpha=0.08, linewidth=0)
    events(ax, {condition: fault})
    ax.set(xlabel="round", ylabel="clustering ARI", ylim=(0, 1.02), xlim=(0, x[-1]))
    tag(ax, "c")

    handles = [
        plt.Line2D([], [], color=c, linestyle=ls, label=lbl)
        for c, ls, lbl in (STYLE[role] for role in LEGEND)
    ]
    handles.append(plt.Line2D([], [], color=MUTED, linestyle=(0, (1, 1.5)), label="no fault"))
    fig.legend(handles=handles, loc="outside lower center", ncol=5, frameon=False,
               handlelength=2.2, columnspacing=1.6, labelspacing=0.25)  # fmt: skip
    for suffix in ("pdf", "png"):
        fig.savefig(out / f"rq2-adaptation.{suffix}")
    plt.close(fig)
    return {"scr": scr_numbers(scr), "scr_clean": scr_numbers(clean)}


def boids_validation(boids, variant, validation):
    """Validation loss every few updates of each seed, from update 0, and the selected update."""
    runs = []
    for path in sorted((boids / "jobs/observed" / variant).glob("seed*/training.json")):
        job = read_json(path)
        with torch.no_grad():  # update 0 is evaluated for selection but not kept in the history
            start = float(objective(make_model(variant, job["seed"]), validation)[1])
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


def rq1_boids_figure(boids, out, *, zoom=200):
    """Validation loss over training (a) and the three learned weights against the teacher's."""
    data = torch.load(boids / "data/traces.pt", weights_only=True)
    validation = combine([transitions(e) for e in data["validation"]])
    errors = {}
    for method in BOIDS_ROLE:
        paths = sorted((boids / "episodes/observed" / method).glob("*/*.json"))
        errors[method] = float(np.mean([read_json(p)["metrics"]["position_rmse"] for p in paths]))

    fig, axes = plt.subplots(1, 4, figsize=(7.16, 1.55), constrained_layout=True,
                             gridspec_kw={"width_ratios": (1.35, 1, 1, 1)})  # fmt: skip
    ax, losses = axes[0], {}
    for variant in ("gnn32", "gnn64", "parametric"):
        runs = boids_validation(boids, variant, validation)
        x = np.array([u for u, _ in runs[0]["points"]])
        curves = np.array([[v for _, v in run["points"]] for run in runs])
        picks = [(run["selected"], dict(run["points"])[run["selected"]]) for run in runs]
        colour, linestyle, label = STYLE[BOIDS_ROLE[variant]]
        ax.fill_between(x, curves.min(0), curves.max(0), color=colour, alpha=0.18, linewidth=0)
        ax.plot(x, np.median(curves, 0), color=colour, linestyle=linestyle, label=label)
        ax.scatter(*zip(*picks, strict=True), s=9, color=colour, edgecolors="white",
                   linewidths=0.4, zorder=3)  # fmt: skip
        losses[variant] = {
            "initial_median": float(np.median(curves[:, 0])),
            "final_median": float(np.median(curves[:, -1])),
            "selected_median": float(np.median([v for _, v in picks])),
        }
        if variant == "parametric":
            initial = float(curves[0, 0])  # the untrained weights: identical for every seed
            weight_runs = runs
    colour, linestyle, label = STYLE["initial"]
    ax.axhline(initial, color=colour, linestyle=linestyle, linewidth=0.9, zorder=1)
    plateau = np.median([losses[v]["final_median"] for v in ("gnn32", "gnn64")])
    ax.text(x[-1], 2 * initial, f"GNNs: {plateau:.2f}", fontsize=FONT - 1, color=DARK_RED,
            ha="right", va="bottom")  # fmt: skip
    ax.set(xlabel="update", ylabel="validation loss", yscale="log")
    ax.set(xlim=(0, x[-1]), ylim=(1e-13, 30))
    ax.set_yticks([1e0, 1e-4, 1e-8, 1e-12])
    ax.set_title("(a) one-step loss", fontweight="normal", loc="left")

    colour = STYLE["diffield"][0]
    for j, (ax, name, label) in enumerate(
        zip(axes[1:], ("separation", "alignment", "cohesion"), "bcd", strict=True)
    ):
        for run in weight_runs:
            ax.plot(range(len(run["weights"])), [w[j] for w in run["weights"]], color=colour,
                    linewidth=0.7, alpha=0.6)  # fmt: skip
        ax.axhline(TEACHER[j], color=INK, linestyle="--", linewidth=0.8)
        ax.text(zoom, TEACHER[j], f"teacher {TEACHER[j]:g} ", fontsize=FONT - 1, color=INK,
                ha="right", va="top")  # fmt: skip
        ax.set(xlabel="update", xlim=(0, zoom))
        ax.set_title(f"({label}) {name}", fontweight="normal", loc="left")
        if j == 0:
            ax.set_ylabel("weight")

    handles = [plt.Line2D([], [], color=STYLE[r][0], linestyle=STYLE[r][1], label=STYLE[r][2])
               for r in ("diffield", "gnn32", "gnn64", "initial")]  # fmt: skip
    handles += [
        plt.Line2D([], [], color=INK, linestyle="--", label="teacher"),
        plt.Line2D([], [], color=MUTED, marker="o", markersize=3, linestyle="none",
                   label="selected checkpoint"),
    ]  # fmt: skip
    fig.legend(handles=handles, loc="outside lower center", ncol=len(handles), frameon=False,
               handlelength=2.2, columnspacing=1.2)  # fmt: skip
    for suffix in ("pdf", "png"):
        fig.savefig(out / f"rq1-boids.{suffix}")
    plt.close(fig)
    return {"position_rmse": errors, "validation_loss": losses, "initial_weights_loss": initial}


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
    ax.set_title(title, fontweight="normal")


def rq1_boids_trajectories(boids, out, *, episode=0, seed=0, early=10, gnn="gnn64"):
    """Free-running trajectories on a test episode: DifField during training, a trained GNN,
    and the teacher."""
    test = torch.load(boids / "data/traces.pt", weights_only=True)["test"][episode]
    training = read_json(boids / f"jobs/observed/parametric/seed{seed}/training.json")
    history = {h["update"]: h["weights"] for h in training["history"]}
    early = min(history, key=lambda update: abs(update - early))  # short runs record fewer
    selected = training["selected_update"]
    stages = (("untrained", FIXED), (f"update {early}", history[early]),
              (f"update {selected}", history[selected]))  # fmt: skip
    rounds = len(test["target_pos"])
    colours = _get_identity_colors(len(test["positions"]))
    fig, axes = plt.subplots(1, 5, figsize=(7.16, 1.62), constrained_layout=True)
    errors = {}
    for ax, (label, weights), letter in zip(axes, stages, "abc", strict=False):
        model = make_model("parametric", seed)
        with torch.no_grad():
            model.log_weights.copy_(torch.tensor(weights).log())
            positions, _ = free_rollout(model, test["positions"], test["velocities"], rounds)
        traj = torch.cat((test["positions"][None], positions)).numpy()
        trajectories_panel(ax, traj, colours, f"({letter}) DifField, {label}")
        errors[label] = float((positions - test["target_pos"]).norm(dim=-1).pow(2).mean().sqrt())
    model = make_model(gnn, seed)
    checkpoint = boids / "jobs/observed" / gnn / f"seed{seed}" / "best.pt"
    model.load_state_dict(torch.load(checkpoint, weights_only=True)["model"])
    with torch.no_grad():
        positions, _ = free_rollout(model.eval(), test["positions"], test["velocities"], rounds)
    traj = torch.cat((test["positions"][None], positions)).numpy()
    trajectories_panel(axes[3], traj, colours, f"(d) {STYLE[gnn][2]}, trained")
    errors[gnn] = float((positions - test["target_pos"]).norm(dim=-1).pow(2).mean().sqrt())
    teacher = torch.cat((test["positions"][None], test["target_pos"])).numpy()
    trajectories_panel(axes[4], teacher, colours, "(e) teacher", background="#f4f8ff")
    for suffix in ("pdf", "png"):
        fig.savefig(out / f"rq1-boids-trajectories.{suffix}")
    plt.close(fig)
    return {"episode": test["key"], "seed": seed, "updates": [0, early, selected],
            "position_rmse": errors}  # fmt: skip


def snapshot_axes(ax, title, box=(0.0, 1.0, 0.0, 1.0), margin=0.03):
    x0, x1, y0, y1 = box
    ax.set(xticks=[], yticks=[], aspect="equal", xlim=(x0 - margin, x1 + margin),
           ylim=(y0 - margin, y1 + margin))  # fmt: skip
    ax.set_title(title, fontsize=7, fontweight="normal", pad=2, linespacing=1.1)
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
    if method in clusters.LEARNERS:
        return load_program(root / scenario / "checkpoints" / method / "seed0" / "best.pt")
    return make_program(method.removeprefix("fixed-"), **norm).requires_grad_(False)


@torch.no_grad()
def rq1_scr_snapshot(root, out, *, scenario="static", index=0):
    """Field, true zones and final regions of the initial and learned programs (batched, hard)."""
    config = clusters.settings(read_json(root / "manifest.json")["profile"])
    episode = clusters.cluster_episode(config, "test", index, moving=scenario == "moving")
    positions, truth = episode.positions.numpy(), clusters.truth_labels(episode)[-1]
    fig, axes = plt.subplots(1, 4, figsize=(3.5, 1.15))
    axes[0].scatter(*positions.T, c=episode.observations[-1].numpy(), cmap="viridis", s=9,
                    linewidths=0)  # fmt: skip
    snapshot_axes(axes[0], "(a) sensed field")
    axes[1].scatter(*positions.T, c=np.array(ZONES)[truth], s=9, linewidths=0)
    snapshot_axes(axes[1], "(b) true zones")
    shown = {"episode": episode.key, "round": episode.rounds - 1}
    panels = (("fixed-combined", "(c) initial"), ("parametric", "(d) learned"))
    for ax, (method, name) in zip(axes[2:], panels, strict=True):
        trace = central(episode, scr_program(root, scenario, method))
        labels = trace.leaders[-1].numpy()
        stored = root / scenario / "episodes" / method / "seed0"
        stored = read_json(stored / f"{scenario}-batched-clean-4-0.3-{index}.json")
        if stored["final_labels"] != labels.tolist():
            raise AssertionError(f"{method}: snapshot differs from the evaluated episode")
        ari, regions = float(adjusted_rand_score(truth, labels)), len(np.unique(labels))
        elected = trace.fields[-1, :, ELECTED].numpy() > 0.5
        regions_panel(ax, positions, labels, truth=truth, elected=elected,
                      active=np.ones(episode.nodes, dtype=bool),
                      title=f"{name}\nARI {ari:.2f}, $K={regions}$")  # fmt: skip
        shown[method] = {"ari": ari, "regions": regions, "leaders": int(elected.sum())}
    fig.subplots_adjust(left=0.005, right=0.995, bottom=0.01, top=0.74, wspace=0.07)
    for suffix in ("pdf", "png"):
        fig.savefig(out / f"rq1-scr-snapshot.{suffix}", dpi=300)
    plt.close(fig)
    return shown


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
    """Asynchronous crash run of the learned program: at the trough and once recovered."""
    config = clusters.settings(read_json(root / "manifest.json")["profile"])
    episode = clusters.cluster_episode(config, "test", index, condition="crash",
                                       moving=scenario == "moving")  # fmt: skip
    trace = decentralized(
        episode, scr_program(root, scenario, "parametric"), activation_probability=0.5,
        seed=seed_for(config.data_seed, episode.key, "schedule"),
    )  # fmt: skip
    name = f"distributed-async-0.5-crash-4-0.3-{index}.json"
    ari = read_json(root / scenario / "episodes/parametric/seed0" / name)["curves"]["ari"]
    truth, fault = clusters.truth_labels(episode), episode.fault_at
    trough = fault + int(np.argmin(ari[fault:]))
    pre = float(np.mean(ari[fault - 5 : fault]))  # recovered: within 0.05 ARI, as in the table
    rounds = (trough, trough + recovery([1 - a for a in ari], trough, 1 - (pre - 0.05))["rounds"])
    positions = episode.positions.numpy()
    for ax, t, label in zip(axes, rounds, ("crash", "recovered"), strict=True):
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
    fig = plt.figure(figsize=(7.16, 1.5))
    grid = fig.add_gridspec(1, 6, width_ratios=(1, 1, 1, 0.14, 1, 1), wspace=0.06)
    axes = [fig.add_subplot(grid[0, k]) for k in (0, 1, 2, 4, 5)]
    shown = {
        "channel": channel_snapshots(axes[:3], channel),
        "regions": regions_snapshots(axes[3:], root, scenario, index),
    }
    fig.subplots_adjust(left=0.003, right=0.997, bottom=0.01, top=0.82)
    cases = ("channel: a large obstacle appears", "coordination regions: devices crash")
    for group, label in zip((axes[:3], axes[3:]), cases, strict=True):
        left, right = group[0].get_position(), group[-1].get_position()
        fig.text((left.x0 + right.x1) / 2, left.y1 + 0.1, label, ha="center", va="bottom",
                 fontsize=FONT, fontweight="semibold", color=INK)  # fmt: skip
    for suffix in ("pdf", "png"):
        fig.savefig(out / f"rq2-snapshots.{suffix}", dpi=300)
    plt.close(fig)
    return shown


def recovery_table(channel, boids, scr, out, size=64):
    """A compact LaTeX tabular with the recovery rounds behind the RQ2 figure."""
    events = read_json(channel / "summary.json")["events"]
    swarm = read_json(boids / "adaptation/summary.json")["summary"]
    devices = read_json(boids / "devices/summary.json")["summary"]

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
    for method in ("oracle", "parametric", "gnn32", "gnn64"):
        e = swarm[f"N{size}/{method}"]
        runs = e["runs"] - e["diverged"]
        lines.append(
            f"Boids, {STYLE[BOIDS_ROLE[method]][2]} & split, join & "
            f"{rounds(e['split'])}, {rounds(e['join'])} & "
            f"{e['split']['recovered']}/{runs}, {e['join']['recovered']}/{runs} \\\\"
        )
    for method in ("parametric", "oracle"):
        e = devices[f"{method}/async"]
        runs = e["runs"] - e["diverged"]
        lines.append(
            f"Boids, {STYLE[BOIDS_ROLE[method]][2]}, async devices & split, join & "
            f"{rounds(e['split'])}, {rounds(e['join'])} & "
            f"{e['split']['recovered']}/{runs}, {e['join']['recovered']}/{runs} \\\\"
        )
    for method in ("parametric", "fixed-combined"):
        e = scr[method]
        lines.append(
            f"SCR, {STYLE[SCR_ROLE[method]][2]} & crash & {rounds(e)} & "
            f"{e['recovered']}/{e['runs']} \\\\"
        )
    lines += [r"\bottomrule", r"\end{tabular}", ""]
    (out / "rq2-recovery.tex").write_text("\n".join(lines))


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
    ax.set_title(f"(a) time, $k={degree}$", fontweight="normal", loc="left")
    ax.legend(frameon=False, loc="upper left", handlelength=2.0)
    ax = axes[1]
    shades = plt.get_cmap("viridis")(np.linspace(0.05, 0.85, len(degrees)))
    for k, shade in zip(degrees, shades, strict=True):
        ax.plot(nodes, [cpu[n, k] / gpu[n, k] for n in nodes], color=shade, marker="o",
                markersize=2.2, label=f"$k={k}$")  # fmt: skip
    ax.axhline(1, color=MUTED, linewidth=0.6, linestyle=":")
    ax.set(ylabel="GPU speedup", ylim=(0, None))
    ax.set_title("(b) GPU speedup", fontweight="normal", loc="left")
    ax.legend(frameon=False, loc="upper left", fontsize=FONT - 1.5, handlelength=1.2,
              labelspacing=0.15)  # fmt: skip
    for ax in axes:
        ax.set_xscale("log")
        ax.set_xticks(nodes, [f"{n // 1000}k" for n in nodes])
        ax.minorticks_off()
        ax.set_xlabel("devices $N$")
    for suffix in ("pdf", "png"):
        fig.savefig(out / f"rq3-scaling.{suffix}", dpi=300)
    plt.close(fig)
    speedup = {f"N{n}-k{k}": cpu[n, k] / gpu[n, k] for n in nodes for k in degrees}
    return {"speedup": speedup, "max_speedup": max(speedup.values()),
            "cpu_seconds": {f"N{n}": cpu[n, degree] for n in nodes},
            "gpu_seconds": {f"N{n}": gpu[n, degree] for n in nodes}}  # fmt: skip


def paper_figures(profile="paper-cpu", figures=None):
    style()
    generated = PROJECT / "generated/seams"
    channel = generated / f"channel-adaptation-{profile}"
    boids = generated / f"boids-{profile}"
    scr_root = generated / "comparison/clusters"
    out = figures or (
        PROJECT / "paper-2026-seams-diffield/figures"
        if profile == "paper-cpu"
        else generated / "figures-smoke"
    )
    out.mkdir(parents=True, exist_ok=True)
    numbers = rq2_figure(channel, boids, scr_root, out)
    numbers["boids_rq1"] = rq1_boids_figure(boids, out)
    numbers["boids_trajectories"] = rq1_boids_trajectories(boids, out)
    numbers["scr_snapshot"] = rq1_scr_snapshot(scr_root, out)
    numbers["rq2_snapshots"] = rq2_snapshots(channel, scr_root, out)
    recovery_table(channel, boids, numbers["scr"], out)
    numbers["channel"] = read_json(channel / "summary.json")["events"]
    numbers["boids_adaptation"] = read_json(boids / "adaptation/summary.json")["summary"]
    numbers["boids_devices"] = read_json(boids / "devices/summary.json")["summary"]
    numbers["rq3"] = rq3_figure(PROJECT / "generated/results", out)
    json_write(generated / f"paper-numbers-{profile}.json", numbers)
    print(f"Paper figures: {out}", flush=True)
    return numbers
