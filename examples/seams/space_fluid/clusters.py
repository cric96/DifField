"""Space-Fluid as distributed spatial clustering of static zones.

Each cluster is the zone dominated by one Gaussian and carries its own level;
boundaries are blurred by a soft dominance and levels are close but distinct.
The program should form one region per zone: regions that grow too much merge
zones, regions that grow too little split them. Training is the reconstruction
objective of the main study (regional estimate error + lambda * leaders), with no
labels; the true partition is its optimum. ARI uses labels only for evaluation.
(Smooth Gaussian bumps are ill-posed for Space-Fluid: its regions follow level
sets, so the objective's optimum and the best clustering disagree.)
"""

import itertools
import math
from dataclasses import replace

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.animation import FuncAnimation, PillowWriter
from scipy.sparse import coo_matrix
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.metrics import adjusted_rand_score
from threadpoolctl import threadpool_limits

from ..artifacts import Budget, json_write, read_json
from ..metrics import interval
from ..randomness import rng, seed_for
from . import cluster_gnn
from .campaign import output_lock
from .config import protocol
from .data import RegionEpisode, faults, topology
from .execution import central, decentralized, equivalence
from .metrics import fragmentation, recovery
from .program import ELECTED, LEADER, make_program
from .report import export, method_color
from .training import load_program, train_job

FIXED = ("combined",)  # the initial configuration of the learned program
LEARNERS = ("parametric", "hybrid")
CENTRAL = ("gnn-central", "kmeans", "ward")  # global view of every round
SCENARIOS = ("static", "moving")  # zones fixed, or drifting during the episode
CONDITIONS = ("clean", "link_loss", "node_stop", "partition", "crash")
DRIFT = 0.08  # radius of each zone centre's circular drift in the moving scenario
# Full-batch rate of the main study (0.1 diverges at high lambda there).
RATE = 0.03
SOFTNESS = 0.3  # boundary blur (dominance softmax temperature), chosen on validation
GAP = 0.12  # minimum difference between zone levels
# (train, validation, test per cell, decentralized per cell) episodes.
SIZES = {"compact-cpu": (64, 16, 8, 4), "smoke": (4, 2, 1, 1)}


def gaussians(seed, count, separation):
    """Centres at least ``separation`` apart, distinct widths and amplitudes."""
    generator = rng(seed, "gaussians")
    centres = []
    for _ in range(10_000):
        centre = 0.15 + 0.7 * torch.rand(2, generator=generator)
        if all(float((centre - other).norm()) >= separation for other in centres):
            centres.append(centre)
            if len(centres) == count:
                break
    else:
        raise ValueError(f"Cannot place {count} Gaussians {separation} apart")
    widths = 0.08 + 0.05 * torch.rand(count, generator=generator)
    amplitudes = 0.4 + 0.6 * torch.rand(count, generator=generator)
    return torch.stack(centres), widths, amplitudes


def zone_levels(seed, count):
    """Distinct levels in [0.2, 1], at least ``GAP`` apart, in random spatial order."""
    generator = rng(seed, "levels")
    for _ in range(10_000):
        levels = 0.2 + 0.8 * torch.rand(count, generator=generator)
        if count < 2 or float(levels.sort().values.diff().min()) >= GAP:
            return levels
    raise ValueError(f"Cannot draw {count} levels {GAP} apart")


def cluster_episode(
    config, split, index, count=4, separation=0.3, *, condition="clean", nodes=None, moving=False
):
    """Zones of ``count`` Gaussians; with ``moving`` each centre drifts on a circle of
    radius DRIFT starting from its place (0.5-1 turns per episode, own phase)."""
    nodes = config.nodes if nodes is None else nodes
    rounds = config.eval_rounds if split == "test" else config.train_rounds
    seed = seed_for(config.data_seed, "clusters", split, count, separation, index)
    points, base = topology(nodes, seed, "jittered", config.degree)
    centres, widths, amplitudes = gaussians(seed, count, separation)
    generator = rng(seed, "drift")
    turns = 0.5 + 0.5 * torch.rand(count, generator=generator)
    phase = torch.rand(count, generator=generator)
    time = torch.linspace(0, 1, rounds)[:, None] if moving else torch.zeros(1, 1)
    angle = 2 * math.pi * (turns * time + phase)
    start = 2 * math.pi * phase
    offset = DRIFT * torch.stack((angle.sin() - start.sin(), angle.cos() - start.cos()), -1)
    distance = (points[None, :, None] - (centres + offset)[:, None]).square().sum(-1)
    dominance = amplitudes.log() - distance / (2 * widths.square())
    labels = dominance.argmax(-1).expand(rounds, -1)
    levels = zone_levels(seed, count)
    field = ((dominance / SOFTNESS).softmax(-1) @ levels).expand(rounds, -1)
    # Peak devices crash: the likeliest leaders, the hardest case for self-healing.
    crashed = distance[min(rounds // 3, len(distance) - 1)].argmin(0)
    active, edges, fault, restore = faults(seed, points, base, rounds, condition, crashed)
    return RegionEpisode(
        f"clusters-{split}-{index}-k{count}-s{separation:g}-{condition}"
        + ("-moving" if moving else ""),
        points,
        field.clone(),
        field.clone(),  # noise-free perception
        torch.rand(nodes, generator=rng(seed, "priorities")),
        active,
        edges,
        fault,
        restore,
        {
            "split": split,
            "index": index,
            "count": count,
            "separation": separation,
            "condition": condition,
            "moving": moving,
            "labels": labels.tolist() if moving else labels[0].tolist(),
            "centres": centres.tolist(),
        },
    )


def truth_labels(episode):
    """True zone of every device in every round."""
    labels = np.array(episode.metadata["labels"])
    return np.broadcast_to(labels, (episode.rounds, episode.nodes)) if labels.ndim == 1 else labels


def cluster_metrics(episode, labels, elected=None):
    """ARI against the dominant Gaussian, cluster count and contiguity per round.

    With ``elected``, also the leaders and the regions whose label is not an elected leader.
    """
    truth = truth_labels(episode)
    ari, clusters, fragments, stable, leaders, orphans = [], [], [], [], [], []
    for t in range(episode.rounds):
        active = episode.active[t]
        predicted = labels[t].numpy()[active.numpy()]
        ari.append(float(adjusted_rand_score(truth[t][active.numpy()], predicted)))
        clusters.append(len(np.unique(predicted)))
        if elected is not None:
            heads = set(np.flatnonzero((elected[t] > 0.5).numpy() & active.numpy()).tolist())
            leaders.append(len(heads))
            orphans.append(len(set(predicted.tolist()) - heads))
        fragments.append(fragmentation(labels[t], episode.edges[t], active))
        if t:
            shared = active & episode.active[t - 1]
            stable.append(float((labels[t, shared] == labels[t - 1, shared]).float().mean()))
    last = slice(-10, None)
    count = episode.metadata["count"]
    pre = float(np.mean(ari[episode.fault_at - 5 : episode.fault_at]))
    target = 1 - (pre - 0.05)  # recover to within 0.05 ARI of the pre-fault level
    errors = [1 - a for a in ari]
    metrics = {
        "ari_final": float(np.mean(ari[last])),
        "ari_mean": float(np.mean(ari)),
        "clusters_final": float(np.mean(clusters[last])),
        "count_error": float(np.mean(np.abs(np.array(clusters[last]) - count))),
        "fragmentation": float(np.mean(fragments)),
        "stability": float(np.mean(stable)),
        "fault_recovery": recovery(errors, episode.fault_at, target),
        "restoration_recovery": recovery(errors, episode.restore_at, target),
    }
    curves = {"ari": ari, "clusters": clusters}
    if elected is not None:
        metrics |= {
            "leaders_final": float(np.mean(leaders[last])),
            "orphans_final": float(np.mean(orphans[last])),
            "orphans_mean": float(np.mean(orphans)),
        }
        curves |= {"leaders": leaders, "orphans": orphans}
    return metrics, curves


@torch.no_grad()
def central_baseline(episode, kind, norm, seed):
    """Central references given the true K (chosen on validation episodes).

    Both cluster positions and the normalized value (weight 1). K-means need not
    be contiguous; Ward uses the communication graph as connectivity.
    """
    count = episode.metadata["count"]
    labels = torch.full((episode.rounds, episode.nodes), -1, dtype=torch.long)
    with threadpool_limits(limits=1):
        for t in range(episode.rounds):
            ids = episode.active[t].nonzero().flatten()
            observed = episode.observations[t, ids]
            value = (observed - norm["mean"]) / norm["scale"]
            features = torch.cat((episode.positions[ids], value[:, None]), -1).numpy()
            k = min(count, len(ids))
            if kind == "kmeans":
                fitted = KMeans(n_clusters=k, n_init=5, random_state=seed).fit_predict(features)
            else:
                position = torch.full((episode.nodes,), -1, dtype=torch.long)
                position[ids] = torch.arange(len(ids))
                a, b = episode.edges[t]
                graph = coo_matrix(
                    (np.ones(len(a)), (position[a].numpy(), position[b].numpy())),
                    shape=(len(ids), len(ids)),
                )
                fitted = AgglomerativeClustering(
                    n_clusters=k, linkage="ward", connectivity=graph
                ).fit_predict(features)
            labels[t, ids] = ids[torch.from_numpy(fitted)]
    return labels


def specs(sizes, scenario):
    """Both test scenarios (clean, batched), then faults in the training scenario."""
    test, distributed = sizes[2], sizes[3]
    for panel, index in itertools.product(SCENARIOS, range(test)):
        yield {
            "panel": panel,
            "moving": panel == "moving",
            "count": 4,
            "separation": 0.3,
            "index": index,
            "condition": "clean",
            "executor": "batched",
        }
    for condition, index, executor in itertools.product(
        CONDITIONS, range(distributed), ("sync", "async-0.5")
    ):
        yield {
            "panel": "distributed",
            "moving": scenario == "moving",
            "count": 4,
            "separation": 0.3,
            "index": index,
            "condition": condition,
            "executor": executor,
        }


def methods():
    return [*(f"fixed-{m}" for m in FIXED), *LEARNERS, *CENTRAL]


def bank(config, sizes, scenario):
    moving = scenario == "moving"
    return {
        split: [cluster_episode(config, split, i, moving=moving) for i in range(count)]
        for split, count in (("train", sizes[0]), ("validation", sizes[1]))
    }


def train(out, config, sizes, norm, budget, *, scenario):
    bank_ = bank(config, sizes, scenario)
    for method, seed in itertools.product(LEARNERS, config.seeds):
        directory = out / "checkpoints" / method / f"seed{seed}"
        if not (directory / "best.pt").exists():
            print(f"train {method} seed={seed}", flush=True)
            train_job(directory, config, bank_, norm, method, seed, config.main_lambda, budget)
    for seed in config.seeds:
        budget.check()
        directory = out / "checkpoints" / "gnn-central" / f"seed{seed}"
        if not (directory / "best.pt").exists():
            print(f"train gnn-central seed={seed}", flush=True)
            cluster_gnn.train(directory, config, bank_, norm, seed=seed, penalty=config.main_lambda)


def evaluate(out, config, sizes, norm, budget, *, scenario):  # noqa: PLR0912 -- one branch per executor
    models = {}
    for spec, method, seed in itertools.product(specs(sizes, scenario), methods(), config.seeds):
        central_method = method in CENTRAL
        if spec["panel"] == "distributed" and central_method:
            continue
        learned = method in (*LEARNERS, "gnn-central")
        if seed != config.seeds[0] and (not learned or spec["executor"] == "sync"):
            continue  # deterministic methods and equivalence checks need one seed
        name = "-".join(
            str(spec[k]) for k in ("panel", "executor", "condition", "count", "separation", "index")
        )
        path = out / "episodes" / method / f"seed{seed}" / f"{name}.json"
        if path.exists():
            continue
        budget.check()
        episode = cluster_episode(
            config,
            "test",
            spec["index"],
            spec["count"],
            spec["separation"],
            condition=spec["condition"],
            moving=spec["moving"],
        )
        print(f"evaluate {out.name} {name} {method} seed={seed}", flush=True)
        check, elected = None, None
        if method == "gnn-central":
            if (method, seed) not in models:
                models[method, seed] = cluster_gnn.load(
                    out / "checkpoints" / method / f"seed{seed}" / "best.pt"
                )
            labels = cluster_gnn.labels(models[method, seed], episode)
        elif central_method:
            labels = central_baseline(episode, method, norm, seed_for(config.data_seed, method))
        else:
            if (method, seed) not in models:
                models[method, seed] = (
                    load_program(out / "checkpoints" / method / f"seed{seed}" / "best.pt")
                    if method in LEARNERS
                    else make_program(method.removeprefix("fixed-"), **norm).requires_grad_(False)
                )
            model = models[method, seed]
            with torch.no_grad():
                if spec["executor"] == "batched":
                    trace = central(episode, model)
                else:
                    probability = 1.0 if spec["executor"] == "sync" else 0.5
                    trace = decentralized(
                        episode,
                        model,
                        activation_probability=probability,
                        seed=seed_for(config.data_seed, episode.key, "schedule"),
                    )
                    if probability == 1.0:
                        check = equivalence(central(episode, model), trace)
                        if not (check["identifiers_equal"] and check["values_close"]):
                            raise AssertionError(f"Synchronous mismatch: {check}")
            labels, elected = trace.leaders, trace.fields[..., ELECTED]
        metrics, curves = cluster_metrics(episode, labels, elected)
        json_write(
            path,
            {
                **spec,
                "method": method,
                "seed": seed,
                "episode": name,
                "metrics": metrics,
                "curves": curves,
                "equivalence": check,
                "final_labels": labels[-1].tolist(),
            },
        )


PANELS = (("static", "batched"), ("moving", "batched"), ("distributed", "async-0.5"))


def report(out, config, scenario):
    rows = [read_json(p) for p in sorted((out / "episodes").glob("*/seed*/*.json"))]
    text = [
        f"# Space-Fluid clustering, trained on {scenario} zones",
        "",
        f"Profile **{config.profile}**; {len(rows)} completed evaluations. Each cluster is the "
        "zone dominated by one Gaussian, with its own level (levels >= "
        f"{GAP:g} apart, boundaries blurred with softness {SOFTNESS:g}); moving zones drift on "
        f"circles of radius {DRIFT:g}. The learned program starts from fixed-combined (its "
        "initial configuration). SCR methods do not know the number of clusters; K-means and "
        "Ward receive the true K. Training: reconstruction + "
        f"{config.main_lambda:g} x leader fraction, no labels. Final values average the last "
        "ten rounds. Intervals: two-way bootstrap over seeds and episodes.",
        "",
        "Panels: static and moving = clean test episodes of each scenario (batched); "
        f"distributed = asynchronous DeviceRuntime (p=0.5) on {scenario} zones, all conditions.",
        "",
    ]
    for metric, label in (
        ("ari_final", "Final ARI (higher is better)"),
        ("count_error", "Final |K - true K| (lower is better)"),
        ("fragmentation", "Fragmentation (extra components per cluster)"),
        ("stability", "Assignment stability"),
        ("clusters_final", "Final regions"),
        ("leaders_final", "Final elected leaders (= regions when every region has its leader)"),
        ("orphans_final", "Final regions without an elected leader"),
    ):
        text += [f"## {label}", "", "| Method | " + " | ".join(p for p, _ in PANELS) + " |"]
        text.append("|---|" + "---:|" * len(PANELS))
        for method in methods():
            cells = []
            for panel, executor in PANELS:
                selected = [
                    r
                    for r in rows
                    if r["method"] == method
                    and r["panel"] == panel
                    and r["executor"] == executor
                    and metric in r["metrics"]
                ]
                if not selected:
                    cells.append("—")
                    continue
                stats = interval(selected, metric)
                ci = stats["ci95"]
                cells.append(
                    f"{stats['mean']:.3f}" + (f" [{ci[0]:.3f}, {ci[1]:.3f}]" if ci else "")
                )
            text.append(f"| {method} | " + " | ".join(cells) + " |")
        text.append("")
    text += [
        "## Faults in decentralized execution (async, p=0.5)",
        "",
        "Recovery: rounds until ARI is back within 0.05 of its pre-fault level for five "
        "rounds. Crash removes the devices nearest each Gaussian peak for good.",
        "",
        "| Method | Condition | ARI before | ARI during | ARI final | Recovered after fault |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for method, condition in itertools.product(methods(), CONDITIONS):
        selected = distributed(rows, method, condition)
        if not selected:
            continue
        curves = np.array([r["curves"]["ari"] for r in selected])
        third = curves.shape[1] // 3
        events = [r["metrics"]["fault_recovery"] for r in selected]
        recovered = [e["rounds"] for e in events if e["rounds"] is not None]
        rate = f"{len(recovered)}/{len(events)}" + (
            f", median {np.median(recovered):.0f} rounds" if recovered else ""
        )
        text.append(
            f"| {method} | {condition} | {curves[:, third - 5 : third].mean():.3f} | "
            f"{curves[:, third : 2 * third].mean():.3f} | {curves[:, -10:].mean():.3f} | {rate} |"
        )
    checks = [r["equivalence"] for r in rows if r.get("equivalence")]
    passed = sum(c["identifiers_equal"] and c["values_close"] for c in checks)
    text += [
        "",
        f"Synchronous DeviceRuntime equals batched execution in {passed}/{len(checks)} "
        "checked episodes.",
        "",
    ]
    (out / "REPORT.md").write_text("\n".join(text))
    figures(out, config, rows, scenario)
    for condition in ("clean", "crash", "partition", "link_loss"):
        animate(out, config, condition, scenario)


def distributed(rows, method, condition):
    return [
        r
        for r in rows
        if r["method"] == method
        and r["panel"] == "distributed"
        and r["executor"] == "async-0.5"
        and r["condition"] == condition
    ]


def figures(out, config, rows, scenario):
    episode = cluster_episode(config, "test", 0, moving=scenario == "moving")
    x, y = episode.positions.T
    shown = [m for m in methods() if any(r["method"] == m for r in rows)]
    fig, axes = plt.subplots(
        1, 2 + len(shown), figsize=(3 * (2 + len(shown)), 3.2), layout="constrained"
    )
    axes[0].scatter(x, y, c=episode.observations[-1], cmap="viridis", s=20)
    axes[0].set(title="field (last round)")
    axes[1].scatter(x, y, c=truth_labels(episode)[-1], cmap="tab10", s=20)
    axes[1].set(title="true clusters")
    for ax, method in zip(axes[2:], shown, strict=True):
        record = next(
            r
            for r in rows
            if r["method"] == method
            and r["panel"] == scenario
            and r["index"] == 0
            and r["seed"] == config.seeds[0]
        )
        labels = np.unique(record["final_labels"], return_inverse=True)[1]
        ax.scatter(x, y, c=labels, cmap="tab20", s=20)
        ax.set(
            title=f"{method}\nARI {record['metrics']['ari_final']:.2f}, "
            f"K={record['metrics']['clusters_final']:.0f}"
        )
    for ax in axes:
        ax.set(xticks=[], yticks=[], aspect="equal")
    export(fig, out, "clusters-snapshot")

    fig, axes = plt.subplots(
        1, len(CONDITIONS), figsize=(3.2 * len(CONDITIONS), 3.2), layout="constrained", sharey=True
    )
    for ax, condition in zip(axes, CONDITIONS, strict=True):
        for method in shown:
            curves = [r["curves"]["ari"] for r in distributed(rows, method, condition)]
            if curves:
                ax.plot(np.mean(curves, 0), color=method_color(method), label=method)
        rounds = config.eval_rounds
        ax.axvspan(rounds // 3, 2 * rounds // 3, alpha=0.06, color="black")
        ax.set(title=condition, xlabel="Round", ylabel="ARI")
        ax.grid(alpha=0.15)
    axes[0].legend(fontsize=7)
    export(fig, out, "clusters-faults")


def overview(out):
    """Both training scenarios side by side: final ARI per test panel and under faults."""
    text = [
        "# Space-Fluid clustering: static and moving zones",
        "",
        "Final ARI (higher is better), mean over seeds and episodes; per-scenario reports with "
        "intervals in `static/REPORT.md` and `moving/REPORT.md`.",
        "",
        "| Trained on | Method | static test | moving test | "
        + " | ".join(f"async {c}" for c in CONDITIONS)
        + " |",
        "|---|---|" + "---:|" * (2 + len(CONDITIONS)),
    ]
    for scenario in SCENARIOS:
        rows = [read_json(p) for p in sorted((out / scenario / "episodes").glob("*/seed*/*.json"))]
        for method in methods():
            cells = []
            for panel in SCENARIOS:
                ari = [
                    r["metrics"]["ari_final"]
                    for r in rows
                    if r["method"] == method and r["panel"] == panel
                ]
                cells.append(f"{np.mean(ari):.3f}" if ari else "—")
            for condition in CONDITIONS:
                ari = [r["metrics"]["ari_final"] for r in distributed(rows, method, condition)]
                cells.append(f"{np.mean(ari):.3f}" if ari else "—")
            text.append(f"| {scenario} | {method} | " + " | ".join(cells) + " |")
    (out / "REPORT.md").write_text("\n".join(text) + "\n")


def settings(profile, device="cpu"):
    """Temperatures as in the main study (calibrated on validation for the composed program:
    at T=0.1 the leader term dominates and validation diverges); Adam rate RATE."""
    return replace(
        protocol(profile),
        validate_every=20 if profile != "smoke" else 1,
        learning_rate=RATE,
        device=device,
    )


def campaign(out, profile="compact-cpu", seconds=14400, *, device="cpu"):
    config = settings(profile, device)
    sizes = SIZES[profile]
    with output_lock(out):
        torch.set_num_threads(config.threads)
        torch.use_deterministic_algorithms(True)
        budget = Budget(seconds)
        json_write(
            out / "manifest.json",
            {
                "profile": profile,
                "sizes": sizes,
                "scenarios": SCENARIOS,
                "config": config.to_dict(),
            },
        )
        try:
            for scenario in SCENARIOS:
                norm = normalization(config, sizes, scenario)
                json_write(out / scenario / "manifest.json", {"normalization": norm})
                train(out / scenario, config, sizes, norm, budget, scenario=scenario)
                evaluate(out / scenario, config, sizes, norm, budget, scenario=scenario)
        except TimeoutError:
            print(f"Incomplete, resumable: {out}", flush=True)
            return 2
        for scenario in SCENARIOS:
            report(out / scenario, config, scenario)
        overview(out)
    print(f"Artifacts: {out}", flush=True)
    return 0


def normalization(config, sizes, scenario):
    values = torch.cat([e.observations.flatten() for e in bank(config, sizes, scenario)["train"]])
    return {"mean": float(values.mean()), "scale": max(float(values.std()), 1e-6)}


@torch.no_grad()
def animate(out, config, condition, scenario, index=0):
    """Regions over time in asynchronous DeviceRuntime execution, seed 0, one GIF per event.

    Colours follow leader IDs, so a region keeps its colour while it keeps its leader.
    """
    path = out / "animations" / f"clusters-{condition}.gif"
    if path.exists():
        return
    norm = read_json(out / "manifest.json")["normalization"]
    episode = cluster_episode(
        config, "test", index, condition=condition, moving=scenario == "moving"
    )
    names = [*(f"fixed-{m}" for m in FIXED), *LEARNERS]
    traces = {}
    for name in names:
        model = (
            load_program(out / "checkpoints" / name / "seed0" / "best.pt")
            if name in LEARNERS
            else make_program(name.removeprefix("fixed-"), **norm).requires_grad_(False)
        )
        traces[name] = decentralized(
            episode,
            model,
            activation_probability=0.5,
            seed=seed_for(config.data_seed, episode.key, "schedule"),
        )
    truth = truth_labels(episode)
    x, y = episode.positions.T.numpy()
    fig, axes = plt.subplots(1, 1 + len(names), figsize=(2.6 * (1 + len(names)), 3.1))
    zones = axes[0].scatter(x, y, c=truth[0], cmap="tab10", s=22, vmin=0, vmax=9)
    axes[0].set_title("true zones", fontsize=9)
    artists = []
    for ax, name in zip(axes[1:], names, strict=True):
        dots = ax.scatter(x, y, c=np.zeros(len(x)), cmap="tab20", s=22, vmin=0, vmax=19)
        stars = ax.scatter([], [], marker="*", s=70, c="black")
        dead = ax.scatter([], [], marker="x", s=22, c="gray")
        artists.append((name, dots, stars, dead, ax))
    for ax in axes:
        ax.set(xticks=[], yticks=[], aspect="equal", xlim=(-0.03, 1.03), ylim=(-0.03, 1.03))
    title = fig.suptitle("")
    fault, restore = episode.fault_at, episode.restore_at

    def draw(t):
        end = episode.rounds if condition == "crash" else restore
        state = "ACTIVE" if fault <= t < end else "off"
        event = "" if condition == "clean" else f" | {condition} {state}"
        title.set_text(f"{scenario} zones, round {t + 1}/{episode.rounds}{event}")
        active = episode.active[t].numpy()
        zones.set_array(truth[t])
        for name, dots, stars, dead, ax in artists:
            fields = traces[name].fields[t]
            leaders = fields[:, LEADER].long().numpy()
            dots.set_array(leaders % 20)
            dots.set_alpha(None)
            dots.set_sizes(np.where(active, 22, 0))
            stars.set_offsets(np.c_[x, y][active & (fields[:, ELECTED].numpy() > 0.5)])
            dead.set_offsets(np.c_[x, y][~active])
            ari = adjusted_rand_score(truth[t][active], leaders[active])
            ax.set_title(f"{name}\nARI {ari:.2f}, K {len(np.unique(leaders[active]))}", fontsize=9)
        return []

    fig.subplots_adjust(left=0.01, right=0.99, bottom=0.02, top=0.74, wspace=0.08)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.stem + ".tmp.gif")
    FuncAnimation(fig, draw, frames=episode.rounds, interval=125).save(
        temporary, writer=PillowWriter(fps=8), dpi=80
    )
    temporary.replace(path)
    draw(episode.rounds - 1)
    fig.savefig(path.with_suffix(".png"), dpi=120)
    plt.close(fig)
