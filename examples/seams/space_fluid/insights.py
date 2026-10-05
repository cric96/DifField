"""Collective insights from gradients that flow through the whole trained program.

For each trained model at the main tradeoff, on fixed held-in (Gaussian) and
held-out (ring) test episodes, the surrogate backward is read as a sensitivity map:
which devices and links the regional estimates depend on, how far back in time,
and which SCR block (S election, G distance, C converge-cast, broadcast) the
signal flows through.
A hard perturbation check tests whether surrogate saliency predicts hard-loss changes.
"""

import matplotlib.pyplot as plt
import numpy as np
import torch

from diffield import with_mode

from ..artifacts import json_write, read_json, tensor_write
from ..randomness import rng
from .data import make_episode
from .execution import central
from .program import ELECTED, LEADER, SAMPLE, make_program
from .report import export, method_color
from .training import load_program, objective, surrogate_objective

METHODS = ("parametric", "neural", "gnn")
FAMILIES = ("gaussian", "ring")
WINDOW = 8  # late-window loss used for the spatial map and temporal horizon


def roles(trace, episode):
    """Per round and device: 0 interior, 1 region boundary, 2 leader (hard trace)."""
    labels = trace.fields[..., LEADER].detach()
    result = torch.zeros(episode.rounds, episode.nodes, dtype=torch.long)
    for t in range(episode.rounds):
        a, b = episode.edges[t]
        result[t, b[labels[t, a] != labels[t, b]]] = 1
    result[trace.fields[..., ELECTED].detach() > 0.5] = 2
    return result


def flat_gradient(model):
    return torch.cat([p.grad.flatten() for p in model.parameters() if p.grad is not None])


def analyse(model, episode, norm, config):
    region = hasattr(model, "metric")
    scale, penalty = norm["scale"], config.main_lambda
    model.requires_grad_(True)
    with torch.no_grad():
        hard = central(episode, model)
    record = {"hard_loss": float(objective(episode, hard, scale, penalty))}
    # Sensitivity of the estimation error to every reading, through the relaxed program.
    observations = episode.observations.detach().requires_grad_()
    episode.observations = observations
    model.probe = [] if region else None
    with with_mode("soft", tau=config.error_temperature):
        trace = central(episode, model)
    for outputs in model.probe or []:
        for tensor in outputs.values():
            tensor.retain_grad()
    error = (trace.fields[..., SAMPLE] - episode.truth).square()
    late = torch.autograd.grad(error[-WINDOW:].mean(), observations, retain_graph=True)[0].abs()
    error[episode.active].mean().backward()
    saliency = observations.grad.abs()
    captured = model.probe or []
    model.probe = None
    # Temporal horizon: how far back a late estimate depends on readings.
    horizon = late.sum(1).flip(0)
    record["horizon"] = (horizon / horizon.sum().clamp_min(1e-30)).tolist()
    # Hard check: perturbing the most salient readings must move the hard loss more.
    count = max(1, saliency.numel() // 10)
    salient = saliency.flatten().topk(count).indices
    random = torch.randperm(saliency.numel(), generator=rng(config.data_seed, "insight"))[:count]
    with torch.no_grad():
        for name, chosen in (("salient", salient), ("random", random)):
            perturbed = observations.detach().clone()
            perturbed.view(-1)[chosen] += 0.1 * scale
            episode.observations = perturbed
            changed = objective(episode, central(episode, model), scale, penalty)
            record[f"perturbation_{name}"] = abs(float(changed) - record["hard_loss"])
    episode.observations = observations.detach()
    labels = hard.fields[..., LEADER]
    spatial = {"map": late.sum(0), "labels": labels[-1]}
    if not region:
        return record, spatial
    role = roles(hard, episode)
    mean = saliency.mean().clamp_min(1e-30)
    record["roles"] = {
        name: float(saliency[role == value].mean() / mean) if bool((role == value).any()) else None
        for value, name in enumerate(("interior", "boundary", "leader"))
    }
    # Gradient reaching each block's output (S membership confidence, G potential,
    # C collected total, B broadcast estimate), summed over rounds.
    record["blocks"] = {
        block: float(sum(o[block].grad.norm() for o in captured if o[block].grad is not None))
        for block in ("S", "G", "C", "B")
    }
    # Parameter gradients of the training estimator.
    model.zero_grad(set_to_none=True)
    surrogate_objective(episode, model, scale, penalty, config).backward()
    record["norms"] = {
        name: float(flat_gradient(getattr(model, name)).norm()) for name in ("metric", "strength")
    }
    spatial["roles"] = role[-1]
    return record, spatial


@torch.no_grad()
def learned_components(models, norm):
    """Cost over (range, signal difference) and strength over (value, variance)."""
    mean, scale = norm["mean"], norm["scale"]
    distance, delta = torch.meshgrid(
        torch.linspace(0, 0.3, 31), torch.linspace(0, 3 * scale, 31), indexing="ij"
    )
    value, variance = torch.meshgrid(
        torch.linspace(mean - 2 * scale, mean + 3 * scale, 31),
        torch.linspace(0, scale**2, 31),
        indexing="ij",
    )
    grids = {}
    for name, model in models.items():
        cost = model.metric(
            distance.flatten(), torch.full_like(delta.flatten(), mean), mean + delta.flatten()
        )
        strength = model.strength(
            value.flatten(), value.flatten(), variance.flatten(), torch.zeros(value.numel())
        )
        grids[name] = {"cost": cost.reshape(31, 31), "strength": strength.reshape(31, 31)}
    return {
        "distance": distance,
        "delta": delta,
        "value": value,
        "variance": variance,
        "grids": grids,
    }


def checkpoint(out, config, method, seed):
    return out / "checkpoints" / f"{method}-l{config.main_lambda:g}" / f"seed{seed}" / "best.pt"


def insights(out, config, norm, budget):
    path = out / "insights.json"
    if path.exists():
        return
    paths = {(m, s): checkpoint(out, config, m, s) for m in METHODS for s in config.seeds}
    if not all(p.exists() for p in paths.values()):
        return  # Pending training; the stage stays incomplete.
    rows, spatial = [], {}
    for (method, seed), checkpoint_path in paths.items():
        model = load_program(checkpoint_path)
        for family in FAMILIES:
            budget.check()
            print(f"insights {method} seed={seed} {family}", flush=True)
            episode = make_episode(config, "test", family, 0)
            record, maps = analyse(model, episode, norm, config)
            rows.append({"method": method, "seed": seed, "family": family, **record})
            if seed == config.seeds[0]:
                spatial[f"{method}/{family}"] = {
                    **maps,
                    "positions": episode.positions,
                    "truth": episode.truth[-1],
                }
    models = {"fixed-combined": make_program("combined", **norm)}
    models |= {m: load_program(paths[(m, config.seeds[0])]) for m in ("parametric", "neural")}
    components = learned_components(models, norm)
    tensor_write(out / "insights.pt", {"spatial": spatial, "components": components})
    json_write(path, {"window": WINDOW, "temperature": config.error_temperature, "rows": rows})
    figures(out, config)


def figures(out, config):  # noqa: PLR0915 -- five fixed panels
    rows = read_json(out / "insights.json")["rows"]
    saved = torch.load(out / "insights.pt", weights_only=True)
    spatial, components = saved["spatial"], saved["components"]

    fig, axes = plt.subplots(
        len(FAMILIES), 1 + len(METHODS), figsize=(13, 6.5), layout="constrained"
    )
    for row, family in zip(axes, FAMILIES, strict=True):
        first = spatial[f"parametric/{family}"]
        x, y = first["positions"].T
        row[0].scatter(x, y, c=first["truth"], cmap="viridis", s=18)
        row[0].set(title=f"{family}: truth (last round)")
        for ax, method in zip(row[1:], METHODS, strict=True):
            maps = spatial[f"{method}/{family}"]
            ax.scatter(x, y, c=maps["map"], cmap="magma", s=18)
            if "roles" in maps:
                leaders = maps["roles"] == 2
                ax.scatter(
                    x[leaders], y[leaders], marker="*", s=90, c="cyan", edgecolors="black", lw=0.4
                )
                border = maps["roles"] == 1
                ax.scatter(
                    x[border], y[border], s=40, facecolors="none", edgecolors="white", lw=0.5
                )
            ax.set(title=f"{method}: |dL_late/d obs|")
    for ax in axes.flat:
        ax.set(xticks=[], yticks=[], aspect="equal")
    export(fig, out, "insights-saliency")

    region_rows = [r for r in rows if "roles" in r]
    fig, left = plt.subplots(figsize=(5.5, 3.6), layout="constrained")
    names = ("interior", "boundary", "leader")
    for offset, method in enumerate(("parametric", "neural")):
        selected = [r for r in region_rows if r["method"] == method]
        for i, name in enumerate(names):
            values = [r["roles"][name] for r in selected if r["roles"][name] is not None]
            left.bar(
                i + 0.35 * offset,
                np.mean(values),
                0.35,
                color=method_color(method),
                alpha=0.7,
                label=method if i == 0 else None,
            )
            left.scatter([i + 0.35 * offset] * len(values), values, s=8, color="black")
    left.set(
        xticks=np.arange(3) + 0.175, xticklabels=names, ylabel="mean |dL/d obs| / overall mean"
    )
    left.legend(fontsize=8)
    export(fig, out, "insights-roles")

    fig, ax = plt.subplots(figsize=(6, 3.6), layout="constrained")
    for method in METHODS:
        curves = np.array([r["horizon"] for r in rows if r["method"] == method])
        ax.plot(curves.mean(0), color=method_color(method), label=method)
    ax.set(
        xlabel=f"Lag (rounds before end; loss on last {WINDOW})",
        ylabel="Share of |dL/d obs|",
        yscale="log",
        xlim=(0, 40),
    )
    ax.legend(fontsize=8)
    ax.grid(alpha=0.15)
    export(fig, out, "insights-horizon")

    fig, (left, right) = plt.subplots(1, 2, figsize=(10, 3.6), layout="constrained")
    for offset, method in enumerate(("parametric", "neural")):
        selected = [r for r in region_rows if r["method"] == method]
        blocks = [[r["blocks"][b] for r in selected] for b in ("S", "G", "C", "B")]
        for i, values in enumerate(blocks):
            left.bar(
                i + 0.35 * offset,
                np.mean(values),
                0.35,
                color=method_color(method),
                alpha=0.7,
                label=method if i == 0 else None,
            )
            left.scatter([i + 0.35 * offset] * len(values), values, s=8, color="black")
        for seed in config.seeds:
            history = read_json(
                out
                / "checkpoints"
                / f"{method}-l{config.main_lambda:g}"
                / f"seed{seed}"
                / "training.json"
            )["history"][1:]
            for block, style in (("metric", "-"), ("strength", ":")):
                right.plot(
                    [r[f"gradient_norm_{block}"] for r in history],
                    style,
                    color=method_color(method),
                    alpha=0.5,
                    label=f"{method} {block}" if seed == config.seeds[0] else None,
                )
    left.set(
        xticks=np.arange(4) + 0.175,
        xticklabels=("S\nelection", "G\ndistance", "C\nconverge-cast", "B\nbroadcast"),
        ylabel="|dL/d block output|, summed over rounds",
        yscale="log",
    )
    left.legend(fontsize=8)
    right.set(xlabel="Adam update", ylabel="Gradient norm per parameter group", yscale="log")
    right.legend(fontsize=7, ncol=2)
    export(fig, out, "insights-attribution")

    names = list(components["grids"])
    fig, axes = plt.subplots(2, len(names), figsize=(4 * len(names), 6.5), layout="constrained")
    for column, name in zip(axes.T, names, strict=True):
        grid = components["grids"][name]
        top = column[0].contourf(
            components["distance"], components["delta"], grid["cost"], levels=12
        )
        fig.colorbar(top, ax=column[0])
        column[0].set(title=f"{name}: link cost", xlabel="range", ylabel="|signal difference|")
        bottom = column[1].contourf(
            components["value"], components["variance"], grid["strength"], levels=12
        )
        fig.colorbar(bottom, ax=column[1])
        column[1].set(
            title=f"{name}: leader strength",
            xlabel="value (= neighbourhood mean)",
            ylabel="neighbourhood variance",
        )
    export(fig, out, "insights-learned")
