#!/usr/bin/env python3
"""Training loss and hard validation over time, from checkpoints/*/seed*/training.json."""

import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from examples.seams.space_fluid.report import method_color

SMOOTH = 10


def runs(checkpoints):
    """{(lambda or None): {method: [history per seed]}}"""
    grouped = defaultdict(lambda: defaultdict(list))
    for path in sorted(checkpoints.glob("*/seed*/training.json")):
        name = path.parent.parent.name
        match = re.fullmatch(r"(.+)-l([\d.]+)", name)
        method, penalty = (match[1], float(match[2])) if match else (name, None)
        grouped[penalty][method].append(json.loads(path.read_text())["history"])
    return grouped


def series(histories, key):
    curves = []
    for history in histories:
        rows = [(r["step"], r[key]) for r in history if key in r]
        curves.append(np.array(rows, dtype=float))
    length = min(len(c) for c in curves)
    stacked = np.stack([c[:length] for c in curves])
    return stacked[0, :, 0], stacked[:, :, 1]


def smooth(values):
    kernel = np.ones(SMOOTH) / SMOOTH
    return np.stack([np.convolve(v, kernel, mode="valid") for v in values])


def band(ax, x, values, color, label):
    ax.fill_between(x, values.min(0), values.max(0), color=color, alpha=0.15, linewidth=0)
    ax.plot(x, values.mean(0), color=color, linewidth=2, label=label)


def figure(grouped, title, target):
    penalties = sorted(grouped, key=lambda p: (p is None, p))
    fig, axes = plt.subplots(
        len(penalties), 3, figsize=(15, 3.6 * len(penalties)), layout="constrained", squeeze=False
    )
    for row, penalty in zip(axes, penalties, strict=True):
        suffix = "" if penalty is None else f" (λ={penalty:g})"
        for method, histories in sorted(grouped[penalty].items()):
            color = method_color(method)
            label = f"{method} ({len(histories)} seeds)"
            if any("training_surrogate" in r for r in histories[0]):
                x, y = series(histories, "training_surrogate")
                band(row[0], x[SMOOTH - 1 :], smooth(y), color, label)
                x, y = series(histories, "validation_hard")
                band(row[1], x, y, color, label)
            x, y = series(histories, "validation_hard")
            budget = x / x[-1]
            band(row[2], budget, np.minimum.accumulate(y, axis=1), color, label)
        row[0].set(title=f"Training loss, surrogate{suffix}", xlabel="Update", yscale="log")
        row[1].set(title=f"Hard validation objective{suffix}", xlabel="Update", yscale="log")
        row[2].set(
            title=f"Best so far, hard validation{suffix}",
            xlabel="Fraction of the method's budget (updates or candidates)",
            yscale="log",
        )
        for ax in row:
            ax.grid(alpha=0.15)
            ax.legend(fontsize=7, frameon=False)
    fig.suptitle(
        f"{title}: mean over seeds, band = min-max; training loss smoothed over {SMOOTH} updates"
    )
    target.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(target.with_suffix(".png"), dpi=160)
    fig.savefig(target.with_suffix(".pdf"))
    plt.close(fig)
    return target.with_suffix(".png")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True, help="Campaign output directory")
    args = parser.parse_args()
    for checkpoints in sorted(args.out.glob("**/checkpoints")):
        study = checkpoints.parent
        relative = study.relative_to(args.out).as_posix()
        name = args.out.name if relative == "." else relative.replace("/", "-")
        print(figure(runs(checkpoints), name, study / "figures" / "training-curves"))


if __name__ == "__main__":
    main()
