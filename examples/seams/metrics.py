"""Paired statistical utilities shared by the SEAMS experiments."""

import math

import numpy as np


def interval(rows, metric, *, reference=None, samples=2000, seed=0):
    """Two-way clustered bootstrap for the crossed seed x shared-episode design.

    Differences are formed before resampling; the same episode draw is used
    for every sampled seed. A single seed or an incomplete/nonfinite panel
    cannot support a seed-level confidence interval.
    """
    table = {(row["seed"], row["episode"]): row["metrics"].get(metric) for row in rows}
    if len(table) != len(rows):
        raise ValueError("Duplicate seed/episode in interval panel")
    if reference is not None:
        baseline = {(row["seed"], row["episode"]): row["metrics"].get(metric) for row in reference}
        if len(baseline) != len(reference):
            raise ValueError("Duplicate seed/episode in reference panel")
        table = {
            key: table[key] - baseline[key]
            if table.get(key) is not None and baseline.get(key) is not None
            else None
            for key in table.keys() | baseline.keys()
        }
    seeds, episodes = sorted({k[0] for k in table}), sorted({k[1] for k in table})
    finite = [value for value in table.values() if value is not None and math.isfinite(value)]
    invalid = len(table) - len(finite)
    result = {
        "mean": float(np.mean(finite)) if finite and not invalid else None,
        "finite_mean": float(np.mean(finite)) if finite else None,
        "ci95": None,
        "seeds": len(seeds),
        "episodes": len(episodes),
        "invalid": invalid,
        "n": len(table),
    }
    if invalid or len(table) != len(seeds) * len(episodes):
        result["ci_reason"] = "nonfinite_or_incomplete_panel"
    elif len(seeds) < 2 or len(episodes) < 2:
        result["ci_reason"] = "insufficient_independent_seeds_or_episodes"
    else:
        matrix = np.array([[table[s, e] for e in episodes] for s in seeds])
        generator = np.random.default_rng(seed)
        draws = []
        for _ in range(samples):
            si = generator.integers(len(seeds), size=len(seeds))
            ei = generator.integers(len(episodes), size=len(episodes))
            draws.append(matrix[np.ix_(si, ei)].mean())
        result["ci95"] = np.quantile(draws, [0.025, 0.975]).tolist()
        result["ci_reason"] = "two_way_cluster_bootstrap"
    return result
