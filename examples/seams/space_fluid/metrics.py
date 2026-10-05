"""Service quality, region structure, reconfiguration and censored recovery."""

import math

import numpy as np
import torch

from .program import COUNT, ELECTED, SAMPLE, TIMESTAMP


def recovery(errors, start, threshold=0.2, consecutive=5):
    compliant = [e is not None and math.isfinite(e) and e <= threshold for e in errors]
    before = compliant[max(0, start - consecutive) : start]
    baseline = len(before) == consecutive and all(before)
    for t in range(start, len(errors) - consecutive + 1):
        if all(compliant[t : t + consecutive]):
            return {
                "status": "recovered" if baseline else "attained",
                "rounds": t - start,
                "baseline_compliant": baseline,
            }
    return {"status": "no_recovery", "rounds": None, "baseline_compliant": baseline}


def fragmentation(labels, edges, active):
    """Extra connected components per label, on the *current* communication graph."""
    selected = active.nonzero().flatten().tolist()
    parent = list(range(len(labels)))

    def root(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    values = labels.tolist()
    for a, b in edges.T.tolist():
        if active[a] and active[b] and values[a] == values[b]:
            parent[root(a)] = root(b)
    components = {}
    for node in selected:
        components.setdefault(values[node], set()).add(root(node))
    return sum(len(c) - 1 for c in components.values()) / max(len(components), 1)


def summarize(episode, trace, scale, config):
    fields, labels = trace.fields.detach(), trace.leaders
    error, regions, leaders, homogeneity, fragments, stability, evidence_age = (
        [] for _ in range(7)
    )
    between, within, collected = [], [], []
    for t in range(episode.rounds):
        active = episode.active[t]
        if not bool(active.any()):
            raise ValueError("An episode must have at least one active node each round")
        values = fields[t, active, SAMPLE]
        if not bool(torch.isfinite(values).all()):
            raise FloatingPointError("Nonfinite reconstruction on an active device")
        error.append(float((values - episode.truth[t, active]).square().mean().sqrt() / scale))
        assigned = labels[t, active]
        unique = assigned.unique()
        regions.append(len(unique))
        leaders.append(float(fields[t, active, ELECTED].mean()))
        variance, means, spreads = 0.0, [], []
        for leader in unique:
            samples = episode.truth[t, active][assigned == leader]
            variance += float((samples - samples.mean()).square().sum())
            means.append(float(samples.mean()))
            spreads.append(float(samples.std(unbiased=False)))
        homogeneity.append(variance / int(active.sum()) / scale**2)
        # Space-Fluid's sigma(mu_s) (between regions) and mu(sigma_s) (within), training scale.
        between.append(float(np.std(means)) / scale)
        within.append(float(np.mean(spreads)) / scale)
        elected = active & (fields[t, :, ELECTED] > 0.5)
        collected.append(float(fields[t, elected, COUNT].sum()) / int(active.sum()))
        fragments.append(fragmentation(labels[t], episode.edges[t], active))
        shared = active & episode.active[max(0, t - 1)]
        stability.append(
            float((labels[t, shared] == labels[max(0, t - 1), shared]).float().mean())
            if bool(shared.any())
            else 1.0
        )
        evidence_age.append(float((t - fields[t, active, TIMESTAMP]).mean()))
    metrics = {
        "nrmse": math.sqrt(float(np.mean(np.square(error)))),
        "regions": float(np.mean(regions)),
        "leader_fraction": float(np.mean(leaders)),
        "homogeneity": float(np.mean(homogeneity)),
        "region_mean_std": float(np.mean(between)),
        "mean_region_std": float(np.mean(within)),
        "mean_region_size": float(
            np.mean([int(a.sum()) / r for a, r in zip(episode.active, regions, strict=True)])
        ),
        "collected_fraction": float(np.mean(collected)),
        "fragmentation": float(np.mean(fragments)),
        "assignment_stability": float(np.mean(stability[1:])),
        "evidence_age": float(np.mean(evidence_age)),
        "message_bytes": sum(trace.traffic),
        "state_bytes": trace.state_bytes,
        "node_activations": sum(trace.activations),
        "inference_seconds": trace.seconds,
        "compliant_fraction": float(np.mean(np.array(error) <= config.recovery_threshold)),
    }
    if trace.executor == "kmeans-central":
        metrics["message_bytes"] = None  # Global access is not a zero-cost distributed protocol.
        metrics["state_bytes"] = None
        metrics["collected_fraction"] = None
    for name, start in (
        ("fault_recovery", episode.fault_at),
        ("restoration_recovery", episode.restore_at),
    ):
        metrics[name] = (
            recovery(error, start, config.recovery_threshold, config.recovery_rounds)
            if episode.metadata["condition"] != "clean"
            else {"status": "not_applicable", "rounds": None, "baseline_compliant": None}
        )
    return metrics, {
        "nrmse": error,
        "regions": regions,
        "leader_fraction": leaders,
        "homogeneity": homogeneity,
        "fragmentation": fragments,
        "assignment_stability": stability,
        "evidence_age": evidence_age,
        "message_bytes": trace.traffic,
        "activations": trace.activations,
    }
