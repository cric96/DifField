"""Service quality, region structure, reconfiguration and censored recovery."""

import math

import numpy as np
import torch
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

from ...shared.metrics import recovery
from .program import COUNT, ELECTED, SAMPLE, TIMESTAMP


def components(labels, edges, active):
    """The component of every device in the graph of links between active devices that share a
    label (numpy arrays: ``labels`` and ``active`` per device, ``edges`` of shape [2, E])."""
    a, b = edges
    joined = active[a] & active[b] & (labels[a] == labels[b])
    links = coo_matrix(
        (np.ones(joined.sum()), (a[joined], b[joined])), shape=(len(labels), len(labels))
    )
    return connected_components(links, directed=False)[1]


def fragmentation(labels, edges, active):
    """Extra connected components per label, on the *current* communication graph."""
    labels, active = labels.numpy(), active.numpy()
    component = components(labels, edges.numpy(), active)
    pieces = {}
    for node in np.flatnonzero(active):
        pieces.setdefault(labels[node], set()).add(component[node])
    return sum(len(c) - 1 for c in pieces.values()) / max(len(pieces), 1)


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
