"""Two executors of program(runtime), with numeric messages and persistent state."""

import time
from dataclasses import dataclass
from types import SimpleNamespace

import torch
from sklearn.cluster import KMeans
from threadpoolctl import threadpool_limits

from diffield import AggregateContext
from diffield.decentralized.runner import build_runtimes
from diffield.decentralized.topology import all_in_edges, require_symmetric

from .program import DISTANCE, ELECTED, LEADER, OUTPUT_DIM, SAMPLE, TIMESTAMP


@dataclass
class RegionTrace:
    fields: torch.Tensor
    traffic: list[int]
    activations: list[int]
    seconds: float
    state_bytes: int
    executor: str

    @property
    def leaders(self):
        return self.fields[..., LEADER].long()

    def payload(self):
        return dict(vars(self))


def ranges(episode, t):
    a, b = episode.edges[t]
    return (episode.positions[a] - episode.positions[b]).norm(dim=-1)


def central(episode, program):
    started = time.perf_counter()
    context = AggregateContext(episode.edges[0], episode.nodes)
    fields, traffic, activations = [], [], []
    for t in range(episode.rounds):
        context.update_topology(episode.edges[t], edge_weight=ranges(episode, t))
        active = episode.active[t]
        carried = context.state.snapshot() if not bool(active.all()) else {}
        with context.round():
            result = program(SimpleNamespace(signals=episode.signals(t), round_idx=t))
        if carried:
            for key, value in carried.items():
                current = context.get_state(key)
                kept = active.view(-1, *[1] * (current.dim() - 1))
                context.state.update(torch.where(kept, current, value), name=key)
        if not bool(active.all()):
            previous = fields[-1] if fields else result.new_zeros(result.shape)
            result = torch.where(active[:, None], result, previous)
        fields.append(result)
        # Numeric wire representation is exactly the exported eight-float iterate row.
        # A device sends its own row of every state slot to each neighbour.
        published = sum(
            state[0].numel() * state.element_size() for state in context.state._states.values()
        )
        traffic.append(episode.edges[t].shape[1] * published)
        activations.append(int(active.sum()))
    return RegionTrace(
        torch.stack(fields),
        traffic,
        activations,
        time.perf_counter() - started,
        episode.nodes * published,
        "batched",
    )


def local_signals(episode, t, node, neighbor_count):
    """Supply only this device's sensors. Neighbour slots intentionally contain zeros."""
    signals = {}
    for key, value in episode.signals(t).items():
        field = value.new_zeros(neighbor_count + 1)
        field[0] = value[node]
        signals[key] = field
    return signals


@torch.no_grad()
def decentralized(episode, program, *, activation_probability=1.0, seed=0):
    if not 0 < activation_probability <= 1:
        raise ValueError("Activation probability must be in (0, 1]")
    started = time.perf_counter()
    devices = build_runtimes(
        edge_index=episode.edges[0],
        num_nodes=episode.nodes,
        program=program,
        signals={},
        edge_weight=ranges(episode, 0),
    )
    bundles = {node: {} for node in devices}
    mailboxes = {node: {} for node in devices}
    output = torch.zeros(episode.nodes, OUTPUT_DIM)
    generator = torch.Generator().manual_seed(seed)
    fields, traffic, activations = [], [], []
    synchronous = activation_probability == 1.0
    for t in range(episode.rounds):
        require_symmetric(episode.edges[t])
        neighbors = all_in_edges(episode.edges[t], episode.nodes, ranges(episode, t))
        for node, device in devices.items():
            device.sync_topology(*neighbors[node])
            device.signals = local_signals(episode, t, node, len(device.neighbor_ids))
            # Removed links cannot keep delivering buffered advertisements.
            mailboxes[node] = {j: v for j, v in mailboxes[node].items() if j in device.neighbor_ids}
        # A common initial execution makes cold-start outputs defined. Thereafter
        # asynchronous nodes receive messages only when their neighbours actually send.
        barrier = synchronous or t == 0
        order = (
            range(episode.nodes)
            if barrier
            else torch.randperm(episode.nodes, generator=generator).tolist()
        )
        pending, bytes_round, activated = {}, 0, 0
        for node in order:
            if not episode.active[t, node]:
                continue
            if not barrier and float(torch.rand((), generator=generator)) >= activation_probability:
                continue
            device = devices[node]
            received = bundles if barrier else mailboxes[node]
            keys = {key for other in device.neighbor_ids for key in received.get(other, {})}
            inbox = {
                key: [received.get(other, {}).get(key) for other in device.neighbor_ids]
                for key in keys
            }
            output[node], bundle = device.step(inbox)
            pending[node] = bundle
            size = sum(v.numel() * v.element_size() for v in bundle.values())
            bytes_round += size * len(device.neighbor_ids)
            activated += 1
            if not barrier:
                for other in device.neighbor_ids:
                    mailboxes[other][node] = bundle
        bundles.update(pending)
        if barrier:
            for node, bundle in pending.items():
                for other in devices[node].neighbor_ids:
                    mailboxes[other][node] = bundle
        fields.append(output.clone())
        traffic.append(bytes_round)
        activations.append(activated)
    state_bytes = sum(v.numel() * v.element_size() for b in bundles.values() for v in b.values())
    return RegionTrace(
        torch.stack(fields),
        traffic,
        activations,
        time.perf_counter() - started,
        state_bytes,
        "sync" if synchronous else f"async-{activation_probability:g}",
    )


def equivalence(expected, actual, tolerance=2e-6):
    a, b = expected.fields.detach(), actual.fields.detach()
    return {
        "identifiers_equal": bool(torch.equal(a[..., LEADER], b[..., LEADER])),
        "elections_equal": bool(torch.equal(a[..., ELECTED], b[..., ELECTED])),
        "values_close": bool(torch.allclose(a, b, atol=tolerance, rtol=tolerance)),
        "max_abs_error": float((a - b).abs().max()),
        "traffic_equal": expected.traffic == actual.traffic,
    }


@torch.no_grad()
def kmeans(episode, k, normalization, seed=0):
    """Conventional global fit on position and normalized observations, every round.

    Reconstruction is the observation mean; representative IDs are medoids.
    It has global instantaneous input and no distributed cost model.
    """
    started = time.perf_counter()
    fields = []
    with threadpool_limits(limits=1):
        for t in range(episode.rounds):
            ids = episode.active[t].nonzero().flatten()
            result = torch.zeros(episode.nodes, OUTPUT_DIM)
            if len(ids):
                features = torch.cat(
                    (
                        episode.positions[ids],
                        (
                            (episode.observations[t, ids] - normalization["mean"])
                            / normalization["scale"]
                        )[:, None],
                    ),
                    -1,
                )
                model = KMeans(n_clusters=min(k, len(ids)), n_init=5, random_state=seed)
                labels = torch.from_numpy(model.fit_predict(features.numpy()))
                for label in labels.unique():
                    members = labels == label
                    center = torch.as_tensor(model.cluster_centers_[int(label)])
                    distances = (features[members] - center).norm(dim=-1)
                    leader = ids[members][distances.argmin()]
                    result[ids[members], LEADER] = leader.float()
                    result[ids[members], SAMPLE] = episode.observations[t, ids[members]].mean()
                    result[ids[members], DISTANCE] = distances
                    result[leader, ELECTED] = 1
                    result[ids[members], TIMESTAMP] = t
            fields.append(result)
    return RegionTrace(
        torch.stack(fields),
        [0] * episode.rounds,
        episode.active.sum(1).tolist(),
        time.perf_counter() - started,
        0,
        "kmeans-central",
    )
