"""Centralised learned clustering: a GNN assigns every active device to one of SLOTS.

A cluster is a connected component of neighbours in the same slot, so clusters are
contiguous like SCR regions. Same inputs as the SCR program (value, neighbourhood mean
and variance on the current graph) plus positions, seen globally at every round.
Trained on the same objective from the phenomenon: error of the cluster mean +
lambda * clusters / N, with no labels and no K. Soft training propagates the
probability of sharing a slot along the graph; with one-hot slots it is exact.
"""

import numpy as np
import torch
from torch import nn

from ...shared.artifacts import json_write, tensor_write
from ...shared.training import fit, frozen
from .metrics import components
from .program import mlp
from .training import history_row, scheduled_rate, training_record

SLOTS = 16
SQUARINGS = 5  # reachability over paths of up to 2**5 hops
EVERY = 16  # training and validation rounds per episode: one every EVERY


class GraphClusterer(nn.Module):
    def __init__(self, mean=0.0, scale=1.0, hidden=32, layers=4):
        super().__init__()
        self.register_buffer("normalization", torch.tensor([mean, scale]))
        self.encode = mlp(5, hidden, hidden)
        self.layers = nn.ModuleList(mlp(2 * hidden, hidden, hidden) for _ in range(layers))
        self.head = nn.Linear(hidden, SLOTS)

    def forward(self, graph):
        mean, scale = self.normalization.unbind()
        value = (graph["values"] - mean) / scale
        source, target = graph["edges"]
        local = torch.cat((value, average(value[source], target, len(value), value)), -1)
        square = average(value[source].square(), target, len(value), value.square())
        variance = (square - local[:, 1:].square()).clamp(0)
        h = self.encode(torch.cat((local, variance, graph["positions"]), -1))
        for layer in self.layers:
            h = h + layer(torch.cat((h, average(h[source], target, len(h), h)), -1))
        return self.head(h)


def average(messages, target, nodes, own):
    """Mean over the neighbours and the device itself."""
    total = own.clone().index_add(0, target, messages)
    count = torch.ones(nodes, 1, dtype=own.dtype).index_add(
        0, target, torch.ones(len(target), 1, dtype=own.dtype)
    )
    return total / count


def graphs(episodes, rounds=None):
    """One disjoint graph per (episode, round) over the active devices."""
    values, positions, truth, edges, group = [], [], [], [], []
    offset = index = 0
    for episode in episodes:
        for t in range(episode.rounds) if rounds is None else rounds(episode):
            ids = episode.active[t].nonzero().flatten()
            local = torch.full((episode.nodes,), -1, dtype=torch.long)
            local[ids] = torch.arange(len(ids))
            a, b = episode.edges[t]
            keep = (local[a] >= 0) & (local[b] >= 0)
            edges.append(torch.stack((local[a[keep]], local[b[keep]])) + offset)
            values.append(episode.observations[t, ids, None])
            truth.append(episode.truth[t, ids])
            positions.append(episode.positions[ids])
            group.append(torch.full((len(ids),), index))
            offset, index = offset + len(ids), index + 1
    sizes = {len(g) for g in group}
    return {
        "size": sizes.pop() if len(sizes) == 1 else None,
        "values": torch.cat(values),
        "positions": torch.cat(positions),
        "truth": torch.cat(truth),
        "edges": torch.cat(edges, 1),
        "group": torch.cat(group),
        "graphs": index,
    }


def terms(assignment, graph, scale):
    """Error of the component mean / scale^2 and components / devices, mean over graphs.

    ``reach[i, j]`` is the probability of the likeliest path of neighbours in the same
    slot joining i and j (a sum over paths saturates and kills the gradient); a
    component of size m contributes m * (1 / m) = 1 to the count.
    """
    count, size = graph["graphs"], graph["size"]
    source, target = graph["edges"]
    same = (assignment[source] * assignment[target]).sum(-1)
    reach = torch.zeros(count, size, size).index_put(
        (source // size, source % size, target % size), same, accumulate=True
    )
    reach = (reach + torch.eye(size)).clamp(max=1)
    for _ in range(SQUARINGS):
        reach = (reach[..., None] * reach[:, None]).amax(2)
    members = reach.sum(-1)
    estimate = (reach @ graph["values"].view(count, size, 1)).squeeze(-1) / members
    error = (estimate - graph["truth"].view(count, size)).square().mean(-1) / scale**2
    return error.mean(), ((1 / members).sum(-1) / size).mean()


def soft_loss(model, graph, scale, penalty):
    error, leaders = terms(model(graph).softmax(-1), graph, scale)
    return error + penalty * leaders


@torch.no_grad()
def hard_scores(model, graph, scale, penalty):
    labels = model(graph).argmax(-1)
    error, leaders = map(float, terms(nn.functional.one_hot(labels, SLOTS).float(), graph, scale))
    return {"objective": error + penalty * leaders, "error": error, "leaders": leaders}


@torch.no_grad()
def labels(model, episode):
    """Smallest device ID of each component, every round; -1 for inactive devices."""
    result = torch.full((episode.rounds, episode.nodes), -1, dtype=torch.long)
    for t in range(episode.rounds):
        graph = graphs([episode], lambda _, t=t: [t])
        ids = episode.active[t].nonzero().flatten().numpy()
        slot = model(graph).argmax(-1).numpy()
        component = components(slot, graph["edges"].numpy(), np.ones(len(ids), dtype=bool))
        smallest = np.full(component.max() + 1, np.iinfo(np.int64).max)
        np.minimum.at(smallest, component, ids)
        result[t, ids] = torch.from_numpy(smallest[component])
    return result


def train(directory, config, bank, norm, *, seed, penalty, budget, rate=0.01):
    """Full-batch Adam with cosine decay and hard validation selection, resumable."""
    if (directory / "best.pt").exists():
        return load(directory / "best.pt")
    directory.mkdir(parents=True, exist_ok=True)
    device = torch.device(config.device)
    with torch.random.fork_rng():
        torch.manual_seed(seed)
        model = GraphClusterer(**norm).to(device)
    sample = lambda episode: range(0, episode.rounds, EVERY)  # noqa: E731
    training, validation = (
        {k: v.to(device) if torch.is_tensor(v) else v for k, v in graphs(split, sample).items()}
        for split in (bank["train"], bank["validation"])
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=rate)
    scale = norm["scale"]

    def update(step):
        for group in optimizer.param_groups:
            group["lr"] = scheduled_rate(rate, config, step, config.updates)
        optimizer.zero_grad(set_to_none=True)
        loss = soft_loss(model, training, scale, penalty)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0, error_if_nonfinite=True)
        optimizer.step()
        return {"step": step, "training_surrogate": float(loss.detach())}

    def scored():
        scores = hard_scores(model, validation, scale, penalty)
        return scores["objective"], history_row(scores)

    with torch.device(device):
        state = fit(
            model,
            optimizer,
            update,
            scored,
            updates=config.updates,
            validate_every=config.validate_every,
            last=directory / "last.pt",
            budget=budget,
        )
    tensor_write(directory / "best.pt", {"normalization": norm, "model": state.best_model})
    json_write(directory / "training.json", training_record(state))
    return model.cpu().eval().requires_grad_(False)


def load(path):
    return frozen(path, lambda checkpoint: GraphClusterer(**checkpoint["normalization"]))
