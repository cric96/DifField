"""Deterministic simulator; truth and global statistics never enter program signals."""

import math
from dataclasses import dataclass

import torch
from sklearn.neighbors import NearestNeighbors
from torch import Tensor

from ..randomness import rng, seed_for
from .config import CONDITIONS, TEST_FAMILIES, TRAIN_FAMILIES


@dataclass
class RegionEpisode:
    key: str
    positions: Tensor
    truth: Tensor
    observations: Tensor
    priorities: Tensor
    active: Tensor
    edges: list[Tensor]
    fault_at: int
    restore_at: int
    metadata: dict

    @property
    def nodes(self):
        return self.observations.shape[1]

    @property
    def rounds(self):
        return self.observations.shape[0]

    def signals(self, t):
        return {
            "observation": self.observations[t],
            "priority": self.priorities,
            "time": self.observations.new_full((self.nodes,), float(t)),
        }

    def payload(self):
        return dict(vars(self))

    def to(self, device):
        moved = {k: v.to(device) if isinstance(v, Tensor) else v for k, v in vars(self).items()}
        return RegionEpisode(**{**moved, "edges": [e.to(device) for e in self.edges]})


def symmetric_edges(pairs, nodes):
    directed = {(int(a), int(b)) for a, b in pairs if a != b}
    directed |= {(b, a) for a, b in directed}
    if not directed:
        return torch.empty(2, 0, dtype=torch.long)
    return torch.tensor(sorted(directed, key=lambda e: e[1] * nodes + e[0])).T.contiguous()


def topology(nodes, seed, layout="jittered", degree=6):
    generator = rng(seed, "positions", layout)
    if layout == "jittered":
        width = math.ceil(math.sqrt(nodes))
        ids = torch.arange(nodes)
        points = torch.stack((ids % width, ids // width), -1).float() + 0.5
        points += 0.5 * (torch.rand(nodes, 2, generator=generator) - 0.5)
        points /= width
    elif layout == "uniform":
        points = torch.rand(nodes, 2, generator=generator)
    elif layout == "uneven":
        points = torch.rand(nodes, 2, generator=generator)
        dense = int(nodes * 0.7)
        points[:dense] = (0.25 + 0.13 * torch.randn(dense, 2, generator=generator)).clamp(0, 1)
    else:
        raise ValueError(layout)
    # Symmetrized kNN: comparable local degree at all network sizes, no long-hop overlay.
    neighbors = NearestNeighbors(n_neighbors=min(degree + 1, nodes)).fit(points.numpy())
    nearest = neighbors.kneighbors(points.numpy(), return_distance=False)
    edges = symmetric_edges(((i, j) for i, row in enumerate(nearest) for j in row), nodes)
    return points, edges


def geometry(seed, split, family, index):
    if split not in ("train", "validation", "test"):
        raise ValueError(split)
    if family not in (TEST_FAMILIES if split == "test" else TRAIN_FAMILIES):
        raise ValueError(f"Family {family} is reserved for test")
    values = torch.rand(8, generator=rng(seed, split, family, index, "geometry")).tolist()
    # Disjoint width intervals make the geometry split auditable, not just a seed split.
    width = {"train": (0.07, 0.14), "validation": (0.145, 0.17), "test": (0.175, 0.23)}[split]
    return {
        "width": width[0] + values[0] * (width[1] - width[0]),
        "amplitude": 0.7 + 0.6 * values[1],
        "phase": 2 * math.pi * values[2],
        "cx": 0.35 + 0.3 * values[3],
        "cy": 0.35 + 0.3 * values[4],
        "speed": 0.5 + values[5],
        "angle": math.pi * values[6],
        "aspect": 1.5 + values[7],
    }


STATIC_TIME = 0.4  # a frozen phenomenon is the moving one at this time (full envelope)


def phenomenon(points, rounds, family, parameters, *, static=False):
    p = parameters
    time = torch.full((rounds, 1), STATIC_TIME) if static else torch.linspace(0, 1, rounds)[:, None]
    phase = 2 * math.pi * p["speed"] * time + p["phase"]
    x = points[None, :, 0] - (p["cx"] + 0.17 * phase.sin())
    y = points[None, :, 1] - (p["cy"] + 0.17 * phase.cos())
    width = p["width"] * (1 + 0.25 * (phase * 1.3).sin())

    def gaussian(a, b):
        return torch.exp(-(a.square() + b.square()) / (2 * width.square()))

    if family == "constant":
        return torch.full((rounds, len(points)), p["amplitude"] * 0.4)
    if family == "gaussian":
        field = gaussian(x, y)
    elif family == "mixture":
        # Continuous split, merge, birth and disappearance of two structures.
        separation = 0.24 * (2 * math.pi * time).sin().square()
        birth = ((time - 0.22) * 35).sigmoid() * ((0.82 - time) * 35).sigmoid()
        field = 0.6 * gaussian(x - separation, y) + birth * gaussian(x + separation, y)
    elif family == "ellipse":
        angle = p["angle"] + phase * 0.5
        rx, ry = angle.cos() * x + angle.sin() * y, -angle.sin() * x + angle.cos() * y
        field = gaussian(rx / p["aspect"], ry * p["aspect"])
    elif family == "ring":
        radius = 0.2 + 0.08 * phase.sin()
        field = torch.exp(
            -(torch.sqrt(x.square() + y.square()) - radius).square() / (2 * (width * 0.25).square())
        )
    elif family == "front":
        field = ((x - 0.18 * (y * 7 + phase).sin()) / (width * 0.2)).sigmoid()
    else:
        raise ValueError(family)
    # Appearance/disappearance without resetting any program state.
    envelope = 0.25 + 0.75 * ((time - 0.13) * 30).sigmoid() * ((0.9 - time) * 30).sigmoid()
    return 0.1 + p["amplitude"] * envelope * field


def faults(seed, points, base, rounds, condition, crashed=None):  # noqa: PLR0917 -- one event model
    """Active devices and current links per round; events at one and two thirds.

    ``crash`` removes ``crashed`` devices permanently at the fault time.
    """
    nodes = len(points)
    fault, restore = rounds // 3, 2 * rounds // 3
    active = torch.ones(rounds, nodes, dtype=torch.bool)
    if condition == "node_stop":
        stopped = torch.randperm(nodes, generator=rng(seed, "stops"))[: max(1, nodes // 5)]
        active[fault:restore, stopped] = False
    if condition == "crash":
        active[fault:, crashed] = False
    a, b = base
    edges = []
    for t in range(rounds):
        keep = active[t, a] & active[t, b]
        if fault <= t < restore:
            if condition == "partition":
                keep &= (points[a, 0] < 0.5) == (points[b, 0] < 0.5)
            elif condition == "link_loss":
                # One random draw per undirected pair; both directions disappear together.
                pairs = base[:, a < b]
                retained = torch.rand(pairs.shape[1], generator=rng(seed, "links", t)) >= 0.3
                edges.append(symmetric_edges(pairs[:, retained].T.tolist(), nodes))
                continue
        edges.append(base[:, keep])
    return active, edges, fault, restore


def make_episode(
    config,
    split="test",
    family="gaussian",
    index=0,
    *,
    nodes=None,
    layout="jittered",
    condition="clean",
    static=False,
):
    """``static`` freezes the phenomenon; ``crash`` removes the four devices on the
    highest readings at the fault time, for good."""
    if condition not in (*CONDITIONS, "crash"):
        raise ValueError(condition)
    nodes = config.nodes if nodes is None else nodes
    rounds = config.eval_rounds if split == "test" else config.train_rounds
    seed = seed_for(config.data_seed, split, family, index)
    points, base = topology(nodes, seed, layout, config.degree)
    parameters = geometry(config.data_seed, split, family, index)
    truth = phenomenon(points, rounds, family, parameters, static=static)
    # Noise-free perception; observations stay a separate array so truth is never a signal.
    observations = truth.clone()
    crashed = truth[rounds // 3].topk(4).indices
    active, edges, fault, restore = faults(seed, points, base, rounds, condition, crashed)
    key = f"{split}-{family}-{index}-n{nodes}-{layout}-{condition}" + ("-static" if static else "")
    return RegionEpisode(
        key,
        points,
        truth,
        observations,
        torch.rand(nodes, generator=rng(seed, "priorities")),
        active,
        edges,
        fault,
        restore,
        {
            "split": split,
            "family": family,
            "index": index,
            "layout": layout,
            "condition": condition,
            "geometry": parameters,
            "seed": seed,
            "static": static,
        },
    )


def training_bank(config, *, static=False):
    return {
        split: [
            make_episode(
                config,
                split,
                family,
                index,
                condition=CONDITIONS[index % len(CONDITIONS)],
                static=static,
            )
            for family in TRAIN_FAMILIES
            for index in range(count)
        ]
        for split, count in (
            ("train", config.train_episodes),
            ("validation", config.validation_episodes),
        )
    }


def normalization(episodes):
    # Only the training bank is supplied; including inactive nodes does not use test truth.
    values = torch.cat([episode.truth.flatten() for episode in episodes])
    return {"mean": float(values.mean()), "scale": max(float(values.std(unbiased=False)), 1e-6)}


def combine(episodes):
    """Disjoint graphs form a true batched sequence; no cross-episode communication."""
    if not episodes or len({e.rounds for e in episodes}) != 1:
        raise ValueError("Batch needs equally long sequences")
    offsets, offset = [], 0
    for episode in episodes:
        offsets.append(offset)
        offset += episode.nodes
    first = episodes[0]
    return RegionEpisode(
        "batch",
        torch.cat([e.positions for e in episodes]),
        torch.cat([e.truth for e in episodes], 1),
        torch.cat([e.observations for e in episodes], 1),
        torch.cat([e.priorities for e in episodes]),
        torch.cat([e.active for e in episodes], 1),
        [
            torch.cat([e.edges[t] + start for e, start in zip(episodes, offsets, strict=True)], 1)
            for t in range(first.rounds)
        ],
        first.fault_at,
        first.restore_at,
        {"batch_offsets": offsets},
    )
