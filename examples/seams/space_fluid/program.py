"""Space-Fluid as a composition of Self-organising Coordination Regions blocks.

S: ``bounded_election`` elects samplers (Space-Fluid Fig. 6; Pianini et al.,
ACSOS 2022) with a strength computed from local readings. Inside each region
(``aligned_on`` the leader): G ``distance_to`` the leader, C ``converge_cast``
of (sum, count) toward it, and ``broadcast`` of the regional mean back. One
learned link metric drives S, G, C and broadcast. Every block is a library
block that runs unchanged on independent devices; training runs the same
composition under ``with_mode("soft")``.
"""

from operator import add

import torch
from torch import nn

from diffield import (
    aligned_on,
    bounded_election,
    broadcast,
    converge_cast,
    distance_to,
    follow,
    gather,
    gather_avg,
    iterate,
    link_cat,
    link_map,
    mid,
    nbr,
    scatter,
    scatter_range,
)

(STRENGTH, DISTANCE, LEADER, SAMPLE, TIMESTAMP, ELECTED, OBSERVATION, TOTAL, COUNT) = range(9)
OUTPUT_DIM = 9
# Floats the SCR program publishes per device and round:
# nbr 1, election 5, region 2, distance 1, converge-cast 5, broadcast 3.
WIRE_FLOATS = 17
# (metric weights, strength weights over normalized value, neighbourhood mean, variance).
FIXED = {
    "spatial": ((4.0, 0.0, 0.0), (0.0, 0.0, 0.0)),
    "combined": ((2.0, 1.0, 1.0), (0.0, 0.0, 0.0)),
    "value": ((2.0, 1.0, 1.0), (4.0, 0.0, 0.0)),
    "variance": ((2.0, 1.0, 1.0), (0.0, 0.0, -4.0)),
}
KINDS = ("fixed", "parametric", "hybrid")
EMBEDDING = 2  # floats of the hybrid neighbourhood embedding
HYBRID_WIRE_FLOATS = WIRE_FLOATS + 3 + EMBEDDING  # plus nbr(features) and nbr(embedding)


class EdgeMetric(nn.Module):
    """Positive symmetric link cost; radius is always one.

    Features are range, normalized absolute difference and their product.
    """

    def __init__(self, kind="parametric", weights=(2.0, 1.0, 1.0), mean=0.0, scale=1.0):
        super().__init__()
        if kind not in KINDS:
            raise ValueError(f"Unknown metric: {kind}")
        self.kind = kind
        self.register_buffer("normalization", torch.tensor([mean, scale]))
        initial = torch.tensor(weights, dtype=torch.float32)
        if kind == "fixed":
            self.register_buffer("fixed_weights", initial)
        else:
            if not bool((initial > 0).all()):
                raise ValueError("Learned metric weights must be positive")
            self.log_weights = nn.Parameter(initial.log())

    @property
    def weights(self):
        return self.log_weights.exp() if hasattr(self, "log_weights") else self.fixed_weights

    def forward(self, distance, a, b):
        delta = (a - b).abs() / self.normalization[1]
        terms = torch.stack((distance, delta, distance * delta), dim=-1)
        return (terms * self.weights).sum(-1) + 1e-4


class LeaderStrength(nn.Module):
    """Candidacy strength from the paper's local options: value, neighbourhood mean, variance.

    ``priority`` has a fixed unit weight: it breaks symmetry and removes the
    scale ambiguity of a lexicographic comparison. Zero weights reproduce the
    random-priority election exactly.
    """

    def __init__(self, kind="parametric", weights=(0.0, 0.0, 0.0), mean=0.0, scale=1.0):
        super().__init__()
        if kind not in KINDS:
            raise ValueError(f"Unknown strength: {kind}")
        self.kind = kind
        self.register_buffer("normalization", torch.tensor([mean, scale]))
        initial = torch.tensor(weights, dtype=torch.float32)
        if kind == "fixed":
            self.register_buffer("weights", initial)
        else:
            self.weights = nn.Parameter(initial)

    def forward(self, value, neighbourhood_mean, neighbourhood_variance, priority):
        mean, scale = self.normalization.unbind()
        features = torch.stack(
            (
                (value - mean) / scale,
                (neighbourhood_mean - mean) / scale,
                neighbourhood_variance / scale**2,
            ),
            dim=-1,
        )
        return priority + (features * self.weights).sum(-1)


class RegionProgram(nn.Module):
    def __init__(self, metric: EdgeMetric, strength: LeaderStrength):
        super().__init__()
        self.metric = metric
        self.strength = strength
        self.probe = None  # A list collects block outputs for the gradient analysis.

    def link_metric(self, neighbours, obs, mean, variance):
        return link_map(self.metric, scatter_range(), neighbours, obs)

    def forward(self, runtime):
        obs, time = runtime.signals["observation"], runtime.signals["time"]
        neighbours = nbr(obs)
        mean = gather_avg(neighbours, include_self=True)
        variance = (gather_avg(neighbours * neighbours, include_self=True) - mean.square()).clamp(0)
        metric = self.link_metric(neighbours, obs, mean, variance)
        strength = self.strength(obs, mean, variance, runtime.signals["priority"])

        election = bounded_election(strength, radius=1.0, metric=metric)  # S
        with aligned_on(election.leader, weight=election.confidence):  # one region per leader
            source = election.leader == mid()
            potential = distance_to(source, metric)  # G
            total, count = converge_cast(potential, (obs, 1.0), add, metric)  # C
            estimate, stamp = broadcast(source, (total / count, time), metric)
        estimate = follow(election, estimate)  # soft: mixes the next region; hard: identity

        if self.probe is not None:
            self.probe.append({"S": election.confidence, "G": potential, "C": total, "B": estimate})
        return torch.stack(
            (strength, election.distance, election.leader, estimate, stamp,
             election.elected, obs, total, count),
            -1,
        )  # fmt: skip


def mlp(inputs, hidden, outputs):
    return nn.Sequential(
        nn.Linear(inputs, hidden), nn.Tanh(), nn.Linear(hidden, hidden), nn.Tanh(),
        nn.Linear(hidden, outputs),
    )  # fmt: skip


class HybridProgram(RegionProgram):
    """SCR program whose link cost is modulated by a GNN over the neighbourhood.

    One message-passing layer embeds each device's (value, mean, variance) with its
    neighbours'; the cost of a link is the parametric cost times
    exp(tanh(g(z_i, z_j) + g(z_j, z_i))), exactly the parametric cost at start.
    """

    def __init__(self, metric: EdgeMetric, strength: LeaderStrength, hidden=8):
        super().__init__(metric, strength)
        self.message = mlp(7, hidden, hidden)
        self.node = mlp(3 + hidden, hidden, EMBEDDING)
        self.edge = mlp(2 * EMBEDDING + 1, hidden, 1)
        nn.init.zeros_(self.edge[-1].weight)
        nn.init.zeros_(self.edge[-1].bias)

    def link_metric(self, neighbours, obs, mean, variance):
        centre, scale = self.metric.normalization.unbind()
        features = torch.stack(
            ((obs - centre) / scale, (mean - centre) / scale, variance / scale**2), -1
        )
        messages = gather_avg(
            link_map(
                lambda d, other, own: self.message(torch.cat((other, own, d[:, None]), -1)),
                scatter_range(), nbr(features), features,
            ),
            fill_value=0.0,
        )  # fmt: skip
        embedding = self.node(torch.cat((features, messages), -1))

        def cost(d, a, b, other, own):
            pair = self.edge(torch.cat((own, other, d[:, None]), -1))
            swap = self.edge(torch.cat((other, own, d[:, None]), -1))
            return self.metric(d, a, b) * (pair + swap).squeeze(-1).tanh().exp()

        return link_map(cost, scatter_range(), neighbours, obs, nbr(embedding), embedding)


class GraphProgram(nn.Module):
    """Pure neural baseline: recurrent message passing with the SCR wire width.

    Same rounds, neighbours, local signals and floats per link as the SCR program;
    no IDs in its computation. It keeps one value per device, i.e. every device is
    its own sampler (a singleton region), so the leader penalty is a constant.
    """

    def __init__(self, mean=0.0, scale=1.0, hidden=32):
        super().__init__()
        self.register_buffer("normalization", torch.tensor([mean, scale]))
        self.edge = nn.Sequential(
            nn.Linear(2 * WIRE_FLOATS + 1, hidden), nn.SiLU(), nn.Linear(hidden, hidden), nn.SiLU()
        )
        self.cell = nn.GRUCell(hidden + 2, WIRE_FLOATS)
        self.head = nn.Linear(WIRE_FLOATS, 1)

    def forward(self, runtime):
        observed = runtime.signals["observation"]
        mean, scale = self.normalization.unbind()
        inputs = torch.stack(((observed - mean) / scale, runtime.signals["priority"]), -1)

        def update(old):
            messages = gather(
                link_cat((scatter(old), old, scatter_range())).map(self.edge),
                aggr="mean",
                fill_value=0.0,
                tag="gnn",
            )
            return self.cell(torch.cat((messages, inputs), -1), old)

        hidden = iterate(observed.new_zeros((len(observed), WIRE_FLOATS)), update, name="gnn")
        output = observed.new_zeros((len(observed), OUTPUT_DIM))
        output[:, LEADER] = mid().to(observed)
        output[:, ELECTED] = output[:, COUNT] = 1
        output[:, TIMESTAMP] = runtime.signals["time"]
        output[:, OBSERVATION] = output[:, TOTAL] = observed
        return torch.cat(
            (output[:, :SAMPLE], (mean + scale * self.head(hidden)), output[:, SAMPLE + 1 :]), -1
        )


def make_program(method, seed=0, *, mean=0.0, scale=1.0, weights=(2.0, 1.0, 1.0)):
    with torch.random.fork_rng():
        torch.manual_seed(seed)
        if method == "gnn":
            return GraphProgram(mean, scale)
        metric_weights, strength_weights = FIXED.get(method, (weights, (0.0, 0.0, 0.0)))
        kind = "fixed" if method in FIXED else method
        program = HybridProgram if method == "hybrid" else RegionProgram
        return program(
            EdgeMetric(kind, metric_weights, mean, scale),
            LeaderStrength(kind, strength_weights, mean, scale),
        )
