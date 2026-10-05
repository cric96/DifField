"""Smooth local flocking in an open world, separate from the historical teacher.

No flock centroid, destination or camera state enters the agent program. The
renderer follows the flock; positions are world coordinates and never wrap.
"""

from dataclasses import dataclass
import math

import torch
from torch import nn

from diffield import AggregateContext, gather_sum, link_cat, scatter

X_AXIS = 0
Y_AXIS = 1
POSITION_DIMENSIONS = 2
MINIMUM_AGENT_COUNT = 2
MERGE_GROUP_COUNT = 2
DEFAULT_HEADING_COMPONENT = 1.0
MAX_ACCELERATION_SCALE = 1.0
MINIMUM_CLOSE_NEIGHBOR_WEIGHT = 1.0

NUMERICAL_EPSILON = 1e-8
COHESION_DELTA = slice(0, 2)
ALIGNMENT_VELOCITY = slice(2, 4)
SEPARATION_REPULSION = slice(4, 6)
NEIGHBOR_WEIGHT = slice(6, 7)
CLOSE_NEIGHBOR_WEIGHT = slice(7, 8)

INITIAL_FLOCK_RADIUS = 0.25
MERGE_GROUP_OFFSET = 0.43
MERGE_HEADING_RADIANS = 0.6
MERGE_HEADING_NOISE_STD = 0.18
FLOCK_HEADING_NOISE_STD = 0.95

WIND_START_TIME = 6.0
WIND_DURATION = 3.0
WIND_CENTER = (0.85, -0.04)
WIND_INFLUENCE_WIDTH = (0.24, 0.28)
WIND_ACCELERATION = 0.12


@dataclass(frozen=True)
class FlockingConfig:
    dt: float = 1 / 60
    radius: float = 0.30
    separation: float = 0.065
    cruise_speed: float = 0.16
    min_speed: float = 0.11
    max_speed: float = 0.20
    cruise_response: float = 1.2
    separation_acceleration: float = 0.12
    max_acceleration: float = 0.35
    max_turn_rate: float = 1.3  # radians / second

    def __post_init__(self):
        if not (0 < self.dt and 0 < self.separation < self.radius):
            raise ValueError("Positive timestep and separation < radius are required")
        if not (0 < self.min_speed <= self.cruise_speed <= self.max_speed):
            raise ValueError("Expected positive min <= cruise <= max speed")
        if min(self.cruise_response, self.max_acceleration, self.max_turn_rate) <= 0:
            raise ValueError("Positive response, acceleration and turn limits are required")


WEIGHTS = (2.0, 1.5, 0.8)  # separation, alignment, cohesion; configured, not learned


def radius_edges(positions, config):
    neighbours = torch.cdist(positions.detach(), positions.detach()) < config.radius
    neighbours.fill_diagonal_(False)
    return neighbours.nonzero().T.contiguous()


def integrate(positions, velocities, steering, config, external=None):
    """Finite angular velocity and acceleration; gradual longitudinal separation."""
    speed = velocities.norm(dim=-1, keepdim=True)
    fallback = torch.zeros_like(velocities)
    fallback[:, X_AXIS] = DEFAULT_HEADING_COMPONENT
    heading = torch.where(
        speed > NUMERICAL_EPSILON,
        velocities / speed.clamp_min(NUMERICAL_EPSILON),
        fallback,
    )
    acceleration = steering + heading * (config.cruise_speed - speed) * config.cruise_response
    if external is not None:
        acceleration = acceleration + external
    magnitude = acceleration.norm(dim=-1, keepdim=True).clamp_min(NUMERICAL_EPSILON)
    acceleration = acceleration * (config.max_acceleration / magnitude).clamp(
        max=MAX_ACCELERATION_SCALE
    )
    proposed = velocities + acceleration * config.dt
    cross = heading[:, X_AXIS] * proposed[:, Y_AXIS] - heading[:, Y_AXIS] * proposed[:, X_AXIS]
    angle = torch.atan2(cross, (heading * proposed).sum(-1).clamp_min(NUMERICAL_EPSILON))
    angle = angle.clamp(-config.max_turn_rate * config.dt, config.max_turn_rate * config.dt)
    cosine, sine = angle.cos(), angle.sin()
    direction = torch.stack(
        [heading[:, X_AXIS] * cosine - heading[:, Y_AXIS] * sine,
         heading[:, X_AXIS] * sine + heading[:, Y_AXIS] * cosine],
        -1,
    )
    next_speed = proposed.norm(dim=-1, keepdim=True).clamp(config.min_speed, config.max_speed)
    velocity = direction * next_speed
    return positions + config.dt * velocity, velocity


class CoherentBoids(nn.Module):
    """Three positive local coefficients, ready for a separately specified training task."""

    def __init__(self, config=None, weights=WEIGHTS):
        super().__init__()
        self.config = config or FlockingConfig()
        if len(weights) != len(WEIGHTS) or any(w <= 0 for w in weights):
            raise ValueError("Three positive coefficients are required")
        self.log_weights = nn.Parameter(torch.tensor(weights).log())

    def messages(self, rows):
        delta = rows[..., :POSITION_DIMENSIONS]
        velocity = rows[..., POSITION_DIMENSIONS:]
        distance = delta.norm(dim=-1, keepdim=True)
        weight = (1 - (distance / self.config.radius).square()).clamp_min(0).square()
        close = (1 - distance / self.config.separation).clamp_min(0).square()
        repulsion = -delta / distance.clamp_min(NUMERICAL_EPSILON) * close
        return torch.cat([delta * weight, velocity * weight, repulsion, weight, close], -1)

    def acceleration(self, positions, velocities):
        values = gather_sum(
            link_cat([scatter(positions) - positions, scatter(velocities)]).map(
                self.messages, label="smooth_flocking_messages"
            )
        )
        total = values[:, NEIGHBOR_WEIGHT]
        cohesion = values[:, COHESION_DELTA] / total.clamp_min(NUMERICAL_EPSILON)
        cohesion = cohesion * self.config.cruise_speed / self.config.radius
        alignment = torch.where(
            total > 0,
            values[:, ALIGNMENT_VELOCITY] / total.clamp_min(NUMERICAL_EPSILON) - velocities,
            0,
        )
        separation = values[:, SEPARATION_REPULSION] / values[:, CLOSE_NEIGHBOR_WEIGHT].clamp_min(
            MINIMUM_CLOSE_NEIGHBOR_WEIGHT
        )
        separation = separation * self.config.separation_acceleration
        separation_weight, alignment_weight, cohesion_weight = self.log_weights.exp()
        return (
            separation_weight * separation
            + alignment_weight * alignment
            + cohesion_weight * cohesion
        )

    def step(self, positions, velocities, edge_index=None, *, external=None):
        edges = radius_edges(positions, self.config) if edge_index is None else edge_index
        context = AggregateContext(edges, len(positions))
        with context.round():
            acceleration = self.acceleration(positions, velocities)
        return integrate(positions, velocities, acceleration, self.config, external)


def dense_step(positions, velocities, config=None, weights=WEIGHTS, external=None):
    """Independent matrix implementation for correctness checks."""
    config = config or FlockingConfig()
    delta = positions[None, :, :] - positions[:, None, :]
    distance = delta.norm(dim=-1)
    mask = ~torch.eye(len(positions), dtype=torch.bool, device=positions.device)
    w = (1 - (distance / config.radius).square()).clamp_min(0).square() * mask
    total = w.sum(-1, keepdim=True)
    alignment = torch.where(
        total > 0,
        w @ velocities / total.clamp_min(NUMERICAL_EPSILON) - velocities,
        0,
    )
    cohesion = (w.unsqueeze(-1) * delta).sum(1) / total.clamp_min(NUMERICAL_EPSILON)
    cohesion *= config.cruise_speed / config.radius
    close = (1 - distance / config.separation).clamp_min(0).square() * mask
    separation = -(
        delta / distance.clamp_min(NUMERICAL_EPSILON).unsqueeze(-1) * close.unsqueeze(-1)
    ).sum(1)
    separation = separation / close.sum(-1, keepdim=True).clamp_min(
        MINIMUM_CLOSE_NEIGHBOR_WEIGHT
    ) * config.separation_acceleration
    separation_weight, alignment_weight, cohesion_weight = weights
    force = (
        separation_weight * separation
        + alignment_weight * alignment
        + cohesion_weight * cohesion
    )
    return integrate(positions, velocities, force, config, external)


def initial_state(scene, nodes=144, seed=3, config=None):
    config = config or FlockingConfig()
    if scene not in ("flock", "merge") or nodes < MINIMUM_AGENT_COUNT:
        raise ValueError("Expected flock or merge, with at least two agents")
    generator = torch.Generator().manual_seed(seed)
    u = torch.rand(nodes, POSITION_DIMENSIONS, generator=generator)
    angle = u[:, X_AXIS] * 2 * math.pi
    radius = u[:, Y_AXIS].sqrt() * INITIAL_FLOCK_RADIUS
    positions = torch.stack([radius * angle.cos(), radius * angle.sin()], -1)
    group = torch.arange(nodes) >= nodes // MERGE_GROUP_COUNT
    if scene == "merge":
        positions[:, Y_AXIS] += torch.where(group, -MERGE_GROUP_OFFSET, MERGE_GROUP_OFFSET)
        heading = torch.where(group, MERGE_HEADING_RADIANS, -MERGE_HEADING_RADIANS)
        heading = heading + torch.randn(nodes, generator=generator) * MERGE_HEADING_NOISE_STD
    else:
        heading = torch.randn(nodes, generator=generator) * FLOCK_HEADING_NOISE_STD
    velocities = config.cruise_speed * torch.stack([heading.cos(), heading.sin()], -1)
    return positions, velocities, group


def wind(positions, time):
    """A labelled local environmental disturbance; no flock state or target used."""
    pulse = (
        math.sin(math.pi * (time - WIND_START_TIME) / WIND_DURATION) ** 2
        if WIND_START_TIME < time < WIND_START_TIME + WIND_DURATION
        else 0.0
    )
    center_x, center_y = WIND_CENTER
    width_x, width_y = WIND_INFLUENCE_WIDTH
    strength = torch.exp(
        -((positions[:, X_AXIS] - center_x) / width_x).square()
        - ((positions[:, Y_AXIS] - center_y) / width_y).square()
    )
    acceleration = torch.zeros_like(positions)
    acceleration[:, Y_AXIS] = WIND_ACCELERATION * pulse * strength
    return acceleration


@torch.no_grad()
def simulate(scene, *, seconds=20, nodes=144, seed=3, config=None):
    config = config or FlockingConfig()
    model = CoherentBoids(config)
    p, v, group = initial_state(scene, nodes, seed, config)
    positions, velocities = [p], [v]
    for step in range(round(seconds / config.dt)):
        gust = wind(p, step * config.dt) if scene == "flock" else None
        p, v = model.step(p, v, external=gust)
        positions.append(p); velocities.append(v)
    return {"positions": torch.stack(positions), "velocities": torch.stack(velocities),
            "group": group, "scene": scene, "seed": seed, "config": vars(config)}


def diagnostics(replay):
    p, v = replay["positions"], replay["velocities"]
    speed = v.norm(dim=-1)
    direction = v / speed.unsqueeze(-1).clamp_min(NUMERICAL_EPSILON)
    coherence = direction.mean(1).norm(dim=-1)
    cross = (
        direction[:-1, :, X_AXIS] * direction[1:, :, Y_AXIS]
        - direction[:-1, :, Y_AXIS] * direction[1:, :, X_AXIS]
    )
    turn = torch.atan2(cross, (direction[:-1] * direction[1:]).sum(-1)).abs() / replay["config"]["dt"]
    spacing = []
    for positions in p:
        d = torch.cdist(positions, positions)
        d.fill_diagonal_(float("inf"))
        spacing.append(float(d.min(1).values.median()))
    return {"coherence": coherence.tolist(), "median_nearest_distance": spacing,
            "mean_speed": speed.mean(1).tolist(), "max_turn_rate": float(turn.max()),
            "min_speed": float(speed.min()), "max_speed": float(speed.max()),
            "finite": bool(torch.isfinite(p).all() and torch.isfinite(v).all())}
