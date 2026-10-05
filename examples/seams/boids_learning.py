"""Models and reproducible one-step training for the Boids experiment."""

import time
from dataclasses import dataclass

import torch
from torch import nn
from torch_geometric.nn import MessagePassing

from .artifacts import json_write, read_json, tensor_write
from .boids_dynamics import RADIUS, SPEED, Boids, dense_step, integrate, radius_edges
from .randomness import rng, seed_for

# Historical RNG namespaces remain fixed to preserve initialization and minibatches.
VARIANTS = ("parametric", "gnn32", "gnn64")
REGIMES = ("observed",)
TUNING_SEED = 10001
RATES = {"parametric": (0.01, 0.03, 0.1), "gnn": (0.0003, 0.001, 0.003)}


class InteractionNetwork(MessagePassing):
    """One-hop learned interactions; no hand-coded Boids forces or oracle inputs."""

    def __init__(self, hidden):
        super().__init__(aggr="mean")
        self.edge_mlp = nn.Sequential(
            nn.Linear(5, hidden), nn.SiLU(), nn.Linear(hidden, hidden), nn.SiLU()
        )
        self.node_mlp = nn.Sequential(
            nn.Linear(hidden + 4, hidden), nn.SiLU(), nn.Linear(hidden, 2)
        )

    def forward(self, positions, velocities, edge_index):
        aggregated = self.propagate(
            edge_index, pos=positions, vel=velocities, size=(len(positions), len(positions))
        )
        # Absolute position is needed for the source teacher's isolated-node rule.
        local = torch.cat((positions, velocities / SPEED, aggregated), dim=-1)
        return SPEED * self.node_mlp(local)

    def message(self, pos_i, pos_j, vel_i, vel_j):
        relative = (pos_j - pos_i) / RADIUS
        features = torch.cat(
            (relative, (vel_j - vel_i) / SPEED, relative.norm(dim=-1, keepdim=True)), dim=-1
        )
        return self.edge_mlp(features)


class GraphBoids(nn.Module):
    def __init__(self, hidden=32):
        super().__init__()
        self.net = InteractionNetwork(hidden)

    def step(self, positions, velocities, edge_index):
        return integrate(positions, velocities, self.net(positions, velocities, edge_index))


def make_model(variant, seed):
    # Initialisation never advances the stream used for shared minibatches.
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed_for(seed, "boids-v2-model", variant))
        if variant == "parametric":
            return Boids()
        if variant in ("gnn32", "gnn64"):
            return GraphBoids(int(variant[3:]))
    raise ValueError(f"Unknown Boids variant: {variant}")


@dataclass
class ObservedBatch:
    """Measured positions and velocities are the only training inputs and targets."""

    positions: torch.Tensor
    velocities: torch.Tensor
    target_pos: torch.Tensor
    target_vel: torch.Tensor
    edge_index: torch.Tensor


def transitions(episode):
    positions = torch.cat((episode["positions"][None], episode["target_pos"][:-1]))
    velocities = torch.cat((episode["velocities"][None], episode["target_vel"][:-1]))
    nodes = positions.shape[1]
    edges = torch.cat([radius_edges(p) + t * nodes for t, p in enumerate(positions)], dim=1)
    return ObservedBatch(
        positions.flatten(0, 1),
        velocities.flatten(0, 1),
        episode["target_pos"].flatten(0, 1),
        episode["target_vel"].flatten(0, 1),
        edges,
    )


def combine(batches):
    offset, edges = 0, []
    for batch in batches:
        edges.append(batch.edge_index + offset)
        offset += len(batch.positions)
    return ObservedBatch(
        *[
            torch.cat([getattr(b, name) for b in batches])
            for name in ("positions", "velocities", "target_pos", "target_vel")
        ],
        torch.cat(edges, dim=1),
    )


def objective(model, batch):
    prediction = model.step(batch.positions, batch.velocities, batch.edge_index)
    position = (prediction.positions - batch.target_pos).square().mean() / SPEED**2
    velocity = (prediction.velocities - batch.target_vel).square().mean() / SPEED**2
    observed = position + velocity
    return observed, observed


def free_rollout(model, positions, velocities, rounds):
    """Differentiable rollout. Topology always follows the predicted positions."""
    ps, vs = [], []
    for _ in range(rounds):
        step = (
            dense_step(positions, velocities)
            if model is None
            else model.step(positions, velocities, radius_edges(positions))
        )
        positions, velocities = step.positions, step.velocities
        ps.append(positions)
        vs.append(velocities)
    return torch.stack(ps), torch.stack(vs)


def weights(model):
    return model.log_weights.exp().detach().tolist() if isinstance(model, Boids) else None


def train_job(directory, config, train, validation, variant, seed, rate, budget):
    """Resume exactly at an update boundary; checkpoint selection uses observed validation."""
    identity = {
        "variant": variant,
        "regime": "observed",
        "seed": seed,
        "rate": rate,
        "updates": config["updates"],
        "batch_size": config["batch_size"],
        "validate_every": config["validate_every"],
    }
    identity_path = directory / "config.json"
    if identity_path.exists() and read_json(identity_path) != identity:
        raise ValueError("Training job configuration changed")
    json_write(identity_path, identity)
    model = make_model(variant, seed)
    optimizer = torch.optim.Adam(model.parameters(), lr=rate)
    last, best_path = directory / "last.pt", directory / "best.pt"
    completed, history, elapsed = 0, [], 0.0
    with torch.no_grad():
        best = float(objective(model, validation)[1])
    if not torch.isfinite(torch.tensor(best)):
        raise FloatingPointError("Nonfinite initial validation")
    if last.exists():
        saved = torch.load(last, weights_only=True)
        model.load_state_dict(saved["model"])
        optimizer.load_state_dict(saved["optimizer"])
        completed, best = saved["completed"], saved["best"]
        history, elapsed = saved["history"], saved["seconds"]
        tensor_write(best_path, saved["best_checkpoint"])
    else:
        tensor_write(
            best_path, {"model": model.state_dict(), "selected_update": 0, "validation_loss": best}
        )
    started = time.perf_counter()
    best_checkpoint = torch.load(best_path, weights_only=True)
    for update in range(completed, config["updates"]):
        budget.check()
        indices = torch.randperm(len(train), generator=rng(seed, "boids-v2-batch", update))
        selected = indices[: config["batch_size"]].tolist()
        batch = combine([train[i] for i in selected])
        optimizer.zero_grad()
        total, observed = objective(model, batch)
        total.backward()
        gradients = [p.grad for p in model.parameters()]
        if not torch.isfinite(total) or any(
            g is None or not torch.isfinite(g).all() for g in gradients
        ):
            json_write(directory / "failure.json", {"update": update, "reason": "nonfinite"})
            raise FloatingPointError(f"Nonfinite training: {directory}")
        norm = float(torch.stack([g.detach().square().sum() for g in gradients]).sum().sqrt())
        optimizer.step()
        validation_loss = None
        if (update + 1) % config["validate_every"] == 0 or update + 1 == config["updates"]:
            with torch.no_grad():
                validation_loss = float(objective(model, validation)[1])
            if not torch.isfinite(torch.tensor(validation_loss)):
                raise FloatingPointError(f"Nonfinite validation: {directory}")
            if validation_loss < best:
                best = validation_loss
                best_checkpoint = {
                    "model": {k: v.detach().clone() for k, v in model.state_dict().items()},
                    "selected_update": update + 1,
                    "validation_loss": best,
                }
                tensor_write(best_path, best_checkpoint)
        history.append(
            {
                "update": update + 1,
                "train_loss": float(total.detach()),
                "observed_train_loss": float(observed.detach()),
                "validation_loss": validation_loss,
                "gradient_norm": norm,
                "weights": weights(model),
                "batch_indices": selected,
            }
        )
        tensor_write(
            last,
            {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "completed": update + 1,
                "best": best,
                "best_checkpoint": best_checkpoint,
                "history": history,
                "seconds": elapsed + time.perf_counter() - started,
            },
        )
    result = {
        **identity,
        "status": "complete",
        "history": history,
        "seconds": elapsed + time.perf_counter() - started,
        "parameters": sum(p.numel() for p in model.parameters()),
        "best_validation": best,
        "selected_update": best_checkpoint["selected_update"],
        "episodes_processed": sum(len(row["batch_indices"]) for row in history),
    }
    json_write(directory / "training.json", result)
    model.load_state_dict(best_checkpoint["model"])
    model.eval()
    return model, result
