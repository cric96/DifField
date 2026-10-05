"""Offline Adam/search, hard validation selection, atomic per-update resumption.

Training differentiates a relaxed forward of the program with one temperature per
loss term. Both were calibrated against hard central differences on validation
data: the error term agrees at a sharp temperature, the leader term at a softer
one (sharp relaxations of the leader term explode through time). Selection and
every reported number use the hard program.
"""

import copy
import time

import torch

from diffield import with_mode

from ..artifacts import json_write, tensor_write
from ..randomness import rng
from .data import combine
from .execution import central
from .program import ELECTED, SAMPLE, make_program

POPULATION, ELITES = 24, 6  # CEM generation size and refit set


def squared_error(episode, trace, scale):
    error = (trace.fields[..., SAMPLE] - episode.truth).square()
    if episode.importance is not None:
        error = error * episode.importance
    return error[episode.active].mean() / scale**2


def objective(episode, trace, scale, penalty):
    leaders = trace.fields[..., ELECTED][episode.active].mean()
    return squared_error(episode, trace, scale) + penalty * leaders


def surrogate_objective(episode, model, scale, penalty, config):
    """Relaxed error at ``error_temperature`` plus relaxed leaders at ``leader_temperature``.

    The relaxed error slope is ``error_gain`` times weaker than the hard central
    difference (fitted on validation data); the gain restores the balance of the terms.
    """
    with with_mode("soft", tau=config.error_temperature):
        trace = central(episode, model)
    error = squared_error(episode, trace, scale)
    if not hasattr(model, "metric"):
        return error + penalty * trace.fields[..., ELECTED][episode.active].mean()
    if penalty:
        with with_mode("soft", tau=config.leader_temperature):
            trace = central(episode, model)
    return config.error_gain * error + penalty * trace.fields[..., ELECTED][episode.active].mean()


@torch.no_grad()
def validate(model, episodes, norm, penalty, batch_size):
    """Hard validation objective and its two terms: error and leader fraction."""
    error = leaders = 0.0
    for offset in range(0, len(episodes), batch_size):
        selected = episodes[offset : offset + batch_size]
        episode = combine(selected)
        trace = central(episode, model)
        error += float(squared_error(episode, trace, norm["scale"])) * len(selected)
        leaders += float(trace.fields[..., ELECTED][episode.active].mean()) * len(selected)
    error, leaders = error / len(episodes), leaders / len(episodes)
    return {"objective": error + penalty * leaders, "error": error, "leaders": leaders}


def history_row(scores):
    return {
        "validation_hard": scores["objective"],
        "validation_error": scores["error"],
        "validation_leaders": scores["leaders"],
    }


def component_norms(model):
    """Gradient norm per program block (metric = G/C, strength = S).

    Recorded after clipping, which preserves their ratio.
    """
    return {
        f"gradient_norm_{name}": float(
            torch.cat([p.grad.flatten() for p in module.parameters() if p.grad is not None]).norm()
        )
        for name in ("metric", "strength")
        if (module := getattr(model, name, None)) is not None
    }


def candidate(model, draws, box):
    """Unit-cube draws to metric and strength weights: the box of search and CEM.

    ``box`` holds the half-widths of log metric weights (around the default) and
    of strength weights.
    """
    knots, (metric, strength) = model.metric.knots, box
    log_weights = metric * (2 * draws[: 3 * knots] - 1)
    weights = torch.tensor((2.0, 1.0, 1.0)).repeat(knots) * log_weights.exp()
    strength = strength * (2 * draws[3 * knots :] - 1)
    model.metric.fixed_weights.copy_(weights)
    model.strength.weights.copy_(strength)
    return {"weights": weights.tolist(), "strength": strength.tolist()}


def cem_draw(state, dimension, seed, step):
    """Member of the current generation from a diagonal Gaussian over the unit cube."""
    cem = state.setdefault("cem", {"mean": [0.5] * dimension, "std": [0.25] * dimension})
    generation, member = divmod(step - 1, POPULATION)
    noise = torch.randn(POPULATION, dimension, generator=rng(seed, "cem", generation))[member]
    return (torch.tensor(cem["mean"]) + torch.tensor(cem["std"]) * noise).clamp(0, 1)


def cem_update(state, draws, score):
    """After a full generation, refit mean and std to its elites."""
    cem = state["cem"]
    cem.setdefault("scored", []).append((score, draws.tolist()))
    if len(cem["scored"]) == POPULATION:
        elites = torch.tensor([d for _, d in sorted(cem["scored"])[:ELITES]])
        cem.update(mean=elites.mean(0).tolist(), std=elites.std(0).clamp_min(0.02).tolist())
        cem["scored"] = []


def train_job(directory, config, bank, norm, method, seed, penalty, budget, *, knots=1):  # noqa: PLR0912, PLR0917, PLR0915 -- one atomic resumable update loop
    directory.mkdir(parents=True, exist_ok=True)
    model = make_program(method, seed, **norm, knots=knots)
    last_path = directory / "last.pt"
    rate = config.gnn_learning_rate if method == "gnn" else config.learning_rate
    black_box = method in ("search", "cem")
    optimizer = None if black_box else torch.optim.Adam(model.parameters(), lr=rate)
    if last_path.exists():
        state = torch.load(last_path, weights_only=True)
        model.load_state_dict(state["model"])
        if optimizer is not None:
            optimizer.load_state_dict(state["optimizer"])
    else:
        budget.check()
        scores = validate(model, bank["validation"], norm, penalty, config.batch_size)
        score = scores["objective"]
        state = {
            "step": 0,
            "model": copy.deepcopy(model.state_dict()),
            "best_model": copy.deepcopy(model.state_dict()),
            "best_score": score,
            "selected_update": 0,
            "history": [{"step": 0, **history_row(scores)}],
            "seconds": 0.0,
            "complete": False,
        }
        if optimizer is not None:
            state["optimizer"] = optimizer.state_dict()
        tensor_write(last_path, state)
    total = config.search_candidates - 1 if black_box else config.updates
    for step in range(state["step"] + 1, total + 1):
        budget.check()
        started = time.perf_counter()
        if black_box:
            dimension = 3 * knots + 3
            draws = (
                torch.rand(dimension, generator=rng(seed, "search", step))
                if method == "search"
                else cem_draw(state, dimension, seed, step)
            )
            row = {"step": step, **candidate(model, draws, config.search_box)}
        else:
            indices = torch.randint(
                len(bank["train"]), (config.batch_size,), generator=rng(seed, "batch", step)
            ).tolist()
            episode = combine([bank["train"][i] for i in indices])
            optimizer.zero_grad(set_to_none=True)
            loss = surrogate_objective(episode, model, norm["scale"], penalty, config)
            if not torch.isfinite(loss):
                raise FloatingPointError("Nonfinite training loss")
            loss.backward()
            gradient_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(), 5.0, error_if_nonfinite=True
            )
            row = {
                "step": step,
                "batch_indices": indices,
                "training_surrogate": float(loss.detach()),
                "gradient_norm": float(gradient_norm),
                **component_norms(model),
            }
            optimizer.step()
            if hasattr(model, "metric"):
                with torch.no_grad():
                    model.metric.log_weights.clamp_(-8, 8)
                    model.strength.weights.clamp_(-8, 8)
        if black_box or step % config.validate_every == 0 or step == total:
            scores = validate(model, bank["validation"], norm, penalty, config.batch_size)
            score = scores["objective"]
            if not torch.isfinite(torch.tensor(score)):
                raise FloatingPointError("Nonfinite hard validation loss")
            row.update(history_row(scores))
            if method == "cem":
                cem_update(state, draws, score)
            if score < state["best_score"]:
                state.update(
                    best_score=score,
                    best_model=copy.deepcopy(model.state_dict()),
                    selected_update=step,
                )
        state.update(step=step, model=copy.deepcopy(model.state_dict()), complete=False)
        if optimizer is not None:
            state["optimizer"] = optimizer.state_dict()
        state["history"].append(row)
        state["seconds"] += time.perf_counter() - started
        tensor_write(last_path, state)
    state["complete"] = True
    tensor_write(last_path, state)
    checkpoint = {
        "method": method,
        "seed": seed,
        "lambda": penalty,
        "knots": knots,
        "normalization": norm,
        "model": state["best_model"],
        "selected_update": state["selected_update"],
        "validation_hard": state["best_score"],
        "completed_updates": total,
    }
    tensor_write(directory / "best.pt", checkpoint)
    json_write(
        directory / "training.json",
        {
            k: state[k]
            for k in ("step", "best_score", "selected_update", "history", "seconds", "complete")
        },
    )
    model.load_state_dict(state["best_model"])
    return model, checkpoint


def sensitivity(model, episode, norm, penalty, config, *, step=0.15):
    """Training gradient against the hard objective, explicitly different objects.

    For each named scalar (metric log-weights, strength weights): the surrogate
    derivative and the hard central difference with ``step``. For all parameters:
    the hard change after a step against the normalized surrogate gradient.
    The hard objective is piecewise constant, so differences are finite steps.
    """
    original = copy.deepcopy(model.state_dict())
    scale = norm["scale"]

    @torch.no_grad()
    def hard():
        return float(objective(episode, central(episode, model), scale, penalty))

    base = hard()
    model.zero_grad(set_to_none=True)
    surrogate_objective(episode, model, scale, penalty, config).backward()
    gradients = [
        p.grad.clone() if p.grad is not None else torch.zeros_like(p) for p in model.parameters()
    ]
    named = []
    for name in ("metric.log_weights", "strength.weights"):
        parameter = model.get_parameter(name)
        for i in range(parameter.numel()):
            values = []
            for sign in (-1, 1):
                model.load_state_dict(original)
                with torch.no_grad():
                    parameter.view(-1)[i] += sign * step
                values.append(hard())
            named.append(
                {
                    "parameter": f"{name}[{i}]",
                    "surrogate": float(parameter.grad.view(-1)[i]),
                    "hard": (values[1] - values[0]) / (2 * step),
                }
            )
    informative = [r for r in named if abs(r["hard"]) > 1e-9]
    length = float(torch.cat([g.flatten() for g in gradients]).norm())
    descent = []
    for epsilon in (0.01, 0.03, 0.1):
        model.load_state_dict(original)
        with torch.no_grad():
            for parameter, gradient in zip(model.parameters(), gradients, strict=True):
                parameter.sub_(epsilon * gradient / max(length, 1e-12))
        descent.append({"epsilon": epsilon, "hard_change": hard() - base})
    model.load_state_dict(original)
    return {
        "hard_loss": base,
        "gradient_norm": length,
        "named": named,
        "sign_agreement": (
            sum((r["surrogate"] > 0) == (r["hard"] > 0) for r in informative) / len(informative)
            if informative
            else None
        ),
        "descent": descent,
    }


def load_program(path):
    checkpoint = torch.load(path, weights_only=True)
    model = make_program(
        checkpoint["method"],
        checkpoint["seed"],
        **checkpoint["normalization"],
        knots=checkpoint.get("knots", 1),
    )
    model.load_state_dict(checkpoint["model"])
    model.eval()
    model.requires_grad_(False)
    return model
