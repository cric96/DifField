"""Offline Adam, hard validation selection, atomic per-update resumption.

Training differentiates a relaxed forward of the program with one temperature per
loss term. Both were calibrated against hard central differences on validation
data: the error term agrees at a sharp temperature, the leader term at a softer
one (sharp relaxations of the leader term explode through time). Selection and
every reported number use the hard program.
"""

import copy
import math

import torch

from diffield import with_mode

from ...shared.artifacts import json_write, tensor_write
from ...shared.randomness import rng
from ...shared.training import fit, frozen
from .data import combine
from .execution import central
from .program import ELECTED, SAMPLE, HybridProgram, make_program


def squared_error(episode, trace, scale):
    error = (trace.fields[..., SAMPLE] - episode.truth).square()
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


def scheduled_rate(rate, config, step, total):
    """Cosine decay from ``rate`` to ``final_rate_fraction * rate``."""
    final = rate * config.final_rate_fraction
    return final + (rate - final) * 0.5 * (1 + math.cos(math.pi * (step - 1) / max(total - 1, 1)))


def parameter_groups(model, rate, config):
    """The hybrid's network learns at ``network_learning_rate``; everything else at ``rate``."""
    if not isinstance(model, HybridProgram):
        return [{"params": list(model.parameters()), "lr": rate, "base_lr": rate}]
    symbolic = [*model.metric.parameters(), *model.strength.parameters()]
    network = [p for p in model.parameters() if all(p is not q for q in symbolic)]
    network_rate = config.network_learning_rate
    return [
        {"params": symbolic, "lr": rate, "base_lr": rate},
        {"params": network, "lr": network_rate, "base_lr": network_rate},
    ]


def train_job(directory, config, bank, norm, method, seed, penalty, budget):
    device = torch.device(config.device)
    with torch.device(device):
        model = _train_job(directory, config, bank, norm, method, seed, penalty, budget, device)
    return model.cpu(), torch.load(directory / "best.pt", weights_only=True)


def _train_job(directory, config, bank, norm, method, seed, penalty, budget, device):
    directory.mkdir(parents=True, exist_ok=True)
    model = make_program(method, seed, **norm).to(device)
    training = [e.to(device) for e in bank["train"]]
    validation = [e.to(device) for e in bank["validation"]]
    full_batch = config.batch_size >= len(training)
    batch = combine(training) if full_batch else None
    rate = config.gnn_learning_rate if method == "gnn" else config.learning_rate
    optimizer = torch.optim.Adam(parameter_groups(model, rate, config))
    total = config.updates

    def update(step):
        if full_batch:
            indices, episode = None, batch
        else:
            indices = torch.randint(
                len(training),
                (config.batch_size,),
                generator=rng(seed, "batch", step),
                device="cpu",
            ).tolist()
            episode = combine([training[i] for i in indices])
        for group in optimizer.param_groups:
            group["lr"] = scheduled_rate(group["base_lr"], config, step, total)
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
        return row

    def scored():
        scores = validate(model, validation, norm, penalty, config.batch_size)
        return scores["objective"], history_row(scores)

    state = fit(
        model,
        optimizer,
        update,
        scored,
        updates=total,
        validate_every=config.validate_every,
        last=directory / "last.pt",
        budget=budget,
    )
    checkpoint = {
        "method": method,
        "seed": seed,
        "lambda": penalty,
        "normalization": norm,
        "model": state.best_model,
        "selected_update": state.selected_update,
        "validation_hard": state.best_score,
        "completed_updates": total,
    }
    tensor_write(directory / "best.pt", checkpoint)
    json_write(directory / "training.json", training_record(state))
    return model


def training_record(state):
    """The training.json of a job: its history starts with the untrained model, as step 0."""
    return {
        "step": state.step,
        "best_score": state.best_score,
        "selected_update": state.selected_update,
        "history": [{"step": 0, **state.initial}, *state.history],
        "seconds": state.seconds,
        "complete": state.complete,
    }


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


def checkpoint_directory(root, method, seed):
    """Where a run at ``root`` keeps the training job of a learned ``method`` and ``seed``."""
    return root / "checkpoints" / method / f"seed{seed}"


def load_program(path):
    return frozen(path, lambda c: make_program(c["method"], c["seed"], **c["normalization"]))


def load_method(root, method, seed, norm):
    """The frozen program of ``method`` in a run at ``root``: ``fixed-<kind>`` builds a fixed
    program with ``norm``, any other method loads the checkpoint trained with ``seed``."""
    if method.startswith("fixed-"):
        model = make_program(method.removeprefix("fixed-"), seed, **norm)
        return model.eval().requires_grad_(False)
    return load_program(checkpoint_directory(root, method, seed) / "best.pt")
