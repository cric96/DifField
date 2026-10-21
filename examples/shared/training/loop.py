"""Resumable training with selection on validation, and the checkpoints it leaves."""

import math
import time
from dataclasses import dataclass, field

import torch

from ..artifacts import tensor_write


@dataclass
class TrainingState:
    """Progress of a training job: all it needs to resume exactly after its last update."""

    initial: dict  # validation fields of the untrained model
    best_score: float  # lowest validation score so far; the earliest wins ties
    best_model: dict  # CPU weights with that score
    selected_update: int = 0  # the update that reached it; 0 for the untrained model
    step: int = 0  # completed updates
    history: list = field(default_factory=list)  # one row per update
    seconds: float = 0.0  # spent in updates and their validations
    complete: bool = False


def fit(model, optimizer, update, validate, *, updates, validate_every, last, budget):
    """Run or resume updates 1 to ``updates`` and keep the model of lowest validation score.

    Args:
        model: The model; on return it holds the selected weights.
        optimizer: The optimizer of the model, saved and restored with it.
        update: ``update(step)`` performs one update and returns its history row. It must depend
            only on ``step`` and the state of model and optimizer, so that a resumed job repeats
            the uninterrupted one exactly.
        validate: Returns the score of the model and the fields it adds to the history row. It
            runs without gradients before the first update, every ``validate_every`` updates and
            after the last.
        updates: How many updates the job makes.
        validate_every: The updates between two validations.
        last: Where the state is saved, atomically, after every update.
        budget: Checked before each update; when it runs out the job stops, resumable.

    Returns:
        The final state of the job.

    Raises:
        FloatingPointError: A validation score is not finite.
        TimeoutError: The budget ran out; ``last`` holds the completed updates.
    """
    if last.exists():
        saved = torch.load(last, weights_only=True, map_location="cpu")
        model.load_state_dict(saved.pop("model"))
        optimizer.load_state_dict(saved.pop("optimizer"))
        state = TrainingState(**saved)
    else:
        budget.check()
        score, fields = _validated(validate)
        state = TrainingState(fields, score, _cpu_state(model))
        _save(last, model, optimizer, state)
    for step in range(state.step + 1, updates + 1):
        budget.check()
        started = time.perf_counter()
        row = update(step)
        if step % validate_every == 0 or step == updates:
            score, fields = _validated(validate)
            row.update(fields)
            if score < state.best_score:
                state.best_score, state.best_model = score, _cpu_state(model)
                state.selected_update = step
        state.step = step
        state.history.append(row)
        state.seconds += time.perf_counter() - started
        _save(last, model, optimizer, state)
    state.complete = True
    _save(last, model, optimizer, state)
    model.load_state_dict(state.best_model)
    return state


def _validated(validate):
    with torch.no_grad():
        score, fields = validate()
    if not math.isfinite(score):
        raise FloatingPointError("Nonfinite validation score")
    return score, fields


def _cpu_state(model):
    return {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}


def _save(last, model, optimizer, state):
    # vars() rather than asdict(): asdict would deep-copy the history at every update.
    tensor_write(
        last, {"model": _cpu_state(model), "optimizer": optimizer.state_dict(), **vars(state)}
    )


def frozen(path, build):
    """The model of a checkpoint, in eval mode and without gradients.

    Args:
        path: A checkpoint holding the weights under ``model``.
        build: Builds the untrained model from the checkpoint.
    """
    checkpoint = torch.load(path, weights_only=True, map_location="cpu")
    model = build(checkpoint)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    model.requires_grad_(False)
    return model
